"""Frozen, validation-only Lumina Phase 2. Reuses Phase 1 loaders/transforms.

Stages: --prepare (no model inference), --stability (one OLD pilot case),
--run [--resume] (NEW cases only). Statistics/reporting is a separate module.
No production imports, training, generative fills, dataset writes or test access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageFilter
from scipy.ndimage import distance_transform_edt

from artifact_sensitivity_lumina_classifiers import (
    ROOT,read_pair,resolve,sha256,define_regions,transform_image,transform_mask,
    perturb,infer,softmax,verify_original,write_json,
)
from lumina_gradcam_all_classifiers import PROFILES,MEAN,STD,load_classifier,preprocess

OUT=ROOT/"data/processed/lumina_reliability_phase2"
P1=ROOT/"data/processed/lumina_reliability_investigation"
BUSI=ROOT/"data/processed/busi_final_reliability_audit"
VAL=ROOT/"data/processed/bra_val.csv"


class AccessGuard:
    def __init__(self,out):
        self.out=out;self.allowed={VAL.resolve()};self.accessed=set();self.denied=[]
        # Prior evidence is read-only; no native BUSI/UCLM or held-out images allowed.
    def __call__(self,event,args):
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT/"data"):return
        if path.is_relative_to(self.out):return
        prior=path.is_relative_to(P1) or path.is_relative_to(BUSI)
        if path not in self.allowed and not prior:
            self.denied.append(str(path));raise PermissionError(f"Not validation/prior-evidence allowlisted: {path}")
        mode,flags=args[1:3]
        if (isinstance(mode,str) and any(x in mode for x in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            raise PermissionError(f"Source write refused: {path}")
        self.accessed.add(str(path))


def protocol(out):
    value=json.loads((out/"frozen_protocol.json").read_text(encoding="utf-8"))
    record=json.loads((out/"protocol_checksum.json").read_text(encoding="utf-8-sig"))
    checksum=sha256(out/"frozen_protocol.json")
    if checksum.upper()!=record["Hash"]:raise ValueError("Frozen protocol checksum mismatch")
    return value,checksum


def manifest(guard):
    frame=pd.read_csv(VAL,dtype={"case_id":str,"image_id":str})
    for col,subdir in (("image_path","Images"),("mask_path","Masks")):
        for raw in frame[col]:
            path=resolve(raw)
            if not path.is_relative_to(ROOT/"data/BUS_BRA/BUSBRA"/subdir):raise ValueError(f"Unexpected validation path: {path}")
            guard.allowed.add(path)
    frame["val_row"]=frame.index
    return frame


def select_new_cases(frame,pilot,max_cases=60,seed=42):
    eligible=frame[~frame.case_id.astype(str).isin(set(pilot.case_id.astype(str)))].copy()
    rng=np.random.default_rng(seed)
    shuffled=eligible.loc[rng.permutation(eligible.index)].drop_duplicates("case_id")
    counts=shuffled.birads.value_counts().sort_index().to_dict();target=min(max_cases,len(shuffled))
    quotas={k:min(3,int(n)) for k,n in counts.items()}
    if sum(quotas.values())>target:raise ValueError("Target below prespecified stratum reserve")
    remaining=target-sum(quotas.values());capacity={k:int(counts[k])-quotas[k] for k in quotas}
    if remaining:
        total=sum(capacity.values());raw={k:remaining*capacity[k]/total for k in quotas}
        floor={k:int(np.floor(raw[k])) for k in quotas}
        for k in quotas:quotas[k]+=floor[k]
        extra=target-sum(quotas.values())
        order=sorted(quotas,key=lambda k:(-(raw[k]-floor[k]),k))
        for k in order[:extra]:quotas[k]+=1
    pieces=[]
    for k in sorted(quotas):
        group=shuffled[shuffled.birads==k]
        pieces.append(group.loc[rng.permutation(group.index)[:quotas[k]]])
    selected=pd.concat(pieces)
    assert selected.case_id.nunique()==len(selected)
    assert not set(selected.case_id)&set(pilot.case_id.astype(str))
    return selected,counts,quotas


def contact(frame,out,name,per_page=12):
    for start in range(0,len(frame),per_page):
        part=frame.iloc[start:start+per_page]
        fig,axes=plt.subplots(3,4,figsize=(14,14))
        for ax in axes.flat:ax.axis("off")
        for ax,(_,row) in zip(axes.flat,part.iterrows()):
            rgb,lesion=read_pair(row);ax.imshow(rgb)
            ax.contour(lesion,levels=[.5],colors=["lime"],linewidths=.7)
            ax.set_title(f"{row.image_id} | case {row.case_id} | BI-RADS {row.birads}",fontsize=10)
        fig.tight_layout();fig.savefig(out/f"{name}_{start//per_page+1:02d}.png",dpi=125);plt.close(fig)


def prepare(frame,out,checksum):
    if (out/"selected_cases.csv").exists():raise FileExistsError("Preparation exists; refusing to overwrite")
    old=pd.read_csv(P1/"audit_rows.csv").set_index("image_id")
    pilot=pd.read_csv(P1/"selected_cases.csv",dtype={"case_id":str})
    valid=[];rows=[]
    for idx,row in frame.iterrows():
        rec=dict(image_id=row.image_id,case_id=row.case_id)
        try:
            if str(row.case_id).strip() in ("","nan") or row.birads not in (2,3,4,5):raise ValueError("Invalid case/BI-RADS")
            if str(row.pathology).lower() not in ("benign","malignant"):raise ValueError("Invalid pathology")
            prior=old.loc[row.image_id]
            ih,mh=sha256(resolve(row.image_path)),sha256(resolve(row.mask_path))
            if ih!=prior.image_sha256 or mh!=prior.mask_sha256:raise ValueError("Inputs differ from Phase 1 audited geometry; investigate before selection")
            if not bool(prior.valid):raise ValueError("Prior audit marked invalid")
            rec.update(image_sha256=ih,mask_sha256=mh,eligible=True)
            valid.append(idx)
        except Exception as exc:rec.update(eligible=False,error=f"{type(exc).__name__}: {exc}")
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/"validation_input_audit.csv",index=False)
    selected,counts,quotas=select_new_cases(frame.loc[valid],pilot)
    again,_,_=select_new_cases(frame.loc[valid],pilot)
    pd.testing.assert_frame_equal(selected,again)
    for _,row in selected.iterrows():read_pair(row)
    selected.to_csv(out/"selected_cases.csv",index=False)
    text=selected.groupby("birads",sort=True,group_keys=False).head(3)
    text.to_csv(out/"text_review_candidates.csv",index=False)
    contact(selected,out,"selected_contact");contact(text,out,"text_candidates_contact")
    summary=dict(protocol_sha256=checksum,validation_rows=len(frame),valid_rows=len(valid),pilot_cases_excluded=int(pilot.case_id.nunique()),
        eligible_new_cases=sum(counts.values()),eligible_by_birads={str(k):int(v) for k,v in counts.items()},
        quotas={str(k):int(v) for k,v in quotas.items()},selected_cases=len(selected),text_review_candidates=len(text),
        selected_manifest_sha256=sha256(out/"selected_cases.csv"),sampling_reproducible=True,
        no_pilot_case_overlap=True,no_prediction_based_selection=True)
    write_json(out/"sampling_audit.json",summary);print(json.dumps(summary,indent=2),flush=True)


def text_mask(row,rgb,config):
    result=np.zeros(rgb.shape[:2],bool);spec=config.get("images",{}).get(str(row.image_id))
    if spec is None:return result,"Not in prespecified text-review subset or no verified text"
    if not spec.get("verified") or not spec.get("review_note"):raise ValueError("Text review must be explicit")
    h,w=result.shape
    if spec["size"]!=[w,h] or spec["image_sha256"]!=sha256(resolve(row.image_path)):raise ValueError("Text geometry/checksum mismatch")
    for x0,y0,x1,y1 in spec["boxes"]:
        if not all(isinstance(v,int) for v in (x0,y0,x1,y1)) or not(0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError("Invalid text rectangle")
        result[y0:y1,x0:x1]=True
    return result,spec["review_note"]


def base_regions(rgb,lesion,row,config):
    # Existing region geometry exactly reused, without its old random controls.
    variants,skips=define_regions(rgb,lesion,row,config,42,0,.05,.10)
    return {v["region"]:v["mask"] for v in variants},skips


def descriptors(coords,gray,distance,shape):
    y,x=coords;h,w=shape;diagonal=float(np.hypot(h,w))
    pixels=gray[y,x].astype(np.float64)
    return dict(cy=float(y.mean()),cx=float(x.mean()),lesion_distance=float(distance[y,x].mean()/diagonal),
                gray_mean=float(pixels.mean()/255),gray_sd=float(pixels.std()/255))


def match_details(reference,candidate,shape):
    diag=float(np.hypot(*shape))
    centroid=float(np.hypot(candidate["cy"]-reference["cy"],candidate["cx"]-reference["cx"])/diag)
    ld=abs(candidate["lesion_distance"]-reference["lesion_distance"])
    mean=abs(candidate["gray_mean"]-reference["gray_mean"]);sd=abs(candidate["gray_sd"]-reference["gray_sd"])
    return dict(centroid_distance=centroid,lesion_distance_mismatch=ld,gray_mean_mismatch=mean,gray_sd_mismatch=sd,
        centroid_matched=centroid<=.20,lesion_distance_matched=ld<=.10,gray_mean_matched=mean<=.15,gray_sd_matched=sd<=.10,
        all_soft_matched=centroid<=.20 and ld<=.10 and mean<=.15 and sd<=.10,
        matching_cost=centroid/.20+ld/.10+mean/.15+sd/.10)


def control_candidates(rgb,lesion,mask,known_text):
    h,w=mask.shape;gray=np.asarray(Image.fromarray(rgb).convert("L"));distance=distance_transform_edt(~lesion)
    y,x=np.nonzero(mask);n=len(y)
    if n==0:raise ValueError("Empty ROI")
    reference=descriptors((y,x),gray,distance,mask.shape)
    y0,x0=int(y.min()),int(x.min());bh,bw=int(y.max()-y0+1),int(x.max()-x0+1)
    ry,rx=y-y0,x-x0;forbidden=lesion|mask|known_text
    def anchors(size,span,original):
        return sorted(set(range(0,size-span+1,max(2,round(size*.05))))|{size-span,original})
    candidates=[]
    for ay in anchors(h,bh,y0):
        for ax in anchors(w,bw,x0):
            yy,xx=ry+ay,rx+ax
            if forbidden[yy,xx].any():continue
            data=descriptors((yy,xx),gray,distance,mask.shape)
            candidates.append(dict(coords=(yy,xx),shape_exact=True,anchor=f"{ay:05d}_{ax:05d}",**match_details(reference,data,mask.shape)))
    if not candidates:
        allowed=~forbidden;available=np.flatnonzero(allowed)
        if len(available)<n:return [],reference,"Insufficient disjoint lesion/text-free area"
        yy,xx=np.unravel_index(available,mask.shape);seen=set()
        for ay in range(0,h,max(2,round(h*.05))):
            for ax in range(0,w,max(2,round(w*.05))):
                d=(yy-ay)**2+(xx-ax)**2
                order=np.argpartition(d,n-1)[:n]
                # Stable tie convention for deterministic exact-area fallback.
                cutoff=d[order].max();lower=np.flatnonzero(d<cutoff);equal=np.flatnonzero(d==cutoff)
                chosen=np.r_[lower,equal[:n-len(lower)]]
                cy,cx=yy[chosen],xx[chosen];key=hashlib.sha256(np.sort(available[chosen]).tobytes()).hexdigest()
                if key in seen:continue
                seen.add(key);data=descriptors((cy,cx),gray,distance,mask.shape)
                candidates.append(dict(coords=(cy,cx),shape_exact=False,anchor=f"{ay:05d}_{ax:05d}",**match_details(reference,data,mask.shape)))
    candidates.sort(key=lambda c:(c["matching_cost"],c["anchor"]))
    return candidates,reference,None


def matched_controls(rgb,lesion,mask,known_text,image_id,region):
    pool,reference,error=control_candidates(rgb,lesion,mask,known_text)
    if not pool:return [],error
    top=pool[:8];selected=[];used=set()
    for seed in (42,2026,3407):
        key=f"{seed}|{image_id}|{region}".encode();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],"little"))
        possible=[i for i in range(len(top)) if i not in used] or list(range(len(top)))
        idx=possible[int(rng.integers(len(possible)))];record=top[idx];reused=idx in used;used.add(idx)
        result=np.zeros_like(mask);result[record["coords"]]=True
        assert result.sum()==mask.sum() and not(result&(lesion|mask|known_text)).any()
        selected.append(dict(mask=result,seed=seed,pool_size=len(pool),top_pool_size=len(top),location_reused=reused,
            **{k:v for k,v in record.items() if k!="coords"}))
    return selected,error


def apply_fill(rgb,mask,lesion,known_text,fill,strength):
    if fill=="blur":
        radius=max(2.,float(strength)*min(mask.shape))
        replacement=np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius)))
        result=rgb.copy();result[mask]=replacement[mask];details=dict(radius_native_px=radius)
    elif fill=="local_median":
        full,details=perturb(rgb,mask,lesion|known_text,"local_median")
        result=rgb.copy()
        values=np.rint(rgb[mask].astype(np.float32)*(1-float(strength))+full[mask].astype(np.float32)*float(strength))
        if values.min()<0 or values.max()>255:raise AssertionError("Unexpected fill overflow")
        result[mask]=values.astype(np.uint8);details["blend_strength"]=float(strength)
    else:raise ValueError(fill)
    assert result.dtype==np.uint8 and result.shape==rgb.shape
    assert np.array_equal(result[~mask],rgb[~mask])
    return result,details


def load_models(device):
    models={};verification={}
    assert PROFILES["busi"].classes==("Normal","Benign","Malignant")
    for name,p in PROFILES.items():
        model,extra=load_classifier(p,device);models[name]=model
        verification[name]=dict(profile=asdict(p),checkpoint_metadata=extra,checkpoint_sha256=sha256(ROOT/"models"/p.checkpoint),strict_load=True)
    return models,verification


def stability(frame,out,checksum):
    if (out/"numerical_stability_report.json").exists():raise FileExistsError("Stability check already exists")
    row=frame[frame.image_id=="bus_0114-s"].iloc[0];rgb,lesion=read_pair(row)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");models,verification=load_models(device)
    variants=[("original",0.,rgb)]
    for fill,strengths in (("local_median",(.25,.5,1.)),("blur",(.01,.025,.05))):
        for strength in strengths:
            changed,_=apply_fill(rgb,lesion,lesion,np.zeros_like(lesion),fill,strength)
            variants.append((fill,strength,changed))
    historical=pd.read_csv(P1/"image_level_results.csv");rows=[]
    for name,model in models.items():
        profile=PROFILES[name];parity=verify_original(model,profile,row,rgb,device,False)
        tensors=[transform_image(image,profile) for _,_,image in variants]
        outputs=infer(model,tensors,device,8);probabilities=softmax(outputs)
        for i,(fill,strength,image) in enumerate(variants):
            tensor=tensors[i];manual=(tensor*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None])
            # Invert then independently renormalize, detecting range/order/multiple normalization errors.
            assert manual.min()>=-2e-6 and manual.max()<=1+2e-6
            expected=transform_image(image,profile)
            torch.testing.assert_close(tensor,expected,rtol=0,atol=0)
            error=None;reference=None
            if fill=="original" or (fill=="local_median" and strength==1) or (fill=="blur" and strength==.025):
                prior=historical[(historical.model==name)&(historical.image_id==row.image_id)&(historical.region==("original" if fill=="original" else "lesion"))&(historical.fill==("none" if fill=="original" else fill))].iloc[0]
                reference=np.array(json.loads(prior.perturbed_logits));error=float(np.abs(outputs[i]-reference).max())
                np.testing.assert_allclose(outputs[i],reference,atol=.001,rtol=.0001)
            rows.append(dict(model=name,fill=fill,strength=strength,raw_dtype=str(image.dtype),raw_shape=list(image.shape),
                raw_min=int(image.min()),raw_max=int(image.max()),tensor_shape=list(tensor.shape),tensor_min=float(tensor.min()),tensor_max=float(tensor.max()),
                logits=outputs[i].tolist(),probabilities=probabilities[i].tolist(),max_abs_logit=float(np.abs(outputs[i]).max()),
                finite=True,phase1_max_logit_error=error,original_parity=parity if i==0 else None,
                extreme=bool(np.abs(outputs[i]).max()>20 or np.abs(outputs[i]).max()>10*max(1.,float(np.abs(outputs[0]).max())))))
    report=dict(protocol_sha256=checksum,case_id=str(row.case_id),image_id=row.image_id,extension_case=False,
        image_sha256=sha256(resolve(row.image_path)),mask_sha256=sha256(resolve(row.mask_path)),models=verification,observations=rows,
        findings="Existing extreme Phase 1 logits reproduced with bounded uint8 RGB pixels, single ImageNet normalization, correct sizes and finite outputs. Evidence supports perturbation-induced out-of-distribution response rather than pixel overflow/double-normalization; internal model cause not established.",
        no_logit_clamp=True,no_weights_changed=True)
    write_json(out/"numerical_stability_report.json",report)
    print(json.dumps({name:{"original":next(r["max_abs_logit"] for r in rows if r["model"]==name and r["fill"]=="original"),
            "full_median":next(r["max_abs_logit"] for r in rows if r["model"]==name and r["fill"]=="local_median" and r["strength"]==1)} for name in models},indent=2),flush=True)


def run(frame,out,checksum,resume=False):
    if (out/"image_level_results.csv").exists():raise FileExistsError("Completed inference exists; use separate summarizer")
    selected=pd.read_csv(out/"selected_cases.csv",dtype={"case_id":str})
    pilot=pd.read_csv(P1/"selected_cases.csv",dtype={"case_id":str})
    assert len(selected)==selected.case_id.nunique() and not(set(selected.case_id)&set(pilot.case_id))
    config=json.loads((out/"text_rois.json").read_text(encoding="utf-8"));roi_sha=sha256(out/"text_rois.json")
    review_targets=pd.read_csv(out/"text_review_candidates.csv")
    assert set(config.get("images",{}))<=set(review_targets.image_id)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");models,verification=load_models(device)
    chunks=out/"chunks";chunks.mkdir(exist_ok=True)
    masks=out/"masks";masks.mkdir(exist_ok=True)
    errors=[];all_audits=[];saved=[]
    signature=dict(protocol_sha256=checksum,text_rois_sha256=roi_sha,selected_cases_sha256=sha256(out/"selected_cases.csv"),
                   checkpoints={n:v["checkpoint_sha256"] for n,v in verification.items()})
    for i,(_,row) in enumerate(selected.iterrows()):
        destination=chunks/f"{row.image_id}.csv";meta_path=chunks/f"{row.image_id}.json"
        image_hash=sha256(resolve(row.image_path));mask_hash=sha256(resolve(row.mask_path))
        if destination.exists():
            if not resume:raise FileExistsError(f"Chunk exists: {destination}; --resume verifies inputs")
            previous=json.loads(meta_path.read_text());assert previous["signature"]==signature
            assert previous["image_sha256"]==image_hash and previous["mask_sha256"]==mask_hash
            saved.append(pd.read_csv(destination));all_audits.extend(previous["region_audits"]);continue
        rgb,lesion=read_pair(row);known_text,note=text_mask(row,rgb,config);known_text &= ~lesion
        regions,skips=base_regions(rgb,lesion,row,config)
        region_audits=[];arrays=[rgb];metas=[dict(region="original",kind="original",fill="none",strength=0.,seed=42,mask=np.zeros_like(lesion),details={})]
        for region,mask in regions.items():
            assert region=="lesion" or not(mask&lesion).any()
            controls,failure=matched_controls(rgb,lesion,mask,known_text,row.image_id,region)
            repeated,_=matched_controls(rgb,lesion,mask,known_text,row.image_id,region)
            assert len(controls)==len(repeated) and all(np.array_equal(a["mask"],b["mask"]) for a,b in zip(controls,repeated))
            variants=[dict(mask=mask,kind="region",seed=42,shape_exact=True,all_soft_matched=True)]+[c|dict(kind="control") for c in controls]
            for k,v in enumerate(variants):
                mask_path=masks/f"{row.image_id}_{region}_{k}.png";Image.fromarray(v["mask"].astype(np.uint8)*255).save(mask_path)
                audit=dict(case_id=str(row.case_id),image_id=row.image_id,birads=int(row.birads),region=region,kind=v["kind"],seed=v["seed"],
                    native_pixels=int(v["mask"].sum()),native_area_fraction=float(v["mask"].mean()),mask_path=str(mask_path),
                    mask_boolean_sha256=hashlib.sha256(v["mask"].tobytes()).hexdigest(),control_failure=failure,
                    **{key:value for key,value in v.items() if key not in ("mask","kind","seed")})
                region_audits.append(audit)
                for fill,strengths in (("blur",(.0125,.025)),("local_median",(.5,1.))):
                    for strength in strengths:
                        changed,details=apply_fill(rgb,v["mask"],lesion,known_text,fill,strength)
                        arrays.append(changed);metas.append(dict(region=region,kind=v["kind"],fill=fill,strength=strength,seed=v["seed"],mask=v["mask"],details=details))
        case_rows=[]
        for name,model in models.items():
            p=PROFILES[name];check=verify_original(model,p,row,rgb,device,False)
            verification[name].setdefault("parity",{})[str(row.image_id)]=check
            tensors=[transform_image(array,p) for array in arrays];logits=infer(model,tensors,device,8);probs=softmax(logits)
            target=int(probs[0].argmax());valid=transform_mask(np.ones_like(lesion),p);mapped_lesion=transform_mask(lesion,p)
            _,reference=preprocess(resolve(row.image_path),p)
            torch.testing.assert_close(tensors[0],reference[0],rtol=0,atol=0)
            with torch.no_grad():single=model(tensors[0].unsqueeze(0).to(device)).cpu().numpy()[0]
            np.testing.assert_allclose(logits[0],single,atol=2e-5,rtol=2e-5)
            if name=="busi" and i==0:
                from audit_busi_final_reliability import original_transform
                original=original_transform()(Image.fromarray(rgb));torch.testing.assert_close(original,tensors[0],rtol=0,atol=0)
                verification[name]["recovered_training_transform_parity"]=True
            for k,meta in enumerate(metas):
                projected=transform_mask(meta["mask"],p);assert not(projected&~valid).any()
                assert meta["region"]=="lesion" or not(projected&mapped_lesion).any()
                actual_changed=(tensors[k]!=tensors[0]).any(0).numpy();assert not(actual_changed&~valid).any()
                predicted=int(probs[k].argmax());delta=float(probs[k,target]-probs[0,target])
                case_rows.append(dict(case_id=str(row.case_id),image_id=row.image_id,birads=int(row.birads),model=name,
                    analysis_role="primary" if name in ("bra","birads") else "secondary_cross_dataset",region=meta["region"],kind=meta["kind"],
                    fill=meta["fill"],strength=meta["strength"],seed=meta["seed"],original_target_index=target,
                    original_predicted_class=p.classes[target],perturbed_predicted_class=p.classes[predicted],prediction_flip=predicted!=target,
                    class_order=json.dumps(p.classes),original_logits=json.dumps(logits[0].tolist()),perturbed_logits=json.dumps(logits[k].tolist()),
                    original_probabilities=json.dumps(probs[0].tolist()),perturbed_probabilities=json.dumps(probs[k].tolist()),
                    original_score=float(probs[0,target]),perturbed_score=float(probs[k,target]),score_delta=delta,abs_score_delta=abs(delta),
                    logit_delta=float(logits[k,target]-logits[0,target]),max_abs_logit=float(np.abs(logits[k]).max()),
                    original_max_abs_logit=float(np.abs(logits[0]).max()),extreme=bool(np.abs(logits[k]).max()>20 or np.abs(logits[k]).max()>10*max(1.,float(np.abs(logits[0]).max()))),
                    native_pixels=int(meta["mask"].sum()),native_area_fraction=float(meta["mask"].mean()),model_region_pixels=int(projected.sum()),
                    model_actual_changed_pixels=int(actual_changed.sum()),model_valid_pixels=int(valid.sum()),
                    raw_dtype=str(arrays[k].dtype),raw_min=int(arrays[k].min()),raw_max=int(arrays[k].max()),
                    tensor_min=float(tensors[k].min()),tensor_max=float(tensors[k].max()),failure_status="PASSED",fill_details=json.dumps(meta["details"])))
        case_frame=pd.DataFrame(case_rows);case_frame.to_csv(destination,index=False);saved.append(case_frame)
        metadata=dict(signature=signature,image_sha256=image_hash,mask_sha256=mask_hash,region_audits=region_audits,skips=skips,text_review_note=note)
        write_json(meta_path,metadata);all_audits.extend(region_audits)
        write_json(out/"model_verification.json",verification)
        print(f"Completed NEW case {i+1}/{len(selected)} {row.image_id}; {len(case_frame)} predictions",flush=True)
    results=pd.concat(saved,ignore_index=True);results.to_csv(out/"image_level_results.csv",index=False)
    pd.DataFrame(all_audits).to_csv(out/"region_control_audit.csv",index=False)
    write_json(out/"run_metadata.json",dict(**signature,case_count=len(selected),prediction_rows=len(results),device=str(device),
        torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0) if device.type=="cuda" else None,errors=errors,
        case_counts_by_birads={str(k):int(v) for k,v in selected.birads.value_counts().sort_index().items()},
        text_case_count=int(results[results.region=="text"].case_id.nunique()),no_pilot_case_overlap=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true");mode.add_argument("--stability",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--resume",action="store_true");args=parser.parse_args();out=OUT
    _,checksum=protocol(out)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    guard=AccessGuard(out);sys.addaudithook(guard)
    stage="prepare" if args.prepare else "stability" if args.stability else "run"
    try:
        frame=manifest(guard)
        if args.prepare:prepare(frame,out,checksum)
        elif args.stability:stability(frame,out,checksum)
        else:run(frame,out,checksum,args.resume)
    except Exception as exc:
        import traceback
        write_json(out/f"{stage}_failure.json",dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        write_json(out/f"{stage}_data_access.json",dict(accessed=sorted(guard.accessed),denied=guard.denied,
            test_manifest_or_image_opened=False,scope="Python open audit hook; validation allowlist/prior research only, not OS/native tracing"))


if __name__=="__main__":main()
