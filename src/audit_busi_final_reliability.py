"""Frozen BUSI train/validation audit. Never reads a test manifest or unlisted image.

Two stages: --scan computes independent similarity evidence; --finalize consumes
an explicit assistant visual review without overwriting the scan or prior reports.
No medical diagnoses or patient identities are inferred from image similarity.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from audit_busi_original import read_manifest, TRUE_CLASSES
from busi_phash_visual_review import perceptual_hash
from lumina_gradcam_all_classifiers import PROFILES, load_classifier, preprocess, target_index_from_arg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/busi_final_reliability_audit"
ORIGINAL = Path("C:/Breast_Cancer_Project/src")
PRIOR = ROOT / "data/processed/busi_phash_visual_review"
CLASSES = tuple(TRUE_CLASSES)
CONFIRMED = "CONFIRMED VISUAL DUPLICATE"
STRONG = "STRONG DUPLICATE CANDIDATE"
UNCERTAIN = "UNCERTAIN"
DISTINCT = "SIMILAR BUT DISTINCT"


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def collect_initial_state(out):
    """Snapshot named protected sources, without opening any held-out manifest."""
    target=out/"initial_state.json"
    if target.exists():return
    files=[p for p in (ROOT/"src").rglob("*") if p.is_file() and p.suffix in (".py",".tsx",".ts",".css",".cjs")]
    files += [ROOT/n for n in ("main.py","lumina_database.db","data/processed/train.csv","data/processed/val.csv",
        "models/best_model.pth","models/uclm_v2_best_model.pth","models/bra_best_model.pth","models/bra_birads_v1_best.pth")]
    files += [ORIGINAL/n for n in ("train.py","dataset.py","evaluate.py","process_data.py")]
    save_json(target,dict(git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).splitlines(),
        hashes=[dict(Path=str(p),Hash=digest(p).upper()) for p in files]))


def collect_provenance(out):
    path=out/"provenance_audit.json"
    if path.exists():return
    original_checkpoint=ORIGINAL.parent/"models/best_model.pth"
    source=(ORIGINAL/"train.py").read_text()
    if "phase == 'val' and epoch_acc > best_acc" not in source:raise ValueError("Unverified original selection rule")
    tree=ast.parse((ORIGINAL/"process_data.py").read_text())
    categories=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="categories" for t in n.targets))
    assert categories==dict(normal=0,benign=1,malignant=2)
    checkpoint=ROOT/"models/best_model.pth"
    save_json(path,dict(recovered_original_sources={str(ORIGINAL/n):digest(ORIGINAL/n) for n in ("train.py","dataset.py","evaluate.py","process_data.py")},
        original_checkpoint=str(original_checkpoint),lumina_checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        recovered_checkpoint_sha256=digest(original_checkpoint),checkpoint_byte_identity=digest(checkpoint)==digest(original_checkpoint),
        architecture="torchvision.efficientnet_b0",classifier="Linear(1280,3)",class_mapping=categories,
        preprocessing="RGB, torchvision Resize((256,256)) bilinear, ToTensor, ImageNet mean/std",
        checkpoint_selection="highest validation accuracy; strict improvement save in original train.py",
        original_split="image-level stratified 80/10/10; initial dataframe shuffle unseeded, split calls random_state=42",
        patient_grouping="Absent; no verified patient IDs",original_evaluate_py_execution="NOT EXECUTED: test-only original evaluator inspected as source",
        original_dataset_factory_execution="NOT EXECUTED: all-split factory inspected as source; only transform AST extracted",
        lineage_caution="Recovered scripts/checkpoint support provenance; immutable original run logs unavailable"))


class DataGuard:
    def __init__(self, out):
        self.out = out.resolve(); self.allowed = set(); self.accessed = set(); self.denied = []
        self.allowed.update((ROOT / f"data/processed/{name}.csv").resolve() for name in ("train", "val"))
        self.allowed.update(p.resolve() for p in PRIOR.glob("*") if p.is_file())

    def __call__(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(ROOT / "data"): return
        if p.is_relative_to(self.out): return
        if p not in self.allowed:
            self.denied.append(str(p)); raise PermissionError(f"Not train/validation allowlisted: {p}")
        mode, flags = args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in "wax+")) or (
            isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            raise PermissionError(f"Research source write denied: {p}")
        self.accessed.add(str(p))


def records(guard):
    items = []
    for split in ("train", "val"):
        frame = read_manifest(ROOT, split+".csv")
        for i, row in frame.iterrows():
            path = row.resolved_path
            if not path.is_relative_to(ROOT/"data/raw") or "_mask" in path.name:
                raise ValueError(f"Unexpected manifest image path: {path}")
            guard.allowed.add(path)
            items.append(dict(image_index=len(items), split=split, row_index=int(i), path=str(path),
                              name=path.name, label=int(row.label), label_name=CLASSES[int(row.label)]))
    return pd.DataFrame(items)


def normalized_gray(path, size=128):
    with Image.open(path) as f:
        gray = f.convert("L").resize((size,size), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32)/255.0


def ssim(a, b):
    """Gaussian SSIM, data_range=1, K1=.01 K2=.03, sigma=1.5, population moments.

    11x11 Gaussian support (truncate=3.5), reflect boundaries, mean over interior
    excluding a 5px rim. Inputs are aligned by resizing whole frames to 128x128;
    central scores use [13:115,13:115] of that same coordinate system. No learned
    registration or optimized alignment. These scores alone never confirm identity.
    """
    a, b = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
    if a.shape != b.shape or min(a.shape) < 11: raise ValueError("SSIM geometry")
    kw = dict(sigma=1.5, truncate=3.5, mode="reflect")
    ma, mb = gaussian_filter(a,**kw), gaussian_filter(b,**kw)
    va = gaussian_filter(a*a,**kw)-ma*ma
    vb = gaussian_filter(b*b,**kw)-mb*mb
    cov = gaussian_filter(a*b,**kw)-ma*mb
    result = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return float(result[5:-5,5:-5].mean())


def feature_vector(a):
    centered = a.ravel()-a.mean()
    norm = np.linalg.norm(centered)
    return centered/max(float(norm),1e-12)


def scan(items, out):
    images, ph, dh, metadata = [], [], [], []
    for row in items.itertuples():
        with Image.open(row.path) as im:
            rgb=im.convert("RGB"); width,height=rgb.size
            pixels=np.asarray(rgb)
            small=np.asarray(rgb.convert("L").resize((9,8),Image.Resampling.LANCZOS))
        images.append(normalized_gray(row.path)); ph.append(perceptual_hash(Path(row.path)))
        dh.append((small[:,1:]>small[:,:-1]).ravel())
        metadata.append(dict(file_sha256=digest(row.path),
            decoded_rgb_sha256=hashlib.sha256(str((width,height)).encode()+pixels.tobytes()).hexdigest(),
            width=width,height=height))
    items=pd.concat([items,pd.DataFrame(metadata)],axis=1)
    items.to_csv(out/"image_inventory.csv",index=False)
    imgs=np.stack(images); ph=np.stack(ph); dh=np.stack(dh)
    full=np.stack([feature_vector(a) for a in imgs]); center=np.stack([feature_vector(a[13:115,13:115]) for a in imgs])
    corr=full@full.T; ccorr=center@center.T
    p_dist=np.count_nonzero(ph[:,None,:]!=ph[None,:,:],axis=2).astype(np.uint8)
    d_dist=np.count_nonzero(dh[:,None,:]!=dh[None,:,:],axis=2).astype(np.uint8)
    n=len(items); same_file=items.file_sha256.to_numpy()[:,None]==items.file_sha256.to_numpy()[None,:]
    same_pixels=items.decoded_rgb_sha256.to_numpy()[:,None]==items.decoded_rgb_sha256.to_numpy()[None,:]
    # Symmetric exhaustive cheap screening over all 246,051 accessible pairs.
    selected=(p_dist<=12)|(d_dist<=8)|(corr>=.94)|(ccorr>=.94)|same_file|same_pixels
    train=items.index[items.split=="train"].to_numpy(); val=items.index[items.split=="val"].to_numpy()
    # Search alternatives beyond nearest pHash: best dHash/full-NCC/central-NCC for EVERY validation image.
    for i in val:
        selected[i,train[np.argmin(d_dist[i,train])]]=True
        selected[i,train[np.argmax(corr[i,train])]]=True
        selected[i,train[np.argmax(ccorr[i,train])]]=True
    selected=selected|selected.T
    prior=pd.read_csv(PRIOR/"review_candidates.csv")
    by_path={Path(p).resolve():int(i) for i,p in enumerate(items.path)}
    previous={tuple(sorted((by_path[Path(r.validation_path)],by_path[Path(r.train_path)]))):int(r.rank) for r in prior.itertuples()}
    all_nearest=pd.read_csv(PRIOR/"all_validation_nearest_matches.csv")
    prior_checks=[]
    for row in all_nearest.itertuples():
        i=by_path[Path(row.validation_path)]; j=by_path[Path(row.train_path)]
        assert int(p_dist[i,j])==int(row.phash_distance)
        assert int(p_dist[i,train].min())==int(row.phash_distance)
        prior_checks.append(dict(validation_path=row.validation_path,verified=True))
    rows=[]
    for i,j in zip(*np.where(np.triu(selected,1))):
        a,b=items.iloc[i],items.iloc[j]
        score=ssim(imgs[i],imgs[j]); central=ssim(imgs[i,13:115,13:115],imgs[j,13:115,13:115])
        exact=bool(same_file[i,j] or same_pixels[i,j])
        strong=(score>=.94 and central>=.94 and corr[i,j]>=.97 and ccorr[i,j]>=.97 and p_dist[i,j]<=12 and d_dist[i,j]<=12)
        credible=exact or strong or p_dist[i,j]<=8 or (score>=.8 and central>=.8 and corr[i,j]>=.9 and ccorr[i,j]>=.9)
        rows.append(dict(pair_id=f"p{i:04d}_{j:04d}",a_index=int(i),b_index=int(j),a_path=a.path,b_path=b.path,
            a_split=a.split,b_split=b.split,a_label=a.label_name,b_label=b.label_name,
            a_dimensions=f"{a.width}x{a.height}",b_dimensions=f"{b.width}x{b.height}",
            cross_split=a.split!=b.split,label_conflict=a.label!=b.label,
            same_filepath=a.path==b.path,sha256_equal=bool(same_file[i,j]),decoded_pixels_equal=bool(same_pixels[i,j]),
            phash_distance=int(p_dist[i,j]),dhash_distance=int(d_dist[i,j]),ssim=score,central_ssim=central,
            ncc=float(corr[i,j]),central_ncc=float(ccorr[i,j]),prior_rank=previous.get((i,j)),
            automated_strong=bool(strong),credible_screen=bool(credible),
            similarity_classification=CONFIRMED if exact else STRONG if strong else UNCERTAIN,
            evidence="Identical file or decoded RGB pixels" if exact else "Multiple aligned metrics support candidate" if strong else "Screen only; identity unresolved",
            reviewed=False,manual_review_remaining=not exact))
    pairs=pd.DataFrame(rows)
    pairs.to_csv(out/"all_similarity_candidates.csv",index=False)
    # Save all cheap pair distances for transparent threshold analyses, not only shortlisted pairs.
    ii,jj=np.triu_indices(n,1)
    pd.DataFrame(dict(a_index=ii,b_index=jj,phash_distance=p_dist[ii,jj],dhash_distance=d_dist[ii,jj],
                      ncc=corr[ii,jj],central_ncc=ccorr[ii,jj])).to_csv(out/"all_pair_screen.csv",index=False)
    nearest=[]
    for i in val:
        j=train[np.argmin(p_dist[i,train])]
        nearest.append(dict(validation_path=items.iloc[i].path,train_path=items.iloc[j].path,
                            phash_distance=int(p_dist[i,j])))
    pd.DataFrame(nearest).to_csv(out/"validation_nearest_matches_verified.csv",index=False)
    summary=dict(accessible_images=n,train_images=len(train),validation_images=len(val),all_pairs_screened=n*(n-1)//2,
        shortlisted_pairs=len(pairs),prior_78_nearest_distances_verified=True,
        existing_30_candidates_preserved=all((pairs.prior_rank==rank).any() for rank in range(1,31)),
        nearest_le4=sum(x["phash_distance"]<=4 for x in nearest),nearest_le8=sum(x["phash_distance"]<=8 for x in nearest),
        missing_files=0,manifest_label_errors=0,
        shared_filepaths=len(set(items[items.split=="train"].path)&set(items[items.split=="val"].path)),
        equal_sha256_cross_split_pairs=int(pairs[pairs.cross_split].sha256_equal.sum()),
        equal_decoded_pixels_cross_split_pairs=int(pairs[pairs.cross_split].decoded_pixels_equal.sum()),
        candidate_cross_split_pairs=int(pairs.cross_split.sum()),candidate_cross_label_pairs=int(pairs.label_conflict.sum()),
        automated_strong_pairs=int(pairs.automated_strong.sum()),credible_screen_pairs=int(pairs.credible_screen.sum()),
        screening="All train/val pairs: pHash<=12 OR dHash<=8 OR whole/central normalized correlation>=.94, plus exact hashes and each val's best dHash/full/central NCC train match.",
        ssim_alignment="Grayscale whole-frame bilinear resize to 128x128; central crop [13:115,13:115]; no registration, same normalized coordinates.")
    save_json(out/"dataset_integrity_summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return items,pairs


class UnionFind:
    def __init__(self,n): self.parent=list(range(n))
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.parent[max(a,b)]=min(a,b)


def make_groups(items,pairs,classes):
    uf=UnionFind(len(items))
    edges=pairs[pairs.similarity_classification.isin(classes)]
    for r in edges.itertuples(): uf.union(int(r.a_index),int(r.b_index))
    members={}
    for i in range(len(items)): members.setdefault(uf.find(i),[]).append(i)
    rows=[]
    for root,ids in sorted(members.items()):
        if len(ids)<2: continue
        group=items.iloc[ids]; gid=f"g{root:04d}"
        group_edges=edges[edges.a_index.isin(ids)&edges.b_index.isin(ids)]
        for i in ids:
            row=items.iloc[i].to_dict()
            row.update(group_id=gid,group_size=len(ids),cross_split_group=group.split.nunique()>1,
                conflicting_label_group=group.label.nunique()>1,group_labels="|".join(sorted(group.label_name.unique())),
                edge_count=len(group_edges),group_confidence="confirmed edges" if set(group_edges.similarity_classification)=={CONFIRMED} else "includes strong candidate edges",
                identity_scope="image-similarity group; patient/examination identity NOT verified")
            rows.append(row)
    columns=list(items.columns)+["group_id","group_size","cross_split_group","conflicting_label_group","group_labels","edge_count","group_confidence","identity_scope"]
    return pd.DataFrame(rows,columns=columns)


def wilson(correct,n):
    if n==0: return (None,None)
    z=1.959963984540054; p=correct/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(mid-radius),float(mid+radius)


def metric_record(name,frame):
    cm=confusion_matrix(frame.label,frame.predicted_index,labels=[0,1,2]) if len(frame) else np.zeros((3,3),int)
    support=cm.sum(1); correct=int(np.trace(cm)); n=int(cm.sum())
    precision,recall,f1,_=precision_recall_fscore_support(frame.label,frame.predicted_index,labels=[0,1,2],zero_division=0) if n else (np.zeros(3),)*4
    lo,hi=wilson(correct,n)
    rec=dict(subset=name,n=n,correct=correct,accuracy=correct/n if n else None,
             balanced_accuracy=float(recall[support>0].mean()) if n else None,
             macro_f1=float(f1.mean()) if n else None,accuracy_ci_low=lo,accuracy_ci_high=hi,
             confusion_matrix=json.dumps(cm.tolist()),class_distribution=json.dumps(dict(zip(CLASSES,map(int,support)))),
             caution="Diagnostic validation reuse; not independent performance. Wilson intervals assume independent images and may understate duplicate dependence.")
    for i,c in enumerate(CLASSES):
        low,high=wilson(int(cm[i,i]),int(support[i]))
        rec.update({c.lower()+"_recall":float(recall[i]) if support[i] else None,
                    c.lower()+"_precision":float(precision[i]) if n else None,
                    c.lower()+"_support":int(support[i]),c.lower()+"_recall_ci_low":low,c.lower()+"_recall_ci_high":high})
    return rec


def extracted_namespace(path,names,namespace):
    tree=ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes=[]
    for node in tree.body:
        is_function=isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names
        assigned=isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)
        if is_function or assigned:
            node=copy.deepcopy(node)
            if hasattr(node,"decorator_list"): node.decorator_list=[]
            nodes.append(node)
    if len(nodes)!=len(names): raise ValueError(f"Requested AST definitions not all found: {names}")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),namespace)
    return namespace


def original_transform():
    tree=ast.parse((ORIGINAL/"dataset.py").read_text())
    loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_dataloaders")
    assignment=next(n for n in loader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="data_transforms" for t in n.targets))
    namespace={"transforms":transforms,"IMAGE_SIZE":256}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),"original_transform","exec"),namespace)
    return namespace["data_transforms"]


def run_immediate_coroutine(coroutine):
    """Drive isolated in-memory Upload.read without creating Windows socket loops.

    Real network I/O is forbidden in this harness; any yielded awaitable is an error.
    The production async function body is unchanged and executes to completion.
    """
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    finally:
        coroutine.close()
    raise RuntimeError("Endpoint unexpectedly requested real asynchronous I/O")


def inference(items,out):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,metadata=load_classifier(PROFILES["busi"],device)
    assert PROFILES["busi"].classes==CLASSES
    direct_transform=original_transform()
    backend=extracted_namespace(ROOT/"main.py",{"BUSI_CLASSES","transform","_build_model","predict_image"},dict(
        torch=torch,nn=nn,efficientnet_b0=efficientnet_b0,Path=Path,transforms=transforms,Image=Image,io=io,
        device=device,UploadFile=object,File=lambda *args,**kwargs:None,HTTPException=ValueError,Any=object))
    assert tuple(backend["BUSI_CLASSES"])==CLASSES
    backend_model=backend["_build_model"](ROOT/"models/best_model.pth",list(CLASSES))
    backend["busi_model"]=backend_model
    # The actual endpoint is executed without production imports/DB or other model loads.
    class Stub(nn.Module):
        def __init__(self,n,k): super().__init__(); self.n=n; self.k=k
        def forward(self,x):
            result=torch.full((len(x),self.n),-2.,device=x.device);result[:,self.k]=2.;return result
    backend.update(uclm_model=Stub(3,1),bra_model=Stub(2,0),UCLM_CLASSES=list(CLASSES),BRA_CLASSES=["Benign","Malignant"])
    captures=[]
    backend["_heatmap_and_cam"]=lambda image,pred,batch,index:(captures.append((pred,index)) or ("",None))
    class Upload:
        def __init__(self,data): self.data=data
        async def read(self): return self.data
    rows=[]; max_logit_error=0.;max_probability_error=0.
    val=items[items.split=="val"]
    baseline={}
    with torch.no_grad():
        for start in range(0,len(items),16):
            block=items.iloc[start:start+16]; tensors=[]
            for r in block.itertuples():
                with Image.open(r.path) as im:tensors.append(direct_transform(im.convert("RGB")))
            values=model(torch.stack(tensors).to(device)).cpu()
            if not torch.isfinite(values).all():raise ValueError("Nonfinite frozen logits")
            baseline.update({int(idx):logits for idx,logits in zip(block.image_index,values)})
    for number,row in enumerate(items.itertuples()):
        with Image.open(row.path) as im: rgb=im.convert("RGB")
        a=direct_transform(rgb).unsqueeze(0);b=backend["transform"](rgb).unsqueeze(0)
        _,unified=preprocess(Path(row.path),PROFILES["busi"])
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(a,unified,rtol=0,atol=0)
        logits=baseline[int(row.image_index)];prob=logits.softmax(0);idx=int(prob.argmax())
        record=dict(image_index=row.image_index,split=row.split,path=row.path,name=row.name,label=row.label,label_name=row.label_name,
            predicted_index=idx,predicted_class=CLASSES[idx],correct=idx==row.label,
            score=float(prob[idx]),logits=json.dumps(logits.cpu().tolist()),
            probabilities=json.dumps({c:float(prob[i]) for i,c in enumerate(CLASSES)}))
        if row.split=="val":
            with torch.no_grad(): app_logits=backend_model(b.to(device))[0].cpu()
            max_logit_error=max(max_logit_error,float((app_logits-logits).abs().max()))
            torch.testing.assert_close(logits,app_logits,rtol=1e-5,atol=1e-5)
            response=run_immediate_coroutine(backend["predict_image"](Upload(Path(row.path).read_bytes())))
            app=response["models"]["BUSI"]
            assert app["prediction"]==CLASSES[idx] and captures[-1]==(CLASSES[idx],idx)
            for c,p in json.loads(record["probabilities"]).items():
                max_probability_error=max(max_probability_error,abs(app["probabilities"][c]-p))
                assert abs(app["probabilities"][c]-p)<1e-5
            record["backend_predicted_class"]=app["prediction"]
        rows.append(record)
        if (number+1)%100==0: print(f"Frozen inference {number+1}/{len(items)}",flush=True)
    predictions=pd.DataFrame(rows)
    predictions[predictions.split=="val"].to_csv(out/"validation_predictions.csv",index=False)
    predictions[predictions.split=="val"].to_json(out/"frontend_validation_inputs.json",orient="records",indent=2)
    predictions.to_csv(out/"all_accessible_predictions.csv",index=False)
    for i,c in enumerate(CLASSES): assert target_index_from_arg(c,PROFILES["busi"],0)==i
    # Verify actual consensus uses names rather than assuming numeric orders.
    consensus_tests=[]
    actual=backend["busi_model"]
    sample=Path(val.iloc[0].path).read_bytes()
    for k in range(3):
        backend["busi_model"]=Stub(3,k)
        backend["uclm_model"]=Stub(3,k)
        backend["bra_model"]=Stub(2,max(0,k-1))
        response=run_immediate_coroutine(backend["predict_image"](Upload(sample)))
        assert response["consensus"]["prediction"]==CLASSES[k]
        assert captures[-1]==(CLASSES[k],k)
        consensus_tests.append(dict(input_index=k,label=CLASSES[k],consensus=response["consensus"]["prediction"],passed=True))
    backend["busi_model"]=actual
    source=(ROOT/"src/App.tsx").read_text(encoding="utf-8")
    display=(ROOT/"src/components/AnalysisResults.tsx").read_text(encoding="utf-8")
    audit=dict(correct_classes=list(CLASSES),backend_class_mapping="PASSED",unified_and_artifact_engine_mapping="PASSED (artifact script imports unified profiles)",
        original_backend_unified_tensor_exact_match_images=len(items),validation_actual_backend_endpoint_cases=len(val),
        backend_logit_max_error=max_logit_error,backend_probability_max_error=max_probability_error,
        gradcam_target_names_and_indices="PASSED all validation endpoint captures and all 3 explicit label targets",
        consensus_name_mapping_tests=consensus_tests,
        frontend_static_label_display="PASSED" if "{result.prediction}" in display and "consensusPrediction ?? data.prediction" in source else "FAILED",
        frontend_runtime="NOT EXECUTED; static source inspection and isolated response adapter test recorded separately",
        actual_production_endpoint_network_call="NOT EXECUTED; AST-extracted actual endpoint used with real BUSI and stub other models/heatmap",
        database="Never imported main.py; init_db and patient endpoints never executed",
        existing_nonmapping_findings=["All-distinct consensus ties use max on a set-derived dictionary; tie resolution is not explicitly specified.",
            "App.tsx associates consensus prediction with BUSI score when constructing/saving analysis; individual model labels are correctly keyed. Score semantics require separate review.",
            "DiagnosticReport.tsx has explicitly labeled DEMO_REPORT_MODE=true; the demo report is not an inference output."],
        production_code_modified=False)
    save_json(out/"application_inference_audit.json",audit)
    return predictions,audit


def review_sheets(items,pairs,out):
    folder=out/"duplicate_review_contact_sheets";folder.mkdir(exist_ok=True)
    review=pairs[pairs.credible_screen | pairs.label_conflict & pairs.automated_strong | pairs.prior_rank.notna()].copy()
    review=review.sort_values(["prior_rank","ssim"],ascending=[True,False],na_position="last")
    for start in range(0,len(review),4):
        group=review.iloc[start:start+4]
        fig,axes=plt.subplots(len(group),3,figsize=(14,3.6*len(group)),squeeze=False)
        for i,row in enumerate(group.itertuples()):
            a=normalized_gray(row.a_path,256);b=normalized_gray(row.b_path,256)
            axes[i,0].imshow(a,cmap="gray",vmin=0,vmax=1)
            axes[i,1].imshow(b,cmap="gray",vmin=0,vmax=1)
            axes[i,2].imshow((a+b)/2,cmap="gray",vmin=0,vmax=1)
            axes[i,0].set_title(f"{row.pair_id} | {row.a_split}: {Path(row.a_path).name}\n{row.a_label}, {row.a_dimensions}",fontsize=10)
            axes[i,1].set_title(f"{row.b_split}: {Path(row.b_path).name}\n{row.b_label}, {row.b_dimensions}",fontsize=10)
            axes[i,2].set_title(f"50% overlay | p/d {row.phash_distance}/{row.dhash_distance}\nSSIM {row.ssim:.3f}, center {row.central_ssim:.3f}",fontsize=10)
            for ax in axes[i]:ax.axis("off")
        fig.suptitle("Train/validation similarity evidence; normalized alignment, no diagnosis or patient inference")
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(folder/f"review_{start//4+1:03d}.png",dpi=125);plt.close(fig)
    review[["pair_id","a_path","b_path","prior_rank","ssim","central_ssim"]].to_csv(out/"visual_review_queue.csv",index=False)
    print(f"Contact sheets: {len(review)} pairs / {(len(review)+3)//4} pages",flush=True)


def plots(metrics,out):
    fig,axes=plt.subplots(2,3,figsize=(14,9)); names=CLASSES
    for ax,row in zip(axes.flat,metrics.itertuples()):
        cm=np.array(json.loads(row.confusion_matrix));ax.imshow(cm,cmap="Blues")
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
        ax.set_xticks(range(3),names,rotation=15);ax.set_yticks(range(3),names)
        ax.set_xlabel("Predicted");ax.set_ylabel("Manifest label");ax.set_title(f"{row.subset}\nn={row.n}, accuracy {row.accuracy:.1%}",fontsize=10)
    for ax in axes.flat[len(metrics):]:ax.axis("off")
    fig.suptitle("Frozen BUSI validation diagnostic subsets; no independent test evaluation")
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/"validation_confusion_matrices.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(metrics))
    mean=metrics.accuracy.to_numpy(float);lo=metrics.accuracy_ci_low.to_numpy(float);hi=metrics.accuracy_ci_high.to_numpy(float)
    ax.bar(x,mean*100,color="#246c91");ax.errorbar(x,mean*100,yerr=np.array([mean-lo,hi-mean])*100,fmt="none",color="black",capsize=5)
    ax.set_xticks(x,[f"{r.subset}\nn={r.n}" for r in metrics.itertuples()],rotation=12,ha="right")
    ax.set_ylim(0,110);ax.set_ylabel("Accuracy (%)");ax.set_title("Diagnostic validation reuse; nominal image-level Wilson 95% intervals")
    fig.tight_layout();fig.savefig(out/"accuracy_by_contamination_group.png",dpi=150);plt.close(fig)


def finalize(items,out,review_file):
    pairs=pd.read_csv(out/"all_similarity_candidates.csv")
    raw_pairs=out/"similarity_screen_candidates.csv"
    if not raw_pairs.exists():raw_pairs.write_bytes((out/"all_similarity_candidates.csv").read_bytes())
    reviews=json.loads(review_file.read_text(encoding="utf-8"))
    for pid,review in reviews["pairs"].items():
        selected=pairs.pair_id==pid
        if selected.sum()!=1:raise ValueError(f"Unknown/nonunique reviewed pair {pid}")
        if review["classification"] not in (CONFIRMED,STRONG,DISTINCT,UNCERTAIN) or not review.get("evidence"):
            raise ValueError("Review requires classification and concrete visual evidence")
        pairs.loc[selected,"similarity_classification"]=review["classification"]
        pairs.loc[selected,"evidence"]=review["evidence"]
        pairs.loc[selected,"reviewed"]=True
        pairs.loc[selected,"manual_review_remaining"]=review["classification"] in (STRONG,UNCERTAIN)
    pairs.to_csv(out/"reviewed_similarity_candidates.csv",index=False)
    pairs.to_csv(out/"all_similarity_candidates.csv",index=False)
    # Preserve raw screen evidence separately; canonical file now contains reviewed classifications.
    confirmed=make_groups(items,pairs,{CONFIRMED})
    groups=make_groups(items,pairs,{CONFIRMED,STRONG})
    groups.to_csv(out/"duplicate_groups.csv",index=False)
    cross=groups[groups.cross_split_group];cross.to_csv(out/"cross_split_duplicate_groups.csv",index=False)
    conflict=groups[groups.conflicting_label_group].copy()
    predictions=pd.read_csv(out/"all_accessible_predictions.csv")
    conflict=conflict.merge(predictions[["image_index","predicted_class","score","probabilities"]],on="image_index",how="left")
    conflict["diagnosis_status"]="UNRESOLVED: expert/annotation-source review required"
    conflict.to_csv(out/"conflicting_label_groups.csv",index=False)
    quarantine=conflict.copy();quarantine["reason"]="Similarity group has conflicting manifest labels; do not automatically relabel"
    quarantine.to_csv(out/"quarantine_manifest.csv",index=False)
    val=predictions[predictions.split=="val"].copy()
    confirmed_affected=set(confirmed[(confirmed.split=="val")&confirmed.cross_split_group].image_index)
    strong_affected=set(cross[cross.split=="val"].image_index)
    ph=pairs[pairs.cross_split & (pairs.phash_distance<=8)]
    ph_affected=set(ph.a_index)&set(val.image_index) | set(ph.b_index)&set(val.image_index)
    upper_pairs=pairs.copy()
    upper_pairs.loc[(upper_pairs.similarity_classification==UNCERTAIN)&upper_pairs.credible_screen,"similarity_classification"]=STRONG
    upper_groups=make_groups(items,upper_pairs,{CONFIRMED,STRONG})
    upper_affected=set(upper_groups[(upper_groups.split=="val")&upper_groups.cross_split_group].image_index)
    subsets={
        "A all validation":val,
        "B confirmed contaminated":val[val.image_index.isin(confirmed_affected)],
        "C no confirmed duplicate":val[~val.image_index.isin(confirmed_affected)],
        "D exclude confirmed+strong":val[~val.image_index.isin(strong_affected)],
        "E exclude all pHash<=8":val[~val.image_index.isin(ph_affected)],
    }
    metrics=pd.DataFrame([metric_record(name,frame) for name,frame in subsets.items()])
    metrics.to_csv(out/"validation_subset_metrics.csv",index=False)
    plots(metrics,out)
    # One validation observation per similarity group, deterministic first path, sensitivity only.
    group_lookup=dict(zip(groups.image_index,groups.group_id))
    cluster=val.copy();cluster["cluster_id"]=[group_lookup.get(i,f"single{i}") for i in cluster.image_index]
    representatives=cluster.sort_values("path").drop_duplicates("cluster_id")
    pd.DataFrame([metric_record("one val image per similarity group",representatives)]).to_csv(out/"cluster_representative_metrics.csv",index=False)
    threshold_rows=[]
    screen=pd.read_csv(out/"all_pair_screen.csv")
    for threshold in (4,8,12):
        e=screen[screen.phash_distance<=threshold].copy();e["similarity_classification"]=STRONG
        g=make_groups(items,e,{STRONG})
        affected=set(g[(g.split=="val")&g.cross_split_group].image_index)
        threshold_rows.append(dict(threshold=threshold,edge_count=len(e),groups=int(g.group_id.nunique()),
            cross_split_groups=int(g[g.cross_split_group].group_id.nunique()),affected_validation=len(affected),
            caution="pHash-only SCREEN groups, not verified duplicates or patient groups"))
    pd.DataFrame(threshold_rows).to_csv(out/"threshold_sensitivity.csv",index=False)
    integrity=json.loads((out/"dataset_integrity_summary.json").read_text())
    raw_summary=out/"similarity_scan_summary.json"
    if not raw_summary.exists():raw_summary.write_bytes((out/"dataset_integrity_summary.json").read_bytes())
    integrity.update(confirmed_groups=int(confirmed.group_id.nunique()),confirmed_strong_groups=int(groups.group_id.nunique()),
        cross_split_confirmed_groups=int(confirmed[confirmed.cross_split_group].group_id.nunique()),
        cross_split_confirmed_strong_groups=int(cross.group_id.nunique()),
        confirmed_affected_validation=len(confirmed_affected),confirmed_strong_affected_validation=len(strong_affected),
        credible_screen_upper_affected_validation=len(upper_affected),affected_validation_percent=100*len(strong_affected)/len(val),
        conflicting_label_groups=int(conflict.group_id.nunique()),conflicting_label_images=len(conflict),
        train_only_groups=int(groups[~groups.cross_split_group & (groups.split=="train")].group_id.nunique()),
        val_only_groups=int(groups[~groups.cross_split_group & (groups.split=="val")].group_id.nunique()),
        largest_group=int(groups.group_size.max()) if len(groups) else 0,
        reviewed_pairs=int(pairs.reviewed.sum()),all_30_prior_candidates_reviewed=bool(pairs[pairs.prior_rank.notna()].reviewed.all()),
        visual_review_scope=reviews.get("reviewer","unspecified"),
        contamination_bounds_caution="Operational lower/upper screen estimates, not statistical bounds; missed transformations/false merges remain possible.",
        by_class={c:dict(total=int((val.label_name==c).sum()),confirmed=sum((val.label_name==c)&val.image_index.isin(confirmed_affected)),
            confirmed_or_strong=sum((val.label_name==c)&val.image_index.isin(strong_affected))) for c in CLASSES})
    # Separate finalized summary to preserve original scan evidence.
    save_json(out/"final_dataset_integrity_summary.json",integrity)
    save_json(out/"dataset_integrity_summary.json",integrity)
    save_json(out/"busi_final_decision.json",dict(decision="B",title="RETAIN BUSI, BUT EXCLUDE IT FROM AUTOMATED CONSENSUS",
        rationale="Frozen model and corrected integration reproduce validation results, but cross-split duplicates, conflicting labels and validation checkpoint-selection reuse prevent independent reliability claims.",
        current_checkpoint="Preserve as historical research baseline",retraining_authorized=False,
        retraining_justified="A future leakage-controlled replacement study is reasonable after annotation review, but this audit does not establish that a replacement is necessary or sufficient.",
        production_consensus_changed=False,phase2="May resume with corrected mapping, regenerated BUSI provenance, and BUSI reported separately/excluded from reliability-dependent automated consensus. Application consensus exclusion requires a separate product change; current code still includes BUSI."))
    print(metrics[["subset","n","correct","accuracy","balanced_accuracy","macro_f1","malignant_recall"]].to_string(index=False),flush=True)
    print(json.dumps(integrity,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--scan",action="store_true");mode.add_argument("--resume-scan",action="store_true");mode.add_argument("--finalize",action="store_true")
    p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--visual-review",type=Path)
    args=p.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(OUT):p.error("Output must stay inside isolated audit root")
    out.mkdir(parents=True,exist_ok=True)
    if args.scan and (out/"all_similarity_candidates.csv").exists():raise FileExistsError("Scan already exists; use a fresh output subfolder")
    if args.finalize and (out/"duplicate_groups.csv").exists():raise FileExistsError("Finalized outputs exist; use fresh output folder")
    if args.scan:collect_initial_state(out)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    guard=DataGuard(out);sys.addaudithook(guard)
    try:
        manifest_items=records(guard)
        if args.scan or args.resume_scan:
            if args.resume_scan:
                if (out/"validation_predictions.csv").exists():raise FileExistsError("Inference already completed; refusing to overwrite")
                items=pd.read_csv(out/"image_inventory.csv");pairs=pd.read_csv(out/"all_similarity_candidates.csv")
                for row in items.itertuples():
                    if digest(row.path)!=row.file_sha256:raise ValueError(f"Image changed since scan: {row.path}")
            else:
                collect_provenance(out)
                items,pairs=scan(manifest_items,out)
            inference(items,out)
            review_sheets(items,pairs,out)
        else:
            if not args.visual_review:p.error("--finalize requires --visual-review")
            items=pd.read_csv(out/"image_inventory.csv")
            finalize(items,out,args.visual_review)
    except Exception as exc:
        import traceback
        save_json(out/("scan_failure.json" if args.scan else "resume_failure.json" if args.resume_scan else "finalize_failure.json"),dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        save_json(out/("scan_data_access.json" if args.scan else "resume_data_access.json" if args.resume_scan else "finalize_data_access.json"),dict(accessed=sorted(guard.accessed),denied=guard.denied,
            test_manifest_opened=False,scope="Python open audit hook; only train/val listed images. Not an OS-wide/native file trace."))


if __name__=="__main__":main()
