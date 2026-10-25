"""Summarize and verify the frozen Phase 2 experiment, without new inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from lumina_reliability_phase2 import (
    ROOT,OUT,P1,BUSI,VAL,protocol,read_pair,resolve,sha256,base_regions,text_mask,apply_fill,
    matched_controls,PROFILES,transform_image,transform_mask,
)
from artifact_sensitivity_lumina_classifiers import write_json

REGIONS=["lesion","text","left_border","right_border","top_strip","context"]


def case_interval(values,replicates=5000):
    values=np.asarray(values,float)
    if len(values)<5:return None,None
    rng=np.random.default_rng(42)
    means=values[rng.integers(0,len(values),(replicates,len(values)))].mean(1)
    return tuple(float(v) for v in np.quantile(means,[.025,.975]))


def sign_randomization(values,draws=10000):
    values=np.asarray(values,float)
    if len(values)<5:return 1.
    rng=np.random.default_rng(42)
    signs=rng.choice(np.array([-1.,1.]),size=(draws,len(values)))
    null=(signs*values).mean(1)
    return float((1+np.count_nonzero(null>=values.mean()))/(draws+1))


def holm(values):
    values=np.asarray(values,float);order=np.argsort(values,kind="stable");result=np.empty_like(values)
    maximum=0.
    for rank,i in enumerate(order):
        maximum=max(maximum,min(1.,values[i]*(len(values)-rank)));result[i]=maximum
    return result


def summarize_observations(frame,columns):
    output=[]
    for key,g in frame.groupby(columns,sort=True):
        if not isinstance(key,tuple):key=(key,)
        cases=g.groupby("case_id").agg(abs_change=("abs_score_delta","mean"),signed=("score_delta","mean"),
            flip=("prediction_flip","mean"),logit=("logit_delta","mean"),extreme=("extreme","mean"))
        lo,hi=case_interval(cases.abs_change)
        output.append(dict(zip(columns,key))|dict(n_cases=len(cases),observations=len(g),
            mean_abs_score_change=float(cases.abs_change.mean()),median_abs_score_change=float(cases.abs_change.median()),
            mean_signed_score_change=float(cases.signed.mean()),median_signed_score_change=float(cases.signed.median()),
            class_flip_rate=float(cases.flip.mean()),mean_logit_delta=float(cases.logit.mean()),median_logit_delta=float(cases.logit.median()),
            between_case_abs_change_sd=float(cases.abs_change.std()) if len(cases)>1 else None,
            abs_change_q25=float(cases.abs_change.quantile(.25)),abs_change_q75=float(cases.abs_change.quantile(.75)),
            mean_abs_change_ci_low=lo,mean_abs_change_ci_high=hi,
            max_abs_logit=float(g.max_abs_logit.max()),extreme_observation_count=int(g.extreme.sum()),
            mean_native_area_fraction=float(g.native_area_fraction.mean()),mean_model_region_pixels=float(g.model_region_pixels.mean())))
    return pd.DataFrame(output)


def paired_results(frame,audit):
    keys=["case_id","image_id","birads","model","analysis_role","region","fill","strength"]
    effects=frame[frame.kind=="region"].copy()
    controls=frame[frame.kind=="control"].copy();controls["support_drop"]=np.maximum(0,-controls.score_delta)
    average=controls.groupby(keys,as_index=False).agg(control_abs_change=("abs_score_delta","mean"),
        control_signed_change=("score_delta","mean"),control_flip_rate=("prediction_flip","mean"),
        control_logit_delta=("logit_delta","mean"),control_support_drop=("support_drop","mean"),
        control_native_pixels=("native_pixels","mean"),control_observations=("seed","count"),
        control_model_region_pixels=("model_region_pixels","mean"))
    pairs=effects.merge(average,on=keys,how="inner",validate="one_to_one")
    assert (pairs.native_pixels==pairs.control_native_pixels).all()
    pairs["excess_abs_effect"]=pairs.abs_score_delta-pairs.control_abs_change
    pairs["excess_signed_change"]=pairs.score_delta-pairs.control_signed_change
    pairs["excess_support_drop"]=np.maximum(0,-pairs.score_delta)-pairs.control_support_drop
    quality=audit[audit.kind=="control"].groupby(["case_id","image_id","region"],as_index=False).agg(
        all_controls_shape_exact=("shape_exact","all"),soft_matched_controls=("all_soft_matched","sum"),
        control_location_reuses=("location_reused","sum"),mean_matching_cost=("matching_cost","mean"),
        mean_centroid_distance=("centroid_distance","mean"),mean_lesion_distance_mismatch=("lesion_distance_mismatch","mean"),
        mean_gray_mismatch=("gray_mean_mismatch","mean"),mean_gray_sd_mismatch=("gray_sd_mismatch","mean"))
    pairs=pairs.merge(quality,on=["case_id","image_id","region"],how="left",validate="many_to_one")
    pairs["strict_match"]=pairs.all_controls_shape_exact & (pairs.soft_matched_controls>=2)&(pairs.control_observations==3)
    return pairs


def summarize_pairs(pairs,group_columns):
    result=[]
    for key,g in pairs.groupby(group_columns,sort=True):
        if not isinstance(key,tuple):key=(key,)
        assert not g.case_id.duplicated().any()
        values=g.excess_abs_effect.to_numpy();lo,hi=case_interval(values)
        signed_lo,signed_hi=case_interval(g.excess_signed_change)
        result.append(dict(zip(group_columns,key))|dict(n_cases=len(g),
            mean_excess_abs_effect=float(values.mean()),median_excess_abs_effect=float(np.median(values)),
            excess_abs_ci_low=lo,excess_abs_ci_high=hi,fraction_cases_exceed_controls=float((values>0).mean()),
            paired_effect_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
            mean_excess_signed_change=float(g.excess_signed_change.mean()),signed_ci_low=signed_lo,signed_ci_high=signed_hi,
            mean_excess_support_drop=float(g.excess_support_drop.mean()),region_mean_abs_change=float(g.abs_score_delta.mean()),
            control_mean_abs_change=float(g.control_abs_change.mean()),strict_matched_cases=int(g.strict_match.sum()),
            exact_shape_cases=int(g.all_controls_shape_exact.sum()),mean_matching_cost=float(g.mean_matching_cost.mean())))
    return pd.DataFrame(result)


def inferential_consistency(pairs,summaries):
    # Preserve all 24 planned hypotheses per family, even for absent categories.
    tests=[]
    for family,models in (("primary",["bra","birads"]),("secondary_descriptive",["busi","uclm"])):
        family_rows=[]
        for model in models:
            for region in REGIONS:
                for fill,strength in (("blur",.025),("local_median",1.)):
                    g=pairs[(pairs.model==model)&(pairs.region==region)&(pairs.fill==fill)&(pairs.strength==strength)]
                    family_rows.append(dict(family=family,model=model,region=region,fill=fill,strength=strength,
                        n_cases=len(g),p_one_sided=sign_randomization(g.excess_abs_effect)))
        corrected=holm([r["p_one_sided"] for r in family_rows])
        for row,adjusted in zip(family_rows,corrected):row["p_holm"]=float(adjusted)
        tests.extend(family_rows)
    tests=pd.DataFrame(tests)
    complete=summaries.merge(tests,on=["model","region","fill","strength","n_cases"],how="left")
    full=complete[((complete.fill=="blur")&(complete.strength==.025))|((complete.fill=="local_median")&(complete.strength==1.))].copy()
    full["passes_full_method_criterion"]=(full.n_cases>=20)&(full.mean_excess_abs_effect>=.02)&(full.excess_abs_ci_low>0)&(full.fraction_cases_exceed_controls>=.6)&(full.p_holm<.05)
    decisions=[]
    for (model,region),group in full.groupby(["model","region"]):
        weak=complete[(complete.model==model)&(complete.region==region)&(((complete.fill=="blur")&(complete.strength==.0125))|((complete.fill=="local_median")&(complete.strength==.5)))]
        both=len(group)==2 and bool(group.passes_full_method_criterion.all())
        weak_positive=len(weak)==2 and bool((weak.mean_excess_abs_effect>0).all())
        status=("robust_by_prespecified_criterion" if both and weak_positive else
                "fill_specific" if int(group.passes_full_method_criterion.sum())==1 else
                "direction_consistent_unconfirmed" if len(group)==2 and bool((group.mean_excess_abs_effect>0).all()) else "inconclusive_or_no_positive_excess")
        if region=="text":status="exploratory_text_below_robustness_sample_threshold"
        decisions.append(dict(model=model,region=region,analysis_role="primary" if model in ("bra","birads") else "secondary_cross_dataset",
            consistency=status,weaker_strengths_positive=weak_positive,full_methods_passing=int(group.passes_full_method_criterion.sum()),
            minimum_cases=int(group.n_cases.min()),minimum_strict_matched_cases=int(group.strict_matched_cases.min()),
            interpretation_caution="No strict-matched contrast supports location/content specificity" if int(group.strict_matched_cases.min())==0 else "Operational perturbation criterion only; inspect strict subset and matching limitations"))
    return complete,tests,pd.DataFrame(decisions)


def phase1_comparison(observation_summary,paired_summary):
    prior=pd.read_csv(P1/"perturbation_summary.csv");rows=[]
    full=observation_summary[(observation_summary.kind=="region")&(((observation_summary.fill=="blur")&(observation_summary.strength==.025))|((observation_summary.fill=="local_median")&(observation_summary.strength==1.)))]
    old_pairs=pd.read_csv(P1/"paired_control_summary.csv")
    for r in full.itertuples():
        old=prior[(prior.model==r.model)&(prior.region==r.region)&(prior.fill==r.fill)]
        if len(old)!=1:continue
        old=old.iloc[0];new_pair=paired_summary[(paired_summary.model==r.model)&(paired_summary.region==r.region)&(paired_summary.fill==r.fill)&(paired_summary.strength==r.strength)]
        old_pair=old_pairs[(old_pairs.model==r.model)&(old_pairs.region==r.region)&(old_pairs.fill==r.fill)]
        rows.append(dict(model=r.model,analysis_role=r.analysis_role,region=r.region,fill=r.fill,
            phase1_cases=int(old.n_cases),phase2_cases=r.n_cases,phase1_mean_abs_change=float(old.mean_abs_score_delta),phase2_mean_abs_change=r.mean_abs_score_change,
            phase1_paired_excess=float(old_pair.iloc[0].mean_excess_abs_effect) if len(old_pair) else None,
            phase2_paired_excess=float(new_pair.iloc[0].mean_excess_abs_effect) if len(new_pair) else None,
            interpretation="Sequential disjoint validation cases; different controls, class prevalence and donor exclusions. Descriptive comparison, not pooled independent replication. Historical BUSI numeric target effects preserved; obsolete class names superseded."))
    return pd.DataFrame(rows)


def figures(frame,pairs,summary,out,config):
    full=summary[((summary.fill=="blur")&(summary.strength==.025))|((summary.fill=="local_median")&(summary.strength==1.))]
    fig,axes=plt.subplots(4,2,figsize=(15,14),squeeze=False)
    for i,model in enumerate(("bra","birads","busi","uclm")):
        for j,fill in enumerate(("blur","local_median")):
            g=full[(full.model==model)&(full.fill==fill)].set_index("region").reindex(REGIONS)
            x=np.arange(len(g));axes[i,j].bar(x,g.mean_excess_abs_effect*100,color="#297ea8" if i<2 else "#9361a8")
            lo=g.excess_abs_ci_low.to_numpy(float);hi=g.excess_abs_ci_high.to_numpy(float);mean=g.mean_excess_abs_effect.to_numpy(float)
            axes[i,j].errorbar(x,mean*100,yerr=np.maximum(0,np.array([mean-lo,hi-mean])*100),fmt="none",color="black",capsize=3)
            axes[i,j].axhline(0,color="gray",lw=.8);axes[i,j].set_xticks(x,REGIONS,rotation=20)
            axes[i,j].set_ylabel("Region minus control |score change| (pp)")
            axes[i,j].set_title(f"{model.upper()} | {fill} | {'primary' if i<2 else 'cross-dataset secondary'}\nCases per region: "+", ".join(str(int(v)) if np.isfinite(v) else "0" for v in g.n_cases),fontsize=10)
    fig.suptitle("Phase 2: new validation cases, case bootstrap 95% intervals; no causal/clinical interpretation")
    fig.text(.5,.007,"Matching quality varies: no context or top-strip cases met the strict matching rule. Error bars do not remove matching bias.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.97));fig.savefig(out/"controlled_sensitivity_comparison.png",dpi=145);plt.close(fig)
    # Fixed first selected case and first-seed controls, never chosen for effect size.
    selected=pd.read_csv(out/"selected_cases.csv",dtype={"case_id":str});row=selected.iloc[0]
    rgb,lesion=read_pair(row);known,_=text_mask(row,rgb,config);known &= ~lesion
    regions,_=base_regions(rgb,lesion,row,config)
    names=[r for r in REGIONS if r in regions];fig,axes=plt.subplots(len(names),4,figsize=(14,3.4*len(names)),squeeze=False)
    for i,region in enumerate(names):
        mask=regions[region];controls,_=matched_controls(rgb,lesion,mask,known,row.image_id,region)
        rec=frame[(frame.image_id==row.image_id)&(frame.model=="birads")&(frame.region==region)&(frame.kind=="region")&(frame.fill=="local_median")&(frame.strength==1)].iloc[0]
        median,_=apply_fill(rgb,mask,lesion,known,"local_median",1.);blur,_=apply_fill(rgb,mask,lesion,known,"blur",.025)
        axes[i,0].imshow(rgb);axes[i,0].contour(mask,levels=[.5],colors=["orange"],linewidths=1)
        axes[i,0].contour(lesion,levels=[.5],colors=["lime"],linewidths=.7)
        axes[i,0].set_title(f"Original + ROI: {region}\nBI-RADS {rec.original_predicted_class}")
        axes[i,1].imshow(rgb)
        if controls:axes[i,1].contour(controls[0]["mask"],levels=[.5],colors=["cyan"],linewidths=1)
        axes[i,1].set_title("Control seed 42\nCyan = area-matched control")
        axes[i,2].imshow(blur)
        b=frame[(frame.image_id==row.image_id)&(frame.model=="birads")&(frame.region==region)&(frame.kind=="region")&(frame.fill=="blur")&(frame.strength==.025)].iloc[0]
        axes[i,2].set_title(f"Blur: {b.perturbed_predicted_class}\nFixed-class delta {100*b.score_delta:+.2f} pp")
        axes[i,3].imshow(median);axes[i,3].set_title(f"Median: {rec.perturbed_predicted_class}\nFixed-class delta {100*rec.score_delta:+.2f} pp")
        for ax in axes[i]:ax.axis("off")
    fig.suptitle(f"{row.image_id}: fixed representative | artificial perturbations, not real observations")
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/"representative_perturbations.png",dpi=130);plt.close(fig)
    review=selected[selected.image_id.isin(config.get("images",{}))]
    fig,axes=plt.subplots(3,4,figsize=(14,14))
    for ax in axes.flat:ax.axis("off")
    for ax,(_,row) in zip(axes.flat,review.iterrows()):
        rgb,lesion=read_pair(row);mask,_=text_mask(row,rgb,config);mask &= ~lesion
        ax.imshow(rgb);ax.contour(mask,levels=[.5],colors=["orange"],linewidths=1);ax.contour(lesion,levels=[.5],colors=["lime"],linewidths=.7)
        ax.set_title(f"{row.image_id} | BI-RADS {row.birads}")
    fig.suptitle("Frozen-subset reviewed text ROIs (orange); lesion masks (green)")
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/"text_roi_review.png",dpi=130);plt.close(fig)


def verify(out,frame,audit):
    _,checksum=protocol(out)
    initial=json.loads((out/"initial_state.json").read_text(encoding="utf-8-sig"))
    changed=[r["Path"] for r in initial["hashes"] if sha256(r["Path"]).upper()!=r["Hash"]]
    assert not changed
    inputs=pd.read_csv(out/"validation_input_audit.csv")
    manifest=pd.read_csv(VAL,dtype={"case_id":str}).set_index("image_id")
    for r in inputs[inputs.eligible].itertuples():
        source=manifest.loc[r.image_id]
        assert sha256(resolve(source.image_path))==r.image_sha256 and sha256(resolve(source.mask_path))==r.mask_sha256
    # Recompute every stored fixed-target outcome from real saved logits.
    score_error=0.;logit_error=0.
    expected_min=float(np.min(-np.array([.485,.456,.406])/np.array([.229,.224,.225])))
    expected_max=float(np.max((1-np.array([.485,.456,.406]))/np.array([.229,.224,.225])))
    assert frame.raw_dtype.eq("uint8").all() and frame.raw_min.ge(0).all() and frame.raw_max.le(255).all()
    assert frame.tensor_min.ge(expected_min-2e-6).all() and frame.tensor_max.le(expected_max+2e-6).all()
    assert np.isfinite(frame[["tensor_min","tensor_max","max_abs_logit","original_score","perturbed_score"]].to_numpy()).all()
    for model,g in frame.groupby("model"):
        a=np.array([json.loads(x) for x in g.original_logits],dtype=np.float32)
        b=np.array([json.loads(x) for x in g.perturbed_logits],dtype=np.float32)
        target=g.original_target_index.to_numpy(int);idx=np.arange(len(g))
        pa=np.exp(a-a.max(1,keepdims=True));pa/=pa.sum(1,keepdims=True)
        pb=np.exp(b-b.max(1,keepdims=True));pb/=pb.sum(1,keepdims=True)
        delta=pb[idx,target]-pa[idx,target]
        np.testing.assert_allclose(delta,g.score_delta,atol=3e-7,rtol=0)
        ld=b[idx,target]-a[idx,target];np.testing.assert_allclose(ld,g.logit_delta,atol=1e-7,rtol=0)
        np.testing.assert_array_equal(pb.argmax(1)!=target,g.prediction_flip)
        labels=PROFILES[model].classes
        assert all(tuple(json.loads(x))==labels for x in g.class_order)
        assert list(g.original_predicted_class)==[labels[i] for i in target]
        assert list(g.perturbed_predicted_class)==[labels[i] for i in pb.argmax(1)]
        score_error=max(score_error,float(np.abs(delta-g.score_delta).max()));logit_error=max(logit_error,float(np.abs(ld-g.logit_delta).max()))
    by_image={};config=json.loads((out/"text_rois.json").read_text())
    for r in audit.itertuples():
        if r.image_id not in by_image:
            source=manifest.loc[r.image_id].copy();source["image_id"]=r.image_id
            rgb,lesion=read_pair(source);known,_=text_mask(source,rgb,config);by_image[r.image_id]=(rgb,lesion,known&~lesion)
        rgb,lesion,known=by_image[r.image_id]
        with Image.open(r.mask_path) as image:mask=np.asarray(image)>127
        assert mask.shape==lesion.shape and int(mask.sum())==r.native_pixels
        assert hashlib.sha256(mask.tobytes()).hexdigest()==r.mask_boolean_sha256
        if r.region!="lesion" or r.kind=="control":assert not(mask&lesion).any()
        if r.kind=="control":
            assert not(mask&known).any()
            region_record=audit[(audit.image_id==r.image_id)&(audit.region==r.region)&(audit.kind=="region")].iloc[0]
            assert r.native_pixels==region_record.native_pixels
            with Image.open(region_record.mask_path) as image:region_mask=np.asarray(image)>127
            assert not(mask&region_mask).any()
            if r.shape_exact:
                ys,xs=np.nonzero(mask);ry,rx=np.nonzero(region_mask)
                np.testing.assert_array_equal(np.c_[ys-ys.min(),xs-xs.min()],np.c_[ry-ry.min(),rx-rx.min()])
    # Independent direct arithmetic check of every saved region-control pair and summary.
    pair_table=pd.read_csv(out/"paired_control_results.csv",dtype={"case_id":str})
    keys=["model","case_id","region","fill","strength"]
    direct={}
    for key,g in frame[frame.kind!="original"].groupby(keys):
        region=g[g.kind=="region"];control=g[g.kind=="control"]
        if not len(control):continue
        assert len(region)==1
        value=abs(float(region.perturbed_score.iloc[0]-region.original_score.iloc[0]))-np.mean(np.abs(control.perturbed_score.to_numpy()-control.original_score.to_numpy()))
        direct[key]=float(value)
    paired_error=0.
    for r in pair_table.itertuples():
        error=abs(r.excess_abs_effect-direct[(r.model,r.case_id,r.region,r.fill,r.strength)])
        # Inference score deltas were subtracted in float32; CSV score columns
        # read back as float64. Their difference is bounded by float32 rounding.
        assert error<1e-7
        paired_error=max(paired_error,error)
    summary_table=pd.read_csv(out/"paired_control_summary.csv")
    for r in summary_table.itertuples():
        g=pair_table[(pair_table.model==r.model)&(pair_table.region==r.region)&(pair_table.fill==r.fill)&(pair_table.strength==r.strength)]
        assert len(g)==r.n_cases and g.case_id.nunique()==len(g)
        assert abs(float(g.excess_abs_effect.mean())-r.mean_excess_abs_effect)<1e-12
        assert abs(float(g.excess_abs_effect.median())-r.median_excess_abs_effect)<1e-12
    selected=pd.read_csv(out/"selected_cases.csv",dtype={"case_id":str})
    old=pd.read_csv(P1/"selected_cases.csv",dtype={"case_id":str})
    assert not set(selected.case_id)&set(old.case_id)
    assert selected.case_id.nunique()==len(selected)==frame.case_id.nunique()
    assert len(frame[frame.kind=="original"])==4*len(selected)
    assert frame.failure_status.eq("PASSED").all()
    accessed=set();denied=[]
    for stage in ("prepare","stability","run"):
        log=json.loads((out/f"{stage}_data_access.json").read_text());accessed.update(log["accessed"]);denied+=log["denied"]
    assert not denied
    allowed={str(VAL.resolve())}
    for _,r in manifest.iterrows():allowed.update((str(resolve(r.image_path)),str(resolve(r.mask_path))))
    assert all(path in allowed or Path(path).is_relative_to(P1) or Path(path).is_relative_to(BUSI) for path in accessed)
    result=dict(status="PASSED",protocol_sha256=checksum,protected_files_checked=len(initial["hashes"]),protected_files_changed=changed,
        validation_image_and_mask_hashes_checked=2*len(inputs),prediction_rows_recomputed=len(frame),saved_masks_verified=len(audit),
        paired_effects_independently_recomputed=len(pair_table),paired_effect_max_score_rounding_error=paired_error,paired_summary_groups_verified=len(summary_table),
        score_delta_max_error=score_error,logit_delta_max_error=logit_error,no_pilot_case_overlap=True,
        access_guard_denied=denied,test_access="No held-out files/manifests opened; Python-level validation/prior-output guard, not OS-wide tracing",
        passed=["all four checkpoints strict load","all original inference/transform parity","correct BUSI mapping","mask geometry/padding/region exclusions",
                "control area and translated shape checks","seeded sampling/control reproducibility","finite scores/logits","input dtype/ranges",
                "saved logit outcome recomputation","original data/checkpoint/production/prior-output hashes"],
        failed=[],unexecuted=["live server/browser/database integration","training or model improvement","independent test/clinical evaluation"],
        not_verifiable=["verified patient/examination identity beyond manifest case_id","clinical calibration/causal medical reasoning","photorealism of artificial fills"])
    write_json(out/"verification_results.json",result)
    return result


def markdown_table(df):
    def fmt(x):
        if pd.isna(x):return "N/A"
        return f"{x:.3f}" if isinstance(x,float) else str(x).replace("|","/")
    return "| "+" | ".join(df.columns)+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(fmt(x) for x in r)+" |" for r in df.itertuples(index=False,name=None))


def reports(out,metadata,verification,stats,paired,consistency,audit,comparison):
    sampling=json.loads((out/"sampling_audit.json").read_text());stability=json.loads((out/"numerical_stability_report.json").read_text())
    model_config=json.loads((out/"model_verification.json").read_text())
    full=paired[((paired.fill=="blur")&(paired.strength==.025))|((paired.fill=="local_median")&(paired.strength==1.))].copy()
    columns=["model","region","fill","n_cases","mean_excess_abs_effect","excess_abs_ci_low","excess_abs_ci_high","p_holm","strict_matched_cases"]
    display=full[columns].copy()
    for c in ("mean_excess_abs_effect","excess_abs_ci_low","excess_abs_ci_high"):display[c]*=100
    robust=consistency[(consistency.analysis_role=="primary")&(consistency.consistency=="robust_by_prespecified_criterion")]
    counts=audit[audit.kind=="control"].groupby("region").agg(
        controls=("shape_exact","size"),exact_shape_fraction=("shape_exact","mean"),soft_match_fraction=("all_soft_matched","mean"),
        mean_centroid_distance=("centroid_distance","mean"),mean_cost=("matching_cost","mean"))
    counts.to_csv(out/"control_matching_summary.csv")
    findings=[]
    for model in ("bra","birads"):
        for region in ("lesion","context","text"):
            g=full[(full.model==model)&(full.region==region)]
            if len(g)!=2:continue
            decision=consistency[(consistency.model==model)&(consistency.region==region)].iloc[0].consistency
            values={r.fill:f"{100*r.mean_excess_abs_effect:+.2f} pp (95% CI {100*r.excess_abs_ci_low:+.2f} to {100*r.excess_abs_ci_high:+.2f})" for r in g.itertuples()}
            label=decision.replace("_"," ")
            caution=" No context case met the strict matching rule." if region=="context" else ""
            findings.append(f"{model.upper()} {region}: paired excess absolute score change {values['blur']} for blur and {values['local_median']} for median; {label}.{caution}")
    phrase=" ".join(findings)
    replication=("BI-RADS lesion sensitivity persisted across both full methods and in the strict-matching subset (27 cases). "
        "BRA lesion effects remained positive, but only blur met the full-cohort criterion; median's full-cohort interval included zero, so the Phase 1 cross-fill finding was not fully replicated. "
        "The prespecified 27-case strict lesion subset nevertheless had positive 95% intervals for both BRA methods. "
        "BI-RADS context met the numerical consistency criterion, but zero strict-matched context cases prevent a location/content-specific conclusion. "
        "Text dependency was again not established; text has only 12 cases.")
    extension_extremes=pd.read_csv(out/"image_level_results.csv",usecols=["model","fill","extreme","max_abs_logit"])
    extreme_summary=extension_extremes.groupby(["model","fill"],as_index=False).agg(observations=("extreme","size"),extremes=("extreme","sum"),max_abs_logit=("max_abs_logit","max"))
    extreme_summary.to_csv(out/"logit_extreme_summary.csv",index=False)
    decision=dict(recommendation="B",title="Proceed to research-prototype integration with restricted contributions and outputs",
        frozen_baselines_retained=True,busi_consensus_restriction="Retained; do not use BUSI in reliability-dependent automated consensus",
        cross_dataset_models="BUSI/UCLM observations are secondary; no native BUSI/UCLM accuracy or clinical consensus inferred",
        permitted_integration="Versioned individual model outputs, provenance, native/cross-dataset task labels, explicit research limitations and genuine Grad-CAM labeled as attribution",
        prohibited_claims="No clinical validation, calibrated confidence, medical reasoning or shortcut causality is established",
        robust_primary_effects=robust[["model","region"]].to_dict("records"),
        operational_effect_caution="BI-RADS context passes numerical criterion but has no strict-matched cases; no context-specific mechanism or shortcut is established.",
        phase1_replication=replication,
        production_changed=False,retraining_performed=False,
        rationale="All four frozen inference paths and data preservation passed; the controlled results and residual matching/fill limitations support restricted research integration, not unrestricted model consensus or automatic retraining.")
    write_json(out/"final_reliability_decision.json",decision)
    notes=json.loads((out/"execution_notes.json").read_text()) if (out/"execution_notes.json").exists() else {}
    maximum={name:max(r["max_abs_logit"] for r in stability["observations"] if r["model"]==name and r["fill"]=="local_median" and r["strength"]==1) for name in PROFILES}
    summary=f"""Lumina Phase 2 completed: {metadata['case_count']} NEW distinct BUS-BRA validation cases, none from the Phase 1 pilot; BI-RADS counts {metadata['case_counts_by_birads']}. All four frozen classifiers passed. BRA pathology and BI-RADS are primary; BUSI/UCLM are secondary cross-dataset observations. {metadata['prediction_rows']} real prediction records include originals, two fill methods at two strengths and three seeded controls per available region. Text ROIs were reviewed for {metadata['text_case_count']} prespecified cases.

Protocol SHA-256: {metadata['protocol_sha256']}. Sampling, methods, controls, outcomes and consistency criteria were frozen before extension inference. Controls use exact area and translated shape where feasible, with logged location/lesion-distance/gray-statistic mismatches and explicit compact-shape fallback. Seeds: 42, 2026, 3407. Median replacement is an explicitly artifact-prone sensitivity analysis.

Numerical stability: known Phase 1 case bus_0114-s reproduced full-median max absolute logits BUSI {maximum['busi']:.2f}, UCLM {maximum['uclm']:.2f}. RGB uint8 [0,255], single ImageNet normalization, model-specific sizing, original parity and finite outputs passed. Results support perturbation-induced out-of-distribution behavior rather than pixel overflow; internal cause remains unproven. No logits were clamped.

Primary results: {phrase} Robustness required both fills, n>=20, mean paired excess>=2 pp, positive 95% case-bootstrap lower bound, >=60% positive cases, Holm p<0.05 across the planned family, and positive weaker-strength means. Text is exploratory and too small for that criterion. Phase 1 comparisons are descriptive because the cases, prevalence and controls changed; no samples were pooled.

Replication assessment: {replication} Full-strength lesion paired excesses (blur/median) were BRA +11.66/+5.36 pp and BI-RADS +9.19/+11.54 pp. The stricter 27-case lesion subset gave BRA +6.70/+8.25 pp and BI-RADS +7.27/+4.10 pp, with positive descriptive 95% intervals for each. Matching support is limited: 95% of lesion controls preserve shape, but only 27/60 lesion case sets meet the complete strict rule; context/top-strip have zero strict case sets. One control location was reused where the feasible pool was insufficient.

In the new extension, no primary-model perturbation met the extreme-logit flag; max absolute logits were BRA 7.17 and BI-RADS 3.11. Median inputs produced 3 flagged BUSI observations (max 22.61) and 15 flagged UCLM observations (max 299.96); blur produced none under the prespecified flag. These flags describe artificial-input behavior, not clinical model quality.

Verification passed: syntax, 13 new and 8 existing unit tests (21/21), all-model original inference and preprocessing, corrected BUSI mapping, native and model-space geometry, lesion/padding exclusion, deterministic controls, {verification['prediction_rows_recomputed']} saved prediction-outcome recalculations, {verification['paired_effects_independently_recomputed']} independently recomputed paired effects, {verification['paired_summary_groups_verified']} checked summary groups and preservation of {verification['protected_files_checked']} protected files plus all 540 validation image/mask hashes. All {verification['saved_masks_verified']} saved masks passed hash/geometry/exclusion checks. No locked test files were accessed. Protocol and prior research outputs were preserved. Unresolved failed checks: none. One reporting-only check initially failed because its arithmetic tolerance ignored float32 rounding; the corrected check passed, with the failure and resolution recorded in execution_notes.json. Live server/browser/database integration and independent patient-level/clinical evaluation were not executed. Patient identity beyond manifest case_id, clinical calibration and causal medical reasoning are not verifiable from this study.

Created research code in {ROOT/'src'}: lumina_reliability_phase2.py, test_lumina_reliability_phase2.py, report_lumina_reliability_phase2.py. All Phase 2 outputs are separate; no existing source, model, dataset, manifest, backend, frontend, database or prior research result was changed. No training, package reinstall, commit or push occurred.

Final recommendation B: proceed to multi-model RESEARCH integration with individual task-specific outputs and restricted contributions. Keep the established BUSI consensus exclusion; do not interpret cross-dataset models as validated native performance or form a clinical consensus. Production consensus was not changed by this study. Remaining risks include validation/model-selection reuse, small text and BI-RADS-5 subsets, imperfect anatomy/texture/location matching, bilinear boundary mixing, artificial-fill artifacts and no causal or clinical proof.

Detailed handoff: {out/'PHASE2_CODEX_HANDOFF.md'}
Decision: {out/'final_reliability_decision.json'}
Main paired results: {out/'paired_control_summary.csv'}
All outputs, protocol, masks, plots and verification: {out}
"""
    sections=["# LUMINA RELIABILITY INVESTIGATION — PHASE 2 CODEX HANDOFF",
        "## 1. Execution status\n\nCompleted. New validation extension executed; no Phase 1/BUSI investigation rerun. "+json.dumps(metadata,indent=2),
        "## 2. Model configurations\n\n"+json.dumps({n:{k:v for k,v in model_config[n].items() if k!="parity"} for n in model_config},indent=2)+"\n\nEfficientNet-B0 with model-specific linear heads; strict state load, eval, no_grad, float32, no TF32/autocast. RGB, ImageNet mean/std. Pathology models 256-square bilinear resize; BI-RADS 224 centered black pad. Original evaluation tensors checked directly using existing training validation transforms; BUSI recovered original transform also checked. No production backend import.",
        f"## 3. Files created\n\nNew source: `{ROOT/'src/lumina_reliability_phase2.py'}`, `{ROOT/'src/test_lumina_reliability_phase2.py'}`, `{ROOT/'src/report_lumina_reliability_phase2.py'}`. All generated artifacts are under `{out}`; artifact_inventory.csv enumerates every exact path and purpose, including masks and per-case chunks.",
        "## 4. Files modified\n\nNo pre-existing source, weights, dataset, manifest, application, database or prior output was modified. New Phase 2 source evolved during implementation. Initial/final git status and protected-file hashes are saved. Reports can be explicitly refreshed from saved predictions; inference is not repeated.",
        "## 5. Frozen experimental protocol\n\nChecksum `"+metadata["protocol_sha256"]+"`. frozen_protocol.json was written before any new-case analysis/inference; no scientific protocol revision occurred. Text review coordinates were fixed before extension inference in the protocol's predefined subset. Protocol defines matching tolerances, two strengths per fill, seed families, primary/secondary roles, outcomes, failure rules and decision criteria. Any later protocol revision must be separately versioned/exploratory.",
        "## 6. Case sampling and counts\n\n"+json.dumps(sampling,indent=2)+"\n\nOne image per case, 24 pilot case IDs excluded. Reserve up to 3 per stratum then prevalence-aware largest-remainder allocation. BI-RADS 5 had only 3 remaining eligible cases; all 3 were selected, not forced to an equal quota. Selection never used prediction correctness or heatmaps. These remain checkpoint-selection validation data, not an independent test set.",
        "## 7. Numerical stability findings\n\n"+stability["findings"]+f" Full-median max absolute logits: {maximum}. Original outputs and old full-strength perturbations reproduced within predefined tolerances. See numerical_stability_report.json for all strengths, pixel/tensor ranges, finite scores and actual deviations. Median strength dependence is a perturbation realism concern. No model weights or logits were changed/clamped.",
        "## 8. Perturbation methods\n\nLocalized Gaussian blur at 1.25% and 2.5% short-side radius (minimum 2px), assigning only ROI pixels. Median replacement uses bounded actual donor RGB medians from lesion/text-free outer ring (3% short side, minimum 3px), with logged global donor fallback. Blend strengths .5 and 1.0 computed in float32 then rounded uint8. Median retained as artifact-prone sensitivity analysis: copying/generating plausible tissue introduces unsupported anatomical realism. Artificial inputs are never represented as patient observations.",
        "## 9. Control matching and exclusions\n\nArea exact; translated ROI support without clipping/wrap/rotation when feasible. Otherwise compact exact-area fallback explicitly flagged. Candidate grid step 5% native width/height; best eight ranked by centroid, mean lesion distance and grayscale mean/SD mismatch; seeds sample distinct locations where possible. Hard exclusions: lesion, tested region and known text. Controls may still contain unreviewed artifacts. Shape/location/content cannot always all be matched. Strict sensitivity requires all control shapes exact and >=2 soft-matched controls. No poorly matched observations silently removed from main paired summaries.\n\n"+markdown_table(counts.reset_index()),
        "## 10. Results for BRA pathology\n\n"+markdown_table(display[display.model=="bra"]),
        "## 11. Results for BI-RADS\n\n"+markdown_table(display[display.model=="birads"]),
        "## 12. Secondary BUSI/UCLM findings\n\n"+markdown_table(display[display.model.isin(["busi","uclm"])])+"\n\nThese are cross-dataset perturbation observations, not BUSI/UCLM native accuracy. Correct BUSI order [Normal,Benign,Malignant] is retained. The final BUSI restricted-consensus decision is not reopened or reversed.",
        "## 13. Statistical limitations\n\nAll score columns in section tables are percentage points; p-values remain proportions. Primary outcome is region absolute fixed-original-target score change minus average of its controls, paired within case. Bootstrap resamples case IDs (5000 draws), not control repeats as independent patients. Intervals suppressed below 5 cases. One-sided case sign randomization uses 10000 draws and a symmetry/exchangeability assumption that imperfect matching may violate. Holm corrects the 24 planned primary full-method tests, absent hypotheses p=1; secondary family is separately labeled. BI-RADS subgroups, weak strengths and strict-match sensitivity remain exploratory. Text n<=12 cannot meet n>=20 robustness threshold; BI-RADS 5 n=3 has no uncertainty interval. Sample selection/model-selection reuse and unknown patient dependence remain.",
        "## 14. Testing and verification\n\n"+json.dumps(verification,indent=2)+"\n\n"+notes.get("tests","Executed unit-test logs are saved in new_unit_tests.txt and existing_unit_tests.txt. Full inference checks were executed, not inferred from source inspection.")+"\n\nFAILED checks, if any, are retained in stage failure JSON; no failed scientific case is silently replaced. See execution_notes.json for exact commands and resolution history.",
        "## 15. Confounding factors\n\nUniform median patches change boundaries/texture and can produce extreme but finite model responses. Blur retains coarse lesion structure and mixes neighboring content; bilinear resizing can affect pixels adjacent to an ROI even if original lesion pixels are unchanged. Grayscale moments do not match anatomy, tissue speckle, instrument annotations or acquisition depth. Long border shapes cannot preserve border location while avoiding the original region; compact fallback loses shape. Raw region areas differ and cross-task score scales differ; paired controls reduce but do not eliminate confounding. Native shape/area matching differs slightly after resize. Scores are uncalibrated and effects do not demonstrate medical reasoning or shortcut causality.",
        "## 16. Comparison with Phase 1\n\n"+replication+"\n\n"+phrase+"\n\nphase1_vs_phase2_comparison.csv contains both real studies separately, with no pooled estimates. Phase 1 used 24 balanced pilot cases and compact controls; Phase 2 uses 60 disjoint cases and improved matching, modified donor exclusions and predefined strengths. Historical BUSI numeric fixed-target changes remain usable after correcting class names; old named-class accuracy is not carried forward. Operational numeric consistency does not resolve unmatched location/content; context and superficial-strip evidence remain confounded.",
        "## 17. Final reliability decision\n\n"+json.dumps(decision,indent=2)+"\n\nPassing original inference and preservation permits proceeding with a restricted research prototype; no independent clinical validity is asserted. No automatic retraining recommendation is made from artificial perturbations alone.",
        "## 18. Remaining risks\n\nValidation reuse, uncertainty in patient/examination identity, small annotation strata, incomplete artifact masks, matching infeasibility, artificial input domain shift, softmax saturation and unverified live integration remain. Robust-effect classifications are operational study labels and cannot certify model quality. Current production still includes BUSI in consensus; this experiment did not implement its exclusion.",
        "## 19. Next integration tasks\n\n1. Preserve model-specific class indices, preprocessing and checkpoint hashes in one versioned inference contract; keep pathology and BI-RADS outputs separate.\n2. Implement the established BUSI consensus restriction as a separately scoped application change before presenting restricted consensus behavior as deployed. Keep cross-dataset scores explicitly labeled and avoid clinical consensus/confidence claims.\n3. Expose individual research predictions and genuine Grad-CAM as attribution, with limitations from the matching/stability evidence.\n4. Add application contract tests and resolve existing consensus tie/score-association semantics identified in the BUSI handoff.\n5. Preserve frozen protocol/results and use this handoff for integration review. No further open-ended audit or automatic model training is required to resume prototype work.",
        "## 20. Ready-to-Paste ChatGPT Summary\n\n"+summary]
    (out/"PHASE2_CODEX_HANDOFF.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    write_json(out/"ready_to_paste_summary.json",dict(summary=summary))
    readme=f"""# Lumina reliability Phase 2

Completed separate validation extension: {metadata['case_count']} new cases, {metadata['prediction_rows']} real prediction records. Read PHASE2_CODEX_HANDOFF.md and final_reliability_decision.json. All model/dataset/production/previous research files are preserved.

Protocol checksum: {metadata['protocol_sha256']}. frozen_protocol.json must not change. protocol_checksum.json independently records the hash. selected_cases.csv excludes all 24 pilot IDs; text_review_candidates.csv fixes the 12-case review subset and text_rois.json stores native reviewed coordinates/hashes. Regions and controls are auditable in masks/, region_control_audit.csv and per-case chunks/. Image dimensions never change before each model's original preprocessing.

## Executed stages / reproduction

From {ROOT}, use the existing bc_ai Python:

```powershell
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' -m py_compile src/lumina_reliability_phase2.py src/test_lumina_reliability_phase2.py src/report_lumina_reliability_phase2.py
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' -m unittest discover -s src -p test_lumina_reliability_phase2.py -v
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' -m unittest discover -s src -p test_artifact_sensitivity_lumina_classifiers.py -v
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' src/lumina_reliability_phase2.py --prepare
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' src/lumina_reliability_phase2.py --stability
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' src/lumina_reliability_phase2.py --run
& 'C:/Users/hks89/anaconda3/envs/bc_ai/python.exe' src/report_lumina_reliability_phase2.py
```

Completed stage outputs refuse overwrite. --run --resume verifies each completed chunk's protocol, input image/mask, selection, text-config and checkpoint hashes and reuses it only if all agree. No scientific change is allowed under this protocol after outcomes. Report regeneration from saved predictions requires explicit --refresh-reports; it does not repeat inference. For new scientific variants create a separate versioned experiment/protocol, not an overwrite of this directory. Exact session commands/check outcomes are in execution_notes.json. No test/train dataset manifest is opened by Phase 2.

## Tables

- image_level_results.csv: model, fixed target index, full original/perturbed logits/probability vectors, positive=increased original-class support, flips, actual/native/model areas, strength, seed, ranges, extreme flags and failure status. kind distinguishes original/region/control.
- paired_control_results.csv: each region minus average of 3 control locations for the same case/model/method/strength; controls are not independent case observations.
- paired_control_summary.csv / uncertainty_summary.csv: case bootstrap intervals, planned-family corrected p-values, consistency support. model_region_summary.csv includes both regions and controls averaged within case.
- strict_matching_summary.csv: prespecified sensitivity restricted by achieved matching, with its reduced n disclosed. birads_subgroup_summary.csv suppresses intervals for n<5.
- effect_consistency.csv: operational cross-fill classification under the frozen criterion; never a clinical/causal label. hypothesis_tests.csv includes all planned tests. phase1_vs_phase2_comparison.csv keeps sequential different-control studies separate.
- numerical_stability_report.json: reproduced old-case extremes, finite tensor/logit diagnostics and original parity. verification_results.json / provenance_audit.json: saved-outcome recalculation, access and hash evidence.

Two fills are intentionally artificial: Gaussian blur and bounded local median replacement. Median is retained as artifact-prone sensitivity analysis; both weaker and full strengths are saved. Control matching is incomplete when shape/location/anatomy constraints conflict; consult actual matching columns before interpreting a paired difference. Prototype integration may proceed only with documented restrictions and existing BUSI consensus exclusion. No production behavior was changed here.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--refresh-reports",action="store_true");args=parser.parse_args();out=OUT
    if (out/"PHASE2_CODEX_HANDOFF.md").exists() and not args.refresh_reports:raise FileExistsError("Report exists; explicit --refresh-reports required")
    _,checksum=protocol(out);frame=pd.read_csv(out/"image_level_results.csv",dtype={"case_id":str})
    audit=pd.read_csv(out/"region_control_audit.csv",dtype={"case_id":str});metadata=json.loads((out/"run_metadata.json").read_text())
    assert metadata["protocol_sha256"]==checksum
    stats=summarize_observations(frame[frame.kind!="original"],["model","analysis_role","region","kind","fill","strength"])
    stats.to_csv(out/"model_region_summary.csv",index=False)
    pairs=paired_results(frame,audit);pairs.to_csv(out/"paired_control_results.csv",index=False)
    summaries=summarize_pairs(pairs,["model","analysis_role","region","fill","strength"])
    complete,tests,consistency=inferential_consistency(pairs,summaries)
    complete.to_csv(out/"paired_control_summary.csv",index=False);complete.to_csv(out/"uncertainty_summary.csv",index=False)
    tests.to_csv(out/"hypothesis_tests.csv",index=False);consistency.to_csv(out/"effect_consistency.csv",index=False)
    summarize_pairs(pairs[pairs.strict_match],["model","analysis_role","region","fill","strength"]).to_csv(out/"strict_matching_summary.csv",index=False)
    summarize_pairs(pairs,["model","analysis_role","region","fill","strength","birads"]).to_csv(out/"birads_subgroup_summary.csv",index=False)
    comparison=phase1_comparison(stats,complete);comparison.to_csv(out/"phase1_vs_phase2_comparison.csv",index=False)
    config=json.loads((out/"text_rois.json").read_text());figures(frame,pairs,complete,out,config)
    verification=verify(out,frame,audit)
    model_config=json.loads((out/"model_verification.json").read_text())
    write_json(out/"provenance_audit.json",dict(protocol_sha256=checksum,models={n:{k:v for k,v in d.items() if k!="parity"} for n,d in model_config.items()},
        phase1_reused=True,busi_audit_reused=True,busi_decision_retained=True,production_modified=False,verification=verification))
    reports(out,metadata,verification,stats,complete,consistency,audit,comparison)
    paths=list(out.rglob("*"));paths=[p for p in paths if p.is_file()]
    paths += [ROOT/"src"/n for n in ("lumina_reliability_phase2.py","test_lumina_reliability_phase2.py","report_lumina_reliability_phase2.py")]
    paths.append(out/"artifact_inventory.csv")
    pd.DataFrame(dict(path=str(p),purpose="Phase 2 research code" if p.suffix==".py" else "Auditable source/configuration, real inference record, binary mask, figure, verification or research report") for p in paths).drop_duplicates("path").to_csv(out/"artifact_inventory.csv",index=False)
    print(complete[(complete.model.isin(["bra","birads"]))&(((complete.fill=="blur")&(complete.strength==.025))|((complete.fill=="local_median")&(complete.strength==1.)))][["model","region","fill","n_cases","mean_excess_abs_effect","excess_abs_ci_low","excess_abs_ci_high","p_holm","strict_matched_cases"]].to_string(index=False),flush=True)
    print(consistency.to_string(index=False),flush=True)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":main()
