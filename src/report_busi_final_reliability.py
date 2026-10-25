"""Verify frozen BUSI audit outputs and build the final research handoff.

Reads only saved audit evidence, train/validation-listed images and known protected
source/checkpoint files. No test manifest, original test evaluator or DB is opened.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from audit_busi_final_reliability import (
    ROOT,OUT,ORIGINAL,CLASSES,CONFIRMED,STRONG,UNCERTAIN,
    digest,save_json,metric_record,extracted_namespace,original_transform,
)


def markdown_table(frame):
    def fmt(v):
        if isinstance(v,float):return "N/A" if not np.isfinite(v) else f"{v:.4f}"
        return str(v).replace("|"," / ").replace("\n"," ")
    return "| "+" | ".join(frame.columns)+" |\n| "+" | ".join(["---"]*len(frame.columns))+" |\n"+"\n".join(
        "| "+" | ".join(fmt(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))


def cluster_intervals(out,predictions,groups,reviewed):
    val=predictions[predictions.split=="val"].copy()
    lower_edges=reviewed[reviewed.similarity_classification==CONFIRMED]
    # Independent sparse graph formulation cross-checks union-find and subset masks.
    def labels(edges):
        n=len(predictions);a=edges.a_index.to_numpy(int);b=edges.b_index.to_numpy(int)
        graph=coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(n,n))
        return connected_components(graph,directed=False)[1]
    primary_edges=reviewed[reviewed.similarity_classification.isin([CONFIRMED,STRONG])]
    primary=labels(primary_edges);lower=labels(lower_edges)
    train_ids=predictions[predictions.split=="train"].image_index.to_numpy(int)
    confirmed_mask=np.isin(lower[val.image_index.to_numpy(int)],lower[train_ids])
    primary_mask=np.isin(primary[val.image_index.to_numpy(int)],primary[train_ids])
    ph_edges=reviewed[reviewed.cross_split & (reviewed.phash_distance<=8)]
    flagged=set(ph_edges.a_index)|set(ph_edges.b_index)
    subsets=[val,val[confirmed_mask],val[~confirmed_mask],val[~primary_mask],val[~val.image_index.isin(flagged)]]
    existing=pd.read_csv(out/"validation_subset_metrics.csv")
    output=[]
    for row,frame in zip(existing.itertuples(),subsets):
        actual=metric_record(row.subset,frame)
        for metric in ("n","correct","accuracy","balanced_accuracy","macro_f1","malignant_recall"):
            assert abs(actual[metric]-getattr(row,metric))<1e-12
        group_cm=[]
        frame=frame.copy();frame["cluster"]=[int(primary[i]) for i in frame.image_index]
        for _,g in frame.groupby("cluster"):
            cm=np.zeros((3,3),int);np.add.at(cm,(g.label.to_numpy(int),g.predicted_index.to_numpy(int)),1);group_cm.append(cm)
        matrices=np.stack(group_cm);rng=np.random.default_rng(42)
        draws=matrices[rng.integers(0,len(matrices),(2000,len(matrices)))].sum(1)
        true=draws.sum(2);pred=draws.sum(1);tp=np.diagonal(draws,axis1=1,axis2=2)
        recall=np.divide(tp,true,out=np.full(tp.shape,np.nan),where=true>0)
        f1=np.divide(2*tp,true+pred,out=np.zeros(tp.shape,float),where=true+pred>0)
        values=dict(accuracy=tp.sum(1)/draws.sum((1,2)),balanced_accuracy=np.nanmean(recall,axis=1),
                    macro_f1=f1.mean(1),malignant_recall=recall[:,2])
        record=dict(subset=row.subset,n=len(frame),similarity_clusters=len(matrices),bootstrap_replicates=2000,seed=42,
            interpretation="Similarity-cluster bootstrap; not verified patient/examination independence. Diagnostic selected-validation sample only.")
        for name,values_array in values.items():
            finite=values_array[np.isfinite(values_array)];lo,hi=np.quantile(finite,[.025,.975])
            too_small=name=="malignant_recall" and int((frame.label==2).sum())<5
            record.update({name+"_ci_low":None if too_small else float(lo),name+"_ci_high":None if too_small else float(hi),name+"_valid_bootstraps":len(finite)})
            if too_small:record["small_class_warning"]="Malignant cluster interval suppressed below 5 images; nonparametric all-correct draws would be misleading. Use broad nominal Wilson interval with dependence caveat."
        output.append(record)
    result=pd.DataFrame(output);result.to_csv(out/"validation_subset_cluster_intervals.csv",index=False)
    return result


def verify(out):
    inventory=pd.read_csv(out/"image_inventory.csv")
    changed=[r.path for r in inventory.itertuples() if digest(r.path)!=r.file_sha256]
    assert not changed
    state=json.loads((out/"initial_state.json").read_text(encoding="utf-8-sig"))
    protected_changed=[v["Path"] for v in state["hashes"] if digest(v["Path"]).upper()!=v["Hash"]]
    assert not protected_changed
    # Actual recovered dataset class executes only val.csv, never its all-split factory.
    namespace=extracted_namespace(ORIGINAL/"dataset.py",{"BUSIDataset"},dict(
        pd=pd,torch=torch,os=os,Image=Image,Dataset=torch.utils.data.Dataset))
    transform=original_transform()
    original_ds=namespace["BUSIDataset"](str(ROOT/"data/processed/val.csv"),str(ROOT),transform)
    val=inventory[inventory.split=="val"].sort_values("row_index")
    assert len(original_ds)==len(val)
    for index,r in enumerate(val.itertuples()):
        tensor,label=original_ds[index]
        with Image.open(r.path) as im:expected=transform(im.convert("RGB"))
        assert label==r.label
        torch.testing.assert_close(tensor,expected,rtol=0,atol=0)
    predictions=pd.read_csv(out/"all_accessible_predictions.csv")
    for row in predictions.itertuples():
        logits=np.array(json.loads(row.logits));probs=np.exp(logits-logits.max());probs/=probs.sum()
        assert int(probs.argmax())==row.predicted_index and CLASSES[row.predicted_index]==row.predicted_class
        np.testing.assert_allclose(list(json.loads(row.probabilities).values()),probs,atol=2e-7,rtol=2e-6)
    reviewed=pd.read_csv(out/"reviewed_similarity_candidates.csv")
    assert reviewed[reviewed.prior_rank.notna()].reviewed.all()
    groups=pd.read_csv(out/"duplicate_groups.csv")
    intervals=cluster_intervals(out,predictions,groups,reviewed)
    for key,g in groups.groupby("group_id"):
        assert len(g)==g.group_size.iloc[0]
        assert bool(g.cross_split_group.iloc[0])==(g.split.nunique()>1)
        assert bool(g.conflicting_label_group.iloc[0])==(g.label.nunique()>1)
    for path in out.glob("*.csv"):pd.read_csv(path)
    for path in out.glob("*.json"):json.loads(path.read_text(encoding="utf-8-sig"))
    images=list((out/"duplicate_review_contact_sheets").glob("*.png"))+list(out.glob("*.png"))
    for path in images:
        with Image.open(path) as im:im.verify()
    accessed=set();denied=[]
    for name in ("scan_data_access.json","resume_data_access.json","finalize_data_access.json"):
        record=json.loads((out/name).read_text());accessed.update(record["accessed"]);denied+=record["denied"]
    expected=set(inventory.path)|{str((ROOT/f"data/processed/{n}.csv").resolve()) for n in ("train","val")}
    prior_expected={str(p.resolve()) for p in (ROOT/"data/processed/busi_phash_visual_review").glob("*.csv")}
    assert accessed <= expected|prior_expected
    assert not denied
    provenance=json.loads((out/"provenance_audit.json").read_text())
    assert digest(Path(provenance["original_checkpoint"]))==provenance["recovered_checkpoint_sha256"]
    result=dict(status="PASSED",original_image_hashes_checked=len(inventory),original_images_changed=changed,
        protected_files_hash_checked=len(state["hashes"]),protected_files_changed=protected_changed,
        recovered_original_validation_dataset_cases=len(original_ds),frozen_predictions_recomputed=len(predictions),
        union_find_independently_verified_by="SciPy sparse connected components; subset metrics exact agreement",
        groups_verified=int(groups.group_id.nunique()),images_decoded=len(images),original_external_checkpoint_unchanged=True,
        source_access_guard_files=len(accessed),denied_accesses=denied,
        accessed_only_train_val_and_prior_csv=True,
        test_split="No test manifest/image accessed; no test membership independently inspected",
        access_scope="Python audit-hook coverage on inference/scan. Supplementary verification reads listed paths only; not OS-wide tracing.")
    save_json(out/"verification_results.json",result)
    return result,intervals


def historical_warnings(out):
    rows=[]
    history=ROOT/"data/processed/lumina_reliability_investigation"
    previous=pd.read_csv(history/"image_level_results.csv")
    busi=previous[previous.model=="busi"].copy()
    # Reassign labels from the saved NUMERIC logits; avoid trusting historical display labels.
    busi["historical_original_display"]=busi.original_predicted_class
    busi["historical_perturbed_display"]=busi.perturbed_predicted_class
    busi["original_predicted_class"]=busi.original_logits.map(lambda s:CLASSES[int(np.argmax(json.loads(s)))])
    busi["perturbed_predicted_class"]=busi.perturbed_logits.map(lambda s:CLASSES[int(np.argmax(json.loads(s)))])
    busi["class_order"]=json.dumps(CLASSES)
    busi["original_correct"]=busi.original_predicted_class==busi.ground_truth_label
    busi["provenance_warning"]="Corrected label interpretation only; original numeric logits, score deltas and masks unchanged. BUS-BRA cross-dataset observations, not native BUSI accuracy."
    busi.to_csv(out/"historical_busi_predictions_corrected.csv",index=False)
    for p in history.rglob("*"):
        if p.is_file() and p.suffix in (".md",".csv",".json") and not p.name.startswith("provenance_"):
            if p.name in ("README.md","CODEX_HANDOFF.md","execution_notes.json","verification.json","investigation_summary.json","image_level_results.csv","selected_cases.csv","model_summary.csv","perturbation_summary.csv","case_level_summary.csv"):
                rows.append(dict(path=str(p),issue="Historical BUSI class display/provenance may reflect old order [Benign,Malignant,Normal].",
                    disposition="Preserved; use new warning and corrected numeric-index-derived BUSI table. Scalar perturbation deltas/flips are invariant; named predictions and correctness must be corrected."))
    cam_rows=[]
    for p in (ROOT/"data/processed/lumina_gradcam_research").glob("*_metadata.json"):
        metadata=json.loads(p.read_text())
        if metadata.get("model")!="busi":continue
        order=metadata.get("class_order",[])
        if tuple(order)!=CLASSES:
            target=metadata.get("target_class");pred=metadata.get("predicted_class")
            cam_rows.append(dict(metadata_path=str(p),old_target=target,corrected_target=CLASSES[order.index(target)],
                old_prediction=pred,corrected_prediction=CLASSES[order.index(pred)],
                interpretation="Numeric CAM target unchanged. Saved plot/file names use obsolete class semantics; relabel/regenerate into a fresh folder before interpretation."))
            rows.append(dict(path=str(p),issue="Grad-CAM class-name mapping obsolete",disposition="Preserved; correction indexed in historical_gradcam_label_corrections.csv"))
    pd.DataFrame(cam_rows,columns=["metadata_path","old_target","corrected_target","old_prediction","corrected_prediction","interpretation"]).to_csv(out/"historical_gradcam_label_corrections.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"historical_provenance_warnings.csv",index=False)
    counts={str(k):int(v) for k,v in busi[busi.region=="original"].original_predicted_class.value_counts().items()}
    return dict(corrected_busi_rows=len(busi),warning_files=len(rows),gradcam_metadata_corrections=len(cam_rows),corrected_prior_pilot_original_classes=counts)


def build_report(out,verification,intervals,history):
    integrity=json.loads((out/"final_dataset_integrity_summary.json").read_text())
    provenance=json.loads((out/"provenance_audit.json").read_text())
    application=json.loads((out/"application_inference_audit.json").read_text())
    frontend=json.loads((out/"frontend_mapping_test.json").read_text())
    decision=json.loads((out/"busi_final_decision.json").read_text())
    metrics=pd.read_csv(out/"validation_subset_metrics.csv")
    pairs=pd.read_csv(out/"reviewed_similarity_candidates.csv")
    groups=pd.read_csv(out/"duplicate_groups.csv")
    conflicts=pd.read_csv(out/"conflicting_label_groups.csv")
    initial=json.loads((out/"initial_state.json").read_text(encoding="utf-8-sig"))
    notes_path=out/"execution_notes.json"
    notes=json.loads(notes_path.read_text()) if notes_path.exists() else {}
    class_conflicts=[]
    for gid,g in conflicts.groupby("group_id"):
        class_conflicts.append(dict(group_id=gid,images="; ".join(f"{r.split}/{r.name} [{r.label_name}] -> {r.predicted_class} ({r.score:.4f})" for r in g.itertuples()),
            status="Label diagnosis unresolved; expert/source adjudication required"))
    compact=metrics[["subset","n","correct","accuracy","balanced_accuracy","macro_f1","normal_recall","benign_recall","malignant_recall","accuracy_ci_low","accuracy_ci_high"]]
    counts=integrity
    summary=f"""BUSI final reliability investigation completed. Decision B: retain models/best_model.pth as a historical research baseline and exclude BUSI from reliability-dependent automated consensus pending independent evaluation. This recommendation was not applied to production consensus; the current application still includes BUSI.

Audited 702 original BUSI images (624 training, 78 validation) and screened all 246,051 accessible pairs using SHA-256, decoded pixel hashes, pHash, dHash, whole/central correlation and explicitly aligned SSIM. Rechecked all 78 prior nearest distances and visually reviewed all 30 old candidate pairs plus 137 additional pairs using actual image-viewing tools. Review was by the assistant, not a clinician.

Found {counts['confirmed_strong_groups']} confirmed-or-strong similarity groups, including {counts['cross_split_confirmed_strong_groups']} cross-split groups affecting 30/78 validation images (38.46%). The conservative confirmed-only count is {counts['confirmed_groups']} groups / {counts['cross_split_confirmed_groups']} cross-split groups / 28 affected validation images (35.90%). Observed-screen contamination estimate: 28–30/78, not verified patient grouping or a statistical bound. Six conflicting-label groups contain 13 images, including 3 validation images; diagnosis remains unresolved. A new derived quarantine manifest was created without relabeling originals.

Correct class order is Normal, Benign, Malignant. Original scripts were recovered at {ORIGINAL}, and that project's checkpoint is byte-identical to Lumina's (SHA-256 {provenance['checkpoint_sha256']}). Full validation reproduced 71/78 correct = 91.03%, balanced accuracy 91.23%, macro-F1 90.88%; recall Normal 12/13, Benign 40/44, Malignant 19/21. Old display mapping would yield 3/78 apparent matches.

Diagnostic subsets: confirmed contaminated 25/28 = 89.29%; no confirmed duplicates 46/50 = 92.00%; excluding confirmed+strong matches 44/48 = 91.67%; excluding all pHash<=8 candidates also 44/48 = 91.67%. Malignant recall in the latter remainder is 15/17 = 88.24% (nominal Wilson 95% interval 65.66–96.71%). The 48-image remainder's nominal accuracy interval is 80.45–96.71%. No clear accuracy inflation is demonstrated by these subgroups; they cannot measure the causal effect of leakage. Every subset still reuses checkpoint-selection validation data and is not an independent test set.

Application mapping passed: 78 actual endpoint-function cases with real frozen BUSI, max probability difference {application['backend_probability_max_error']:.2g}; exact original/backend/unified preprocessing tensors on 702 images; Grad-CAM label/target-index checks; all three named consensus paths; frontend response adapter on all 78 real cases. Other models and heatmap output were explicit stubs only in isolated endpoint tests. No database initialization or production network calls occurred. Separate pre-existing concerns remain: consensus tie policy, consensus-label/BUSI-score association and explicitly marked demo reports.

Created research code: src/audit_busi_final_reliability.py, src/test_audit_busi_final_reliability.py, src/test_busi_frontend_mapping.cjs and src/report_busi_final_reliability.py under {ROOT}. No existing project source, checkpoint, original image, manifest, database or previous research output changed; {verification['protected_files_hash_checked']} protected file hashes and all 702 image hashes matched. Historical incorrect BUSI labels are indexed in provenance warnings and corrected derived tables; old outputs remain intact. Corrected prior BUS-BRA pilot BUSI originals are Normal 14, Benign 1, Malignant 9; previous class-name interpretations must be replaced while numeric perturbation effects remain unchanged.

Passed: syntax, 11 isolated unit tests, checkpoint/label checks, reproduced validation, backend/frontend inference tests, independent sparse-graph verification of union-find, subset metric recomputation, CSV/JSON/image checks and protected-file hashes. No locked test manifest or test images were accessed. Browser rendering, full server/database integration, expert diagnostic adjudication and independent patient-level reliability remain unverified. No retraining, package installation, commit or push occurred.

Phase 2 can resume as controlled research using the corrected BUSI mapping, with BUSI reported separately and excluded from reliability-dependent consensus. No automatic retraining is necessary to resume. Any future replacement study must resolve/quarantine conflicts, keep similarity groups together before fitting and use a fresh versioned checkpoint and evaluation protocol; never overwrite best_model.pth.

Detailed handoff: {out/'BUSI_CODEX_HANDOFF.md'}
Decision: {out/'busi_final_decision.json'}
Metrics: {out/'validation_subset_metrics.csv'}
All results, contact sheets and quarantine manifest: {out}
"""
    sections=[
        "# LUMINA BUSI — FINAL RELIABILITY INVESTIGATION",
        "## 1. Executive Summary\n\nThe original model technically works with the corrected output mapping. Its 91.03% validation result is reproducible. Substantial same-frame duplication and conflicting image labels compromise the evaluation design. Diagnostic exclusion does not lower accuracy, so a numerical leakage penalty cannot be estimated. Retain the checkpoint for restricted research use (Decision B) and resume Phase 2 with explicit separation from automated-consensus reliability claims.",
        "## 2. Execution Status\n\nCompleted. All required audit phases and deliverables were executed/generated. Runtime checks are isolated from production data. Expert diagnosis, verified patient identity and independent generalization remain unverified; no clinical certification is claimed.",
        "## 3. Original BUSI Configuration\n\n"+json.dumps(provenance,indent=2)+"\n\nCurrent architecture strictly loads all state-dictionary tensors as EfficientNet-B0 / Linear(1280,3). Mean [0.485,0.456,0.406], std [0.229,0.224,0.225], RGB PIL Resize((256,256)), ToTensor. Model eval, no_grad/inference_mode, float32, no TF32, deterministic algorithms. Recovered original dataset class was directly exercised on all 78 validation rows without calling the all-split factory. Original training source selects best validation accuracy; fixed split random_state=42 does not make the unseeded initial dataframe shuffle fully reproducible.",
        "## 4. Initial Problems\n\nThe old class order [Benign,Malignant,Normal] cyclically misnamed the true output indices [Normal,Benign,Malignant]. That correction was already present before this audit and was preserved. Old apparent accuracy is 3/78; correct mapping gives 71/78. A prior image-level split allowed differently encoded/resized versions of the same frame to cross into validation; label conflicts were also suspected.",
        "## 5. Dataset Integrity Results\n\n"+json.dumps(integrity,indent=2)+"\n\nCounts distinguish shortlisted edges from grouped images: 317 screened candidate pairs include low-quality nearest alternatives retained as uncertain; only confirmed/strong edges enter the primary 117 groups. Six conflicting groups / 13 images are quarantined, without changing data. Exact file and decoded-pixel cross-split equality are zero; one exact SHA-identical conflicting pair occurs within training. All 702 original files exist and manifest numeric/name labels agree. Dataset medical labels were not adjudicated.",
        "## 6. Visual Duplicate Findings\n\nAll 30 prior pairs were independently checked computationally and viewed in the six existing sheets. New review pages 008–042 cover all 137 additional shortlisted plausible pairs (and repeated prior cases). Review classifications: "+str(pairs[pairs.reviewed].similarity_classification.value_counts().to_dict())+". Remaining broad-screen alternatives stay UNCERTAIN without invented review claims. Visual judgments compare spatial texture, superficial bands, internal structures, calipers/text and overlays; they are assistant image-similarity judgments, not clinician diagnosis. Ambiguous crops and weak alignment retain STRONG/UNCERTAIN. `visual_review.json` records every viewed pair and evidence; reviewed_similarity_candidates.csv is the authoritative final classification table. all_similarity_candidates.csv contains the same final reviewed classifications; similarity_screen_candidates.csv preserves the initial pre-review evidence.\n\nSSIM uses documented Gaussian population moments, sigma 1.5, constants .01/.03, 11x11 support and 5px valid rim. Whole-frame grayscale bilinear resize to 128 square fixes correspondence; central [13:115] reduces background influence. No SSIM-optimizing registration was performed. This alignment can lower similarity for genuine crop variants and cannot establish identity alone.",
        "## 7. Cross-Split Leakage Assessment\n\nConfirmed-only: 106 groups, 28 cross-split groups, 28/78 affected validation images (35.90%). Confirmed+strong: 117 groups, 29 cross-split groups, 30/78 images (38.46%); 87 train-only groups and 1 validation-only group. Largest group has 7 accessible images. Normal contamination 6/13 (46.15%), Benign 20/44 (45.45%), Malignant 4/21 (19.05%) under the primary definition. The observed conservative range is 28–30, not a population bound: transformations may be missed and similarity edges can falsely merge frames. No verified patient or examination metadata exists. Same bytes, same decoded pixels, same underlying frame, similar acquisition and same patient are distinct claims. Group transitivity was independently checked with SciPy connected components. pHash-only threshold sensitivity is saved separately and is not duplicate proof.",
        "## 8. Original Model Evaluation\n\n"+markdown_table(compact.iloc[:1])+"\n\nConfusion matrix (true rows / predicted columns Normal, Benign, Malignant): [[12,1,0],[1,40,3],[0,2,19]]. Precision: Normal 92.31%, Benign 93.02%, Malignant 86.36%. Recall: 92.31%,90.91%,90.48%. Scores are softmax outputs, not calibrated clinical confidence.",
        "## 9. Leakage-Aware Diagnostic Evaluation\n\n"+markdown_table(compact)+"\n\nAll 0–1 metrics are proportions. Accuracy intervals above are nominal Wilson 95% image-level intervals; dependence can make them too narrow. Supplementary similarity-cluster bootstrap intervals (2000 draws, seed 42):\n\n"+markdown_table(intervals)+"\n\nThe remainder after primary exclusions is 44/48 (91.67%), slightly higher than full validation by 0.64 percentage points. Confirmed contaminated cases are 25/28 (89.29%); confirmed+strong contaminated cases are 27/30 (90%). This pattern does not demonstrate a simple inflation effect. Cross-label duplicates contribute errors as well as correct matches; class mix and difficulty differ. No corrected independent accuracy, arbitrary subtraction or causal leakage estimate is defensible. Malignant remainder recall 15/17=88.24%, Wilson 95% 65.66–96.71%, is too imprecise for reliability certification. All subsets retain validation checkpoint-selection reuse. Similarity-cluster resampling still does not establish patient independence. `cluster_representative_metrics.csv` also collapses repeated validation frames for a descriptive sensitivity check.",
        "## 10. Conflicting Labels\n\n"+markdown_table(pd.DataFrame(class_conflicts))+"\n\nThe two previously identified pairs are confirmed visually: benign (406) / malignant (94), and benign (437) / normal (34). Additional conflicts include benign (131)/benign (42)/malignant (51), normal (1)/benign (269), malignant (93)/benign (399), and exact-identical malignant (145)/benign (433). Model scores can differ across crop/encoding variants and must not resolve diagnosis. Expert/annotation-source review is required. quarantine_manifest.csv is a new derived list, not file deletion or relabeling.",
        "## 11. Application Integration Verification\n\n"+json.dumps(application,indent=2)+"\n\nFrontend pure-function runtime result:\n\n"+json.dumps(frontend,indent=2)+"\n\nThe actual AST-extracted backend model builder and async endpoint ran with the real BUSI checkpoint on all 78 validation inputs. Other models/heatmap generation were explicit test stubs and are not claimed as full production inference. Original evaluator was inspected but not executed because its factory reads the locked test split. Original transform/architecture and actual recovered BUSIDataset validation loading supplied the reference. No remaining BUSI class-index defect was found; no production fix was made. Existing score association and unspecified consensus tie policy are separate concerns. Browser rendering and full HTTP/database routes were not exercised.",
        f"## 12. Files Created and Modified\n\nNew code under `{ROOT/'src'}`: audit_busi_final_reliability.py (scan/inference/grouping/subsets), test_audit_busi_final_reliability.py (11 mathematical/safety tests), test_busi_frontend_mapping.cjs (actual pure frontend adapter tests), report_busi_final_reliability.py (verification, historical corrections and report). All new artifacts are under `{out}`. `artifact_inventory.csv` enumerates every exact absolute path and purpose. No existing project source or prior result was modified. Initial/final Git states: initial_state.json and final_git_status.txt. Initial pre-existing changes were preserved. History: {history}; corrected derived BUSI rows and CAM label index are in this folder. Warnings identify historical reports requiring revised label interpretation. Numeric fixed-target perturbation scores/flips remain unchanged; named predictions, task correctness and named CAM targets require correction.",
        "## 13. Commands Executed\n\n"+notes.get("commands","See execution_notes.json and README for actual commands. No inferred command outcomes.")+"\n\nNo training/evaluation test-set command, package installation, database launch, commit or push was run.",
        "## 14. Tests and Verification\n\n"+json.dumps(verification,indent=2)+"\n\nPASSED: Python syntax; 11 isolated tests; all-source hash preservation; train/val loading; strict checkpoint load; 71/78 reproduction; backend class/probability/Grad-CAM-index consistency; 3 consensus class fixtures; frontend adapter 78 real cases; prior similarity reproduction; SSIM identity/symmetry/constant checks; independent graph/subset cross-check; CSV/JSON/image integrity.\n\nResolved failures: initial endpoint harness lacked a positional File argument; fixed in new audit code only. A second attempt stalled while using the asyncio.run endpoint wrapper on Windows (the exact blocked call was not stack-traced); it was interrupted, then driven as an immediate in-memory coroutine, with any real asynchronous I/O rejected. No image-similarity computation repeated on resume; frozen inference was batched for completion. `scan_failure.json` preserves the first failure. scikit-image was unavailable; installed SciPy implements tested SSIM, with no installation. Process-inspection CIM access was unavailable and not needed.\n\nNOT EXECUTED: browser rendering, live HTTP server, DB/patient routes, original test-only evaluator/factory, retraining, independent external evaluation. NOT VERIFIABLE: medical ground truth, patient/examination identity, missed duplicate transformations and complete immutable historical training-run lineage. No current unresolved executed check failure.",
        "## 15. Known Limitations\n\nValidation checkpoint selection persists in every subgroup; no locked test split was opened. Similarity graphs are not patient groups; single-link chains can overmerge. Multi-metric screening can miss crop/rotation/brightness variants, and black backgrounds/ultrasound texture can create false matches. Assistant visual review is not independent expert adjudication and source labels were visible. Normalized-square alignment distorts aspect ratios; native dimensions and central metrics are saved. Bound counts are operational definitions, not truth guarantees. Small malignant subgroups and uncertain labels prevent clinical conclusions. Native BUSI results do not establish cross-dataset performance. The previous BUS-BRA perturbation evidence used incorrect BUSI names; numerical changes survive but semantic interpretation must use corrected outputs. Nominal and similarity-cluster intervals do not resolve selection bias, unknown patient dependence or domain shift.",
        "## 16. FINAL BUSI DECISION\n\nB — RETAIN BUSI, BUT EXCLUDE IT FROM AUTOMATED CONSENSUS. The checkpoint and corrected integration reproduce source-validation performance. Dataset duplication, conflicting labels and validation reuse leave generalization unverified. Preserve it as a historical research baseline and report it separately. This audit does not establish that replacement training is necessary, nor would re-splitting and retraining alone solve label errors or prove clinical reliability. Production consensus was not changed because the request prohibits changing it solely from research assumptions; current inclusion remains visible and is not presented as an applied restriction.",
        "## 17. Recommended Next Actions\n\n1. Resume Phase 2 controlled perturbation research with corrected BUSI labels and versioned outputs; report BUSI separately and exclude its vote from reliability-dependent consensus analyses.\n2. Apply the new historical provenance warnings; use corrected saved BUSI label tables. No new inference is necessary to fix historical numeric-index naming.\n3. Keep conflicting groups quarantined in derived research manifests and obtain annotation-source/expert adjudication before any new BUSI fitting/evaluation.\n4. If Lumina will continue to use automatic product consensus, explicitly implement the restricted participation policy as a separate scoped product change and resolve score semantics/tie behavior; this was not silently applied.\n5. Only if a replacement is subsequently approved: create a separate, versioned protocol and checkpoint such as models/busi_v2_grouped_best.pth; group duplicate variations before fitting, resolve label conflicts first, keep old best_model.pth immutable and reserve a genuinely untouched evaluation source. No further audit is required merely to return to Phase 2.",
        "## 18. Impact on Lumina's Reliability Investigation\n\nPhase 2 can resume as research. Corrected current loader mapping is verified, original BUSI provenance is recovered, and historical named-class interpretations are superseded by the new warning/correction files. BUSI may be analyzed as a frozen baseline but should not supply an automated consensus vote used as evidence of reliability. Numeric prior perturbation effects are not proof of shortcuts. The current application still includes BUSI: this final research recommendation and deployed behavior are explicitly distinguished. Other classifiers, segmentation work, datasets and historical experiments remain intact.",
        "## 19. Ready-to-Paste ChatGPT Summary\n\n"+summary,
    ]
    (out/"BUSI_CODEX_HANDOFF.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    save_json(out/"ready_to_paste_summary.json",dict(summary=summary))
    readme=f"""# BUSI final reliability audit

Completed: 624 train + 78 validation images; 246,051 pair screens; 167 actually viewed plausible pairs. Final decision B: retain frozen BUSI for research, restrict automated-consensus use. This is not independent or clinical validation.

Read BUSI_CODEX_HANDOFF.md for all 19 requested sections. Authoritative reviewed pair classifications are in all_similarity_candidates.csv and its reviewed_similarity_candidates.csv alias. similarity_screen_candidates.csv preserves the raw pre-review screen. dataset_integrity_summary.json and final_dataset_integrity_summary.json contain final grouping totals; similarity_scan_summary.json preserves the raw scan summary. duplicate_groups.csv uses CONFIRMED + STRONG edges; cross_split_duplicate_groups.csv filters those groups. Conflicting-label records include real frozen predictions and unresolved diagnoses; quarantine_manifest.csv is derived and never moves/relabels images.

## Reproduce (PowerShell, repository root)

```powershell
& 'C:\\Users\\hks89\\anaconda3\\envs\\bc_ai\\python.exe' -m py_compile src/audit_busi_final_reliability.py src/test_audit_busi_final_reliability.py src/report_busi_final_reliability.py
& 'C:\\Users\\hks89\\anaconda3\\envs\\bc_ai\\python.exe' -m unittest discover -s src -p test_audit_busi_final_reliability.py -v
& 'C:\\Users\\hks89\\anaconda3\\envs\\bc_ai\\python.exe' src/audit_busi_final_reliability.py --scan --output-dir data/processed/busi_final_reliability_audit/new_run
& 'C:\\Users\\hks89\\anaconda3\\envs\\bc_ai\\python.exe' src/audit_busi_final_reliability.py --finalize --output-dir data/processed/busi_final_reliability_audit/new_run --visual-review data/processed/busi_final_reliability_audit/visual_review.json
```

Original source location is C:\\Breast_Cancer_Project\\src. No original all-split factories or test evaluator are called. Review JSON pair IDs depend on exact manifest ordering/hashes; re-review if inputs change. --resume-scan reuses saved similarities after verifying all source image hashes, and refuses to overwrite completed inference. Output folders must remain beneath this audit root and be fresh for new runs. Use the saved frontend_validation_inputs.json with `node src/test_busi_frontend_mapping.cjs <completed audit folder>` to reproduce the adapter test. Report generation uses `python src/report_busi_final_reliability.py --output-dir <completed audit>`; it refuses to overwrite reports unless --refresh-reports is explicit. Root verification and source lineage are recorded separately, not fabricated for new runs.

## Interpretation

Similarity metrics use whole/central normalized coordinates, fixed SSIM alignment, no patient metadata and conservative evidence labels. Initial screening is deliberately broad. No metric alone proves duplicate identity. Component chains can overmerge. pHash-only threshold_sensitivity.csv is a screen sensitivity analysis, not verified grouping. All full/subset metrics are actual frozen predictions with fixed Normal/Benign/Malignant indices; none is an independent test set. Metrics are proportions (multiply by 100 for percent). Wilson image-level intervals are nominal; validation_subset_cluster_intervals.csv supplies exploratory similarity-cluster bootstrap intervals. Both retain validation model-selection bias and unknown patient dependence.

Historical outputs are preserved. historical_busi_predictions_corrected.csv recomputes names from the original numeric logits, not new predictions; original scalar perturbation deltas remain valid numerically. historical_gradcam_label_corrections.csv corrects named numeric targets without altering CAM arrays. See historical_provenance_warnings.csv before using prior reports. No training, original data mutation, database execution or product consensus change occurred.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    files=sorted(p for p in out.rglob("*") if p.is_file())
    files += [ROOT/"src"/n for n in ("audit_busi_final_reliability.py","test_audit_busi_final_reliability.py","test_busi_frontend_mapping.cjs","report_busi_final_reliability.py")]
    files.append(out/"artifact_inventory.csv")
    pd.DataFrame(dict(path=str(p),purpose="New research source" if p.suffix in (".py",".cjs") else "Audit evidence, derived results, review image or report; see filename and README") for p in files).drop_duplicates("path").to_csv(out/"artifact_inventory.csv",index=False)
    print(json.dumps(dict(verification=verification,history=history,decision=decision["decision"]),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--refresh-reports",action="store_true");args=parser.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(OUT):raise ValueError("Output outside isolated audit root")
    if (out/"BUSI_CODEX_HANDOFF.md").exists() and not args.refresh_reports:raise FileExistsError("Report exists; use --refresh-reports explicitly")
    verification,intervals=verify(out)
    history=historical_warnings(out)
    build_report(out,verification,intervals,history)


if __name__=="__main__":main()
