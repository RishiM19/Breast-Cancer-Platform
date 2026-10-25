"""Run Stage 1 contract tests and a small frozen-inference regression, never SQL.

This is integration verification of four existing Phase 2 images, not another
reliability study, training run, or independent accuracy evaluation.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import ast
import difflib
import hashlib
import importlib.util
import io
import json
import sqlite3
import subprocess
import sys
import unittest
import py_compile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from .research_inference import ROOT, SPECS, PROFILES, ResearchEngine, ResearchResponse, contract_descriptor, sha256
from .research_api import create_research_router

OUT=ROOT/"data/processed/lumina_integration_stage1"


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def unit_tests():
    sys.path.insert(0,str(ROOT/"src"))
    names=["src.test_research_integration","test_artifact_sensitivity_lumina_classifiers",
           "test_lumina_reliability_phase2","test_audit_busi_final_reliability"]
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    with (OUT/"python_regression_tests.txt").open("w",encoding="utf-8") as log:
        with patch.object(sqlite3,"connect",side_effect=AssertionError("Database access forbidden in tests")):
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    record=dict(status="PASSED" if result.wasSuccessful() else "FAILED",tests_run=result.testsRun,
        suite_counts={name:unittest.defaultTestLoader.loadTestsFromName(name).countTestCases() for name in names},
        failures=[dict(test=str(t),traceback=trace) for t,trace in result.failures],
        errors=[dict(test=str(t),traceback=trace) for t,trace in result.errors],
        skipped=[dict(test=str(t),reason=why) for t,why in result.skipped],database_connections=0,
        note="New policy/API fixtures use explicit stub logits; previous mathematical/geometry unit tests reuse original sources. Real checkpoint verification is separate.")
    write("python_test_results.json",record)
    if not result.wasSuccessful():raise RuntimeError("Unit/regression tests failed; inspect log")
    return record


def real_inference():
    torch.set_num_threads(2)
    engine=ResearchEngine()
    prior=ROOT/"data/processed/lumina_reliability_phase2"
    selected=pd.read_csv(prior/"selected_cases.csv",dtype={"case_id":str}).groupby("birads",sort=True).head(1)
    originals=pd.read_csv(prior/"image_level_results.csv")
    originals=originals[originals.kind=="original"]
    responses=[];checks=[];db_attempts=[]
    def deny_db(*args,**kwargs):
        db_attempts.append("sqlite3.connect");raise AssertionError("Production database must not be accessed")
    # Real main.py application assembly, with the production router and a real
    # engine injected for observable provenance. TestClient is intentionally not
    # a context manager: database startup/lifespan is not executed.
    with patch.object(sqlite3,"connect",side_effect=deny_db):
        with patch("src.research_api.create_research_router",side_effect=lambda:create_research_router(engine)):
            spec=importlib.util.spec_from_file_location("integration_real_main",ROOT/"main.py")
            main=importlib.util.module_from_spec(spec);spec.loader.exec_module(main)
        client=TestClient(main.app)
        for i,row in enumerate(selected.itertuples()):
            path=Path(row.image_path);path=path if path.is_absolute() else ROOT/path
            raw=path.read_bytes();response=client.post("/predict",data={"source_domain":"BUS-BRA"},
                files={"file":(path.name,raw,"image/png")})
            assert response.status_code==200,response.text
            payload=response.json();result=ResearchResponse.model_validate(payload)
            assert result.status=="ok" and result.success and result.consensus.score is None
            assert result.source_domain=="BUS-BRA" and not result.consensus.eligible_model_ids
            for k,s in SPECS.items():
                m=result.models[s.display_name];old=originals[(originals.image_id==row.image_id)&(originals.model==k)].iloc[0]
                expected_logits=np.asarray(json.loads(old.original_logits))
                expected_scores=np.asarray(json.loads(old.original_probabilities))
                np.testing.assert_allclose(m.logits,expected_logits,atol=4e-5,rtol=5e-5)
                scores=np.array([m.probabilities[c] for c in PROFILES[k].classes])
                np.testing.assert_allclose(scores,expected_scores,atol=3e-6,rtol=0)
                assert m.predicted_class_index==old.original_target_index and m.prediction==old.original_predicted_class
                assert m.confidence==m.probabilities[m.prediction] and m.checkpoint.verification=="verified"
                assert not engine._models[k].training
                checks.append(dict(image_id=row.image_id,case_id=str(row.case_id),model_id=k,status="PASSED",
                    max_abs_logit_difference=float(np.max(np.abs(np.array(m.logits)-expected_logits))),
                    max_abs_score_difference=float(np.max(np.abs(scores-expected_scores))),
                    prediction=m.prediction,predicted_index=m.predicted_class_index,score=m.score,
                    checkpoint_sha256=m.checkpoint.sha256))
            assert result.attribution.status=="ok",result.attribution
            assert result.attribution.target_label==result.models["BUS-BRA"].prediction
            if i==0:
                import base64
                (OUT/"real_gradcam_attribution.jpg").write_bytes(base64.b64decode(result.heatmap_base64.split(",",1)[1]))
                repeat=engine.analyze(raw,"unknown",attribution=False)
                for name,m in result.models.items():
                    np.testing.assert_allclose(repeat.models[name].logits,m.logits,atol=2e-5,rtol=2e-5)
                assert all(m.eligibility.domain_relation=="unknown" for m in repeat.models.values())
            responses.append(dict(image_id=row.image_id,case_id=str(row.case_id),image_sha256=hashlib.sha256(raw).hexdigest(),response=payload))
            print(f"Real API regression {i+1}/{len(selected)}: {row.image_id}; four models and genuine Grad-CAM passed",flush=True)
        assert not db_attempts
    write("real_inference_responses.json",responses)
    record=dict(status="PASSED",cases=len(selected),model_predictions=len(checks),checks=checks,
        device=str(engine.device),torch_version=torch.__version__,database_connection_attempts=db_attempts,
        source="Four existing Phase 2 images, first selected image per BI-RADS category; no accuracy estimate or repeat reliability experiment.",
        asgi="Actual main.py app and production research router exercised in-process without database startup or external network.",
        attribution="Genuine BRA Grad-CAM succeeded for all four requests; original inference rechecked after hooks.",
        unexecuted=["Database startup/persistence endpoints","External HTTP server","Interactive browser workflow"])
    write("real_inference_test_results.json",record)
    return record


def protected_audit():
    initial=json.loads((OUT/"initial_state.json").read_text());changed=[];allowed=[];missing=[]
    for r in initial["files"]:
        p=Path(r["path"])
        if not p.is_file():missing.append(str(p));continue
        current=sha256(p)
        if current!=r["sha256"]:
            (allowed if r["may_change"] else changed).append(dict(path=str(p),before=r["sha256"],after=current))
    metadata=[]
    for folder in (ROOT/"data").iterdir():
        if folder.name!="processed" and folder.is_dir():
            metadata += [dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()]
    assert sorted(metadata,key=lambda r:r["path"])==sorted(initial["dataset_metadata"],key=lambda r:r["path"])
    manifest=pd.read_csv(ROOT/"data/processed/bra_val.csv").set_index("image_id")
    checked=0
    for r in initial["prior_validation_hash_audit"]:
        row=manifest.loc[r["image_id"]]
        for field,col in (("image_path","image_sha256"),("mask_path","mask_sha256")):
            p=Path(row[field]);p=p if p.is_absolute() else ROOT/p
            assert sha256(p)==r[col];checked+=1
    # Ensure the persistence schema/SQL and patient/feedback function bodies did
    # not change, independent of line movement and old inference helper removal.
    before=ast.parse((OUT/"before/main.py").read_text(encoding="utf-8"));after=ast.parse((ROOT/"main.py").read_text(encoding="utf-8"))
    names={"_get_db","init_db","PatientUpsert","FeedbackSubmission","list_patients","upsert_patient","delete_patient","submit_feedback","get_feedback_stats"}
    def blocks(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    assert blocks(before)==blocks(after) and len(blocks(before))==len(names)
    model_files=[r for r in initial["files"] if Path(r["path"]).is_relative_to(ROOT/"models")]
    database=next(r for r in initial["files"] if Path(r["path"])==ROOT/"lumina_database.db")
    research=[r for r in initial["files"] if Path(r["path"]).is_relative_to(ROOT/"data/processed")]
    record=dict(status="PASSED" if not changed and not missing else "FAILED",files_hashed=len(initial["files"]),
        preserved_files=len(initial["files"])-len(allowed),authorized_source_changes=allowed,unexpected_changes=changed,missing_files=missing,
        all_model_files_unchanged=not any(Path(r["path"]).is_relative_to(ROOT/"models") for r in changed),
        model_files_checked=len(model_files),model_file_hashes=model_files,
        database_bytes_unchanged=sha256(ROOT/"lumina_database.db")==database["sha256"],database_sha256=database["sha256"],
        prior_research_outputs_unchanged=not any(Path(r["path"]).is_relative_to(ROOT/"data/processed") for r in changed),
        prior_research_files_checked=len(research),
        native_dataset_files_metadata_unchanged=len(metadata),validation_image_and_mask_hashes_verified=checked,
        persistence_schema_and_route_bodies="AST-identical to pre-integration snapshot",database_queried_or_written=False,
        limitations="Raw dataset files outside the prior validation audit were compared by path/length/mtime, without reading held-out images/manifests. Checkpoint/database/prior-output files were SHA-256 checked.")
    write("preserved_files_audit.json",record)
    assert not changed and not missing
    return record


def reports():
    initial=json.loads((OUT/"initial_state.json").read_text())
    python=json.loads((OUT/"python_test_results.json").read_text())
    front=json.loads((OUT/"frontend_test_results.json").read_text())
    real=json.loads((OUT/"real_inference_test_results.json").read_text())
    audit=json.loads((OUT/"preserved_files_audit.json").read_text())
    assert all(r["status"]=="PASSED" for r in (python,front,real,audit))
    sources=["src/research_inference.py","src/research_api.py","src/test_research_integration.py","src/verify_integration_stage1.py"]
    for name in ["main.py"]+sources:py_compile.compile(str(ROOT/name),doraise=True)
    build=(OUT/"frontend_build.txt").read_text(encoding="utf-8-sig")
    assert "built in" in build and "error during build" not in build
    diff=subprocess.run(["git","diff","--check"],capture_output=True,text=True)
    assert diff.returncode==0,diff.stdout+diff.stderr
    (OUT/"final_git_status.txt").write_text(subprocess.check_output(["git","status","--short"],text=True),encoding="utf-8")
    failures=[dict(check="Initial in-process TestClient run",status="RESOLVED_ENVIRONMENT_BLOCK",
        finding="Restricted Windows event loop hung creating an internal socket pair; interrupted after diagnosis.",
        resolution="Database-isolated tests reran successfully with local event-loop networking enabled; no external API/network or database requests."),
        dict(check="First real Grad-CAM integration check",status="RESOLVED_IMPLEMENTATION_FAILURE",
        finding="Lazy model construction occurred inside torch.inference_mode, making parameters unsuitable for later Grad-CAM autograd.",
        resolution="Moved model load outside inference_mode, added a specific regression test, then all four real Grad-CAM requests and inference parity checks passed. No checkpoint change.")]
    unexecuted=["Database startup, SQL patient/feedback CRUD and live persistence; SQLite connect forbidden in tests",
        "External HTTP server and interactive browser upload/reload/print workflow",
        "Standalone TypeScript type-check (compiler/tsconfig not installed); Vite build and actual TS/React tests passed",
        "Historical test_busi_frontend_mapping.cjs: asserts obsolete vote semantics and writes protected BUSI-audit outputs; replaced by 78-record regression against the new actual adapter",
        "Training, package installation, clinical/independent test evaluation"]
    results=dict(status="PASSED",contract_version="1.0.0",syntax=dict(status="PASSED",files=["main.py"]+sources),
        python=python,frontend=front,frontend_build=dict(status="PASSED",log=str(OUT/"frontend_build.txt")),
        real_inference=real,protected_audit=str(OUT/"preserved_files_audit.json"),diff_check="PASSED",
        failed_unresolved=[],resolved_failures=failures,unexecuted=unexecuted,
        not_verifiable=["Clinical correctness/calibration/medical reasoning/generalization","Caller-declared upload domain or verified patient identity","Full concurrent server/load/browser behavior"],
        production_database_opened=False,models_retrained=False,packages_installed=False,committed_or_pushed=False)
    write("integration_test_results.json",results)
    descriptions={
        "main.py":"Mount the shared versioned research router; remove superseded three-model vote and old inference/CAM/diagnostic-fabrication helpers. Defer the unchanged database initializer to server startup. Patient/feedback schemas, SQL and route bodies are unchanged.",
        "src/App.tsx":"Use the tested response adapter; preserve complete multi-model metadata when saving. Never borrow BUSI score for abstention. Store only a documented zero placeholder in the legacy NOT NULL confidence column, retain null consensus in JSON, and handle save failures. Keep human review separate from model scores.",
        "src/types/patient.ts":"Represent four task-specific model records, nullable model/consensus scores, availability statuses, provenance, eligibility and optional human review while allowing historical rows.",
        "src/components/AnalysisResults.tsx":"Render shared pathology/BI-RADS research cards, BUSI restriction, uncalibrated scores and abstention; label the genuine BRA attribution and research review action.",
        "src/components/DiagnosticReport.tsx":"Versioned real results bypass demo substitution; show actual model cards, attribution and contract version without synthetic lesion metrics or clinical BI-RADS actions. Preserve the explicitly labeled historical demo view.",
        "src/research_inference.py":"Single versioned runtime contract/policy, pinned checkpoint hashes, shared frozen loaders/preprocessing, lazy serialized inference, explicit failure statuses and genuine BRA Grad-CAM without synthetic fallback.",
        "src/research_api.py":"Database-free /predict and /inference-contract router shared by the actual FastAPI app and tests. Validate uploads/domains and publish response schemas.",
        "src/lib/inferenceAdapter.ts":"Pure tested live/stored adapters: reject invalid score associations/unknown contracts, preserve individual results, separate human review and hide obsolete consensus scores.",
        "src/components/ResearchModelResults.tsx":"Shared research display groups pathology and BI-RADS separately and labels model scores, availability, domains and exclusions.",
        "src/test_research_integration.py":"Isolated Python contract/policy/API/preprocessing/precision/failure/side-effect tests; explicit stub logits only in unit fixtures.",
        "src/test_research_frontend.cjs":"Actual TypeScript/React/persistence-shape regressions using four saved real responses and 78 historical BUSI records; no database/network calls.",
        "src/verify_integration_stage1.py":"Run regression suites, real four-case API/Grad-CAM parity checks, protected-file audit and reproducible Stage 1 reports without querying a database.",
    }
    changed=[Path(r["path"]).relative_to(ROOT).as_posix() for r in audit["authorized_source_changes"]]
    created=[p for p in descriptions if p not in initial["authorized_existing_changes"]]
    patches=[];changes=[]
    for name in changed+created:
        before=(OUT/"before"/name).read_text(encoding="utf-8") if name in changed else ""
        after=(ROOT/name).read_text(encoding="utf-8")
        patchlines=list(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile="before/"+name,tofile="after/"+name))
        patches.extend(patchlines)
        changes.append(dict(path=name,change="modified" if name in changed else "created",description=descriptions[name],sha256=sha256(ROOT/name)))
    (OUT/"integration_changes.patch").write_text("".join(patches),encoding="utf-8")
    write("source_changes.json",changes)
    text="# Stage 1 source changes\n\nCompared against this session's captured source, including all pre-existing edits, not against Git HEAD.\n\n"
    text+="\n\n".join(f"- **{c['change'].upper()} `{ROOT/c['path']}`**: {c['description']}" for c in changes)
    text+="\n\nAll new report/test/snapshot artifacts are under `"+str(OUT)+"`. `artifact_inventory.csv` lists them. `integration_changes.patch` is the exact session diff; original source snapshots are in `before/`. No Git commit/push/reset or checkpoint/database/data/prior-report edit occurred.\n"
    text+="\nPre-existing Git modifications to the database and three sources remain visible in git diff; they are not all changes made by this stage. Database SHA-256 remained identical to session start.\n"
    (OUT/"files_changed.md").write_text(text,encoding="utf-8")
    commands=["python -m py_compile main.py src/research_inference.py src/research_api.py src/test_research_integration.py src/verify_integration_stage1.py",
        "python -m src.verify_integration_stage1 --unit --real", "node src/test_research_frontend.cjs", "npm.cmd run build",
        "python -m src.verify_integration_stage1 --audit --report", "git diff --check"]
    write("execution_notes.json",dict(python="C:/Users/hks89/anaconda3/envs/bc_ai/python.exe",cwd=str(ROOT),commands=commands,
        resolved_failures=failures,scope="No SQL/database/server startup; no training, installations, independent dataset evaluation, commits or pushes.",
        checks="Real inference uses all four frozen checkpoints through main.py's actual router. Unit tests use labeled stubs for failures; frontend uses saved real payloads. No reliability investigation was rerun."))
    max_logits=max(c["max_abs_logit_difference"] for c in real["checks"])
    max_scores=max(c["max_abs_score_difference"] for c in real["checks"])
    summary=f"""Lumina Integration Stage 1 is implemented and tested in {ROOT}. This stage did not retrain models or repeat the completed reliability investigations.

The FastAPI /predict endpoint now returns research inference contract 1.0.0, with model IDs, SHA-256 checkpoint versions, ordered labels, preprocessing, source-training domains, individual logits/per-class scores, eligibility and ok/missing/not_run/error statuses. GET /inference-contract publishes configuration, policy and the response schema. The four frozen models remain available: BUSI [Normal, Benign, Malignant], UCLM [Normal, Benign, Malignant], BUS-BRA pathology [Benign, Malignant], and BUS-BRA BI-RADS [2, 3, 4, 5]. Existing Phase 2 loaders and preprocessing are reused; pathology uses RGB 256-square bilinear resize and BI-RADS uses RGB centered black 224-square pad, each with one ImageNet normalization.

BUSI is always excluded from automated consensus but its individual prediction remains visible. UCLM has native-domain individual research support only for UCLM, never assumed eligibility on BUS-BRA uploads. BUS-BRA pathology and BI-RADS have separate individual research support on BUS-BRA, and BI-RADS never votes on pathology. Unknown uploads default to unknown; supplied domains are explicitly unverified. No current model/combination is approved for a validated consensus. Agreement, disagreement, ties, missing models and high scores therefore return exactly “No validated consensus available,” with zero participants and null consensus score/confidence. No label is paired with another model's score.

The React analysis and saved-report paths preserve all four results and display pathology separately from BI-RADS, with uncalibrated model scores, availability and exclusions. Versioned real results bypass the demo report. Genuine BRA Grad-CAM targets the actual predicted class; failures return unavailable attribution without a synthetic heatmap. New inference does not invent size, grade, stage, morphology or clinical BI-RADS recommendations. Human review remains separate from frozen predictions.

Compatibility: /predict still accepts multipart file-only uploads and retains models, consensus, heatmap_base64 and image fields; source_domain is optional. Three existing model display keys remain, with BUS-BRA BI-RADS added. Per-model confidence remains an explicitly uncalibrated alias for that model's own predicted-class score; BRA Normal stays null for older clients, not a third class. Consensus vote semantics intentionally changed to abstention. Patient/feedback request schemas, SQL and route bodies are unchanged. Database initialization now occurs at server startup rather than import. No database migration or patient-record edit was performed. The existing NOT NULL confidence column requires a zero storage placeholder for abstentions; versioned JSON retains null, and adapters display Unavailable rather than treating zero as a probability. Historical rows are not rewritten; obsolete consensus scores are suppressed in display.

Verification passed: {python['tests_run']} Python tests ({python['suite_counts']['src.test_research_integration']} new plus {python['tests_run']-python['suite_counts']['src.test_research_integration']} existing), {front['tests_run']} frontend/React/persistence-shape tests, syntax checks, Vite production build and Git diff check. The actual main.py application/router ran in-process on four existing Phase 2 validation images, one per BI-RADS category: all {real['model_predictions']} real model predictions matched saved original labels/indices and scores; maximum absolute logit difference {max_logits:.3g}, score difference {max_scores:.3g}. Production configures float32, no TF32 and deterministic algorithms in the shared engine itself. Genuine Grad-CAM passed all four requests, and original inference remained stable after hooks. Frontend tests also covered all 78 saved historical BUSI records. SQLite connections were forbidden; no database was queried or written.

Preservation passed: {audit['preserved_files']} protected files unchanged out of {audit['files_hashed']} initially hashed files, with only five authorized existing sources changed. All model files, database bytes and prior research outputs are unchanged. All 540 previously audited validation image/mask hashes matched; metadata for {audit['native_dataset_files_metadata_unchanged']} native dataset files was unchanged. Initial/final Git status and exact session diffs are saved. No locked test images/manifests were opened for evaluation, and no training, package installation, commit or push occurred.

Modified existing files: {', '.join(changed)}. Created source/test files: {', '.join(created)}. All new integration artifacts are under {OUT}; files_changed.md and source_changes.json enumerate exact changes, and before/ preserves the starting source contents.

Resolved issues: the initial Windows in-process test run stalled on internal event-loop socket creation and passed after isolated execution with local networking enabled. The first real attribution check caught model construction inside inference_mode; loading was moved outside that context and a regression test added. Final unresolved test failures: none.

Not executed: database startup/SQL CRUD, external HTTP server, interactive browser upload/reload/printing, standalone TypeScript type-check (compiler/tsconfig absent), or independent/clinical evaluation. Unit/React/ASGI results do not establish full UI/server/database integration or clinical readiness. Existing dashboard/feedback statistics and the legacy demonstration view remain outside this stage's inference validation. Remaining risks include the legacy database confidence placeholder, unverified upload domains, validation reuse, unsupported cross-dataset generalization and untested concurrent server/browser behavior.

Next milestone: Stage 2 research-workflow integration testing against an explicitly isolated database copy, covering upload → save → reload → human review → report/print, unavailable states and dashboard/feedback semantics. Keep the same frozen checkpoints and abstention policy; do not resume automatic majority voting or retraining.

Handoff: {OUT/'INTEGRATION_STAGE1_HANDOFF.md'}
Contract: {OUT/'integration_contract.json'}
Policy: {OUT/'model_eligibility_policy.json'}
Tests: {OUT/'integration_test_results.json'}
Preservation audit: {OUT/'preserved_files_audit.json'}
Changes: {OUT/'files_changed.md'}
"""
    write("ready_to_paste_summary.json",dict(summary=summary))
    sections=["# Lumina Integration Stage 1 handoff",
        "## 1. Execution status\n\nImplemented and verified. Research-only individual outputs; all automated consensus abstains. Reliability studies/checkpoints were reused, not repeated or trained.",
        "## 2. Model eligibility\n\n| Model | Task | Native individual research context | Consensus |\n|---|---|---|---|\n| BUSI | Pathology, Normal/Benign/Malignant | Historical display only | Always excluded |\n| UCLM V2 | Pathology, Normal/Benign/Malignant | UCLM; other domains are secondary cross-dataset observations | Excluded; no validated ensemble |\n| BUS-BRA pathology | Pathology, Benign/Malignant | BUS-BRA | Excluded; no validated ensemble |\n| BUS-BRA BI-RADS V1 | Categories 2/3/4/5 | BUS-BRA, distinct from pathology | Never a pathology vote; no validated ensemble |\n\nDomain claims are unverified and do not unlock voting. All four inference outputs are available for research. Complete preprocessing/labels/hash versions and domain matrix are in integration_contract.json/model_eligibility_policy.json.",
        "## 3. Contract and BUSI enforcement\n\nSingle runtime policy in src/research_inference.py reuses the frozen PROFILES/loader/preprocess functions from the preserved unified Grad-CAM module. Checkpoints are SHA-256 pinned and lazily loaded once per process, outside inference_mode, with strict state loading. Availability failures are per-model and do not fabricate scores. Contract/Pydantic validators protect identity, classes and own-label score association. BUSI exclusion is recomputed from immutable policy, never trusted from client flags.",
        "## 4. Consensus, disagreement and scores\n\nNo validated compatible ensemble exists. resolve_consensus never performs a majority vote, random/set-order tie break, score average or winner replacement. All outcomes return the exact abstention, empty eligible IDs, zero participants and null confidence/score. Individual scores remain paired to their own model and predicted index. Even unanimous high-score stubs and untrusted eligibility flags abstain deterministically.",
        "## 5. Frontend and report behavior\n\nShared research cards separate tasks, label domains/exclusions/availability and show uncalibrated per-class scores. Live-to-patient conversion now retains the full inference envelope. Versioned real reports bypass DEMO_REPORT_MODE; the old explicitly marked demonstration remains for historical unversioned views. BRA Grad-CAM is genuine attribution, not a lesion/size/staging measurement. No synthetic diagnostic narrative or clinical action is derived from class scores. Human review metadata is separate from original model predictions.",
        "## 6. Endpoint and storage compatibility\n\n/predict remains multipart POST; source_domain defaults to unknown. New /inference-contract is read-only. models/consensus/heatmap_base64/image and three old display keys remain, with a fourth BI-RADS key and version/provenance/status fields added. Consensus status/score semantics intentionally break unsafe voting assumptions. BRA probability Normal:null is only a compatibility sentinel.\n\nThe database SQL, request schemas and patient/feedback route bodies are AST-identical to the captured initial source. Startup now runs the unchanged initializer; importing main.py no longer opens a database or loads checkpoint weights. New clients must read versioned JSON: confidence=0 in the old NOT NULL storage column is explicitly a placeholder for abstention, not a calibrated/zero model score. No stored rows or schema were migrated. Historical records are retained and old consensus scores are not promoted as valid confidence.",
        "## 7. Passed tests\n\n"+json.dumps({k:v for k,v in results.items() if k in ("status","syntax","python","frontend_build","diff_check")},indent=2)+f"\n\nFrontend: {front['tests_run']}/{front['tests_run']}, actual TypeScript adapters/React static rendering plus isolated persistence payload tests; includes four actual API responses and 78 historical BUSI records. Full per-test results are in frontend_test_results.json.\n\nReal API: 4 images, 16 real predictions, all four checkpoints, genuine BRA Grad-CAM on each request. Maximum original-logit difference {max_logits:.3g}; maximum score difference {max_scores:.3g}. main.py's real application/router was exercised in-process without database startup. No model was stubbed in these real checks. See real_inference_test_results.json and real_inference_responses.json. This is a small regression, not a new accuracy/generalization study.",
        "## 8. Resolved failures and unexecuted checks\n\n"+json.dumps(dict(resolved_failures=failures,failed_unresolved=[],unexecuted=unexecuted,not_verifiable=results["not_verifiable"]),indent=2),
        "## 9. Files created/modified and preservation\n\n"+text+f"\n\nProtected hash verification: {audit['preserved_files']} unchanged files, including all model files/database/prior research; five authorized existing source changes. Native dataset metadata unchanged for {audit['native_dataset_files_metadata_unchanged']} files; all 540 previous validation image/mask hashes verified. See preserved_files_audit.json and initial_state.json. Starting database modifications were pre-existing and were preserved byte-for-byte. Generated report inventory is artifact_inventory.csv.",
        "## 10. Remaining risks\n\nNo clinical readiness, clinical calibration or validated generalization is established. Full interactive browser/server/persistence and concurrency/load are untested. Source-domain metadata is caller-declared. The legacy numeric confidence column has a documented placeholder; external clients and existing feedback/dashboard statistics must not reinterpret it as a model probability or count abstention as diagnostic failure. Existing legacy/demo/static statistics remain outside this stage. Frozen historical audit scripts are not silently rewritten to match the new API; use the captured before/main.py snapshot for historical implementation context.",
        "## 11. Next milestone and reproduction\n\nStage 2 should exercise the research upload/save/reload/review/report/print workflow using an explicitly isolated database copy, and align feedback/dashboard semantics with abstention. Keep the frozen models and current policy. No clinical consensus or retraining is authorized by this handoff.\n\nUse the existing bc_ai Python from the repository root; exact commands and resolved environment issues are in execution_notes.json. Run --unit --real, then the Node frontend tests/build, then --audit --report. Real/ASGI tests require Windows local event-loop socket creation; no external application calls are needed. SQL connections are blocked in tests.\n\n"+"\n".join("- `"+c+"`" for c in commands),
        "## READY-TO-PASTE CHATGPT SUMMARY\n\n"+summary,
        "LUMINA INTEGRATION STAGE 1 COMPLETE — CHATGPT HANDOFF READY"]
    (OUT/"INTEGRATION_STAGE1_HANDOFF.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    inventory=[str(p) for p in OUT.rglob("*") if p.is_file()]
    inventory += [str(ROOT/p) for p in changed+created]
    inventory.append(str(OUT/"artifact_inventory.csv"))
    pd.DataFrame(dict(path=p) for p in sorted(set(inventory))).to_csv(OUT/"artifact_inventory.csv",index=False)
    print(json.dumps(dict(status="PASSED",python_tests=python["tests_run"],frontend_tests=front["tests_run"],
        real_predictions=real["model_predictions"],protected_files_preserved=audit["preserved_files"],
        existing_sources_modified=changed,new_sources=created,handoff=str(OUT/"INTEGRATION_STAGE1_HANDOFF.md")),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--unit",action="store_true");p.add_argument("--real",action="store_true");p.add_argument("--audit",action="store_true");p.add_argument("--report",action="store_true");args=p.parse_args()
    descriptor=contract_descriptor();write("integration_contract.json",descriptor)
    write("model_eligibility_policy.json",dict(policy_version=descriptor["policy_version"],
        consensus=descriptor["consensus"],models={k:{c:v for c,v in r.items() if c in ("model_id","task","source_training_domain","consensus_eligible","exclusion_reason","eligibility_by_domain")} for k,r in descriptor["models"].items()},
        restriction="No validated ensemble. Never infer eligibility from caller-declared domain, agreement or score; BUSI remains excluded."))
    if args.unit:print(json.dumps(unit_tests(),indent=2),flush=True)
    if args.real:print(json.dumps(real_inference(),indent=2),flush=True)
    if args.audit:print(json.dumps(protected_audit(),indent=2),flush=True)
    if args.report:reports()


if __name__=="__main__":main()
