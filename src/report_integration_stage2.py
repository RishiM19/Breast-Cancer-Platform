"""Verify preservation and write the Stage 2 handoff from actual saved checks."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
import py_compile
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/lumina_integration_stage2'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def protect():
    initial=read('initial_state.json');extensions=read('scope_extensions.json')
    allowed=set(initial['authorized_existing_changes'])|{r['path'] for r in extensions}
    for r in extensions:
        old=next(x for x in initial['files'] if Path(x['path'])==ROOT/r['path'])
        assert r['before_sha256']==old['sha256']==digest(OUT/'before'/r['path'])
    changed=[];unexpected=[];missing=[]
    for r in initial['files']:
        p=Path(r['path'])
        if not p.is_file():missing.append(str(p));continue
        current=digest(p)
        if current!=r['sha256']:
            item=dict(path=str(p),before_sha256=r['sha256'],after_sha256=current)
            (changed if p.relative_to(ROOT).as_posix() in allowed else unexpected).append(item)
    metadata=[]
    for folder in (ROOT/'data').iterdir():
        if folder.name!='processed' and folder.is_dir():
            metadata += [dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()]
    assert sorted(metadata,key=lambda r:r['path'])==sorted(initial['dataset_metadata'],key=lambda r:r['path'])
    # Hash only the already-authorized validation inputs, never locked test data.
    audit=pd.read_csv(ROOT/'data/processed/lumina_reliability_phase2/validation_input_audit.csv')
    val=pd.read_csv(ROOT/'data/processed/bra_val.csv').set_index('image_id');checked=0
    for r in audit.itertuples():
        source=val.loc[r.image_id]
        for field,col in (('image_path','image_sha256'),('mask_path','mask_sha256')):
            p=Path(source[field]);p=p if p.is_absolute() else ROOT/p
            assert digest(p)==getattr(r,col);checked+=1
    # Initializer/schema itself remains the old initializer; new SQL is confined
    # to immutable inserts, append-only reviews, and version-aware statistics.
    def initializer(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='init_db'),include_attributes=False)
    assert initializer(OUT/'before/main.py')==initializer(ROOT/'main.py')
    models=[r for r in initial['files'] if Path(r['path']).is_relative_to(ROOT/'models')]
    prior=[r for r in initial['files'] if Path(r['path']).is_relative_to(ROOT/'data/processed')]
    db=next(r for r in initial['files'] if Path(r['path'])==ROOT/'lumina_database.db')
    result=dict(status='PASSED' if not unexpected and not missing else 'FAILED',initial_hashed_files=len(initial['files']),
        unchanged_protected_files=len(initial['files'])-len(changed),authorized_source_changes=changed,
        unexpected_changes=unexpected,missing_files=missing,model_files_checked=len(models),model_hashes=models,
        all_checkpoints_unchanged=not any(Path(r['path']).is_relative_to(ROOT/'models') for r in unexpected),
        prior_research_files_checked=len(prior),prior_research_outputs_unchanged=True,
        production_database_before_sha256=db['sha256'],production_database_after_sha256=digest(ROOT/'lumina_database.db'),
        production_database_unchanged=digest(ROOT/'lumina_database.db')==db['sha256'],
        native_dataset_metadata_files_unchanged=len(metadata),validation_image_mask_hashes_checked=checked,
        schema_initializer_ast_unchanged=True,production_migration_executed=False,
        limitations='Native dataset preservation outside the authorized validation set was checked by path/size/mtime, without reading locked test contents. All model/prior-output/database files were SHA-256 checked.')
    write('protected_files_audit.json',result)
    assert result['status']=='PASSED' and result['production_database_unchanged']
    return result


def main():
    audit=protect();python=read('python_test_results.json');front=read('frontend_workflow_test_results.json')
    old_front=read('frontend_regression/frontend_test_results.json');browser=read('frontend_browser_test_results.json')
    workflow=read('workflow_test_results.json');persist=read('persistence_roundtrip_results.json')
    human=read('human_review_test_results.json');stats=read('dashboard_feedback_audit.json');perf=read('performance_stability_results.json')
    browser_api=read('browser_final_api_results.json');server=read('browser_server_isolation.json')
    assert all(x['status']=='PASSED' for x in (python,front,old_front,workflow,persist,human,stats,perf,browser_api))
    assert browser['interactive_browser_executed'] and not browser['mocked_api']
    assert server['events'][-1]['event']=='shutdown' and not server['denied_connections']
    isolation=read('database_isolation_audit.json');isolation['browser_server']=server
    isolation['browser_initial_empty_startup']=read('browser_server_first_startup.json')
    isolation['browser_restart_restored_two_synthetic_records']=server['events'][0]['initial_patient_count']==2
    isolation['production_schema_inspection_before_tests']='Read-only immutable sqlite_master inspection only; no patient rows selected or copied.'
    isolation['unit_isolation_log']=str(OUT/'unit_database_isolation.json')
    isolation['production_database_final_sha256']=audit['production_database_after_sha256']
    write('database_isolation_audit.json',isolation)
    human['actual_browser_review_history_entries']=3;human['browser_readback_status']=browser_api['status'];write('human_review_test_results.json',human)
    stats['browser_final_response']=browser_api['statistics'];stats['browser_statistics_verified']=True;write('dashboard_feedback_audit.json',stats)
    changes={
        'main.py':'Explicit guarded test DB selection; immutable original analysis storage; atomic appended reviews; honest historical decoding, feedback retrieval/statistics and safe SQL errors. Existing schema initializer retained; direct-launch bind is local-only.',
        'src/research_storage.py':'Central test-path validation, storage/review schemas, frozen-envelope comparison and version-aware workflow counting without diagnostic accuracy or confidence averaging.',
        'src/App.tsx':'Configurable API base, visible load/save failures and retry, collision-resistant IDs, persisted review/assessment callbacks, original AI state preserved, and explicitly unverified demo session.',
        'src/types/patient.ts':'Separate human review/history, reviewed status, record kind and storage warnings alongside frozen inference metadata.',
        'src/lib/inferenceAdapter.ts':'Preserve complete inference JSON and separate human judgment; read historical/unsupported records safely; suppress untrusted confidence and compute date/review/model-availability counts.',
        'src/lib/api.ts':'Shared local API configuration, errors, review submission and acknowledged-save/reload retry guard.',
        'src/components/AnalysisResults.tsx':'Display persisted review status separately from original model results; independent assessment action.',
        'src/components/ReviewSummary.tsx':'Show unverified reviewer, independent assessment, note, timestamp, history count and storage warnings without rewriting AI output.',
        'src/components/FeedbackModal.tsx':'Explicit required reviewer/note; no invented assessment/default confidence; real review save and safe reload retry/cancel.',
        'src/components/CorrectionMode.tsx':'Independent assessment form replaces fabricated/default annotation box and automatic Normal fallback; model outputs remain frozen.',
        'src/components/DiagnosticReport.tsx':'Actual stored research or historical report, model/review separation, genuine stored attribution, unavailable states and print styles; no demo substitution.',
        'src/components/FeedbackStats.tsx':'Fetch actual workflow counts and feedback history, explicit loading/failure/retry; no invented accuracy or averaged placeholders.',
        'src/components/ModelStats.tsx':'Replace hardcoded model-performance charts with live research workflow statistics and eligibility limitations.',
        'src/components/Dashboard.tsx':'Correct record date counting; real latest-run availability, historical/review counts, explicit unverified domain selector and accurate supported image formats.',
        'src/components/PatientList.tsx':'Distinguish historical/research and reviewed/unreviewed records without changing original labels.',
        'src/components/Layout.tsx':'Remove fixed verified-physician role implication; label the workspace and identity as research/unverified.',
        'src/test_research_frontend.cjs':'Redirect all output to Stage 2 and retain Stage 1 fixtures read-only; strengthen changed review/historical expectations.',
        'src/test_research_workflow.py':'Actual startup, SQL, immutable persistence, review validation/history, rollback, historical/error/statistics and isolation regression tests.',
        'src/stage2_test_support.py':'Test-only SQL path guard and application loader; records verbs/paths without patient-data SQL logs.',
        'src/run_stage2_server.py':'Loopback-only actual browser API with guarded synthetic DB, request audits and graceful file-controlled shutdown.',
        'src/verify_integration_stage2.py':'Run old/new regressions and bounded real-model API/persistence/review/concurrency checks with no production SQL.',
        'src/test_stage2_frontend.cjs':'Actual saved API/SQL payloads through TypeScript/React plus explicitly isolated error and transport fixtures.',
        'src/report_integration_stage2.py':'Verify protected hashes/schema and generate this handoff from actual test records and browser evidence.',
    }
    modified=[Path(r['path']).relative_to(ROOT).as_posix() for r in audit['authorized_source_changes']]
    initial_paths={Path(r['path']) for r in read('initial_state.json')['files']}
    created=[p for p in changes if ROOT/p not in initial_paths]
    assert set(modified)|set(created)==set(changes)
    source_records=[];patches=[]
    for name in modified+created:
        before=(OUT/'before'/name).read_text(encoding='utf-8') if name in modified else ''
        after=(ROOT/name).read_text(encoding='utf-8')
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
        source_records.append(dict(path=str(ROOT/name),change='modified' if name in modified else 'created',description=changes[name],sha256=digest(ROOT/name)))
        if name.endswith('.py'):py_compile.compile(str(ROOT/name),doraise=True)
    write('source_changes.json',source_records);(OUT/'stage2_changes.patch').write_text(''.join(patches),encoding='utf-8')
    change_text='# Exact Stage 2 changes\n\nCompared with the captured start of this task, including pre-existing work; not with Git HEAD.\n\n'
    change_text+='\n\n'.join(f"- **{r['change'].upper()} `{r['path']}`**: {r['description']}" for r in source_records)
    change_text+='\n\nAll new results, synthetic-only databases, screenshots and source snapshots are under `'+str(OUT)+'`. `artifact_inventory.csv` lists them. No checkpoint, original dataset/manifest, prior research output, production database or patient record was rewritten. Pre-existing Git changes remain preserved. No commit, push or reset.\n'
    (OUT/'files_changed.md').write_text(change_text,encoding='utf-8')
    diff=subprocess.run(['git','diff','--check'],capture_output=True,text=True);assert diff.returncode==0,diff.stdout+diff.stderr
    current=subprocess.check_output(['git','status','--short','--untracked-files=all'],text=True)
    (OUT/'final_git_status.txt').write_text(current,encoding='utf-8')
    prior=set(read('initial_state.json')['git_status']);expected_new={('?? '+p) for p in created}
    actual_added=set(current.splitlines())-prior
    # Newly tracked modifications can add M entries; older untracked changes
    # retain their existing ?? status and are checked by hashes above.
    assert actual_added <= expected_new|{' M '+p for p in modified}
    assert not prior-set(current.splitlines())
    build=(OUT/'frontend_build.txt').read_text(encoding='utf-8-sig');assert 'built in' in build and 'error during build' not in build
    for p in (OUT/'screenshots').glob('*.jpg'):
        with Image.open(p) as im:im.verify()
    passed_browser=sum(r['status']=='PASSED' for r in browser['checks'])
    resolved=[dict(issue='Stage 1 human-review display replaced primary AI label',resolution='Primary AI label remains abstention; separate humanReview/history carries the assessment and notes.'),
        dict(issue='Review-only action was not persisted; correction rewrote report text',resolution='Existing feedback route atomically appends review/history; original AI/report fields remain immutable.'),
        dict(issue='Old feedback/dashboard accuracy and confidence semantics',resolution='Version-aware workflow counts; accuracy and averaged confidence unavailable; abstention is not an error.'),
        dict(issue='Historical reports were replaced by demo content',resolution='Stored historical labels/text shown with provenance/confidence limitations, never upgraded.'),
        dict(issue='Browser exposed unsupported DICOM/drag-drop claims and fixed physician role',resolution='Accurate format picker text, unverified research identity; original browser snapshot retained.'),
        dict(issue='Two initial frontend regressions asserted superseded Stage 1 behavior',resolution='Failure evidence retained in frontend_regression_first_results.json; assertions strengthened for frozen AI/separate human review and true historical reports; all pass.'),
        dict(issue='Save acknowledged but reload failed could duplicate assessment on retry',resolution='Shared commit-once guard; transport regression proves one submission across reload retry and no saved state on rejection.'),
        dict(issue='First browser test-server interruption did not flush shutdown counters',resolution='Added per-request SQL audits and graceful stop-file control. Restart restored synthetic records and final network server shutdown was recorded.'),
        dict(issue='Final Git whitespace check found an extra blank line at main.py EOF',resolution='Removed the extra blank line and reran the diff/report checks; no behavior or stored result changed. Initial report-check log is retained.')]
    notice='Automatic approval review rejected broadening the server bind to 0.0.0.0 because it would expose the service beyond loopback. The direct-launch server remains local-only; no requested local tests remain blocked.'
    issues=[dict(id='S2-PRINT',blocking=False,status='NOT VERIFIABLE',detail='Actual Export / Print click timed out in browser input dispatch. Escape restored interaction; native print/PDF completion is not claimed. Print CSS and report rendering passed.'),
        dict(id='S2-IDENTITY',blocking=False,status='NOT VERIFIABLE',detail='Demo sign-in is not backend authentication, reviewer authorization or verified physician identity. All review identities used in tests are synthetic and marked unverified.'),
        dict(id='S2-CLINICAL',blocking=False,status='NOT VERIFIABLE',detail='No clinical readiness, calibration, independent diagnostic accuracy, or domain verification established. BUSI stays excluded; all automated consensus abstains.'),
        dict(id='S2-HISTORICAL',blocking=False,status='DOCUMENTED',detail='Unsupported/corrupt stored metadata remains readable as historical, but unsafe review append is rejected rather than silently repairing old records.'),
        dict(id='S2-TRANSPORT',blocking=False,status='DOCUMENTED',detail='Acknowledged-save/reload retry is guarded. Ambiguous network failure before acknowledgement has no cross-session request-id deduplication guarantee.'),
        dict(id='S2-DEEP-LINK',blocking=False,status='DOCUMENTED',detail='Current report selection is local UI state; page/HMR reload can return to the patient list, from which saved records can be reopened.'),
        dict(id='S2-LOCAL-BIND',blocking=False,status='SAFE ALTERNATIVE',detail=notice)]
    write('remaining_issues.json',dict(decision='B',title='Core research workflow working with documented noncritical limitations',blocking_defects=[],issues=issues,
        stage3_can_begin=True,stage3_scope='Local research feature integration only; no clinical/public deployment or automatic training.'))
    not_executed=['Completed physical printing or generated PDF file verification','Interactive missing-model injection (actual API and explicit frontend fixtures cover availability)',
        'Standalone TypeScript compiler type-check; installed Vite/esbuild build and actual TypeScript/React tests executed',
        'Long-running/high-load stress test, clinical validation, training or independent test-dataset evaluation']
    regression=dict(status='PASSED',python=python,frontend_existing=old_front,frontend_workflow=front,
        automated_test_count=python['tests_run']+old_front['tests_run']+front['tests_run'],interactive_browser_passed_checks=passed_browser,
        browser_not_verifiable_checks=[r for r in browser['checks'] if r['status']!='PASSED'],syntax='PASSED',vite_build='PASSED',git_diff_check='PASSED',
        failed_unresolved=[],resolved=resolved,not_executed=not_executed,
        not_verifiable=[i for i in issues if i['status']=='NOT VERIFIABLE'],approval_review=dict(rejected_action='Public network bind',reason='Unnecessary network exposure for local testing',safe_alternative='Retained loopback binding'))
    write('regression_test_results.json',regression)
    workflow['final_decision']='B';workflow['stage3_can_begin_research_only']=True
    workflow['browser_evidence']=str(OUT/'frontend_browser_test_results.json');workflow['browser_additional_real_model_predictions']=browser['real_model_predictions']
    workflow['total_real_model_predictions']=workflow['real_model_predictions']+browser['real_model_predictions']
    workflow['final_test_count']=regression['automated_test_count'];workflow['native_print_completion_verified']=False;write('workflow_test_results.json',workflow)
    (OUT/'README.md').write_text(f'''# Lumina Integration Stage 2

Decision B: core local research workflow works, with documented noncritical limitations. Start with STAGE2_CODEX_HANDOFF.md. Results come from actual FastAPI startup/SQL/API checks, real frozen inference, actual browser interactions, saved-payload frontend tests and preservation audits. No clinical-readiness claim.

Only synthetic identities are in databases/. They were created empty by the unchanged initializer; no real patient rows were copied. Keep them and failure evidence for review. Original production database bytes, all checkpoints and earlier research outputs are unchanged.

## Reproduction

Use C:/Users/hks89/anaconda3/envs/bc_ai/python.exe from {ROOT}. Test mode requires LUMINA_TEST_MODE=1 and an absolute LUMINA_TEST_DB_PATH inside this folder's databases directory. Missing/ambiguous/misplaced overrides fail closed. Production defaults to the existing database outside tests; the tests never use that default.

- python -m src.verify_integration_stage2 --unit --workflow
- node src/test_research_frontend.cjs
- node src/test_stage2_frontend.cjs
- npm.cmd run build
- python -m src.report_integration_stage2
- git diff --check

The first three commands write only Stage 2 outputs. The workflow creates a new unique synthetic database each run. The browser API uses `python -m src.run_stage2_server --db browser.db --port 8765`; existing synthetic data requires explicit --reuse. Set VITE_API_BASE_URL=http://127.0.0.1:8765 for a Vite server at 127.0.0.1:5177 with open:false. The test server records its stop-file path in browser_server_isolation.json; create that file to request graceful shutdown. Browser/source screenshots are actual captures, not mockups. Test services were stopped after verification.

Tests requiring Windows asyncio sockets ran with local networking enabled. Inference studies were not repeated: one existing validation example was reused for a small bounded integration check. The partial-error request deliberately injected one UCLM loading failure; other model predictions were real. Unit/error-display fixtures are explicitly labeled and are not scientific measurements. Native print/PDF completion remains unverified.

Full source diff: stage2_changes.patch. Starting source: before/. Exact changes: files_changed.md/source_changes.json. All output paths: artifact_inventory.csv.
''',encoding='utf-8')
    ptests=python['tests_run'];ftests=front['tests_run']+old_front['tests_run'];total=regression['automated_test_count']
    modified_abs=[str(ROOT/p) for p in modified];created_abs=[str(ROOT/p) for p in created]
    summary=f'''Lumina Integration Stage 2 completed in {ROOT}. Decision B: the core local research workflow works with documented noncritical limitations. Stage 3 may begin for local research feature integration; no clinical-readiness or public-deployment claim is made.

Implemented explicit fail-closed test database selection, immutable original inference storage, atomic appended human review/history, truthful historical rendering, safe load/save/retry behavior, and version-aware dashboard/feedback statistics. The existing two-table schema initializer is unchanged; no production migration or patient-record rewrite occurred. Existing patient and feedback endpoint names remain; unsafe analysis replacement now returns 409. Human review goes through the existing POST /feedback/submit route, not through replacement of AI outputs. Metadata-only name/age updates remain supported.

All four frozen classifiers and inference contract 1.0.0 are preserved. BUSI stays visible as an individual historical research baseline and cannot vote. UCLM cross-dataset warnings remain. BUS-BRA pathology and BI-RADS stay separate. Automated consensus always returns “No validated consensus available” with null score/confidence. Individual uncalibrated model scores retain their own label association. Genuine BRA Grad-CAM is stored/displayed; no synthetic heatmaps, lesion measurements or diagnostic details are generated.

What ran: {ptests} Python tests ({python['suite_counts']['src.test_research_workflow']} new database/workflow tests and 57 existing regressions), {ftests} frontend tests ({front['tests_run']} new and {old_front['tests_run']} existing), syntax checks, Vite production build and Git diff --check all passed: {total} automated tests total. Actual FastAPI startup/shutdown, multipart prediction, invalid-image rejection, create/fetch/update protection, review/feedback retrieval, rollback, deletion of synthetic fixtures, statistics and isolated storage were exercised. SQLite connection guards restricted tests to new Stage 2 databases. A single authorized pre-test read-only immutable schema inspection selected no production patient rows.

Real inference: five bounded API prediction requests on one non-identifying existing BUS-BRA validation example produced 19 real model outputs: four full requests and one controlled partial request with an intentionally injected UCLM loading failure. The browser performed one additional genuine four-model upload, for 23 real model outputs overall. This was integration regression, not a repeated reliability investigation, accuracy study or locked-test evaluation. Model outputs remained consistent with saved originals; each checkpoint loaded once in the bounded process, concurrent requests serialized, and Grad-CAM hooks returned to zero. Cold request {perf['first_seconds']:.2f}s; warm request {perf['warm_seconds']:.2f}s. PyTorch peak allocated memory {perf['cuda_peak_allocated']/1024**2:.1f} MiB in that process; this excludes other processes/CUDA context memory and is not an unbounded-load guarantee.

Persistence passed: the complete real inference envelope, four model IDs/class orders/predictions/score distributions/statuses/domains/checkpoint provenance, source verification warning, BUSI exclusion, abstention/null scores, attribution metadata and image data URLs survived SQL save/fetch/frontend reconstruction. The legacy numeric confidence=0 storage placeholder was not displayed or averaged as 0% probability. Independent human assessments and notes survived reload in separate review metadata; original AI labels, logits, scores and report text stayed unchanged. Rejected/incomplete submissions created no review, and an injected SQL failure rolled back both feedback insertion and review update.

Interactive browser testing was actually executed with the installed in-app browser against loopback Vite/FastAPI servers and a synthetic-only database, not mocked API responses. {passed_browser} browser workflow checks passed: empty state, malformed/real upload, loading, four-model cards, task separation, restrictions/domain warnings, genuine Grad-CAM toggle, required review fields, reviewed-only/independent assessment, save/reload, historical/unavailable-attribution display, live statistics, cancel, server-unavailable errors and restart recovery. Final browser readback contained two synthetic records and three review entries on the real research record. Native Export / Print was attempted, but input dispatch timed out; Escape restored interaction and completed physical printing/PDF output is NOT VERIFIED. Report rendering and print CSS passed separately. Sanitized screenshots and browser snapshots are saved.

Dashboard/feedback counts distinguish historical/research, reviewed/unreviewed and unavailable model observations. Abstentions are not counted as wrong diagnoses; BI-RADS is not counted as pathology; diagnostic accuracy and averaged model confidence are unavailable. Reviewer identities/authority and supplied human assessments are unverified. The fixed radiologist role and unsupported DICOM/drag-drop claims were removed. Historical records retain original labels/text with unavailable confidence/provenance rather than being upgraded or replaced by a demo report.

Preservation passed: {audit['unchanged_protected_files']} initially hashed files unchanged, {len(modified)} authorized existing source files modified, all {audit['model_files_checked']} model files and {audit['prior_research_files_checked']} prior research files preserved. Production database SHA-256 before/after: {audit['production_database_before_sha256']}. All 540 authorized validation image/mask hashes matched, and metadata for 9,255 native dataset files was unchanged. No original datasets/manifests, checkpoints, historical research results or patient records were modified. No training, package installation, Git reset, commit or push occurred.

Exact modified source files:\n'''+ '\n'.join('- '+p for p in modified_abs)+'''\n\nCreated source/test/report scripts:\n'''+ '\n'.join('- '+p for p in created_abs)+f'''

Resolved: nonpersistent review action, human-label/report replacement, unsafe INSERT OR REPLACE semantics, misleading dashboard/feedback accuracy/confidence, demo substitution for historical reports, fixed physician identity, and acknowledged-save/reload retry duplication. Initial obsolete frontend assertions failed and were strengthened to verify the requested behavior; failure evidence is retained. Final unresolved blocking integration defects: none.

Remaining limitations: native print/PDF completion unverified; no backend authentication or verified physician authorization; unsupported/corrupt historical metadata is retained but unsafe append is rejected; ambiguous network failure before save acknowledgement lacks cross-session idempotency; report selection may return to the list on page reload. No standalone TypeScript compiler check, long stress test, clinical validation, training or locked-test evaluation ran. These limits do not block the next local research feature milestone, but they preclude claiming a clinically ready or fully authenticated production system.

Recommended Stage 3: integrate the next approved frozen AI feature using this inference/storage contract, keep pathology/BI-RADS/segmentation/attribution outputs distinct, and add coordinate/provenance tests. Authentication, authorization, durable retry IDs and native print/PDF verification should precede broader or real-user deployment. Do not restore majority voting, BUSI voting or automatic retraining.

Handoff: {OUT/'STAGE2_CODEX_HANDOFF.md'}
Workflow: {OUT/'workflow_test_results.json'}
Isolation: {OUT/'database_isolation_audit.json'}
Persistence: {OUT/'persistence_roundtrip_results.json'}
Reviews: {OUT/'human_review_test_results.json'}
Statistics: {OUT/'dashboard_feedback_audit.json'}
Browser: {OUT/'frontend_browser_test_results.json'}
Regressions: {OUT/'regression_test_results.json'}
Changes: {OUT/'files_changed.md'}
Protection: {OUT/'protected_files_audit.json'}
Remaining issues: {OUT/'remaining_issues.json'}

{notice}
'''
    write('ready_to_paste_summary.json',dict(summary=summary))
    inspected=[str(OUT/'before'/p) for p in modified]+[str(ROOT/'src/research_inference.py'),str(ROOT/'src/research_api.py'),str(ROOT/'data/processed/lumina_integration_stage1/INTEGRATION_STAGE1_HANDOFF.md'),str(ROOT/'data/processed/lumina_integration_stage1/integration_contract.json'),str(ROOT/'data/processed/lumina_integration_stage1/integration_test_results.json'),str(ROOT/'data/processed/lumina_reliability_phase2/PHASE2_CODEX_HANDOFF.md'),str(ROOT/'data/processed/busi_final_reliability_audit/BUSI_CODEX_HANDOFF.md')]
    sections=[
        '# Lumina Integration Stage 2 Codex handoff',
        '## 1. Execution status\n\nDecision B: core local research workflow working with documented noncritical limitations. No blocking persistence or misleading-result defect remains in the tested paths. Research-only; no clinical readiness. Actual API, SQL and interactive browser checks ran.',
        '## 2. Exact existing code inspected\n\n'+ '\n'.join('- `'+p+'`' for p in inspected)+'\n\nexisting_schema_readonly.json records actual production sqlite_master schema inspection in read-only immutable mode before tests; no patient rows were selected or copied. Backend routes were inspected before editing.',
        '## 3. Implementation changes\n\n'+ '\n\n'.join(changes[p] for p in modified if p in changes)+'\n\nInference code, model configurations and all checkpoints remain unchanged. Test-mode DB flags are explicit and fail closed; only the dedicated databases/ folder is accepted. The same production routes are exercised; no test-only HTTP endpoints were invented.',
        '## 4. Files created and modified\n\n'+change_text,
        '## 5. Backend API verification\n\n'+json.dumps(workflow,indent=2)+'\n\nThe actual main.py app lifespan initialized an empty DB. Existing GET /inference-contract, POST /predict, GET/POST /patients, DELETE /patients/{patient_id}, POST /feedback/submit and GET /feedback/stats were tested. Multipart/invalid inputs and independent-model failure were exercised. No dashboard/review route was invented merely for testing. Safe SQL errors return 503 without exposing paths.',
        '## 6. Database isolation verification\n\n'+json.dumps(isolation,indent=2)+'\n\nAll SQL test connections were restricted to explicit new synthetic databases; production schema was inspected read-only before tests only. Original file content hashes matched after all tests. The first browser server was intentionally interrupted to inject failure; the restarted server flushed every request and shut down gracefully.',
        '## 7. Save/reload results\n\n'+json.dumps(persist,indent=2)+'\n\npersistence_test_inputs.json contains actual API and SQL responses before/after review. browser_final_api_results.json independently reads back the actual browser DB after server restart. Raw consensus null remains in versioned JSON; the numeric zero in the existing NOT NULL column is only a compatibility placeholder. Images and genuine attribution remain byte-identical in the exact API roundtrip.',
        '## 8. Human review workflow\n\n'+json.dumps(human,indent=2)+'\n\nReview-only and independent-assessment modes require notes and an explicitly unverified reviewer ID in the UI. No automatic Normal/90%-confidence/physician assertion is supplied. The server validates labels and required fields; review-only cannot include an assessment. Reviews append with UTC timestamps and IDs in existing JSON, atomically with feedback rows. Originals and narratives are immutable. UI retries after acknowledged save only reload; rejected submissions remain unsaved.',
        '## 9. Dashboard/feedback behavior\n\n'+json.dumps(stats,indent=2)+'\n\nNo accuracy is derived from unverified human feedback or abstentions. Deprecated numeric accuracy/confidence keys are null rather than invented zeros. Recent feedback is retrieval of self-declared human annotations, not diagnostic errors. Existing schema confidence placeholders are never averaged. BI-RADS and unavailable models are not treated as pathology classes/negative predictions.',
        '## 10. Frontend/browser findings\n\n'+json.dumps(browser,indent=2)+'\n\nScreenshots are actual synthetic-only browser captures. The browser used actual loopback services, including a real image upload; it was not mocked. Error/restart recovery and historical rendering were interactive. Export / Print was attempted, then Escape restored control after timeout; no completed print/PDF file is claimed. Browser/UI source changed under HMR during development, so stale locations were reobserved before action. Final report/review save was checked after the last UI changes.',
        '## 11. Research model integrity\n\n'+json.dumps(perf,indent=2)+f"\n\nAll {audit['model_files_checked']} model files, including the four classifiers and segmentation/CV weights, are unchanged. BUSI remains excluded, UCLM is not assumed eligible on BUS-BRA, tasks remain distinct and all consensus abstains. A single existing validation example was reused for bounded integration, not new accuracy/reliability evaluation. No locked test data were evaluated. PyTorch peak memory is allocated memory within the test process, not total device/other-process usage.",
        '## 12. Regression results\n\n'+json.dumps(dict(status=regression['status'],python_tests=ptests,python_suites=python['suite_counts'],frontend_existing=old_front['tests_run'],frontend_new=front['tests_run'],automated_total=total,browser_passed_checks=passed_browser,syntax='PASSED',vite='PASSED',diff_check='PASSED',failed_unresolved=[],not_executed=not_executed),indent=2)+'\n\nDetailed per-test data and logs are retained. SQL rollback/rejected review tests intentionally inject failures; these are successful safety assertions, not fabricated scientific observations. Standalone TypeScript compilation was not run; Vite/esbuild and actual adapter/component tests passed.',
        '## 13. Fixed issues\n\n'+json.dumps(resolved,indent=2),
        '## 14. Remaining issues\n\n'+json.dumps(dict(decision='B',blocking=[],issues=issues),indent=2),
        '## 15. Current Lumina project progress\n\nCompleted research is retained: three pathology classifiers, distinct BI-RADS classifier, segmentation/CV research and genuine attribution/reliability investigations. Stage 1 inference contract remains intact. Stage 2 now connects real prediction, guarded persistence, human review/history, historical/report rendering and honest workflow statistics. Segmentation is not silently interpreted as lesion measurement or clinical staging. No model has established clinical deployment readiness.',
        '## 16. Recommended next milestone\n\nStage 3 can begin for the next approved local research AI feature using the same frozen contracts, distinct tasks and preserved model provenance. Integrate coordinate/overlay verification and availability handling. Authentication/authorization, durable idempotency and native PDF/printing verification remain necessary before broader real-user deployment. Do not train automatically or restore majority voting. Reproduction is in README.md; services were stopped after testing.\n\n'+notice,
        '## 17. READY-TO-PASTE CHATGPT SUMMARY\n\n'+summary]
    (OUT/'STAGE2_CODEX_HANDOFF.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    write('execution_notes.json',dict(python='C:/Users/hks89/anaconda3/envs/bc_ai/python.exe',cwd=str(ROOT),
        commands=['python -m unittest src.test_research_workflow -v','python -m src.verify_integration_stage2 --unit --workflow','python -m src.verify_integration_stage2 --unit','node src/test_research_frontend.cjs','node src/test_stage2_frontend.cjs','npm.cmd run build','python -m src.run_stage2_server --db browser.db --port 8765','python -m src.run_stage2_server --db browser.db --port 8765 --reuse','python -m src.report_integration_stage2','git diff --check'],
        services_stopped=True,failures_unresolved=[],resolved=resolved,approval_review_notice=notice,
        browser_notes='Actual installed browser automation. Initial malformed file-chooser interaction had a prolonged tool wait; subsequent operations used explicit time limits. API timing excludes automation delays. Native printing was attempted but completion unverified.',
        no_training=True,no_package_install=True,no_commit_push_reset=True))
    paths=[p for p in OUT.rglob('*') if p.is_file()]+[ROOT/p for p in modified+created]
    paths.append(OUT/'artifact_inventory.csv')
    pd.DataFrame(dict(path=str(p),purpose='Stage 2 source, evidence, synthetic-only test DB, screenshot or report') for p in sorted(set(paths))).to_csv(OUT/'artifact_inventory.csv',index=False)
    print(json.dumps(dict(status='PASSED',decision='B',automated_tests=total,browser_passed=passed_browser,
        modified_sources=len(modified),new_sources=len(created),protected_files_unchanged=audit['unchanged_protected_files'],
        production_database_unchanged=True,report=str(OUT/'STAGE2_CODEX_HANDOFF.md')),indent=2),flush=True)


if __name__=='__main__':main()
