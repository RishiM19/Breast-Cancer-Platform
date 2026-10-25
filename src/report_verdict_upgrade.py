"""Build the verdict/UI handoff from executed evidence; never alters prior research outputs."""
import difflib,hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/lumina_ui_verdict_upgrade'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    initial=read('initial_state.json');extra=read('scope_extension.json');allowed=set(initial['authorized_existing_changes'])|{r['path'] for r in extra}
    modified=[];protected=[];diff=[]
    for r in initial['files']:
        p=Path(r['path']);rel=p.relative_to(ROOT).as_posix();actual=sha(p) if p.is_file() else None
        row=dict(path=str(p),before_sha256=r['sha256'],after_sha256=actual,unchanged=actual==r['sha256'])
        if rel in allowed:
            if not row['unchanged']:
                modified.append(row);before=OUT/'before'/rel
                assert before.is_file() and sha(before)==r['sha256']
                diff+=difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='before/'+rel,tofile='after/'+rel)
            else:protected.append(row)
        else:protected.append(row)
    unexpected=[r for r in protected if not r['unchanged']]
    metadata_changed=[]
    for r in initial['dataset_metadata']:
        p=Path(r['path']);stat=p.stat() if p.exists() else None
        if stat is None or stat.st_size!=r['bytes'] or stat.st_mtime_ns!=r['mtime_ns']:metadata_changed.append(r)
    raw_paths={str(p) for p in (ROOT/'data').rglob('*') if p.is_file() and not p.is_relative_to(ROOT/'data/processed')}
    assert raw_paths=={r['path'] for r in initial['dataset_metadata']}
    assert not unexpected and not metadata_changed
    checkpoints=[r for r in protected if Path(r['path']).is_relative_to(ROOT/'models') and Path(r['path']).suffix=='.pth']
    production=next(r for r in protected if Path(r['path'])==ROOT/'lumina_database.db')
    development=json.loads((ROOT/'data/processed/lumina_integration_stage3/development_input_hashes.json').read_text())
    input_hashes=[dict(path=r['path'],recorded_sha256=r['sha256'],current_sha256=sha(Path(r['path']))) for r in development]
    assert len(checkpoints)==15 and all(r['recorded_sha256']==r['current_sha256'] for r in input_hashes)
    write('protected_files_audit.json',dict(status='PASSED',baseline='Task-start dirty workspace snapshot, not Git HEAD',protected_files_hashed=len(protected),all_protected_hashes_unchanged=True,checkpoints=checkpoints,production_database=production,
        raw_dataset_files_metadata_checked=len(initial['dataset_metadata']),raw_dataset_file_inventory_unchanged=True,raw_dataset_metadata_changes=[],
        raw_scope='Size, mtime and file inventory only for raw datasets; no blanket content-hash claim and no locked test image contents opened.',permitted_development_input_hashes=input_hashes,
        prior_research_outputs_unchanged=True,unauthorized_changes=[],details=protected))
    status=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True);(OUT/'final_git_status.txt').write_text(status,encoding='utf-8')
    (OUT/'upgrade_changes.patch').write_text(''.join(diff),encoding='utf-8')
    new=['src/research_decision_engine.py','src/components/ResearchVerdictSummary.tsx','src/styles/research-workspace.css','src/test_research_decision_engine.py','src/verify_verdict_upgrade.py','src/run_verdict_server.py','src/run_verdict_vite.cjs','src/test_verdict_frontend.cjs','src/report_verdict_upgrade.py']
    write('source_changes.json',dict(modified=modified,created=[dict(path=str(ROOT/n),sha256=sha(ROOT/n)) for n in new]))
    descriptions={
        'src/research_decision_engine.py':'New deterministic primary-selection policy and strict verdict/observation schema. No ensemble weights.',
        'src/research_inference.py':'Add contract1.2 response, frozen-model consistency validator and independent verdict generation. Preserve1.0/1.1 and all model math.',
        'src/research_api.py':'Accept explicit1.2 requests/discovery; keep default1.0 and legacy response shapes.',
        'src/research_storage.py':'Permit only the new existing dedicated synthetic-test DB directory in explicit test mode; production path and immutable storage unchanged.',
        'src/App.tsx':'Request1.2; retain canonical abstention/zero compatibility columns and full envelope. Refine login, errors and verdict-aware review display; link to saved report; default light with saved theme honored.',
        'src/types/patient.ts':'Typed independent verdict and observations on the inference envelope.',
        'src/lib/inferenceAdapter.ts':'Validate1.2 primary label/score and fallback/domain metadata. Keep old record semantics; add display-only finding and compact-ID helpers.',
        'src/components/AnalysisResults.tsx':'Replace oversized abstention hero with image evidence, primary research result, separate BI-RADS, model comparison, segmentation and review/report actions.',
        'src/components/Dashboard.tsx':'Explicit image selection/preview/Analyze flow, honest progress/errors, advanced origin settings, real counts/recent records and clear interpretation guidance.',
        'src/components/Layout.tsx':'Cohesive responsive shell, clear research identity, navigation and accessible desktop/mobile theme control.',
        'src/components/PatientList.tsx':'Compact displayed IDs with accessible original ID, both-ID search, finding/review/history filters, pagination, responsive scroll area, report and confirmed Delete actions.',
        'src/components/DiagnosticReport.tsx':'Stored verdict, review and images separated; complete provenance, older/historical handling and print-only lineage/history. Native PDF not claimed.',
        'src/components/ResearchModelResults.tsx':'Real per-model score distributions and statuses, separate tasks, accessible checkpoint details and compact consensus panel.',
        'src/components/ReviewSummary.tsx':'Shared card styling and expandable actual review history.',
        'src/components/ExploratorySegmentation.tsx':'Shared card/control styles and responsive image grid; original mask/overlay logic and semantics preserved.',
        'src/components/Settings.tsx':'Remove inert clinical-threshold/notification switches and invented department defaults; retain honest local display-name settings and fixed-policy explanation.',
        'src/components/FeedbackModal.tsx':'Shared form styles and separate display of saved verdict/score versus consensus; review submission logic retained.',
        'src/components/ModelStats.tsx':'Use the shared page container; actual research statistics unchanged.',
        'src/styles/globals.css':'Import the new token/component stylesheet; retain the existing utility system.',
        'src/test_research_frontend.cjs':'Allow output only within the new dedicated upgrade artifact root as well as prior roots; assertions unchanged.',
        'src/test_stage2_frontend.cjs':'Same confined output-root extension; prior Stage2 inputs remain read-only and all assertions retained.',
        'src/test_stage3_frontend.cjs':'Separate immutable Stage3 inputs from new output destination; assertions retained.',
        'src/test_research_workflow.py':'Allow new dedicated synthetic-test root; all prior workflow assertions retained.',
        'src/components/ResearchVerdictSummary.tsx':'New reusable finding, primary model/score, applicability, comparison, limitations and separate BI-RADS summary.',
        'src/styles/research-workspace.css':'New shared tokens, semantic component styles, responsive breakpoints, keyboard focus, reduced motion and print styling.',
        'src/test_research_decision_engine.py':'28 routing, failure, no-vote, score, immutability and contract tests with explicitly synthetic fixtures.',
        'src/verify_verdict_upgrade.py':'131-case Python runner plus genuine inference, contract and isolated SQL/review verification.',
        'src/run_verdict_server.py':'Loopback browser test backend with guarded synthetic DB, cache/hook audit and graceful shutdown.',
        'src/run_verdict_vite.cjs':'Loopback frontend service without automatic native windows.',
        'src/test_verdict_frontend.cjs':'20 new actual adapter/React checks using genuine saved outputs and explicit malformed-state fixtures.',
        'src/report_verdict_upgrade.py':'Generate final evidence, preservation audit, source inventory and handoff.',
    }
    changes=['# Exact verdict and UI upgrade changes','',f'{len(modified)} existing files modified; {len(new)} source/test/style/report files created. Compared with the preserved task-start workspace, including pre-existing Git changes.','', '| Source | Change |','|---|---|']
    for r in modified:
        rel=Path(r['path']).relative_to(ROOT).as_posix();changes.append(f'| [{rel}]({Path(r["path"]).as_posix()}) | {descriptions[rel]} |')
    changes+=['','New files:','']+[f'- [{n}]({(ROOT/n).as_posix()}): {descriptions[n]}' for n in new]
    changes+=['','No checkpoint, training source, dataset, split, production DB or previous research output changed. `upgrade_changes.patch` captures only this task\'s changes to existing source. `source_changes.json` records source hashes. All generated evidence and synthetic databases are confined to this upgrade output directory. The ignored Vite build was regenerated.']
    (OUT/'files_changed.md').write_text('\n'.join(changes)+'\n',encoding='utf-8')
    groups=[('Python (103 existing + 28 new decision)',read('python_test_results.json')),
        ('Existing frontend adapter/rendering',read('frontend_regression/frontend_test_results.json')),
        ('Existing Stage2 frontend workflow',read('frontend_workflow_test_results.json')),
        ('Existing Stage3 frontend',read('stage3_frontend_test_results.json')),
        ('New verdict frontend',read('verdict_frontend_test_results.json')),
        ('Contract compatibility',read('contract_compatibility_results.json')),
        ('Genuine inference/API',read('api_inference_results.json')),
        ('SQL persistence and review',read('persistence_roundtrip_results.json')),
        ('Interactive browser',read('browser_test_results.json'))]
    counts=[dict(group=n,tests_run=r['tests_run'],passed=r['passed']) for n,r in groups];total=sum(r['tests_run'] for r in counts);passed=sum(r['passed'] for r in counts);assert total==passed
    write('ui_regression_results.json',dict(status='PASSED',tests_run=total,passed=passed,failed=0,groups=counts,
        counting='Named unit cases and executed verification checks, not raw assertions. Decision28 is included in Python131; browser persistence hash audit, build and preservation are separate, not double counted.',
        old_assertions_weakened=False,legacy_test_edits='Artifact-output paths only; immutable prior inputs retained. The old1.2-without-verdict rejection remains valid for an incomplete new-contract payload; new unknown9.0 tests explicitly verify unsupported versions.',
        production_build='PASSED; build.log',browser_automation_corrections=read('browser_test_results.json')['resolved_automation_issues']))
    browser=read('browser_test_results.json');browser_storage=read('browser_persistence_verification.json');server=read('browser_server_isolation.json')
    assert server['events'][-1]['event']=='shutdown' and not server['denied_connections'] and server['hook_count']==0
    remaining=dict(final_engineering_decision='B: working for restricted local research with explicit limitations',blocking_implementation_defects=[],
        limitations=['The selection policy is transparent and deterministic but not clinically validated. No weighted ensemble or calibrated disease probability exists.',
            'Caller-declared dataset origin is unverified; experimental UCLM fallback on BUSI/unknown images is not established external-domain generalization or a reliable negative Normal finding.',
            'BUS-BRA cannot classify Normal; BUSI remains historical comparison only and BI-RADS remains a separate task.',
            'The unchanged exploratory segmentation baseline can produce false-positive foreground; original resize/fidelity/domain and EXIF restrictions remain.',
            'No model-produced Normal verdict occurred in the three fixed validation examples; all three-class routing states, including Normal, were covered with explicit decision fixtures. No new accuracy estimate was produced.',
            'Demo sign-in and reviewer identities remain unverified; no production authentication or external-service readiness claim.',
            'Native printing/generated PDF and permanent Delete acceptance were not executed. Delete confirmation retained; browser records were preserved.',
            'Only bounded local checks were performed; no long-duration load, clinical calibration, prospective or untouched external evaluation.',
            'Embedded image/probability artifacts retain existing JSON payload/storage costs; further storage/authentication/provenance work remains.'],
        recommended_next_milestone='Restricted local research engineering: authenticated roles and provenance, immutable bounded artifact storage, and a separately authorized untouched evaluation protocol against this primary-model baseline. No automatic retraining.')
    write('remaining_issues.json',remaining)
    manifests=read('development_manifest_inventory.json')
    feasibility='''# Weighted ensemble feasibility

**Weights cannot be justified for activation at this stage. No weights were fitted, no weighted ensemble was activated, and no locked test images or test labels were used.**

The existing permitted development manifests contain:

| Manifest | Images | Labels | Grouping fields observed |
|---|---:|---|---|
'''
    for r in manifests:feasibility+=f'| `{r["manifest"]}` | {r["rows"]} | {r["labels"]} | '+('patient_id' if r['patient_id_column_present'] else 'case_id' if r['case_id_column_present'] else 'Neither')+' |\n'
    feasibility+='''
This is a manifest inventory, not a new leakage audit or performance study. Existing BUSI audit evidence documents duplication and label conflicts across training/validation, and no verified patient grouping. Its historical91.03% result does not supply a defensible ensemble weight. BUSI stays comparison-only.

UCLM provides three pathology classes, while BUS-BRA has only Benign/Malignant and its270-row validation manifest has no Normal cohort. Assigning an artificial Normal probability of zero to BUS-BRA or interpreting Benign as Normal would distort the class problem. BI-RADS2/3/4/5 is another target and cannot be averaged into pathology labels. Uncalibrated softmax outputs are not comparable clinical probabilities.

The110 UCLM validation images could support an explicitly exploratory frozen-model comparison or a pre-specified hierarchical hypothesis. They do not establish an independent clean development/evaluation cohort after prior checkpoint selection and investigation. Existing domain-specific validation results and repeated uses do not justify40% BRA +30% UCLM +30% BUSI, or any other activated weights. No calibration/generalization advantage is established by nominally better scores on reused data.

A future study needs auditable patient/frame grouping and cross-split duplicate/conflict controls; clearly defined harmonized three-class references including genuine Normal examples; domain-specific applicability; a pre-specified calibrated or hierarchical method; and untouched evaluation patients. Compare it against domain_aware_research_prediction_v1 on the same eligible cases, including failure/coverage and disagreement handling, before considering activation. This upgrade does not authorize that training/evaluation or any clinical deployment.

Sources inspected: Phase2, BUSI final audit, Integration Stages1-3 handoffs and the three permitted validation manifests. No locked-test evaluation was repeated.
'''
    (OUT/'weighted_ensemble_feasibility.md').write_text(feasibility,encoding='utf-8')
    genuine=read('genuine_inference_responses.json');examples=[]
    for i,r in enumerate(genuine):
        v=r['response'].get('research_verdict')
        if v:examples.append(dict(response_index=i,declared_domain=r['declared_domain'],prediction=v['prediction'],primary_model=v['primary_model_id'],primary_model_score=v['primary_model_score'],score_semantics=v['score_semantics'],comparison=v['agreement']['indicator'],experimental_fallback=v['experimental_fallback']))
    write('genuine_output_examples.json',examples)
    summary=f'''Implemented a separate Final AI Research Prediction and a cohesive Lumina interface redesign. Contract1.2.0 adds immutable research_verdict while preserving models, abstained consensus, attribution and segmentation. Clients1.0.0/1.1.0 remain functional; default API behavior remains1.0.0 and the frontend explicitly requests1.2.0. Previous records were not rewritten or reclassified.

The versioned domain_aware_research_prediction_v1 policy selects UCLM for declared UCLM and BUS-BRA binary pathology for declared BUS-BRA. BUSI/unknown use an explicitly experimental UCLM three-class fallback. BUS-BRA cannot classify Normal. BUSI remains a historical comparison, BI-RADS never supplies a pathology vote, and all validated-consensus outputs remain abstained. Primary labels and scores come from the same selected model; unavailable primary inference produces null label/score. Agreement is descriptive and never adjusts confidence. Unknown-domain Normal is explicitly not a reliable negative or verified Normal finding.

The redesigned light interface uses shared tokens, typography, cards and controls; preview plus explicit Analyze; Advanced Research Settings with self-declared origin; prominent class/primary score; separate BI-RADS and four-model comparison; genuine Grad-CAM and eligible UCLM segmentation; accessible provenance and review history. Records have compact displayed IDs, full-ID access/search, finding/review/history filters, pagination and confirmation before Delete. Reports separate the frozen finding from human assessment. Print styles were improved; native PDF output is not claimed.

Genuine frozen inference ran on the three preselected permitted UCLM validation examples. Routing across all declared domains used the same permitted UCLM input with different self-declared metadata; this was not an external-dataset generalization study. The first example produced UCLM Benign with its own99.18% uncalibrated score, while historical BUSI predicted Malignant; BUSI did not affect selection. Another example produced Malignant with50.47%. The Normal-labelled example produced Benign with64.13%, illustrating a model limitation, not a performance estimate. A genuine Normal output was not obtained from these three examples; Normal routing was tested with explicit fixtures. No fabricated results appear in success screenshots.

{total}/{total} named tests/checks passed:131 Python (103 existing +28 new decision),63 existing frontend,20 new frontend,11 contract compatibility,10 genuine inference/API,13 SQL persistence/review and34 interactive browser checks. Production build and preservation audit passed separately. Browser tests covered desktop,390px mobile and820px tablet, three genuine uploads, unknown fallback, invalid upload, search/filter, reports, historical/older records and human review. The browser SQL audit independently confirmed the entire original envelope, verdict, model outputs, masks and original record fields remained unchanged after a deliberately different unverified human assessment.

All15 checkpoint files, production SQLite and previous research outputs retained their hashes. {len(protected)} protected files were hash-verified;9,255 raw dataset files retained size/mtime and file inventory, with six permitted development input hashes unchanged. Raw datasets were not blanket content-hashed. Locked test image contents were not accessed; no retraining, checkpoint replacement, production-record deletion/rewrite, commit or push occurred. Changed {len(modified)} existing files and created{len(new)} source/test/style/report files; files_changed.md and upgrade_changes.patch contain the exact task-only changes.

Weighted-ensemble weights cannot be justified from the current contaminated/mismatched/reused development evidence; no weights were activated or tuned. Final engineering decision B: working for restricted local research with documented domain/model limits. Clinical validation, calibrated probabilities, prospective external generalization, long-run load, native print/PDF and permanent Delete acceptance remain unverified. Test services were stopped and the viewport reset. Authentication, provenance and bounded artifact storage remain suitable next engineering work; no clinical deployment or automatic retraining is authorized.'''
    # Improve spacing without changing scientific values or counts.
    for old,newtext in [('Contract1.2.0','Contract 1.2.0'),('Clients1.0.0/1.1.0','Clients 1.0.0/1.1.0'),('remains1.0.0','remains 1.0.0'),('requests1.2.0','requests 1.2.0'),('own99.18%','own 99.18%'),('with50.47%','with 50.47%'),('with64.13%','with 64.13%'),('passed:131','passed: 131'),('+28','+ 28'),('desktop,390px','desktop, 390px'),('and820px','and 820px'),('All15','All 15'),(';9,255','; 9,255'),('created9','created 9')]:summary=summary.replace(old,newtext)
    for old,newtext in [(',63 existing',', 63 existing'),(',20 new',', 20 new'),(',11 contract',', 11 contract'),(',10 genuine',', 10 genuine'),(',13 SQL',', 13 SQL'),('and34','and 34')]:summary=summary.replace(old,newtext)
    (OUT/'READY_TO_PASTE_CHATGPT_SUMMARY.md').write_text(summary+'\n',encoding='utf-8')
    table='\n'.join(f'| {r["group"]} | {r["passed"]}/{r["tests_run"]} |' for r in counts)
    handoff='''# Lumina verdict and UI upgrade handoff

## Engineering outcome

Implemented and tested. Decision B: restricted local research functionality works with documented model/domain limitations; no clinical-validation claim.

## Inspected evidence and preservation baseline

Read Integration Stage1/2/3, reliability Phase2 and BUSI final handoffs; inspected current engine, router, storage, immutable feedback flow, frontend adapters/types/views, styles and tests. Captured initial dirty Git status, protected file hashes and raw metadata before edits. Preserved selected Stage3 genuine before screenshots without altering their originals.

## Decision architecture and routing

research_decision_engine.py defines strict ResearchVerdict, secondary observations and descriptive comparison metadata. Method domain_aware_research_prediction_v1 /1.0.0 selects UCLM on declared UCLM, BRA binary on declared BUS-BRA, and explicitly experimental UCLM fallback on BUSI/unknown. No secondary model replaces a failed primary. Every successful verdict is Normal, Benign or Malignant; unavailable status has null label/score. Primary model ID and own score are validated against the frozen models on creation and reload. BUSI is historical-only; BI-RADS is excluded from comparisons. Secondary binary outputs cannot assess a primary Normal class and are explicitly marked not_comparable. Historical BUSI disagreements are shown separately and excluded from the active comparison indicator. No voting, averaging or score adjustment occurs.

## Contract and storage compatibility

1.2.0 extends1.1.0;1.0.0/1.1.0 remain supported, and default API negotiation stays1.0.0. Classifier and consensus-policy versions remain1.0.0. The new frontend requests1.2.0. Legacy SQL prediction/confidence remain abstention/zero compatibility fields, while the full new verdict lives in immutable multi_model_json. No DB schema migration. Human review appends only separate review metadata. Old rows are displayed from stored data and never receive a recomputed verdict. Malformed primary scores, hidden fallback flags, false native-domain claims and unknown versions fail closed.

## Cohesive interface changes

New semantic design tokens/components replace the old full-height split and oversized abstention hero. Light neutral surfaces, restrained blue, readable typography, consistent spacing/cards/controls, focus outlines and reduced motion. Shared mobile/tablet layouts keep the primary finding first; table scrolling stays inside its card. Existing chosen theme is honored and the default is light. Upload now has selection/preview and explicit Analyze; advanced origin is discoverable but secondary, defaults Unknown and remains unverified. Actual record/review counts and real model-score bars are used; no decorative invented performance chart.

Patient management shows compact display IDs with full originals accessible and searchable, alongside name/age/date, original research finding, historical/review status, filters, pagination and View Report. Delete still asks confirmation and no browser deletion was accepted. Older research records show no stored verdict; historical records remain historical. Report/review UI distinguishes the provisional research finding from consensus and independent assessment. Model hashes, class order, limitations and stored review history are accessible. Print-only lineage/history and print CSS were added; native PDF was not generated.

Inert settings suggesting clinical confidence thresholds/ensemble controls and fabricated profile defaults were removed; actual local display-name functionality remains. Four-model inference, genuine BRA Grad-CAM, native-eligible UCLM V3 masks/overlays and all prior domain/EXIF restrictions remain unchanged.

## Genuine results and scope

genuine_output_examples.json and genuine_inference_responses.json contain actual selected-model outputs. Three fixed permitted validation images were reused; no locked images, image search, tuning or new performance study. First-image UCLM Benign99.18%, BRA Benign51.17%, historical BUSI Malignant88.96%, BI-RADS2 with37.82% own-category score. These scores are uncalibrated. The Normal-labelled validation image produced Benign64.13%; the third produced Malignant50.47%. This is functional evidence, not accuracy/calibration/generalization evidence. No genuine Normal prediction occurred among these examples; all class-routing states passed explicit unit fixtures.

## Executed tests

'''+f'Final **{total}/{total} named tests/checks passed**,0 failed. Build and preservation verified separately.\n\n| Group | Passed |\n|---|---|\n{table}\n\n'+'''The28 decision tests are included in131 Python and are not counted again. Prior frontend assertions were retained; only output roots changed. The previous1.2-without-verdict negative fixture remains rejected as an incomplete new contract; separate9.0 tests verify unknown-version rejection. One browser exact-text locator included a nested Retry-button label; fresh alert text confirmed correct product behavior, and the locator was corrected. Final browser checks passed. Viewport reset after390x844 and820x1180 tests.

## Persistence and real browser verification

Isolated guarded api.db and browser.db only.13 named SQL checks verify full envelope roundtrip, immutable verdict, human-review separation, rejected overwrite409 and legacy save/reload.34 interactive checks covered actual uploads/inference, source settings, prominent score/class, separate BI-RADS, historical disagreement, Grad-CAM/segmentation switches, mobile inference, record search/filter, reports, review history, historical/older records and invalid PNG errors. Three browser success uploads used genuine inference; no success results were mocked. Two older-format browser fixtures use genuine previously executed outputs with explicitly synthetic metadata. browser_persistence_verification.json confirms all original fields and the frozen envelope after a deliberately different human Malignant assessment against original Benign. All five browser records were preserved. SQL guard observed no production/other DB connections. Cache audit records four classifiers, one segmentation load and zero remaining hooks. Services stopped gracefully.

## Before/after screenshots

Before: screenshots/before_uclm_analysis.jpg and before_reviewed_uclm_report.jpg are genuine Stage3 captures. After: dashboard.jpg, upload.jpg, advanced_settings.jpg, results.jpg, model_comparison.jpg, patient_management.jpg, report.jpg, report_summary.jpg, mobile_upload.jpg, mobile_results.jpg, mobile_records.jpg and tablet_results.jpg. Additional unknown_fallback, invalid_upload and historical_report images document actual states. See browser_test_results.json for checks and evidence. The source before copies and upgrade_changes.patch identify exact baseline differences.

## Exact source changes

'''+(OUT/'files_changed.md').read_text(encoding='utf-8')+f'''
## Model, data and database preservation

{len(protected)} protected file hashes unchanged, including15 checkpoints, production database and prior research outputs.9,255 raw files retain size/mtime and complete file inventory. Six permitted development image/mask hashes still match. No blanket raw content-hash claim, no locked image contents accessed. Existing Git changes retained; no commit/push/reset. Model/training/dataset/split files unchanged during this upgrade.

## Weighted ensemble feasibility

See weighted_ensemble_feasibility.md and development_manifest_inventory.json. BUSI78-row validation is contaminated; UCLM110 contains Normal/Benign/Malignant; BRA270 contains only binary pathology and separate BI-RADS. These mismatched/reused development sets do not justify activated weights or demonstrate calibrated/generalizable improvements. No weights fitted or locked data used. A future weighted/hierarchical study must use auditable grouping, harmonized labels, suitable calibration and untouched evaluation, with this explicit primary-selection policy as baseline.

## Remaining limits and final decision

No clinical validation, calibrated disease probability, reliable unknown-domain negative, verified origin/reviewer identity, physical lesion measurement, grade/stage or treatment recommendation. The segmentation baseline's existing false-positive/fidelity limitations remain. Authentication/provenance and embedded-artifact storage costs remain. No native PDF/print verification or permanent Delete acceptance; no long-run or prospective external evaluation. Decision B: implementation working for restricted local research, with no blocking defect found in executed scope. Further engineering can proceed under these constraints; no retraining or clinical deployment is authorized.

## READY-TO-PASTE CHATGPT SUMMARY

'''+summary+'\n'
    (OUT/'VERDICT_UI_CODEX_HANDOFF.md').write_text(handoff,encoding='utf-8')
    readme='''# Lumina verdict and UI upgrade

Start with VERDICT_UI_CODEX_HANDOFF.md and READY_TO_PASTE_CHATGPT_SUMMARY.md. All12 requested deliverables are present. Decision B: restricted local research, not clinical deployment.

Contract1.2 adds research_verdict, preserving consensus/models/segmentation/attribution. Default API contract stays1.0; frontend requests1.2. Dataset origin is self-declared; unknown/BUSI fallback is explicitly experimental. No ensemble weights were fitted or activated. Original rows were not upgraded.

Existing dependencies only. From project root:

```
python -m src.verify_verdict_upgrade --unit
node src/test_verdict_frontend.cjs
npm run build
```

Prior frontend suites were executed with LUMINA_TEST_ARTIFACT_ROOT set to this absolute upgrade directory; test_research_frontend.cjs also receives its frontend_regression subdirectory as a positional output argument. Their prior input artifacts remain read-only. Do not run prior report/verifier entry points that overwrite earlier research outputs.

The real workflow verifier (`python -m src.verify_verdict_upgrade`) refuses an existing databases/api.db to preserve this evidence. Browser service (`python -m src.run_verdict_server`) binds127.0.0.1:8765 and guards databases/browser.db; an existing file requires explicit --reuse. `node src/run_verdict_vite.cjs` binds127.0.0.1:5177 without opening a native window. Both services were stopped after verification. The viewport override was reset. Demo sign-in is not authentication.

Screenshots include real success outputs and clearly labeled legacy/error states. before_ images are preserved genuine Stage3 captures. No prediction was mocked for a success screenshot. Native printing/PDF export and permanent Delete acceptance were not verified. See protected_files_audit.json for hash scope; raw dataset checks are metadata-only except six selected input hashes. No models were trained or replaced.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    required=['VERDICT_UI_CODEX_HANDOFF.md','research_decision_policy.json','decision_engine_test_results.json','ui_regression_results.json','browser_test_results.json','contract_compatibility_results.json','persistence_roundtrip_results.json','weighted_ensemble_feasibility.md','protected_files_audit.json','files_changed.md','remaining_issues.json','README.md']
    assert all((OUT/n).is_file() for n in required)
    print(json.dumps(dict(named_tests_checks=total,passed=passed,protected_hashes=len(protected),checkpoint_count=len(checkpoints),modified_source_files=len(modified),new_files=len(new),deliverables=len(required),decision='B')))


if __name__=='__main__':main()
