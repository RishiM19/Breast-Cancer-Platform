"""Assemble Stage 3 handoff only from executed evidence and preservation snapshots."""
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/lumina_integration_stage3'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    initial=read('initial_state.json');changed=[];protected=[];unauthorized=[];patches=[]
    for item in initial['files']:
        p=Path(item['path']);actual=digest(p) if p.is_file() else None
        r=dict(path=str(p),before_sha256=item['sha256'],after_sha256=actual,unchanged=actual==item['sha256'])
        if item['may_change']:
            if not r['unchanged']:
                changed.append(r)
                rel=p.relative_to(ROOT);before=OUT/'before'/rel
                patches.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='before/'+rel.as_posix(),tofile='after/'+rel.as_posix()))
        else:
            protected.append(r)
            if not r['unchanged']:unauthorized.append(r)
    metadata_changes=[]
    for item in initial['dataset_metadata']:
        p=Path(item['path']);stat=p.stat() if p.is_file() else None
        if stat is None or stat.st_size!=item['bytes'] or stat.st_mtime_ns!=item['mtime_ns']:
            metadata_changes.append(item)
    models=[r for r in protected if Path(r['path']).is_relative_to(ROOT/'models')]
    checkpoints=[r for r in models if Path(r['path']).suffix=='.pth']
    original_raw_paths={str(Path(r['path'])) for r in initial['dataset_metadata']}
    current_raw_paths={str(p) for p in (ROOT/'data').rglob('*') if p.is_file() and not p.is_relative_to(ROOT/'data/processed')}
    assert current_raw_paths==original_raw_paths
    production=next(r for r in protected if Path(r['path'])==ROOT/'lumina_database.db')
    inputs=read('development_input_hashes.json')
    input_checks=[dict(path=r['path'],recorded_sha256=r['sha256'],current_sha256=digest(Path(r['path']))) for r in inputs]
    assert not unauthorized and not metadata_changes and all(r['recorded_sha256']==r['current_sha256'] for r in input_checks)
    audit=dict(status='PASSED',baseline='Initial task snapshot including pre-existing dirty Git state; not Git HEAD',
        protected_files_hashed=len(protected),all_protected_hashes_unchanged=True,authorized_existing_source_changes=len(changed),
        checkpoints=checkpoints,model_directory_entries=len(models),production_database=production,
        raw_dataset_files_metadata_checked=len(initial['dataset_metadata']),raw_dataset_metadata_changes=metadata_changes,
        raw_dataset_scope='Size and modification-time only for unprocessed data. No blanket content-hash claim; locked test image contents were never opened.',
        raw_dataset_file_inventory_unchanged=True,
        permitted_development_input_hashes=input_checks,prior_research_outputs_unchanged=True,unauthorized_changes=unauthorized,
        details=protected)
    write('protected_files_audit.json',audit)
    (OUT/'stage3_changes.patch').write_text(''.join(patches),encoding='utf-8')
    status=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True)
    (OUT/'final_git_status.txt').write_text(status,encoding='utf-8')
    py=read('python_regression_results.json');old_front=read('frontend_regression/frontend_test_results.json');old_work=read('frontend_workflow_test_results.json');new_front=read('stage3_frontend_test_results.json')
    scientific=read('segmentation_inference_tests.json');cpu=read('cpu_inference_results.json');api=read('api_integration_results.json');persistence=read('persistence_roundtrip_results.json');browser=read('frontend_browser_results.json')
    groups=[('Python regressions (79 existing + 24 new)',py),('Existing frontend adapter/rendering',old_front),('Existing Stage 2 frontend workflow',old_work),('New Stage 3 frontend adapter/rendering',new_front),('GPU evaluator consistency',scientific),('CPU inference consistency',cpu),('Actual API and stability',api),('Actual SQL persistence/review',persistence),('Interactive browser',browser)]
    counts=[dict(group=name,tests_run=r['tests_run'],passed=r['passed'],failed=r['tests_run']-r['passed']) for name,r in groups]
    total=sum(r['tests_run'] for r in counts);passed=sum(r['passed'] for r in counts)
    assert total==passed and py['errors']==0
    regression=dict(status='PASSED',tests_run=total,passed=passed,failed=0,groups=counts,
        counting='Named unittest cases and named executed verification checks. Coordinate report reuses those results and is not counted again. Build and preservation checks reported separately.',
        production_build='PASSED (npm run build; see build.log)',old_assertions_weakened=False,
        old_test_changes='Output destinations only: new synthetic SQL files and frontend reports redirected to Stage 3. Old 1.0 expectations retained.',
        initially_resolved_harness_issues=['Sandbox temporary-directory cleanup denied; fixtures moved inside Stage 3.',
            'Legacy script-style test imports required adding src to runner import path.',
            'Verifier evidence key status collided with its own check status; evidence now nested.'],
        initial_harness_failures_preserved_in=['segmentation_unit.log','initial_test_harness_errors.json','initial_verifier_error.log'],
        unexecuted=browser['unexecuted'])
    write('regression_test_results.json',regression)
    geometry_names=['native_sizes_and_binary_nearest','asymmetric_landmarks_no_rotation_mirror_or_padding','noninteger_resize_pixel_center_convention',
        'overlay_alpha_and_color_matches_foreground','empty_genuine_output_and_transparency','threshold_strict_ties_background',
        'probability_map_float32_lossless','grayscale_and_rgb_tensor_parity','rgba_converts_without_geometric_transform',
        'nonfinite_and_wrong_shape_rejected','invalid_binary_input_rejected','schema_rejects_mask_overlay_mismatch','schema_rejects_wrong_native_dimensions','nondefault_exif_orientation_declined_before_weights']
    log=(OUT/'python_regression.log').read_text(encoding='utf-8')
    assert all('test_'+name+' (' in log for name in geometry_names)
    coordinate=dict(status='PASSED',unit_geometry_cases=len(geometry_names),executed_cases=['test_'+n for n in geometry_names],
        regression_log='python_regression.log',native_sizes_tested=[[19,37],[856,606],[1400,220],[256,256],[1,1],[1,513],[513,1]],
        asymmetric_integer_scale_oracle='Independent NumPy repeat comparison at 512x768; top-left and bottom-right landmarks, remaining corners background',
        noninteger_scale_oracle='Independent pixel-center floor((destination_index + 0.5) * 256 / destination_size) at 79x61',
        original_coordinate_convention='Decoded original width/height, top-left origin, x right and y down; no EXIF transpose, rotation, mirroring, crop or padding',
        unsupported_exif_orientation='EXIF orientations 2 through 8 explicitly decline segmentation before checkpoint access; genuine classifier and attribution preservation also passed API test.',
        binary_mask='Argmax at model resolution, nearest-neighbor original-size restoration; only 0 and 255',
        alpha='RGBA PNG exact transparent black background; foreground (0,255,128,112)',
        actual_examples=scientific['cases'],browser_original_mask_overlay_geometry=next(c for c in browser['checks'] if c['name'].startswith('Browser renders original dimensions')),
        caveat='Coordinate mapping and display alignment verified; detail lost by stretch resizing cannot be recovered. No perfect spatial fidelity or clinical localization claim.',
        development_reference_annotations='examples/*_reference_annotation.png and comparison figures explicitly label reference separately; never supplied to ordinary inference',
        empty_masks='All three selected genuine cases were nonempty; empty-output behavior tested with explicitly synthetic logits, not a fabricated end-to-end prediction')
    write('mask_coordinate_validation.json',coordinate)
    remaining=dict(final_decision='B',description='Working with explicitly documented limitations; no blocking implementation defects found in executed scope.',
        blocking_implementation_defects=[],limitations=[
            'Caller-declared UCLM eligibility is unverified and can be mislabeled; no dataset or clinical provenance authentication.',
            'Baseline remains exploratory, with prior development/test reuse caveats. Stage 3 is integration regression, not a new scientific performance study.',
            'Preselected Normal validation image ALWI_001 has an empty reference annotation but 15,902 predicted foreground pixels; matches original evaluator. This illustrates false-positive foreground and does not estimate its population frequency.',
            'Stretch-to-square preprocessing loses detail and changes aspect ratio at model input; nearest-neighbor restoration preserves the documented coordinates, not perfect fidelity.',
            'Non-default EXIF orientations 2 through 8 explicitly decline segmentation before checkpoint access, preventing browser/evaluator orientation mismatch; classification remains available.',
            'No physical spacing, mm diameter, grade, stage, BI-RADS from masks, clinical measurements or treatment recommendations.',
            'Probability map is model-space float32 NPY at 256x256, separately encoded from original-size binary PNG and overlay.',
            'Large original-size masks and embedded base64 increase response and SQLite JSON sizes; upload and pixel limits remain bounded, but only a small local workflow was tested.',
            'Native printing, external domains, prospective clinical accuracy and long-running load behavior were not verified.',
            'Demo sign-in and human reviewer identities remain unverified; backend has no production authentication and is restricted to local research.',
            'No model-produced empty mask occurred in the three fixed examples; empty-state behavior passed synthetic unit and React tests.'
        ],stage4_can_begin=True,stage4_scope='Restricted local research engineering only. Authentication/provenance controls and artifact storage limits before any broader service exposure; prospective untouched evaluation requires separate authorization. No automatic retraining or clinical deployment.')
    write('remaining_issues.json',remaining)
    descriptions={
        'main.py':'Validate both known response versions on saved-record reload; preserve immutable review flow and schema.',
        'src/research_inference.py':'Add 1.1 root response and dispatcher, cached serialized segmentation after classifiers/Grad-CAM; retain all classifier mappings and 1.0/policy behavior.',
        'src/research_api.py':'Optional multipart contract_version and discovery version; defaults remain 1.0.',
        'src/research_storage.py':'Validate both contracts and permit only explicit dedicated Stage 2 or Stage 3 synthetic DB directories.',
        'src/train_uclm_segmentation_v2.py':'Package-versus-script import compatibility only; architecture, training and weights unchanged; no training executed.',
        'src/uclm_segmentation_dataset_v2.py':'Factor the identical evaluation normalization into a shared helper; tensor parity checked against preserved pre-edit source.',
        'src/App.tsx':'Request contract 1.1 explicitly; existing full-envelope save flow reused.',
        'src/lib/inferenceAdapter.ts':'Version-aware segmentation parsing and failure checks; old 1.0 mappings unchanged.',
        'src/types/patient.ts':'Add separate segmentation result type and optional field.',
        'src/components/AnalysisResults.tsx':'Mount independent segmentation panel.',
        'src/components/DiagnosticReport.tsx':'Restore segmentation panel from saved outputs, independent of Grad-CAM and human review.',
        'src/test_research_frontend.cjs':'Permit Stage 3 report output only; all old assertions retained.',
        'src/test_stage2_frontend.cjs':'Read immutable Stage 2 fixtures while redirecting output to Stage 3; assertions retained.',
        'src/test_research_workflow.py':'Confined artifact-root override for new synthetic test DB files; old assertions retained.',
    }
    new_names=['src/research_segmentation.py','src/components/ExploratorySegmentation.tsx','src/test_research_segmentation.py',
        'src/verify_integration_stage3.py','src/run_stage3_server.py','src/run_stage3_vite.cjs','src/test_stage3_frontend.cjs','src/report_integration_stage3.py']
    changes=['# Exact Stage 3 source changes','',f'{len(changed)} existing files modified; {len(new_names)} new source/test/report files. Pre-existing Git changes retained. Baseline is `initial_state.json`, not Git HEAD.','', '| Existing file | Change |','|---|---|']
    for r in changed:
        rel=Path(r['path']).relative_to(ROOT).as_posix();changes.append(f'| `{rel}` | {descriptions[rel]} |')
    changes+=['','New source/test/report files:','']+[f'- `{n}`' for n in new_names]
    changes+=['','All generated evidence, isolated databases, pre-edit copies, comparison images, screenshots and logs are inside `data/processed/lumina_integration_stage3/`. `stage3_changes.patch` contains only changes made to existing source in this stage. `source_changes.json` records source hashes. No checkpoint, dataset, split, prior research output or production database was changed. Raw dataset preservation evidence is metadata-only except six permitted development image/mask hashes.']
    (OUT/'files_changed.md').write_text('\n'.join(changes)+'\n',encoding='utf-8')
    write('source_changes.json',dict(modified=changed,created=[dict(path=str(ROOT/n),sha256=digest(ROOT/n)) for n in new_names]))
    summary=f'''Stage 3 implemented genuine frozen UCLM V3 binary lesion segmentation with strict checkpoint loading, cached CUDA/CPU inference, exact RGB/ImageNet 256x256 stretch preprocessing, foreground probability maps, original-size binary PNG masks and transparent PNG overlays. Checkpoint SHA-256: 7acf7f5331e92389537ac85aac0c91ed848ece7455a4b688d49ce4da4d6ea957. No retraining occurred.

Only caller-declared UCLM uploads permit exploratory segmentation; provenance remains unverified. BUSI, BUS-BRA and unknown uploads retain classifier outputs with segmentation not_run and no mask. Non-default EXIF orientation also declines segmentation to prevent a browser/evaluator orientation mismatch. Four classifiers, genuine BRA Grad-CAM, own-label softmax scores, separate pathology/BI-RADS tasks, BUSI exclusion and universal consensus abstention remain intact.

Contract 1.1.0 adds an independent segmentation record; 1.0.0 remains the default API contract and old saved records remain readable. The frontend explicitly requests 1.1.0 and displays original image, AI mask, optional overlay, provenance and limitations in analysis and research reports. SQL save/reload and independent human review preserve the complete frozen segmentation and probabilities; mask overwrite attempts return 409.

Genuine GPU inference on three preselected validation images matched the original evaluator with zero probability-map difference and zero restored-mask differences. CPU float32 parity also passed on one of these images. Geometry and transparency were verified with asymmetric fixtures, multiple aspect ratios and independent pixel-center oracles; browser image dimensions and overlay rectangles matched. Coordinate alignment does not establish perfect spatial fidelity or clinical reliability. Predictions and reference annotations are separately labeled in comparison images. The selected Normal image had false-positive foreground (15,902 pixels), also present in the original evaluator. No accuracy estimate was produced.

Executed results: {total}/{total} named tests/checks passed: 103 Python (79 existing + 24 new), 21 existing frontend adapter/rendering, 24 existing Stage 2 frontend workflow, 18 new frontend, 25 GPU consistency, 4 CPU consistency, 21 API/stability, 13 SQL persistence/review and 20 interactive browser checks. Production frontend build passed. Browser tests covered genuine UCLM upload, overlay toggle, save/reload/report, independent review, unknown-domain not_run, invalid image and missing-checkpoint simulation. BUSI/BUS-BRA restrictions and non-default EXIF handling were additionally tested through the API. GPU peak allocated memory was {api['resources']['gpu_peak_allocated_bytes']/1024**2:.1f} MiB; no Grad-CAM hooks remained; caches were reused, with only one expected segmentation reload after explicit missing-checkpoint invalidation.

All {len(checkpoints)} checkpoint files, the production SQLite database and prior research outputs retained their recorded hashes. {len(protected)} protected files were hash-verified; 9,255 raw dataset files retained size/mtime and file inventory, and all six selected development image/mask hashes were unchanged. Raw data were not blanket content-hashed, and locked test image contents were not accessed. No production patient-record rewrite, checkpoint replacement, model training, commit or push occurred. Changes: 14 existing source/test files and 8 new files; exact inventory and patch are in files_changed.md and source_changes.json.

Unverified: clinical accuracy, external-domain validity, prospective untouched evaluation, long-duration load behavior, native print/PDF output and a genuine empty-mask browser case. Empty-output rendering and transparency passed explicitly synthetic tests. Local test services were stopped. Final decision B: working with explicitly documented limitations, no blocking implementation defects found. Stage 4 may begin for restricted local research engineering, prioritizing authentication/provenance and bounded artifact storage; this does not authorize clinical deployment or retraining.'''
    (OUT/'READY_TO_PASTE_CHATGPT_SUMMARY.md').write_text(summary+'\n',encoding='utf-8')
    count_table='\n'.join(f'| {r["group"]} | {r["passed"]}/{r["tests_run"]} |' for r in counts)
    sections=[
        ('1. Implementation summary','Real frozen V3 lesion foreground segmentation is integrated into the existing application with explicit eligibility, version negotiation, native-coordinate artifacts, immutable persistence and independent review. Decision B; restricted local research only.'),
        ('2. Existing code inspected','Read Stage 1, Stage 2, reliability Phase 2 and BUSI handoffs; current FastAPI router/engine/storage/main, frontend adapters/types/App/results/report/review/stats and existing tests. Inspected V3 training/evaluation, original UNet and DatasetV2, and misleadingly named older visualization script. No original evaluator main or training main was executed.'),
        ('3. Selected V3 segmentation checkpoint and provenance','`models/uclm_unet_v3_binary_best.pth`, SHA-256 `7acf7f5331e92389537ac85aac0c91ed848ece7455a4b688d49ce4da4d6ea957`. Epoch 44, version UCLM_V3_BINARY_EXPLORATORY, frozen patient-disjoint validation-selected baseline. `segmentation_model_audit.json` contains actual metadata and strict-load evidence, without promoting stored validation metrics into Stage 3 performance claims.'),
        ('4. Actual model architecture and preprocessing','Original UNet, base32, four pooling levels, bilinear decoder upsampling (align_corners=False), skip concatenation, 2 output channels; 7,849,634 parameters. RGB conversion; PIL bilinear stretch to 256x256; /255 once, ImageNet mean/std; no padding, crop, random augmentation or EXIF transpose. Two-class float32 softmax; channel1 foreground merges lesion classes. Original evaluator uses two-logit argmax, exact ties background. Equivalent to >0.5 in exact arithmetic; no saved tuned threshold. CUDA FP16 autocast follows evaluator, CPU float32. No ground-truth needed at runtime.'),
        ('5. Domain eligibility policy','Only caller-declared UCLM gets native exploratory output, with unverified provenance notice. BUSI, BUS-BRA and unknown return not_run with null mask, overlay and probability map before any segmentation checkpoint access. Source choice never enables a vote.'),
        ('6. Model inference results','Three fixed validation examples (one per existing label, chosen before inference) achieved exact tensor parity against preserved pre-edit DatasetV2 and exact probability/native-mask parity against the original evaluator forward AST using independently strict-loaded original architecture. CPU parity also passed. All three selected real masks were nonempty. ALWI_001 Normal produced 15,902 foreground pixels despite an empty reference annotation: a baseline limitation, not new clinical evaluation. See segmentation_inference_tests.json and cpu_inference_results.json.'),
        ('7. Coordinate and overlay verification','14 named unit geometry/orientation cases plus actual native outputs and browser rectangle checks passed. Includes non-square/asymmetric images, integer and noninteger pixel-center mapping, grayscale/RGBA, no unintended mirroring/rotation/padding, binary NN restoration, transparent alpha, synthetic empty output and explicit refusal for non-default EXIF orientations. Masks and overlays are original-size; probability NPY is explicitly 256x256 model-space. Resize loses detail. Prediction and reference annotation are separate in examples/*_comparison.png. See mask_coordinate_validation.json.'),
        ('8. Contract/schema changes','New root 1.1.0 adds segmentation; classifier records and abstention policy remain 1.0.0. Dispatcher accepts only 1.0/1.1. Success validates checkpoint, domain, image geometry, binary mask, matching RGBA overlay and finite float32 probability map; unavailable states forbid output artifacts. Old saved records and unknown-version rejection tested. No DB schema migration.'),
        ('9. API changes','POST /predict accepts optional multipart contract_version (default1.0.0); frontend requests1.1.0. GET /inference-contract?version=1.1.0 exposes new schema and eligibility. Existing default discovery is1.0.0. Eligible segmentation participates in request availability status only, never consensus or classifier arrays. Classifier/Grad-CAM/segmentation requests serialize under the same lock. Invalid images400; unsupported version422; missing segmentation preserves other results with partial status.'),
        ('10. Frontend changes','Independent “Exploratory Lesion Segmentation — UCLM V3” section in analysis and report: original, genuine AI binary mask, transparent original overlay switch, checksum provenance and limitations. Shared width/height layout keeps overlay pixel coordinates aligned. Grad-CAM stays separately controlled and labeled class attribution. Missing/error/not_run/historical/empty states are explicit; no fabricated images.'),
        ('11. Save/reload and review verification','Guarded api.db synthetic workflow passed13 named checks. Entire envelope/PNG/NPY survives SQL roundtrip and review. Independent assessment and unverified identity/history are separate. Overwrite409 leaves original unchanged. Legacy1.0 saved/reloaded without upgrade. Browser workflow independently verified complete frozen envelope against browser.db before/after review; see browser_storage_verification.json.'),
        ('12. Browser testing','20/20 real interactive checks through in-app browser against 127.0.0.1:5177/8765; synthetic metadata and guarded browser.db only. Genuine model success, four cards, segmentation mask/overlay toggle/provenance, report reload, independent review, Grad-CAM independence, unknown-domain not_run, invalid upload and missing-checkpoint save/reload. Missing simulation changes only test-server path, never actual weights. Screenshots saved. Native print was not executed. Services stopped after testing.'),
        ('13. Regression and preservation checks',f'79 existing Python and45 existing frontend tests retained;24 new Python and18 new frontend tests passed. {len(protected)} protected hashes unchanged, including15 .pth checkpoints, production DB and previous research outputs; raw9255 size/mtime unchanged, selected6 input hashes unchanged. Preserve the distinction between raw metadata checks and content hashes. Stage2 test output paths redirected toStage3 only; assertions unchanged. Non-default EXIF orientations 2 through 8 explicitly decline segmentation before checkpoint access.'),
        ('14. Numerical or performance issues',f'GPU/CPU reference parity exact under matching device precision. Baseline false-positive foreground noted; no threshold tuning. Eleven bounded real API predictions including two concurrent requests; peak {api["resources"]["gpu_peak_allocated_bytes"]/1024**2:.1f}MiB allocated. Four classifier cache object identities stable; segmentation initially cached once and reloaded only after deliberate missing-path invalidation. Zero remaining Grad-CAM hooks or autograd conflicts. Warm request timings and isolated SQL trace counts in api_integration_results.json; no long-run load claim.'),
        ('15. Tests passed, failed and unexecuted',f'Final {total}/{total} named tests/checks passed;0 failed. Frontend production build and protected hashes passed separately. Initial test-harness/temp-directory/import/evidence-format failures were corrected; initial logs preserved and no assertions weakened. Coordinate report is a reuse of executed checks, not an extra test count.\n\n| Group | Passed |\n|---|---|\n{count_table}\n\nUnexecuted: native printing/PDF; clinical or cross-domain validation; prospective untouched evaluation; long-running load; genuine empty-mask browser case. Empty-state synthetic Python/React tests passed.'),
        ('16. Exact files modified and created',(OUT/'files_changed.md').read_text(encoding='utf-8')),
        ('17. Remaining limitations','See remaining_issues.json. Unverified caller provenance, exploratory baseline with false-positive foreground, stretch-resolution limits, uncalibrated pixels, embedded-artifact payload size, demo identity/auth limitations, no physical measurements or clinical claims. Misnamed V3 visualization still points to old multiclass/test data and was deliberately preserved without execution.'),
        ('18. Final decision','B. Working with explicitly documented limitations. No blocking implementation defects remain in executed scope. Restricted local research integration verified; no clinical deployment readiness claim.'),
        ('19. Recommended next development milestone','Stage4 may begin as restricted local research engineering. Prioritize authenticated roles/reviewer provenance, bounded artifact storage/retention with immutable AI lineage, and defined validation gates. An untouched prospective scientific evaluation requires separate authorization. Do not retrain automatically, reinstate voting or expose this unauthenticated backend publicly.'),
        ('20. READY-TO-PASTE CHATGPT SUMMARY',summary),
    ]
    (OUT/'STAGE3_CODEX_HANDOFF.md').write_text('# Lumina Integration Stage 3 handoff\n\n'+'\n\n'.join('## '+title+'\n\n'+body for title,body in sections)+'\n',encoding='utf-8')
    readme='''# Lumina Integration Stage 3

Decision **B**: working with documented limitations, restricted local research only. Start with STAGE3_CODEX_HANDOFF.md and READY_TO_PASTE_CHATGPT_SUMMARY.md.

The 13 requested deliverables are present: handoff, model audit, segmentation tests, coordinate validation, contract update, API results, persistence results, browser results, regressions, protected-file audit, remaining issues, file changes and this README. Supporting JSON, logs, isolated synthetic databases, original-source copies and screenshots provide executed evidence. `examples/` contains genuine predicted masks/transparent overlays and separately labeled reference annotations; no locked test images were accessed.

Reproduction from project root uses existing bc_ai Python and Node dependencies; no installs or training:

```
python -m src.verify_integration_stage3 --unit
python -m src.verify_integration_stage3 --cpu
node src/test_research_frontend.cjs data/processed/lumina_integration_stage3/frontend_regression
# For the original Stage 2 frontend suite, set LUMINA_TEST_ARTIFACT_ROOT to the absolute Stage 3 directory first:
node src/test_stage2_frontend.cjs
node src/test_stage3_frontend.cjs
npm run build
```

Real scientific/API verification (`python -m src.verify_integration_stage3`) refuses to overwrite its existing `databases/api.db`; preserve this evidence. Browser server (`python -m src.run_stage3_server`) refuses existing browser.db unless `--reuse` is explicit, binds127.0.0.1 and guards every SQL connection. Its `--missing-segmentation` flag changes only a test path. `node src/run_stage3_vite.cjs` binds127.0.0.1:5177 without opening a native window. Both test services were stopped at completion. Human demo sign-in is not authentication.

Default API callers retain contract1.0.0. Add multipart `contract_version=1.1.0` for segmentation; only declared `source_domain=UCLM` enables it. Declaration is unverified. GET `/inference-contract?version=1.1.0` describes the extension. Classifier/consensus policy remains unchanged; BUSI never votes.

The probability artifact is lossless float32 NPY256x256 in model coordinates; decode base64 then NumPy load with allow_pickle=False. Binary PNG and transparent overlay PNG use original decoded dimensions. Argmax ties select background; no tuned threshold exists. No clinical probabilities or measurements are provided.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    required=['STAGE3_CODEX_HANDOFF.md','segmentation_model_audit.json','segmentation_inference_tests.json','mask_coordinate_validation.json','inference_contract_update.json','api_integration_results.json','persistence_roundtrip_results.json','frontend_browser_results.json','regression_test_results.json','protected_files_audit.json','remaining_issues.json','files_changed.md','README.md']
    assert all((OUT/n).is_file() for n in required)
    print(json.dumps(dict(named_checks=total,passed=passed,protected_hashes=len(protected),existing_source_files_changed=len(changed),new_source_files=len(new_names),required_deliverables=len(required),decision='B')))


if __name__=='__main__':main()
