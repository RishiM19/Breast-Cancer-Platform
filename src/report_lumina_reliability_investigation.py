"""Generate readable reports from saved reliability results; never runs inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/processed/lumina_reliability_investigation"
CODE = ["artifact_sensitivity_lumina_classifiers.py", "test_artifact_sensitivity_lumina_classifiers.py",
        "report_lumina_reliability_investigation.py"]


def table(df):
    def fmt(value):
        if pd.isna(value): return "not estimated"
        if isinstance(value, float): return f"{value:.3f}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join(["---"]*len(df.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))


def build_reports(out, refresh=False):
    out = Path(out).resolve()
    if not out.is_relative_to(DEFAULT):
        raise ValueError("Reports must remain in the isolated investigation directory")
    for name in ("README.md", "CODEX_HANDOFF.md"):
        if (out/name).exists() and not refresh:
            raise FileExistsError(f"{out/name} exists; --refresh-reports explicitly regenerates reports only")
    summary = json.loads((out/"investigation_summary.json").read_text())
    audit = json.loads((out/"audit_summary.json").read_text())
    verification = json.loads((out/"verification.json").read_text())
    results = pd.read_csv(out/"image_level_results.csv")
    stats = pd.read_csv(out/"perturbation_summary.csv")
    paired = pd.read_csv(out/"paired_control_summary.csv")
    notes_path = out/"execution_notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    original = results[results.region == "original"]
    primary = stats[stats.region != "random_control"].copy()
    numerical = primary[["model", "region", "fill", "n_cases", "class_flip_rate", "mean_score_delta",
                         "median_score_delta", "mean_abs_score_delta", "mean_logit_delta", "median_logit_delta",
                         "abs_score_delta_ci_low", "abs_score_delta_ci_high"]].copy()
    for col in ("class_flip_rate", "mean_score_delta", "median_score_delta", "mean_abs_score_delta",
                "abs_score_delta_ci_low", "abs_score_delta_ci_high"):
        numerical[col] *= 100
    metric_table = table(numerical)
    pair_table = paired.copy()
    for col in ("mean_excess_abs_effect", "median_excess_abs_effect", "ci_low", "ci_high", "mean_excess_support_drop"):
        pair_table[col] *= 100
    # Rankings are descriptive within each intervention; no accuracy comparison across tasks.
    ranking = []
    for fill in ("local_median", "blur"):
        for region in ("text", "left_border", "right_border", "top_strip", "lesion", "context"):
            g = primary[(primary.fill == fill) & (primary.region == region)]
            if len(g):
                best = g.loc[g.mean_abs_score_delta.idxmax()]
                ranking.append(f"- {region}, {fill}: {best.model.upper()}, mean absolute change {100*best.mean_abs_score_delta:.2f} percentage points, n={int(best.n_cases)}.")
        controls = results[(results.fill == fill) & (results.region == "random_control")]
        case_means = controls.groupby(["model", "case_id"]).abs_score_delta.mean().groupby("model").mean()
        if len(case_means):
            ranking.append(f"- Random controls, {fill}: {case_means.idxmax().upper()}, case-weighted pooled absolute change {100*case_means.max():.2f} pp; mixed control sizes, descriptive only.")
    model_lines = []
    for name, data in verification.items():
        p = data["profile"]
        model_lines.append(f"- {name.upper()}: torchvision EfficientNet-B0, Linear(1280,{len(p['classes'])}) classifier; ordered classes {p['classes']}; RGB {p['preprocessing']} {p['input_size']}x{p['input_size']}, bilinear. Checkpoint `{ROOT/'models'/p['checkpoint']}`; strict state load passed. SHA-256 `{data['checkpoint_sha256']}`.")
    command = "& '" + summary["command"][0] + "' " + " ".join(summary["command"][1:])
    n = summary["cases"]
    key_metrics = []
    for model in summary["models"]:
        lesion = primary[(primary.model == model) & (primary.region == "lesion") & (primary.fill == "local_median")].iloc[0]
        text = primary[(primary.model == model) & (primary.region == "text") & (primary.fill == "local_median")]
        key_metrics.append(f"{model.upper()}: lesion median-fill mean absolute score change {100*lesion.mean_abs_score_delta:.2f} pp, flips {100*lesion.class_flip_rate:.1f}%" +
                           (f"; text {100*text.iloc[0].mean_abs_score_delta:.2f} pp, flips {100*text.iloc[0].class_flip_rate:.1f}%" if len(text) else "; text unavailable") + ".")
    findings = notes.get("interpretation", "Compare region effects against paired controls and across fills in section 9. Rankings alone do not establish artifact reliance; stronger median-fill than blur effects may reflect the operation itself.")
    recommendations = notes.get("recommendations", "B. Perform a more controlled artifact investigation. Prioritize disjoint shape- and location-matched controls, multiple fill strengths, blinded text/marker review, and larger validation case replication. Resolve BUSI provenance and conflicting image labels. No retraining or production changes are justified by this pilot alone.")
    limitations = summary["limitations"] + [
        "Text analysis targets anatomical/acquisition words on reviewed images, not all typography; logos, calipers and numeric measurement readouts were not isolated.",
        "The 10% context patch and random controls may contain text or instrument artifacts; they are lesion-free, not artifact-free. Native lesion pixels are preserved, but bilinear preprocessing may mix neighboring pixels at boundaries.",
        "Mask geometry checks establish coordinate consistency, not clinical annotation accuracy. Text rectangles contain surrounding tissue and can therefore change more than glyphs.",
        "One image per case; balanced strata differ from the natural validation distribution. Patient-level distinctness relies on manifest case_id; no external patient identity verification.",
        "Control locations average three repeats; percentile bootstrap resamples cases (2000 draws), not repeated images as independent observations. Intervals are unstratified, exploratory and not population/clinical guarantees.",
        "Raw lesion/artifact effects involve unequal areas; use each region's own exact-area paired controls. Cross-classifier softmax sensitivity is influenced by different tasks, class counts, saturation and margins.",
        "Read-only existing code inspection included evaluation source; no test manifest or test image was opened. Python audit log is not an operating-system-wide file trace.",
    ]
    short = f"""Lumina reliability investigation completed: {n} distinct BUS-BRA validation cases, strata {summary['selected_cases_by_birads']}, classifiers {summary['models']}. Text analysis used {summary['text_cases']} manually reviewed cases. Seed {summary['seed']}; two fills; three exact-native-area lesion-free random controls per region. {summary['observations']} prediction records including originals.

Created research code: {', '.join(str(ROOT/'src'/f) for f in CODE)}. No existing application, checkpoint, dataset, manifest or database files were modified by this investigation.

Executed verification: {', '.join(summary['checks_passed'])}. {notes.get('test_summary', 'See verification.json for executed per-model checks; separate smoke/unit-test status is not inferred for this run.')}

Main metrics (uncalibrated original-predicted-class score, percentage points): {' '.join(key_metrics)}

Main findings: {findings}

Limitations: validation-only exploratory results, BUSI/UCLM domain shift, different prediction tasks, unequal raw region areas, imperfect shape/texture control, masking artifacts, small reviewed-text subset, missing original BUSI training provenance, and image-level BI-RADS disagreement for case 856. No clinical validation or causal proof of shortcut learning.

Recommended next step: {recommendations}

Detailed handoff: {out/'CODEX_HANDOFF.md'}
Results and plots: {out}
Core data: {out/'image_level_results.csv'}; {out/'perturbation_summary.csv'}; {out/'paired_control_summary.csv'}.
"""
    short = notes.get("ready_to_paste_summary", short)
    sections = [
        "# LUMINA RELIABILITY INVESTIGATION — CODEX HANDOFF",
        "## 1. Execution Status\n\nCompleted: requested pilot executed. Scope is a validation research pilot, not resolution of all reliability questions. " + f"{n} cases; {summary['models']}; {summary['observations']} rows. Larger runs are supported but were not executed.",
        f"## 2. Original Project State\n\nRepository `{ROOT}`. Windows, bc_ai interpreter `{summary['command'][0]}`, torch {summary['torch_version']}, device {summary['device']}, GPU {summary['gpu']}. Existing unified Grad-CAM loader reused. Original training/dataset implementations inspected without calling split dataloader factories or importing main.py. Initial tracked modifications: main.py, lumina_database.db, src/components/AnalysisResults.tsx, src/components/DiagnosticReport.tsx; multiple pre-existing untracked research scripts and images. See `{DEFAULT/'provenance_start.json'}` for original git status and file hashes.",
        "## 3. Files Created\n\n" + "\n".join(f"- `{ROOT/'src'/f}` — " + purpose for f,purpose in zip(CODE, ["audit, deterministic perturbations, model inference, metrics and figures", "eight isolated geometry/safety tests", "reproducible report generation from saved results"])) +
        f"\n\nAll result files reside under `{out}`. Exact absolute paths and purposes for every new output (including individual region PNGs, examples, smoke/failed-run evidence and report files) are enumerated in `{out/'artifact_inventory.csv'}`. Core outputs: audit_summary.json, image_level_results.csv, model_summary.csv, perturbation_summary.csv, case_level_summary.csv, representative_examples.png, sensitivity_comparison.png, investigation_summary.json, README.md and CODEX_HANDOFF.md. Additional outputs include audit_rows.csv, selected_cases.csv, birads_summary.csv, paired_control_results.csv, paired_control_summary.csv, verification.json, data_access_log.json, region_definitions.json, skipped_regions.json and text ROI configuration.",
        "## 4. Files Modified\n\nNo pre-existing project files were modified by this investigation. New research files were revised during development. " + notes.get("integrity_summary", "Refer to provenance_start.json and any integrity_verification.json for the executed hash comparison; a production end-to-end run is not implied."),
        "## 5. Model Configuration\n\n" + "\n".join(model_lines) + "\n\nAll models use ImageNet normalization mean [0.485,0.456,0.406], std [0.229,0.224,0.225], eval mode, float32 and no_grad for ordinary inference. No autocast; TF32 convolution and matmul disabled; deterministic algorithms enabled. BI-RADS centered ImageOps.pad uses black padding. Actual evaluation tensors exactly matched bra_dataset.get_bra_transforms, uclm_dataset.get_uclm_transforms and train_bra_birads_v1.BRADataset(training=False). BUSI training/evaluation source is absent; class mapping and preprocessing follow main.py and the unified loader and cannot be inferred independently from a plain state dictionary. No weights were downloaded or changed.",
        f"## 6. Experiment Design\n\nAudited {audit['rows']} images / {audit['unique_cases']} cases; {audit['missing_images']} missing images, {audit['missing_masks']} missing masks, {len(audit['errors'])} invalid rows, {audit['duplicate_content_or_paths']} duplicate images. Case(s) with conflicting image-level BI-RADS: {audit['cases_with_conflicting_birads']}. Selected {n} unique cases by deterministic shuffled BI-RADS round robin, one image per case, seed {summary['seed']}, with no outcome-based selection. BI-RADS baseline correct: {summary['baseline_birads_correct']}/{n}; both correct and incorrect examples are retained when present. Selection and smoke overlap are not independent replication.\n\nNative-coordinate regions: left/right 5% width, top 10% height, original full lesion, central 10% context patch excluding lesion/top/side strips, and documented reviewed text rectangles minus lesion. Borders/top are location probes, not confirmed artifacts. Text available in {summary['text_cases']} cases. Original masks must be binary 0/255, nonempty, and exactly image-sized; invalid rows are skipped and logged. No mask rescaling at native resolution. Nearest-neighbor projection is used only to verify model-space geometry and padding exclusion.\n\nLocal median fill uses a lesion-free outer ring of radius 3% of the short image side (at least 3px), falling back to all available lesion-free donors if necessary. Localized Gaussian blur uses radius 2.5% of the short side (at least 2px), assigning only ROI pixels. Three seeded compact lesion-free controls match each ROI's exact native pixel count; avoid the tested region where feasible, otherwise record overlap. Shapes and underlying textures are not matched. All native masks and rules are saved. Positive delta means increased support for the original class; negative means decreased support, even after a class flip. Softmax scores are not calibrated confidence.",
        "## 7. Commands Executed\n\nCurrent inference invocation (completed):\n\n```powershell\n"+command+"\n```\n\n" + notes.get("commands", "Other commands are not inferred by the report generator; see execution_notes.json when present."),
        "## 8. Test and Verification Results\n\nPassed during inference: " + ", ".join(summary["checks_passed"]) + ". Per-case details: verification.json.\n\n" + notes.get("test_summary", "Separate unit/smoke executions not inferred for this run.") + "\n\n" + notes.get("resolved_failures", "Any retained failure.json files document failed prior attempts; do not use them as completed experiments.") + "\n\nUnexecuted: production end-to-end UI/API tests, 60/all-case extension, independent train/test membership audit, calibration/clinical validation, original BUSI training transform verification. Dataset guard and saved access logs restrict research access to validation-listed source files; no held-out split was loaded.",
        "## 9. Quantitative Results\n\nScore changes and intervals below are percentage points; flip rates are percent; logits remain in raw units. Mean/median values are case weighted, not independent perturbation repeats. n_cases is the denominator. Intervals are 2000-draw case bootstrap 95% percentile estimates (suppressed for n<5).\n\n"+metric_table+"\n\nPaired region minus own area-matched controls (score columns in pp):\n\n"+table(pair_table)+"\n\nDescriptive rankings by mean absolute score change:\n\n"+"\n".join(ranking)+"\n\nAll control-level signed changes, flips, logits, BI-RADS strata, case variation and native/model-space areas are in the CSVs. model_summary.csv averages a mixture of interventions within each case and must not be used as an overall model-quality ranking. No accuracy is pooled across tasks.\n\n"+findings,
        f"## 10. Visual Findings\n\n`{out/'representative_examples.png'}` shows the first preselected case with original, modified-region overlay and median-filled input, predicted classes and fixed-target score changes. `{out/'sensitivity_comparison.png'}` compares actual region and control effects by model and fill, with descriptive case intervals. selection_contact_*.png shows native lesion alignment; regions/ holds exact masks; perturbed_examples/ holds two fills for the first case. Smoke figures are in smoke_fp32/.\n\n" + notes.get("visual_review", "Figures are generated from actual saved measurements; automated image decoding passed. Human/assistant visual-review status is not inferred."),
        "## 11. Reliability Concerns\n\n" + findings + "\n\nResolve missing BUSI provenance and image-level label inconsistency. Non-lesion sensitivity can reflect tissue context, text, preprocessing or perturbation artifacts; these experiments do not identify learned causal shortcuts. Measurement calipers were not separately tested, and original lesion annotations can include caliper pixels.",
        "## 12. Limitations\n\n" + "\n".join("- "+v for v in limitations),
        "## 13. Recommendations\n\n" + recommendations,
        "## 14. Current Lumina Project Status\n\nThe four trained classifiers and existing segmentation/Grad-CAM work remain intact. This adds a reproducible validation reliability checkpoint before further integration decisions. It does not change the production backend, frontend, database, checkpoints, datasets or existing Grad-CAM outputs, and is not an integration or clinical-readiness approval.",
        "## 15. Questions Requiring ChatGPT Review\n\n- Can the original BUSI training code and class provenance be recovered?\n- Are different image-level labels for case 856 intentional?\n- Which region/fill effects merit a preregistered larger validation study?\n- Should controls match shape, relative lesion distance, texture and text-free status?\n- Should calipers, device logos and measurement readouts be separately annotated?\n- What region-specific effect threshold and replication criterion should be fixed before any cleanup/retraining study?",
        "## 16. Ready-to-Paste ChatGPT Summary\n\n" + short,
    ]
    (out/"CODEX_HANDOFF.md").write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    readme = f"""# Lumina validation reliability investigation

Actual completed run: {n} distinct cases; {summary['observations']} observations; all configured models loaded. Read CODEX_HANDOFF.md for interpretation, checks and limitations. This is research, not clinical validation.

## Reproduction (PowerShell from {ROOT})

Use the existing bc_ai Python; no installation is required. Every inference destination must be fresh: prior runs are never overwritten.

```powershell
& '{summary['command'][0]}' -m py_compile src/artifact_sensitivity_lumina_classifiers.py src/test_artifact_sensitivity_lumina_classifiers.py src/report_lumina_reliability_investigation.py
& '{summary['command'][0]}' -m unittest discover -s src -p test_artifact_sensitivity_lumina_classifiers.py -v
& '{summary['command'][0]}' src/artifact_sensitivity_lumina_classifiers.py --audit-only --max-cases 24 --seed 42 --output-dir data/processed/lumina_reliability_investigation/new_audit
& '{summary['command'][0]}' src/artifact_sensitivity_lumina_classifiers.py --smoke-test --models all --seed 42 --text-rois data/processed/lumina_reliability_investigation/text_rois.json --output-dir data/processed/lumina_reliability_investigation/new_smoke
& '{summary['command'][0]}' src/artifact_sensitivity_lumina_classifiers.py --run --max-cases 24 --models all --seed 42 --text-rois data/processed/lumina_reliability_investigation/text_rois.json --output-dir data/processed/lumina_reliability_investigation/new_pilot
```

For extension use `--max-cases 60` or `--max-cases 0` (one selected image per all eligible validation cases) and a new output directory. These extensions were NOT executed. Additional text annotations are required for broader text inference; unseen cases are skipped, never assigned guessed coordinates. `--models busi,uclm,bra,birads`, `--device cpu`, `--batch-size 8`, `--control-repeats 3`, `--border-fraction 0.05`, `--top-fraction 0.10` are supported. CPU execution is implemented but not benchmarked in this investigation.

The report helper reads saved results only. `python src/report_lumina_reliability_investigation.py --output-dir <run>` generates reports; `--refresh-reports` explicitly refreshes Markdown/inventory only. Optional execution_notes.json records session-specific verification and interpretation. No test or training split is opened.

## Output interpretation

- image_level_results.csv: original and perturbed class/logits/margins, fixed-original-target score deltas, native/model pixel fractions, fill/region rules; class_order specifies logits order. Positive delta increases original-class support. Score units are 0–1; multiply by 100 for percentage points.
- perturbation_summary.csv: model/region/fill metrics; random_control rows are grouped by matched_region. Three repeats are averaged within each case before case-level statistics/bootstrap.
- paired_control_summary.csv: positive excess_abs_effect means region changes exceeded its own same-area random controls, not proof of artifact use. Its 95% intervals are exploratory and unadjusted for many comparisons.
- case_level_summary.csv, birads_summary.csv: case and BI-RADS subgroup views; intervals omitted below five cases. model_summary.csv averages unlike perturbations and is not a model-quality score.
- text_rois.json: manually reviewed native `[x0,y0,x1,y1)` rectangles, image dimensions, hashes and review notes. Definition targets visible anatomical/acquisition text; no OCR. No verified ROI means skip.
- regions/: exact binary native masks for every tested region and control; region_definitions.json records selection rules and control overlaps. No lesion/padding confusion is permitted.
- audit_rows.csv/audit_summary.json: file existence, checksums, geometry, duplicates and labels for validation only.
- verification.json/data_access_log.json: model/checkpoint/preprocessing checks and Python-level data accesses. Not an OS-level access trace.
- artifact_inventory.csv: complete paths/purposes, including retained failed first smoke evidence. Use smoke_fp32 for the successful smoke results.

Different tasks/class counts, source domain shift, score saturation, mask shape, blur strength and fill edges confound cross-model/raw-region rankings. Context and controls may contain text. Large changes do not establish shortcut learning or clinical validity. No retraining occurred.
"""
    (out/"README.md").write_text(readme, encoding="utf-8")
    purpose = {".png":"actual figure, exact region mask or perturbed image (see parent directory)",
               ".csv":"audit, prediction, summary or provenance table", ".json":"configuration, audit, verification or execution metadata",
               ".md":"research documentation", ".txt":"command/test output log"}
    paths = sorted(p for p in out.rglob("*") if p.is_file())
    paths += [ROOT/"src"/f for f in CODE]
    inventory_path = out/"artifact_inventory.csv"
    if inventory_path not in paths: paths.append(inventory_path)
    pd.DataFrame([dict(path=str(p), purpose=purpose.get(p.suffix, "research source/evidence")) for p in paths]).to_csv(inventory_path,index=False)
    print(f"Reports saved: {out/'CODEX_HANDOFF.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--refresh-reports", action="store_true")
    args = parser.parse_args()
    build_reports(args.output_dir, args.refresh_reports)
