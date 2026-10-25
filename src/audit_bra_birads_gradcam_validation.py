"""Lumina research-only stratified Grad-CAM audit on BUS-BRA *validation* cases.

From the repository root:
    python src/audit_bra_birads_gradcam_validation.py --audit-only
    python src/audit_bra_birads_gradcam_validation.py --run

Requires the previously verified src/gradcam_bra_birads_v1.py; does not modify it.
No test data are loaded, no training is performed, no fabricated attributions.
The script samples up to three images per annotated BI-RADS class from distinct
cases, prioritizing two correct and one incorrect predictions per class when
possible. Results are DESCRIPTIVE and NOT clinical explanations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from gradcam_bra_birads_v1 import (
    ROOT, VAL_CSV, CHECKPOINT, IMAGE_SIZE, LABELS,
    image_path_from_manifest, preprocess, load_model, get_gradcam,
)

OUTPUT_DIR = ROOT / 'data' / 'processed' / 'bra_birads_v1_gradcam_validation_audit'


def validate_manifest() -> pd.DataFrame:
    df = pd.read_csv(VAL_CSV)
    required = {'case_id', 'image_path', 'mask_path', 'birads'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Validation manifest missing columns: {sorted(missing)}')
    if df['case_id'].isna().any() or df['birads'].isna().any():
        raise ValueError('Missing case ID or BI-RADS label in validation manifest')
    if not df['birads'].isin(LABELS).all():
        raise ValueError('Unexpected BI-RADS label in validation manifest')
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Model checkpoint missing: {CHECKPOINT}')
    if not len(df):
        raise ValueError('Empty validation manifest')
    print(f'Validation images: {len(df)} | unique cases: {df.case_id.nunique()}')
    print('Category counts:', df['birads'].value_counts().sort_index().to_dict())
    print('Checkpoint:', CHECKPOINT)
    print('Test manifest is not accessed.')
    return df


def prepare_mask(img_path: Path, mask_path: Path):
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    with Image.open(img_path) as im, Image.open(mask_path) as m:
        rgb = im.convert('RGB')
        mask = m.convert('L')
        if rgb.size != mask.size:
            raise ValueError(f'Image/mask mismatch: {img_path.name} {rgb.size} vs {mask.size}')
        pixels = np.asarray(mask)
        vals = np.unique(pixels)
        if not set(vals.tolist()).issubset({0, 255}):
            raise ValueError(f'Nonbinary mask for {img_path.name}: {vals[:12]}')
        padded_mask = ImageOps.pad(mask, (IMAGE_SIZE, IMAGE_SIZE),
                                   method=Image.Resampling.NEAREST,
                                   color=0, centering=(0.5, 0.5))
        valid = ImageOps.pad(Image.new('L', rgb.size, 255),
                             (IMAGE_SIZE, IMAGE_SIZE),
                             method=Image.Resampling.NEAREST,
                             color=0, centering=(0.5, 0.5))
    lesion = np.asarray(padded_mask) > 127
    valid_image = np.asarray(valid) > 127
    if not np.any(lesion):
        raise ValueError(f'Empty lesion mask: {img_path.name}')
    if (lesion & ~valid_image).any():
        raise ValueError(f'Lesion in padding: {img_path.name}')
    return lesion, valid_image


def attribution_stats(cam: np.ndarray, lesion: np.ndarray, valid: np.ndarray) -> dict:
    if cam.shape != lesion.shape or lesion.shape != valid.shape:
        raise ValueError('CAM / lesion / image-valid-mask shape mismatch')
    if not np.isfinite(cam).all():
        raise ValueError('Non-finite Grad-CAM values')
    total = float(cam[valid].sum())
    if total <= 1e-10:
        raise ValueError('Grad-CAM has no positive mass in imaged area')
    area = float(lesion.sum() / valid.sum())
    mass = float(cam[lesion].sum() / total)
    viable = np.flatnonzero(valid.reshape(-1))
    k = max(1, round(0.2 * len(viable)))
    # As in the existing single-case quantifier; top-K ties are not unique.
    ordered = np.argpartition(cam.reshape(-1)[viable], -k)[-k:]
    selected = viable[ordered]
    top_inside = int(lesion.reshape(-1)[selected].sum())
    return {
        'lesion_area_fraction': area,
        'cam_mass_inside_lesion': mass,
        'enrichment_vs_uniform': mass / area,
        'top20_fraction_inside_lesion': top_inside / k,
        'lesion_fraction_in_top20': top_inside / int(lesion.sum()),
    }


def select_rows(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Deterministic, unique-case stratification with correctness diversity."""
    rng = np.random.default_rng(seed)
    chosen = []
    used_cases = set()
    for label in LABELS:
        group = df[df['birads'] == label]
        choices = group.index.to_numpy()
        choices = rng.permutation(choices).tolist()
        correct = [i for i in choices if bool(df.at[i, 'correct'])]
        incorrect = [i for i in choices if not bool(df.at[i, 'correct'])]
        # At least one wrong, when available, but not claiming representativeness.
        preference = ([True, False] if per_class == 2 else
                      [True, False] + [True] * (per_class - 2))
        for prefer_correct in preference:
            primary = correct if prefer_correct else incorrect
            other = incorrect if prefer_correct else correct
            found = None
            for idx in primary + other:
                case = str(df.at[idx, 'case_id'])
                if case not in used_cases and idx not in chosen:
                    found = idx
                    break
            if found is None:
                raise ValueError(f'Cannot find {per_class} distinct cases for BI-RADS {label}')
            chosen.append(found)
            used_cases.add(str(df.at[found, 'case_id']))
    result = df.loc[chosen].copy()
    if result['case_id'].astype(str).duplicated().any():
        raise RuntimeError('Duplicate validation case selected')
    return result


def create_audit(df: pd.DataFrame, per_class: int, seed: int, device):
    print('Loading selected BI-RADS V1 checkpoint...')
    model, ckpt = load_model(device)
    # Predict once over validation data, to deliberately sample both correct and
    # incorrect predictions without inspecting or using the test set.
    preds = []
    scores = []
    with torch.inference_mode():
        for i, row in df.iterrows():
            path = image_path_from_manifest(row['image_path'])
            _, tensor = preprocess(path)
            logits = model(tensor.to(device))
            probs = logits.softmax(dim=1)[0].cpu().numpy()
            preds.append(LABELS[int(np.argmax(probs))])
            scores.append([float(v) for v in probs])
    df['predicted'] = preds
    df['correct'] = df['birads'].to_numpy() == np.asarray(preds)
    df['softmax_scores'] = scores
    selected = select_rows(df, per_class, seed)
    print('Selected annotated/predicted cases (one image per case):')
    for idx, row in selected.iterrows():
        print(f'  val index {idx:3d} | case {row.case_id} | label {row.birads} | '
              f'pred {row.predicted} | {"correct" if row.correct else "incorrect"}')

    reports = []
    pictures = []
    failures = []
    for idx, row in selected.iterrows():
        path = image_path_from_manifest(row['image_path'])
        mask_path = image_path_from_manifest(row['mask_path'])
        try:
            lesion, valid = prepare_mask(path, mask_path)
            img, tensor = preprocess(path)
            tensor = tensor.to(device)
            pred_idx = LABELS.index(int(row.predicted))
            pred_cam, actual_probs = get_gradcam(model, tensor, pred_idx)
            gt_idx = LABELS.index(int(row.birads))
            gt_cam = pred_cam if gt_idx == pred_idx else get_gradcam(model, tensor, gt_idx)[0]
            pred_stats = attribution_stats(pred_cam, lesion, valid)
            gt_stats = attribution_stats(gt_cam, lesion, valid)
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({'index': int(idx), 'case_id': str(row.case_id),
                             'reason': str(exc)})
            print('  SKIPPED with recorded error:', idx, str(exc))
            continue
        output = {
            'val_row': int(idx), 'case_id': str(row.case_id),
            'image_id': Path(path).stem,
            'annotated_birads': int(row.birads),
            'predicted_birads': int(row.predicted),
            'correct': bool(row.correct),
            'predicted_class_softmax': float(actual_probs[pred_idx]),
            'annotated_class_softmax': float(actual_probs[gt_idx]),
        }
        for key, value in pred_stats.items():
            output['pred_' + key] = value
        for key, value in gt_stats.items():
            output['annotated_' + key] = value
        reports.append(output)
        pictures.append((img, lesion, pred_cam, gt_cam, output))

    if not reports:
        raise RuntimeError('All examples failed the annotation/CAM audit; no output created')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(reports).to_csv(OUTPUT_DIR / 'case_results.csv', index=False)
    (OUTPUT_DIR / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(len(pictures), 3, figsize=(13, 3.1 * len(pictures)),
                             squeeze=False)
    for ri, (img, lesion, pred_cam, gt_cam, record) in enumerate(pictures):
        axes[ri, 0].imshow(img)
        axes[ri, 0].contour(lesion.astype(float), levels=[0.5], colors=['lime'], linewidths=1.0)
        axes[ri, 0].set_title(f"Case {record['case_id']} | GT {record['annotated_birads']} "
                              f"→ pred {record['predicted_birads']}", fontsize=10)
        for ci, (cam, col, target) in enumerate([
            (pred_cam, 1, record['predicted_birads']),
            (gt_cam, 2, record['annotated_birads']),
        ]):
            axes[ri, col].imshow(img)
            axes[ri, col].imshow(cam, cmap='jet', vmin=0, vmax=1, alpha=0.48)
            axes[ri, col].contour(lesion.astype(float), levels=[0.5], colors=['lime'], linewidths=1.0)
            enrichment = record['pred_enrichment_vs_uniform'] if ci == 0 else record['annotated_enrichment_vs_uniform']
            axes[ri, col].set_title(f'Target {target} | enrichment {enrichment:.2f}×', fontsize=10)
        for ax in axes[ri]:
            ax.axis('off')
    fig.suptitle('Lumina — validation-only, case-distinct Grad-CAM audit (green = lesion border)',
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.993))
    fig.savefig(OUTPUT_DIR / 'contact_sheet.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    results = pd.DataFrame(reports)
    summary = {
        'scope': 'BUS-BRA validation only, one image per case, exploratory selection of correct and incorrect predictions',
        'checkpoint': str(CHECKPOINT),
        'checkpoint_best_epoch': ckpt.get('best_epoch'),
        'seed': seed,
        'intended_examples_per_class': per_class,
        'successful_examples': len(reports),
        'recorded_failures': len(failures),
        'mean_predicted_cam_enrichment': float(results['pred_enrichment_vs_uniform'].mean()),
        'median_predicted_cam_enrichment': float(results['pred_enrichment_vs_uniform'].median()),
        'mean_annotated_cam_enrichment': float(results['annotated_enrichment_vs_uniform'].mean()),
        'by_annotated_class': {},
        'by_correctness': {},
        'cautions': [
            'Descriptive convenience sample, not an unbiased estimate of explanation quality.',
            'CAM is per-target min-max normalized and only indicates positive gradient-weighted attribution.',
            'CAM lesion overlap does not establish causation or diagnostic correctness.',
            'Validation cases only; test set not accessed.',
        ],
    }
    for category, group in results.groupby('annotated_birads'):
        summary['by_annotated_class'][str(int(category))] = {
            'n': len(group), 'mean_predicted_enrichment': float(group['pred_enrichment_vs_uniform'].mean())}
    for correct, group in results.groupby('correct'):
        summary['by_correctness'][str(bool(correct)).lower()] = {
            'n': len(group), 'mean_predicted_enrichment': float(group['pred_enrichment_vs_uniform'].mean())}
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSummary:', json.dumps(summary, indent=2))
    print('Saved:', OUTPUT_DIR / 'case_results.csv')
    print('Saved:', OUTPUT_DIR / 'contact_sheet.png')
    print('Saved:', OUTPUT_DIR / 'summary.json')
    if failures:
        print('WARNING: Some examples failed validation. See failures.json.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--audit-only', action='store_true')
    action.add_argument('--run', action='store_true')
    ap.add_argument('--per-class', type=int, default=3, choices=(2, 3, 4),
                    help='Number of one-per-case examples per class (default 3)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    df = validate_manifest()
    if args.audit_only:
        print('AUDIT PASS: source manifests/checkpoint present. No inference performed.')
        return
    if OUTPUT_DIR.exists():
        raise FileExistsError(f'Results directory already exists: {OUTPUT_DIR}\n'
                              'Move/rename it explicitly before generating a new audit; '
                              'previous results will never be overwritten.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    create_audit(df, args.per_class, args.seed, device)


if __name__ == '__main__':
    main()
