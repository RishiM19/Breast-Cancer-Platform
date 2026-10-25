"""Lumina UCLM V3 binary U-Net: patient-grouped 5-fold exploratory CV.

Install into PROJECT_ROOT/src/train_uclm_v3_cv.py.
Requires the unchanged local files:
  src/train_uclm_segmentation_v2.py
  src/train_uclm_segmentation_v3_binary.py
  src/uclm_segmentation_dataset_v2.py
  data/processed/uclm_v3_cv_folds/fold_01_{train,val}.csv ... fold_05

Usage from project root:
  python src/train_uclm_v3_cv.py --audit-only
  python src/train_uclm_v3_cv.py --fold 1
  python src/train_uclm_v3_cv.py --fold 2   # etc.
  python src/train_uclm_v3_cv.py --summarize
  python src/train_uclm_v3_cv.py --all       # optional sequential 5-fold run

Safety: no test image/label reads; no checkpoint overwrites; all output paths
are CV-specific; original V1/V2/V3 models and reports are untouched.
CV is exploratory: V2 test findings informed the V3 design. Early stopping
chooses checkpoints on each fold's own validation patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_uclm_segmentation_v2 import UNet
from uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2
from train_uclm_segmentation_v3_binary import (
    SIZE, BATCH_SIZE, MAX_EPOCHS, PATIENCE, MIN_DELTA, SEED, BASE_CHANNELS,
    scan_split, run_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
FOLD_DIR = ROOT / 'data' / 'processed' / 'uclm_v3_cv_folds'
MODEL_DIR = ROOT / 'models' / 'uclm_v3_cv'
RESULT_DIR = ROOT / 'data' / 'processed' / 'uclm_v3_cv_results'
FOLDS = (1, 2, 3, 4, 5)


def patient_id(path: str) -> str:
    return Path(str(path)).stem.split('_')[0]


def read_manifest(path: Path) -> pd.DataFrame:
    """Load either original or CV fold CSVs without modifying them on disk.

    The fold preparation script deliberately drops helper columns such as
    patient_id and annotation. Infer patient IDs from the image filename,
    and cross-check IDs if an explicit patient_id column is available.
    """
    df = pd.read_csv(path)
    required = {'image_path', 'mask_path'}
    if not required.issubset(df.columns):
        raise ValueError(f'Columns missing from {path}: {sorted(required - set(df.columns))}')
    if df.empty or df['image_path'].isna().any() or df['mask_path'].isna().any():
        raise ValueError(f'Empty manifest or missing image/mask paths: {path}')
    if df.image_path.duplicated().any():
        raise ValueError(f'Duplicate images: {path}')

    # Use exactly the same patient-ID convention as prepare_uclm_v3_cv_folds.py.
    if not df['image_path'].astype(str).map(lambda p: '_' in Path(p).stem).all():
        raise ValueError(f'Cannot infer patient ID from one or more filenames: {path}')
    derived = df['image_path'].map(patient_id)
    if 'patient_id' in df.columns:
        if df['patient_id'].isna().any():
            raise ValueError(f'Missing patient ID in: {path}')
        inconsistent = df['patient_id'].astype(str) != derived
        if inconsistent.any():
            raise ValueError(f'Patient ID and filename mismatch: {path} row {int(np.flatnonzero(inconsistent)[0])}')
    else:
        df['patient_id'] = derived
    return df



class CVFoldDataset(UCLMSegmentationDatasetV2):
    """Load generated fold manifests without changing the original V2 loader.

    The fold builder intentionally emits only image_path and mask_path. The
    original V2 loader requires label and patient_id in its __init__, although
    its __getitem__ actually reads only the image/mask paths. Supply the same
    initialized attributes and inherit V2's tested synchronized transforms,
    strict color decoder, and image-mask geometry checks unchanged.
    """

    def __init__(self, csv_path, root_dir, size=256, augment=False):
        self.csv_path = Path(csv_path)
        self.root = Path(root_dir)
        self.size = int(size)
        self.augment = bool(augment)
        self.data = read_manifest(self.csv_path).reset_index(drop=True)
        self.excluded_patients = ()
        self.excluded_rows = 0


def audit_manifests() -> dict[int, tuple[Path, Path]]:
    """Validate patient identities and folds; never open test images/masks."""
    original_train = read_manifest(ROOT / 'data' / 'processed' / 'uclm_train.csv')
    original_val = read_manifest(ROOT / 'data' / 'processed' / 'uclm_val.csv')
    development = pd.concat([original_train, original_val], ignore_index=True)
    excluded = development.patient_id.eq('HESN')
    if int(excluded.sum()) != 13:
        raise ValueError(f'Expected 13 HESN exclusions, got {int(excluded.sum())}')
    development = development.loc[~excluded].copy()
    if development.image_path.duplicated().any():
        raise ValueError('Duplicate image paths in combined development manifests')
    dev_images = set(development.image_path.astype(str))
    dev_patients = set(development.patient_id.astype(str))
    if len(dev_images) != 565 or len(dev_patients) != 31:
        raise ValueError('Expected 565 development images from 31 patients')

    # Patient-ID-only test check. Test images and segmentation masks remain untouched.
    test = read_manifest(ROOT / 'data' / 'processed' / 'uclm_test.csv')
    overlapping_test_patients = dev_patients & set(test.patient_id.astype(str))
    if overlapping_test_patients:
        raise ValueError(f'Development/test patient overlap: {sorted(overlapping_test_patients)}')

    summary = pd.read_csv(FOLD_DIR / 'fold_summary.csv').set_index('fold')
    assignments = pd.read_csv(FOLD_DIR / 'patient_assignments.csv')
    if set(summary.index.astype(int)) != set(FOLDS) or len(summary) != len(FOLDS):
        raise ValueError('Expected fold_summary.csv with 5 distinct folds')
    if (assignments.patient_id.duplicated().any() or len(assignments) != 31 or
            set(assignments.patient_id.astype(str)) != dev_patients):
        raise ValueError('Patient assignments do not match the 31 development patients')
    assigned = dict(zip(assignments.patient_id.astype(str), assignments.validation_fold.astype(int)))
    if set(assigned.values()) != set(FOLDS):
        raise ValueError('Expected every fold in patient_assignments.csv')

    result = {}
    all_val_images = set()
    print('Five-fold audit:')
    for fold in FOLDS:
        tr_path = FOLD_DIR / f'fold_{fold:02d}_train.csv'
        va_path = FOLD_DIR / f'fold_{fold:02d}_val.csv'
        tr = read_manifest(tr_path)
        va = read_manifest(va_path)
        train_patients, val_patients = set(tr.patient_id.astype(str)), set(va.patient_id.astype(str))
        train_images, val_images = set(tr.image_path.astype(str)), set(va.image_path.astype(str))
        if train_patients & val_patients or train_images & val_images:
            raise ValueError(f'Patient/image leakage within fold {fold}')
        if train_images | val_images != dev_images:
            raise ValueError(f'Fold {fold} does not contain exactly the 565 development images')
        if all_val_images & val_images:
            raise ValueError(f'An image is validated in more than one fold: {fold}')
        all_val_images.update(val_images)
        expected_val_patients = {pid for pid, num in assigned.items() if num == fold}
        if val_patients != expected_val_patients:
            raise ValueError(f'Validation patient assignment mismatch in fold {fold}')
        expected = summary.loc[fold]
        for col, actual in {
            'train_patients': len(train_patients),
            'train_images': len(tr),
            'val_patients': len(val_patients),
            'val_images': len(va),
        }.items():
            if int(expected[col]) != actual:
                raise ValueError(f'Fold {fold} {col}: {actual} differs from summary {int(expected[col])}')
        if not train_patients.isdisjoint(set(test.patient_id.astype(str))) or not val_patients.isdisjoint(set(test.patient_id.astype(str))):
            raise ValueError(f'Test leakage in fold {fold}')
        print(f'  Fold {fold}: train {len(tr)} images/{len(train_patients)} patients | '
              f'validate {len(va)} images/{len(val_patients)} patients')
        result[fold] = (tr_path, va_path)
    if all_val_images != dev_images:
        raise ValueError('Not every development image has exactly one validation fold')
    print('PASS: 565 images, 31 patients; each patient validated once; no test patients included.')
    return result


def artifact_paths(fold: int) -> tuple[Path, Path, Path]:
    return (
        MODEL_DIR / f'fold_{fold:02d}_best.pth',
        RESULT_DIR / f'fold_{fold:02d}_history.json',
        RESULT_DIR / f'fold_{fold:02d}_best_val_metrics.json',
    )


def completion_marker(fold: int) -> Path:
    return RESULT_DIR / f'fold_{fold:02d}_COMPLETED.json'


def run_fold(fold: int, train_csv: Path, val_csv: Path) -> None:
    checkpoint_path, history_path, metrics_path = artifact_paths(fold)
    collisions = [p for p in (checkpoint_path, history_path, metrics_path, completion_marker(fold)) if p.exists()]
    if collisions:
        raise FileExistsError('Refusing to overwrite existing fold artifacts: ' + ', '.join(map(str, collisions)))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Fix VS Code/Conda interpreter before starting CV.')

    # Independent training from a fresh random initialization, same V3 hyperparameters.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda')
    print(f'\n===== V3 CV FOLD {fold}/5 =====')
    print('Device:', device, '| GPU:', torch.cuda.get_device_name(0))
    print('Training manifest:', train_csv)
    print('Validation manifest:', val_csv)

    train_ds = CVFoldDataset(train_csv, ROOT, size=SIZE, augment=True)
    plain_ds = CVFoldDataset(train_csv, ROOT, size=SIZE, augment=False)
    val_ds = CVFoldDataset(val_csv, ROOT, size=SIZE, augment=False)
    print(f'Images: train={len(train_ds)}, val={len(val_ds)}')
    train_summary = scan_split(plain_ds, f'Fold {fold} train audit')
    val_summary = scan_split(val_ds, f'Fold {fold} val audit')
    print('Training annotation summary:', train_summary)
    print('Validation annotation summary:', val_summary)
    if train_summary['lesion_pixels'] == 0 or val_summary['lesion_pixels'] == 0:
        raise ValueError(f'Fold {fold} does not have lesion pixels in train/validation data')

    # Balanced foreground/background pixel CE weights from TRAINING DATA ONLY.
    counts = torch.tensor([train_summary['background_pixels'], train_summary['lesion_pixels']], dtype=torch.float64)
    weights = 1.0 / torch.sqrt((counts / counts.sum()).clamp_min(1e-12))
    weights = (weights / weights.mean()).float().to(device)
    print('Pixel counts [background, lesion]:', counts.tolist())
    print('Cross-entropy weights:', weights.cpu().tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = UNet(base=BASE_CHANNELS, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler('cuda')

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {
        'train_sha256': hashlib.sha256(train_csv.read_bytes()).hexdigest(),
        'val_sha256': hashlib.sha256(val_csv.read_bytes()).hexdigest(),
    }
    best_dice, no_improve, history = -1.0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, scaler, device, True)
        val_metrics = run_epoch(model, val_loader, None, criterion, None, device, False)
        score = float(val_metrics['pooled_binary_dice'])
        scheduler.step(score)
        improved = score > best_dice + MIN_DELTA
        if improved:
            best_dice, no_improve = score, 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'model_version': 'UCLM_V3_BINARY_CV_EXPLORATORY',
                'cv_fold': fold,
                'num_classes': 2,
                'class_names': ['Background', 'Lesion'],
                'base_channels': BASE_CHANNELS,
                'image_size': SIZE,
                'best_val_binary_dice': best_dice,
                'best_val_metrics': val_metrics,
                'train_annotation_summary': train_summary,
                'val_annotation_summary': val_summary,
                'class_weights': weights.cpu().tolist(),
                'seed': SEED,
                'manifest_hashes': manifest_hashes,
                'selected_using': 'held-out patients of this fold; validation pooled binary Dice',
                'loss': '0.5*weighted_cross_entropy + 0.5*(1-foreground_soft_dice)',
            }
            torch.save(checkpoint, checkpoint_path)
            metrics_payload = {
                'fold': fold,
                'best_epoch': epoch,
                'best_val_binary_dice': best_dice,
                'train_images': len(train_ds),
                'val_images': len(val_ds),
                'val_metrics': val_metrics,
                'manifest_hashes': manifest_hashes,
                'research_status': 'EXPLORATORY CV: V3 design influenced by prior test inspection',
            }
            metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')
        else:
            no_improve += 1
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
            'best_val_binary_dice': best_dice,
            'no_improve': no_improve,
        })
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'Fold {fold} epoch {epoch:02d}: train Dice={train_metrics["pooled_binary_dice"]:.4f}; '
              f'val Dice={score:.4f}; best={best_dice:.4f}; '
              f'val misses={val_metrics["fully_missed_lesion_images"]}/{val_metrics["lesion_images"]}; '
              f'val false positives={val_metrics["normal_images_with_false_positive"]}/{val_metrics["normal_images"]}; '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}; patience={no_improve}/{PATIENCE}', flush=True)
        if no_improve >= PATIENCE:
            print(f'Fold {fold}: early stopping triggered.')
            break
    completion_marker(fold).write_text(json.dumps({
        'fold': fold,
        'epochs_completed': len(history),
        'best_val_binary_dice': best_dice,
        'early_stopping_triggered': no_improve >= PATIENCE,
    }, indent=2), encoding='utf-8')
    print('Best checkpoint:', checkpoint_path)
    print('Best validation metrics:', metrics_path)
    print('Training history:', history_path)
    print('Test images/masks were not loaded.')


def summarize() -> None:
    payloads = []
    for fold in FOLDS:
        path = artifact_paths(fold)[2]
        if not path.is_file() or not completion_marker(fold).is_file():
            print(f'Not all folds are complete: fold {fold} lacks metrics or completion marker')
            return
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['fold'] != fold:
            raise ValueError(f'Fold metric identity mismatch: {path}')
        payloads.append(data)
    dice = [d['best_val_binary_dice'] for d in payloads]
    tp = sum(d['val_metrics']['true_positive_pixels'] for d in payloads)
    pred = sum(d['val_metrics']['predicted_lesion_pixels'] for d in payloads)
    gt = sum(d['val_metrics']['ground_truth_lesion_pixels'] for d in payloads)
    n_lesions = sum(d['val_metrics']['lesion_images'] for d in payloads)
    weighted_img_dice = sum(d['val_metrics']['mean_image_dice_on_lesions'] * d['val_metrics']['lesion_images'] for d in payloads) / n_lesions
    summary = {
        'experiment': 'UCLM_V3_BINARY_5FOLD_CV_EXPLORATORY',
        'folds': 5,
        'total_development_validation_images': sum(d['val_images'] for d in payloads),
        'mean_of_fold_best_pooled_dice': float(np.mean(dice)),
        'std_of_fold_best_pooled_dice_sample': float(np.std(dice, ddof=1)),
        'pooled_out_of_fold_binary_dice': float(2 * tp / (pred + gt)) if pred + gt else None,
        'mean_lesion_image_dice_all_folds': float(weighted_img_dice),
        'lesion_images': n_lesions,
        'fully_missed_lesion_images': sum(d['val_metrics']['fully_missed_lesion_images'] for d in payloads),
        'normal_images_with_false_positive': sum(d['val_metrics']['normal_images_with_false_positive'] for d in payloads),
        'fold_best_scores': [
            {'fold': d['fold'], 'epoch': d['best_epoch'], 'val_images': d['val_images'],
             'pooled_binary_dice': d['best_val_binary_dice']}
            for d in payloads
        ],
        'interpretation': ('EXPLORATORY. V3 model choices were influenced by prior V2 test analysis; '
                           'epoch selection on validation patients introduces selection bias. '
                           'Independent untouched patient evaluation is needed for unbiased final claims.'),
    }
    out = RESULT_DIR / 'cv_summary.json'
    if out.exists():
        raise FileExistsError(f'Preserving existing CV summary, refusing to overwrite: {out}')
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved CV summary:', out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--audit-only', action='store_true', help='Validate CSV folds, no training')
    operation.add_argument('--fold', type=int, choices=FOLDS, help='Train exactly one fold (1..5)')
    operation.add_argument('--all', action='store_true', help='Train every missing fold sequentially')
    operation.add_argument('--summarize', action='store_true', help='Summarize all five completed folds')
    args = parser.parse_args()
    if args.summarize:
        summarize()
        return
    folds = audit_manifests()
    if args.audit_only:
        return
    if args.fold is not None:
        run_fold(args.fold, *folds[args.fold])
        return
    for fold in FOLDS:
        paths = (*artifact_paths(fold), completion_marker(fold))
        if all(not p.exists() for p in paths):
            run_fold(fold, *folds[fold])
        elif all(p.exists() for p in paths):
            print(f'Skipping fold {fold}: all artifacts and completion marker exist.')
        else:
            raise FileExistsError(f'Partial fold artifacts found for fold {fold}; inspect before proceeding.')
    summarize()


if __name__ == '__main__':
    main()
