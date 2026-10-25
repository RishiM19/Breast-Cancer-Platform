"""Lumina UCLM V3 binary segmentation — exploratory held-out test evaluation.

Run from the project root AFTER finalizing your V3 checkpoint:
    python src/evaluate_uclm_segmentation_v3_binary.py

The same held-out test patients were previously inspected for V2 errors, which
informed V3. V3 results are therefore exploratory, not an unbiased final test.
A new untouched patient cohort is needed for an independent final estimate.

This evaluator NEVER modifies training data, model weights, or existing reports.
It refuses to overwrite a previous V3 evaluation.
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_uclm_segmentation_v2 import UNet
from uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2, validate_patient_splits
from train_uclm_segmentation_v3_binary import ROOT, binary_targets

BEST = ROOT / 'models/uclm_unet_v3_binary_best.pth'
TEST = ROOT / 'data/processed/uclm_test.csv'
OUT_JSON = ROOT / 'data/processed/uclm_segmentation_v3_binary_test_metrics.json'
OUT_CSV = ROOT / 'data/processed/uclm_segmentation_v3_binary_test_predictions.csv'
V2_JSON = ROOT / 'data/processed/uclm_segmentation_v2_test_metrics.json'


def safe_ratio(numer, denom):
    return float(numer / denom) if denom else None


def compute_v2_binary_comparator():
    """Derive pooled *binary* lesion Dice from existing 3-class V2 confusion."""
    if not V2_JSON.exists():
        return None
    with V2_JSON.open('r', encoding='utf-8') as f:
        old = json.load(f)
    cm = np.asarray(old.get('confusion_matrix'), dtype=np.int64)
    if cm.shape != (3, 3):
        print('Existing V2 confusion matrix is not 3x3; skipping comparison.')
        return None
    tp = int(cm[1:, 1:].sum())
    gt = int(cm[1:, :].sum())
    pred = int(cm[:, 1:].sum())
    return {
        'method': 'Merge benign and malignant as lesion from stored V2 test confusion matrix',
        'images': old.get('images'),
        'pooled_binary_lesion_dice': safe_ratio(2 * tp, gt + pred),
        'mean_binary_lesion_dice_on_lesion_images': old.get('mean_binary_lesion_dice_on_lesion_images'),
        'fully_missed_lesion_images': old.get('fully_missed_lesion_images'),
        'normal_images_with_false_positive': old.get('normal_images_with_false_positive'),
    }


def main():
    if OUT_JSON.exists() or OUT_CSV.exists():
        raise FileExistsError(
            'A V3 test report already exists. Preserve your previous evaluation; '
            'do not overwrite it. Paths: ' + str(OUT_JSON) + ', ' + str(OUT_CSV))
    if not BEST.is_file():
        raise FileNotFoundError(f'V3 checkpoint does not exist: {BEST}')

    counts = validate_patient_splits(ROOT / 'data/processed/uclm_train.csv',
                                     ROOT / 'data/processed/uclm_val.csv', TEST)
    print('Patient-disjoint split counts:', counts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    checkpoint = torch.load(BEST, map_location='cpu', weights_only=True)
    if checkpoint.get('num_classes') != 2 or checkpoint.get('model_version') != 'UCLM_V3_BINARY_EXPLORATORY':
        raise ValueError('Wrong checkpoint format: expected the V3 2-class exploratory checkpoint.')

    print('Best checkpoint epoch:', checkpoint.get('epoch'))
    print('Best validation pooled binary Dice:', checkpoint.get('best_val_binary_dice'))
    model = UNet(base=int(checkpoint.get('base_channels', 32)), num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    dataset = UCLMSegmentationDatasetV2(
        TEST, ROOT, size=int(checkpoint.get('image_size', 256)), augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    tp = fp = fn = tn = 0
    lesion_images = missed = normal_images = normal_fp = 0
    rows = []
    offset = 0
    with torch.inference_mode():
        for images, multiclass_masks in tqdm(loader, desc='V3 exploratory test'):
            target = binary_targets(multiclass_masks)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                pred = model(images.to(device)).argmax(1).cpu()
            if pred.shape != target.shape:
                raise RuntimeError(f'Prediction/mask size mismatch: {pred.shape} vs {target.shape}')

            gt_fg, pr_fg = target == 1, pred == 1
            tp += int((gt_fg & pr_fg).sum().item())
            fp += int((~gt_fg & pr_fg).sum().item())
            fn += int((gt_fg & ~pr_fg).sum().item())
            tn += int((~gt_fg & ~pr_fg).sum().item())

            for i in range(images.size(0)):
                gt = gt_fg[i].numpy()
                pr = pr_fg[i].numpy()
                g = int(gt.sum())
                p = int(pr.sum())
                overlap = int(np.count_nonzero(gt & pr))
                present = bool(g)
                if present:
                    lesion_images += 1
                    missed += int(p == 0)
                else:
                    normal_images += 1
                    normal_fp += int(p > 0)
                path = str(dataset.data.iloc[offset + i]['image_path'])
                patient = Path(path).stem.split('_')[0]
                rows.append({
                    'image_path': path,
                    'patient_id': patient,
                    'has_gt_lesion': present,
                    'gt_lesion_pixels': g,
                    'pred_lesion_pixels': p,
                    'true_positive_lesion_pixels': overlap,
                    'binary_lesion_dice_if_present': safe_ratio(2 * overlap, g + p) if present else None,
                    'lesion_recall_if_present': safe_ratio(overlap, g) if present else None,
                    'false_positive_on_normal': bool(not present and p > 0),
                })
            offset += images.size(0)

    lesion_dices = [r['binary_lesion_dice_if_present'] for r in rows if r['has_gt_lesion']]
    metrics = {
        'checkpoint': str(BEST),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'best_validation_pooled_binary_dice': checkpoint.get('best_val_binary_dice'),
        'test_interpretation': 'EXPLORATORY: V2 test cases influenced V3 development; seek untouched external patient cohort.',
        'images': len(dataset),
        'patient_count': len(set(r['patient_id'] for r in rows)),
        'pooled_binary_lesion_dice': safe_ratio(2 * tp, 2 * tp + fp + fn),
        'binary_lesion_iou': safe_ratio(tp, tp + fp + fn),
        'binary_lesion_precision': safe_ratio(tp, tp + fp),
        'binary_lesion_recall': safe_ratio(tp, tp + fn),
        'mean_binary_lesion_dice_on_lesion_images': float(np.mean(lesion_dices)) if lesion_dices else None,
        'lesion_images': lesion_images,
        'fully_missed_lesion_images': missed,
        'normal_images': normal_images,
        'normal_images_with_false_positive': normal_fp,
        'foreground_pixel_confusion': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
        'v2_exploratory_binary_comparison': compute_v2_binary_comparator(),
    }
    if len(rows) != len(dataset):
        raise RuntimeError(f'Unexpected number of per-image rows: {len(rows)} vs {len(dataset)}')
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))
    print('Saved:', OUT_JSON)
    print('Saved:', OUT_CSV)


if __name__ == '__main__':
    main()
