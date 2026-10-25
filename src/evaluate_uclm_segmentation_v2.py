"""Locked-test evaluation of saved UCLM V2 checkpoint; run ONCE after selection.
Usage: python src/evaluate_uclm_segmentation_v2.py
Requires train_uclm_segmentation_v2.py and uclm_segmentation_dataset_v2.py.
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_uclm_segmentation_v2 import ROOT, UNet, lesion_dice_from_confusion
from uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2, validate_patient_splits

BEST = ROOT / 'models/uclm_unet_v2_best.pth'
TEST = ROOT / 'data/processed/uclm_test.csv'
OUT_JSON = ROOT / 'data/processed/uclm_segmentation_v2_test_metrics.json'
OUT_CSV = ROOT / 'data/processed/uclm_segmentation_v2_test_predictions.csv'


def main():
    print('Patient split counts:', validate_patient_splits(ROOT / 'data/processed/uclm_train.csv',
                                                         ROOT / 'data/processed/uclm_val.csv', TEST))
    if not BEST.is_file():
        raise FileNotFoundError(f'No V2 checkpoint yet: {BEST}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(BEST, map_location='cpu', weights_only=True)
    if checkpoint.get('num_classes') != 3:
        raise ValueError('Checkpoint class count mismatch')
    model = UNet(base=checkpoint.get('base_channels', 32)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    dataset = UCLMSegmentationDatasetV2(TEST, ROOT, size=checkpoint.get('image_size', 256), augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    rows = []
    normal_count = normal_fp_count = 0
    lesion_count = missed_count = 0
    with torch.inference_mode():
        offset = 0
        for images, masks in tqdm(loader, desc='Locked test'):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                pred = model(images.to(device)).argmax(1).cpu()
            ids = (masks.flatten() * 3 + pred.flatten()).long()
            confusion += torch.bincount(ids, minlength=9).reshape(3, 3)
            for i in range(images.size(0)):
                gt, p = masks[i].numpy(), pred[i].numpy()
                gt_lesion = gt != 0
                p_lesion = p != 0
                present = bool(gt_lesion.any())
                if present:
                    lesion_count += 1
                    missed_count += int(not p_lesion.any())
                else:
                    normal_count += 1
                    normal_fp_count += int(p_lesion.any())
                # Dice only when GT class is present; absent classes are not rewarded with 1.
                dices = {}
                for c, name in ((1, 'benign'), (2, 'malignant')):
                    gt_c, p_c = gt == c, p == c
                    denom = int(gt_c.sum() + p_c.sum())
                    dices[f'{name}_dice_if_present'] = float(2 * np.logical_and(gt_c, p_c).sum() / denom) if gt_c.any() and denom else None
                both_den = gt_lesion.sum() + p_lesion.sum()
                binary_dice = float(2 * np.logical_and(gt_lesion, p_lesion).sum() / both_den) if present and both_den else None
                path = str(dataset.data.iloc[offset + i]['image_path'])
                rows.append({'image_path': path, 'has_gt_lesion': present,
                             'gt_lesion_pixels': int(gt_lesion.sum()), 'pred_lesion_pixels': int(p_lesion.sum()),
                             'binary_lesion_dice_if_present': binary_dice, **dices})
            offset += images.size(0)
    cm = confusion.numpy().astype(np.float64)
    per_class = {}
    for c, name in enumerate(('Background', 'Benign', 'Malignant')):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        per_class[name] = {'dice': float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None,
                           'iou': float(tp/(tp+fp+fn)) if tp+fp+fn else None,
                           'precision': float(tp/(tp+fp)) if tp+fp else None,
                           'recall': float(tp/(tp+fn)) if tp+fn else None}
    present_dices = [x['binary_lesion_dice_if_present'] for x in rows if x['binary_lesion_dice_if_present'] is not None]
    metrics = {'checkpoint': str(BEST), 'checkpoint_epoch': checkpoint.get('epoch'),
               'images': len(dataset), 'pixel_accuracy': float(cm.trace()/cm.sum()),
               'global_mean_lesion_dice': lesion_dice_from_confusion(confusion),
               'per_class': per_class, 'confusion_matrix': confusion.tolist(),
               'mean_binary_lesion_dice_on_lesion_images': float(np.mean(present_dices)) if present_dices else None,
               'lesion_images': lesion_count, 'fully_missed_lesion_images': missed_count,
               'normal_images': normal_count, 'normal_images_with_false_positive': normal_fp_count}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JSON.exists() or OUT_CSV.exists():
        raise FileExistsError('V2 test evaluation output already exists; preserve it, do not overwrite.')
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(metrics, indent=2))
    print('Saved', OUT_JSON, OUT_CSV)


if __name__ == '__main__':
    main()
