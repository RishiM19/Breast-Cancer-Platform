"""Lumina UCLM V3: exploratory binary lesion segmentation (not clinical use).

Run from project root:
    python src/train_uclm_segmentation_v3_binary.py

Dependencies, all already used by V2:
    src/train_uclm_segmentation_v2.py          (UNet architecture, imported only)
    src/uclm_segmentation_dataset_v2.py        (same validated preprocessing)

V3 differs from V2 by the output targets and loss/metric:
    0 = background (black)
    1 = any annotated lesion (green benign OR red malignant)

Patient HESN (13 mismatched image/mask pairs) is excluded from training.
Validation alone is used for checkpoint selection. Test data is not loaded.
Existing V1/V2 checkpoints, histories, and predictions remain untouched.
"""
from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_uclm_segmentation_v2 import UNet
from uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2, validate_patient_splits

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / 'data/processed/uclm_train.csv'
VAL = ROOT / 'data/processed/uclm_val.csv'
TEST = ROOT / 'data/processed/uclm_test.csv'  # split ID verification only; never load images
CHECKPOINT = ROOT / 'models/uclm_unet_v3_binary_best.pth'
HISTORY = ROOT / 'data/processed/uclm_segmentation_v3_binary_history.json'

SIZE = 256
BATCH_SIZE = 8
MAX_EPOCHS = 60
PATIENCE = 10
MIN_DELTA = 0.001
SEED = 42
BASE_CHANNELS = 32
TRAIN_EXCLUDED_PATIENTS = ('HESN',)


def binary_targets(multiclass_masks):
    """Combine benign (1) and malignant (2) into foreground (1)."""
    if torch.any((multiclass_masks < 0) | (multiclass_masks > 2)):
        raise ValueError('Unexpected mask class ID: expected only 0, 1, and 2')
    return (multiclass_masks != 0).long()


def scan_split(ds, desc):
    """Read-only mask audit using the exact same decoder/resizing as training."""
    background = lesion = empty = positives = 0
    for _, m in tqdm(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
                     desc=desc, leave=False):
        y = binary_targets(m)
        n_fg = y.sum(dim=(1, 2))
        positives += int((n_fg > 0).sum().item())
        empty += int((n_fg == 0).sum().item())
        lesion += int(n_fg.sum().item())
        background += int(y.numel() - n_fg.sum().item())
    return {'images': len(ds), 'background_images': empty, 'lesion_images': positives,
            'background_pixels': background, 'lesion_pixels': lesion}


def binary_composite_loss(logits, targets, cross_entropy):
    """Weighted cross-entropy + foreground soft Dice; no reward for empty masks."""
    foreground_probability = torch.softmax(logits.float(), dim=1)[:, 1]
    target_fg = targets.float()
    intersection = (foreground_probability * target_fg).sum()
    denominator = foreground_probability.sum() + target_fg.sum()
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * cross_entropy(logits, targets) + 0.5 * (1.0 - soft_dice)


def run_epoch(model, loader, optimizer, criterion, scaler, device, training):
    model.train(training)
    total_loss, count = 0.0, 0
    tp = pred_pixels = true_pixels = 0
    gt_lesion_images = missed_images = normal_images = false_positive_images = 0
    image_lesion_dices = []

    for images, multiclass_masks in tqdm(loader,
                                         desc='Train' if training else 'Val', leave=False):
        images = images.to(device, non_blocking=True)
        targets = binary_targets(multiclass_masks).to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                logits = model(images)
                loss = binary_composite_loss(logits, targets, criterion)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            predicted = logits.argmax(1)
            gt = targets == 1
            pred = predicted == 1
            tp += int((gt & pred).sum().item())
            pred_pixels += int(pred.sum().item())
            true_pixels += int(gt.sum().item())
            per_gt = gt.flatten(1).sum(1)
            per_pred = pred.flatten(1).sum(1)
            per_overlap = (gt & pred).flatten(1).sum(1)
            gt_lesion_images += int((per_gt > 0).sum().item())
            missed_images += int(((per_gt > 0) & (per_pred == 0)).sum().item())
            normal_images += int((per_gt == 0).sum().item())
            false_positive_images += int(((per_gt == 0) & (per_pred > 0)).sum().item())
            # On a lesion-containing image only: no artificial 1.0 for empty masks.
            present = per_gt > 0
            if bool(present.any().item()):
                dice = (2 * per_overlap[present].float() /
                        (per_gt[present] + per_pred[present]).float())
                image_lesion_dices.extend(dice.cpu().tolist())
        total_loss += float(loss.detach().item()) * images.size(0)
        count += images.size(0)

    denominator = pred_pixels + true_pixels
    pooled_dice = (2.0 * tp / denominator) if denominator else 0.0
    return {
        'loss': total_loss / count,
        'pooled_binary_dice': pooled_dice,
        'mean_image_dice_on_lesions': float(np.mean(image_lesion_dices)) if image_lesion_dices else None,
        'lesion_images': gt_lesion_images,
        'fully_missed_lesion_images': missed_images,
        'normal_images': normal_images,
        'normal_images_with_false_positive': false_positive_images,
        'true_positive_pixels': tp,
        'predicted_lesion_pixels': pred_pixels,
        'ground_truth_lesion_pixels': true_pixels,
    }


def main():
    if CHECKPOINT.exists() or HISTORY.exists():
        raise FileExistsError(
            f'An existing V3 artifact was found: {CHECKPOINT} or {HISTORY}. '
            'Preserve it and use a new experiment name rather than overwrite it.')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    split_counts = validate_patient_splits(TRAIN, VAL, TEST)
    print('Original patient-disjoint splits:', split_counts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    train_ds = UCLMSegmentationDatasetV2(
        TRAIN, ROOT, size=SIZE, augment=True, exclude_patients=TRAIN_EXCLUDED_PATIENTS)
    plain_train_ds = UCLMSegmentationDatasetV2(
        TRAIN, ROOT, size=SIZE, augment=False, exclude_patients=TRAIN_EXCLUDED_PATIENTS)
    val_ds = UCLMSegmentationDatasetV2(VAL, ROOT, size=SIZE, augment=False)

    print('Training exclusions:', TRAIN_EXCLUDED_PATIENTS, 'excluded rows:', train_ds.excluded_rows)
    if train_ds.excluded_rows != 13 or len(train_ds) != 455 or len(val_ds) != 110:
        raise ValueError('Unexpected split sizes/exclusions. Review dataset before proceeding.')
    print('Images:', len(train_ds), 'training,', len(val_ds), 'validation')
    train_summary = scan_split(plain_train_ds, 'Training annotation audit')
    val_summary = scan_split(val_ds, 'Validation annotation audit')
    print('Training annotation summary:', train_summary)
    print('Validation annotation summary:', val_summary)
    if (train_summary['lesion_images'], train_summary['background_images']) != (175, 280):
        raise ValueError('Training mask-count audit changed; investigate before training.')
    if (val_summary['lesion_images'], val_summary['background_images']) != (42, 68):
        raise ValueError('Validation mask-count audit changed; investigate before training.')

    counts = torch.tensor([train_summary['background_pixels'], train_summary['lesion_pixels']],
                          dtype=torch.float64)
    # Same inverse-sqrt-frequency CE weighting strategy as V2, now for two classes.
    weights = 1.0 / torch.sqrt((counts / counts.sum()).clamp_min(1e-12))
    weights = (weights / weights.mean()).float().to(device)
    print('Pixel counts [background, lesion]:', counts.tolist())
    print('Cross-entropy weights:', weights.cpu().tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')
    model = UNet(base=BASE_CHANNELS, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_dice, no_improve = -1.0, 0
    history = []
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, scaler, device, True)
        val_metrics = run_epoch(model, val_loader, None, criterion, None, device, False)
        score = val_metrics['pooled_binary_dice']
        scheduler.step(score)
        improved = score > best_dice + MIN_DELTA
        if improved:
            best_dice = score
            no_improve = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'model_version': 'UCLM_V3_BINARY_EXPLORATORY',
                'num_classes': 2,
                'class_names': ['Background', 'Lesion'],
                'base_channels': BASE_CHANNELS,
                'image_size': SIZE,
                'best_val_binary_dice': best_dice,
                'validation_mean_image_dice_on_lesions': val_metrics['mean_image_dice_on_lesions'],
                'excluded_training_patients': list(TRAIN_EXCLUDED_PATIENTS),
                'train_images': len(train_ds),
                'train_annotation_summary': train_summary,
                'val_annotation_summary': val_summary,
                'class_weights': weights.cpu().tolist(),
                'seed': SEED,
                'loss': '0.5*weighted_cross_entropy + 0.5*(1-foreground_soft_dice)',
                'selected_using': 'patient-disjoint validation pooled binary Dice only',
            }
            torch.save(checkpoint, CHECKPOINT)
        else:
            no_improve += 1
        record = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics,
                  'lr': optimizer.param_groups[0]['lr'], 'best_val_binary_dice': best_dice,
                  'no_improve': no_improve}
        history.append(record)
        HISTORY.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'Epoch {epoch:02d}: train Dice={train_metrics["pooled_binary_dice"]:.4f}; '
              f'val Dice={score:.4f}; best={best_dice:.4f}; '
              f'val misses={val_metrics["fully_missed_lesion_images"]}/42; '
              f'val false positives={val_metrics["normal_images_with_false_positive"]}/68; '
              f'lr={record["lr"]:.2e}; patience={no_improve}/{PATIENCE}')
        if no_improve >= PATIENCE:
            print('Early stopping triggered.')
            break
    print('Best checkpoint:', CHECKPOINT)
    print('History:', HISTORY)
    print('Locked test images were not loaded.')
    print('V3 is an exploratory experiment guided by prior V2 test analysis; '
          'independent external patient evaluation is needed for an unbiased final claim.')


if __name__ == '__main__':
    main()
