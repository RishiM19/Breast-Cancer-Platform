"""Train Lumina UCLM segmentation V2 without changing original checkpoints.
Run from project root: python src/train_uclm_segmentation_v2.py
Selection uses validation only; test manifest is checked for leakage, never evaluated.
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
if __package__:
    from .uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2, validate_patient_splits
else:
    from uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2, validate_patient_splits

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / 'data/processed/uclm_train.csv'
VAL = ROOT / 'data/processed/uclm_val.csv'
TEST = ROOT / 'data/processed/uclm_test.csv'
CHECKPOINT = ROOT / 'models/uclm_unet_v2_best.pth'
HISTORY = ROOT / 'data/processed/uclm_segmentation_v2_history.json'
SIZE, BATCH, MAX_EPOCHS = 256, 8, 60
PATIENCE, MIN_DELTA = 10, 0.001
SEED = 42
# Deliberately exclude the 13 HESN training pairs whose image/mask dimensions disagree.
TRAIN_EXCLUDED_PATIENTS = ("HESN",)


class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                   nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(a, b)
    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, a, skip, b):
        super().__init__()
        self.conv = DoubleConv(a + skip, b)
    def forward(self, x, skip):
        return self.conv(torch.cat([skip, F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)], dim=1))


class UNet(nn.Module):
    def __init__(self, base=32, num_classes=3):
        super().__init__()
        self.inc = DoubleConv(3, base)
        self.down1 = DownBlock(base, base * 2)
        self.down2 = DownBlock(base * 2, base * 4)
        self.down3 = DownBlock(base * 4, base * 8)
        self.down4 = DownBlock(base * 8, base * 16)
        self.up1 = UpBlock(base * 16, base * 8, base * 8)
        self.up2 = UpBlock(base * 8, base * 4, base * 4)
        self.up3 = UpBlock(base * 4, base * 2, base * 2)
        self.up4 = UpBlock(base * 2, base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)
    def forward(self, x):
        a = self.inc(x)
        b = self.down1(a)
        c = self.down2(b)
        d = self.down3(c)
        e = self.down4(d)
        return self.outc(self.up4(self.up3(self.up2(self.up1(e, d), c), b), a))


def lesion_dice_from_confusion(cm):
    values = []
    for c in (1, 2):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        values.append(2 * tp / denom if denom else float('nan'))
    valid = [v for v in values if np.isfinite(v)]
    return float(np.mean(valid)) if valid else 0.0


def composite_loss(logits, target, ce):
    """Weighted CE plus class-wise soft Dice with explicit absent-class FP penalty.

    Empty target classes retain a gradient via denominator; no `.item()` gating.
    """
    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    # Sum over batch and spatial dimensions; prediction on an absent class raises loss.
    overlap = (probabilities * one_hot).sum(dim=(0, 2, 3))
    mass = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * overlap[1:] + 1.0) / (mass[1:] + 1.0)
    return 0.5 * ce(logits, target) + 0.5 * (1 - dice).mean()


def run_epoch(model, loader, optimizer, criterion, scaler, device, training):
    model.train(training)
    cm = torch.zeros((3, 3), dtype=torch.int64, device=device)
    total_loss, n = 0., 0
    for images, masks in tqdm(loader, desc='Train' if training else 'Val', leave=False):
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(images)
                loss = composite_loss(logits, masks, criterion)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        with torch.no_grad():
            p = logits.argmax(dim=1)
            ids = (masks.reshape(-1) * 3 + p.reshape(-1)).long()
            cm += torch.bincount(ids, minlength=9).reshape(3, 3)
        total_loss += loss.detach().item() * images.size(0)
        n += images.size(0)
    return total_loss / n, lesion_dice_from_confusion(cm), cm.cpu()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if CHECKPOINT.exists():
        raise FileExistsError(f'V2 checkpoint exists: {CHECKPOINT}; move/version it before training again.')
    print('Patient split counts:', validate_patient_splits(TRAIN, VAL, TEST))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    train_ds = UCLMSegmentationDatasetV2(TRAIN, ROOT, size=SIZE, augment=True,
                                          exclude_patients=TRAIN_EXCLUDED_PATIENTS)
    val_ds = UCLMSegmentationDatasetV2(VAL, ROOT, size=SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    print('Training exclusions:', TRAIN_EXCLUDED_PATIENTS, 'excluded rows:', train_ds.excluded_rows)
    if train_ds.excluded_rows != 13:
        raise ValueError(f'Expected 13 HESN training rows; found {train_ds.excluded_rows}. Review manifest before training.')
    print('Images:', len(train_ds), 'train,', len(val_ds), 'validation')
    # Pixel-frequency weights from training masks only, without stochastic augmentation.
    plain_train = UCLMSegmentationDatasetV2(TRAIN, ROOT, size=SIZE, augment=False,
                                             exclude_patients=TRAIN_EXCLUDED_PATIENTS)
    counts = torch.zeros(3, dtype=torch.float64)
    for _, m in tqdm(DataLoader(plain_train, batch_size=BATCH, num_workers=0), desc='Class weights'):
        counts += torch.bincount(m.flatten(), minlength=3).double()
    weights = 1.0 / torch.sqrt((counts / counts.sum()).clamp_min(1e-12))
    weights = (weights / weights.mean()).float().to(device)
    print('Pixel counts:', counts.tolist(), 'CE weights:', weights.tolist())
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    history, best, no_improve = [], -1., 0
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_dice, _ = run_epoch(model, train_loader, optimizer, criterion, scaler, device, True)
        val_loss, val_dice, val_cm = run_epoch(model, val_loader, None, criterion, None, device, False)
        scheduler.step(val_dice)
        improved = val_dice > best + MIN_DELTA
        if improved:
            best = val_dice
            no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'best_val_lesion_dice': best, 'num_classes': 3, 'base_channels': 32,
                        'image_size': SIZE, 'class_names': ['Background', 'Benign', 'Malignant'],
                        'class_weights': weights.cpu().tolist(), 'model_version': 'UCLM_V2',
                        'excluded_training_patients': list(TRAIN_EXCLUDED_PATIENTS),
                        'train_images': len(train_ds)}, CHECKPOINT)
        else:
            no_improve += 1
        row = {'epoch': epoch, 'train_loss': train_loss, 'train_lesion_dice': train_dice,
               'val_loss': val_loss, 'val_lesion_dice': val_dice,
               'lr': optimizer.param_groups[0]['lr'], 'no_improve': no_improve,
               'val_confusion': val_cm.tolist()}
        history.append(row)
        HISTORY.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'Epoch {epoch:02d}: train Dice={train_dice:.4f}; val Dice={val_dice:.4f}; '
              f'best={best:.4f}; lr={row["lr"]:.2e}; patience={no_improve}/{PATIENCE}')
        if no_improve >= PATIENCE:
            print('Early stopping triggered.')
            break
    print('Saved best:', CHECKPOINT)
    print('History:', HISTORY)
    print('Locked test set has not been used for training or checkpoint selection.')


if __name__ == '__main__':
    main()
