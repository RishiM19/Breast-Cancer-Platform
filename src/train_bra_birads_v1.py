"""Lumina BUS-BRA BI-RADS 2/3/4/5 EfficientNet-B0 exploratory baseline.

Run from any directory:
    python src/train_bra_birads_v1.py --audit-only
    python src/train_bra_birads_v1.py --train

Audit examines metadata, checks file existence; train uses *only* bra_train.csv / bra_val.csv.
The existing pathology classifier checkpoints and test images are untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parents[1]
CLASS_VALUES = (2, 3, 4, 5)
TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
SEED = 2026
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resolve_image_path(raw: object) -> Path:
    text = str(raw).strip()
    if not text or text.lower() in ('nan', 'none'):
        raise ValueError(f'Empty image_path: {raw!r}')
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [ROOT / path, ROOT / 'data' / 'BUS_BRA' / 'BUSBRA' / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_split(name: str) -> pd.DataFrame:
    path = ROOT / 'data' / 'processed' / f'bra_{name}.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {'case_id', 'image_path', 'birads'}
    if not required <= set(df.columns):
        raise ValueError(f'{path}: missing columns {sorted(required - set(df.columns))}')
    if df[['case_id', 'image_path', 'birads']].isna().any().any():
        raise ValueError(f'{path}: null case_id, image_path or birads')
    labels = pd.to_numeric(df['birads'], errors='coerce')
    if labels.isna().any() or (~labels.isin(CLASS_VALUES)).any():
        raise ValueError(f'{path}: BI-RADS labels must be 2, 3, 4, or 5')
    if (labels != labels.astype(int)).any():
        raise ValueError(f'{path}: non-integer BI-RADS value')
    df['birads'] = labels.astype(int)
    df['case_id'] = df['case_id'].astype(str).str.strip()
    if (df['case_id'] == '').any():
        raise ValueError(f'{path}: blank case_id')
    df['__image_abs__'] = df['image_path'].map(resolve_image_path)
    return df


def audit(train: pd.DataFrame, val: pd.DataFrame) -> bool:
    print('=== Lumina BI-RADS v1: development-only audit ===')
    okay = True
    for name, df in [('train', train), ('val', val)]:
        print(f'\n{name.upper()}: {len(df)} images / {df.case_id.nunique()} cases')
        print('Class counts:', {k: int((df.birads == k).sum()) for k in CLASS_VALUES})
        print('Cases per class:', {k: df.loc[df.birads == k, 'case_id'].nunique() for k in CLASS_VALUES})
        if 'device' in df.columns:
            print('Device by BI-RADS:')
            print(pd.crosstab(df['device'], df['birads']).to_string())
        missing = df.loc[~df['__image_abs__'].map(Path.is_file), ['image_path', '__image_abs__']]
        print('Missing image files:', len(missing))
        if len(missing):
            print(missing.head(5).to_string(index=False))
            okay = False
        if df['__image_abs__'].duplicated().any():
            print('WARNING: repeated image paths inside', name)
        conflicting = df.groupby('case_id')['birads'].nunique()
        conflict_cases = conflicting[conflicting > 1].index.tolist()
        print('Cases with multiple image-level BI-RADS labels:', len(conflict_cases))
        if conflict_cases:
            print(df.loc[df.case_id.isin(conflict_cases), ['case_id', 'image_path', 'birads']].to_string(index=False))
    cases = set(train.case_id) & set(val.case_id)
    images = set(train['__image_abs__']) & set(val['__image_abs__'])
    print('\nShared train/validation cases:', len(cases))
    print('Shared train/validation paths:', len(images))
    if 'image_sha256' in train.columns and 'image_sha256' in val.columns:
        def hashes(df):
            return {v.strip().lower() for v in df['image_sha256'].dropna().astype(str) if v.strip()}
        overlaps = hashes(train) & hashes(val)
        print('Shared train/validation image hashes:', len(overlaps))
        if overlaps:
            okay = False
    if cases or images:
        okay = False
    print('\nAUDIT:', 'PASS' if okay else 'FAILED; do not train')
    return okay


class BRADataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool):
        self.rows = list(zip(df['__image_abs__'].tolist(), df['birads'].map(TO_INDEX).tolist()))
        # Pad, don't center-crop: keep the entire original ultrasound image.
        steps = []
        if training:
            steps += [transforms.RandomHorizontalFlip(p=0.5),
                      transforms.RandomRotation(degrees=5, fill=(0, 0, 0))]
        steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, y = self.rows[index]
        with Image.open(path) as source:
            rgb = source.convert('RGB')
            image = ImageOps.pad(rgb, (IMAGE_SIZE, IMAGE_SIZE),
                                 method=Image.Resampling.BILINEAR,
                                 color=(0, 0, 0), centering=(0.5, 0.5))
        return self.transform(image), int(y)


def metrics(true, pred):
    cm = confusion_matrix(true, pred, labels=list(range(4)))
    return {
        'accuracy': float(accuracy_score(true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(true, pred)),
        'macro_f1': float(f1_score(true, pred, labels=list(range(4)), average='macro', zero_division=0)),
        'per_class': classification_report(true, pred, labels=list(range(4)),
                                           target_names=[str(x) for x in CLASS_VALUES],
                                           zero_division=0, output_dict=True),
        'confusion_matrix': cm.tolist(),
    }


def one_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    n, total_loss = 0, 0.0
    ys, preds = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.detach().item()) * len(labels)
        n += len(labels)
        ys.extend(labels.detach().cpu().tolist())
        preds.extend(logits.detach().argmax(1).cpu().tolist())
    m = metrics(ys, preds)
    m['loss'] = total_loss / n
    return m


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, args):
    output_dir = ROOT / 'models'
    data_dir = ROOT / 'data' / 'processed'
    ckpt_path = output_dir / f'{args.run_name}_best.pth'
    hist_path = data_dir / f'{args.run_name}_history.json'
    val_path = data_dir / f'{args.run_name}_best_val_metrics.json'
    for file in (ckpt_path, hist_path, val_path):
        if file.exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment output: {file}. Use a new --run-name.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Verify VS Code is using conda bc_ai. No CPU fallback for this experiment.')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    # Different random seeds can cause variance even with these settings.
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0))
    print('Training images:', len(train_df), '| validation images:', len(val_df))
    train_data = BRADataset(train_df, training=True)
    val_data = BRADataset(val_df, training=False)
    tr_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    counts = np.bincount(train_df['birads'].map(TO_INDEX).to_numpy(), minlength=4)
    if np.any(counts == 0):
        raise ValueError(f'One or more training classes missing: {counts.tolist()}')
    weights = (1.0 / np.sqrt(counts.astype(float)))
    weights = weights / weights.mean()
    print('Class counts:', counts.tolist())
    print('Cross-entropy class weights:', [round(float(x), 4) for x in weights])
    if args.scratch:
        print('WARNING: using randomly initialized EfficientNet weights (--scratch).')
        initial_weights = None
    else:
        print('Loading ImageNet pretrained EfficientNet-B0 weights (first run may download them)...')
        initial_weights = EfficientNet_B0_Weights.DEFAULT
    try:
        model = efficientnet_b0(weights=initial_weights)
    except Exception as e:
        raise RuntimeError('Could not load pretrained weights. Check internet/cache; '
                           'or deliberately use --scratch for a different baseline.') from e
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                            patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    best = -1.0
    best_epoch, patience = -1, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_m = one_epoch(model, tr_loader, criterion, device, optimizer, scaler)
        val_m = one_epoch(model, va_loader, criterion, device)
        score = val_m['macro_f1']
        scheduler.step(score)
        lr = optimizer.param_groups[0]['lr']
        record = {'epoch': epoch, 'train': train_m, 'val': val_m, 'lr': lr}
        history.append(record)
        if score > best + 1e-4:
            best, best_epoch, patience = score, epoch, 0
            ckpt = {
                'state_dict': model.state_dict(),
                'architecture': 'torchvision.efficientnet_b0',
                'num_classes': 4,
                'class_values': list(CLASS_VALUES),
                'class_names': [f'BI-RADS {v}' for v in CLASS_VALUES],
                'best_epoch': best_epoch,
                'best_val_macro_f1': best,
                'best_val_metrics': val_m,
                'pretrained_imagenet': not args.scratch,
                'image_size': IMAGE_SIZE,
                'preprocessing': 'RGB, ImageOps.pad preserve-aspect 224x224, ImageNet normalization',
                'random_seed': SEED,
                'train_csv': 'data/processed/bra_train.csv',
                'val_csv': 'data/processed/bra_val.csv',
                'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                'research_only': True,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, ckpt_path)
            best_val = val_m
            print(f'Epoch {epoch:02}: train macro-F1={train_m["macro_f1"]:.4f}, '
                  f'val macro-F1={score:.4f}, val balanced acc={val_m["balanced_accuracy"]:.4f}, '
                  f'best={best:.4f} [checkpoint saved], lr={lr:.2e}', flush=True)
        else:
            patience += 1
            print(f'Epoch {epoch:02}: train macro-F1={train_m["macro_f1"]:.4f}, '
                  f'val macro-F1={score:.4f}, val balanced acc={val_m["balanced_accuracy"]:.4f}, '
                  f'best={best:.4f}, lr={lr:.2e}, patience={patience}/{args.patience}', flush=True)
        if patience >= args.patience:
            print('Early stopping triggered.')
            break
    data_dir.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps({
        'experiment': args.run_name, 'history': history,
        'metadata': {'best_epoch': best_epoch, 'best_val_macro_f1': best, 'seed': SEED,
                     'exploratory': True, 'test_untouched': True}
    }, indent=2), encoding='utf-8')
    val_path.write_text(json.dumps({'best_epoch': best_epoch, **best_val}, indent=2), encoding='utf-8')
    print('Best model:', ckpt_path)
    print('Best validation metrics:', val_path)
    print('Training history:', hist_path)
    print('TEST SPLIT NOT USED IN TRAINING OR MODEL SELECTION.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-only', action='store_true', help='Metadata, split and image-path audit only')
    group.add_argument('--train', action='store_true', help='Train and validate; does not touch test images')
    parser.add_argument('--run-name', default='bra_birads_v1', help='Unique experiment basename')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--max-epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--scratch', action='store_true', help='Disable pretrained weights (not recommended)')
    args = parser.parse_args()
    if not (1 <= args.batch_size <= 128 and 1 <= args.max_epochs <= 200 and 1 <= args.patience <= 100):
        parser.error('Invalid batch-size/max-epochs/patience')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('--run-name may only include letters, numbers, underscore and hyphen')
    train_df, val_df = load_split('train'), load_split('val')
    passed = audit(train_df, val_df)
    if not passed:
        sys.exit(2)
    if args.audit_only:
        print('Read-only audit complete. Review any conflicting cases before running --train.')
    else:
        print('\nWARNING: check any mixed-label cases and investigate image-level BI-RADS discrepancies before publishing results.')
        train(train_df, val_df, args)


if __name__ == '__main__':
    main()
