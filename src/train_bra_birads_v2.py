"""Lumina BUS-BRA BI-RADS V2: staged EfficientNet-B0 exploratory experiment.

Install alongside src/train_bra_birads_v1.py; this script uses exactly the
same case-disjoint train/validation manifests, labels, image resolution and
image preprocessing as V1. Does not load test split.

    python src/train_bra_birads_v2.py --audit-only
    python src/train_bra_birads_v2.py --train

V2 differences: head-only warmup, fine-tune last 2 feature blocks with lower
backbone learning rate, modest intensity augmentation and stronger regularization.
All checkpoints/results are V2-specific. For research only, not clinical use.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Import the proven V1 dataset/image preprocessing and metadata audit.
from train_bra_birads_v1 import (
    ROOT, CLASS_VALUES, TO_INDEX, SEED, BRADataset, load_split, audit,
)

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class BRADatasetV2(BRADataset):
    def __init__(self, df: pd.DataFrame, training: bool):
        super().__init__(df, training=False)
        if training:
            # Horizontal flip/rotation retained from V1. Moderate brightness
            # and contrast jitter added only to training; no hue or saturation.
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5, fill=(0, 0, 0)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])


def v2_metrics(y_true: list[int], y_pred: list[int], case_ids: list[str]):
    labels = list(range(4))
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'per_class': classification_report(
            y_true, y_pred, labels=labels, target_names=[str(x) for x in CLASS_VALUES],
            zero_division=0, output_dict=True,
        ),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    # One weight per case, split equally among its images. Unlike majority-vote
    # collapse, this correctly retains image-specific labels for mixed cases.
    cases = pd.Series(case_ids, dtype='str')
    counts = cases.value_counts().to_dict()
    case_weights = np.array([1.0 / counts[case] for case in case_ids], dtype=float)
    out['case_equal_weighted_image_accuracy'] = float(
        accuracy_score(y_true, y_pred, sample_weight=case_weights)
    )
    out['case_equal_weighted_image_macro_f1'] = float(f1_score(
        y_true, y_pred, labels=labels, average='macro',
        sample_weight=case_weights, zero_division=0,
    ))
    out['case_count'] = len(counts)
    return out


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None,
              freeze_batchnorm=True, case_ids=None, return_predictions=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_batchnorm:
        # Keep BN running statistics frozen for small-batch medical images.
        model.features.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    probs_chunks = []
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
        bs = labels.size(0)
        total_loss += float(loss.detach().item()) * bs
        n += bs
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(1).cpu().tolist())
        if return_predictions:
            probs_chunks.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
    if case_ids is None:
        case_ids = [str(i) for i in range(len(y_true))]
    if len(case_ids) != len(y_true):
        raise ValueError('case_id order does not match loader output')
    m = v2_metrics(y_true, y_pred, list(case_ids))
    m['loss'] = total_loss / n
    if return_predictions:
        return m, y_pred, np.concatenate(probs_chunks, axis=0)
    return m


def create_model(*, scratch: bool = False):
    if scratch:
        print('WARNING: randomly initialized backbone (--scratch). Different experiment!')
        weights = None
    else:
        print('Loading pretrained EfficientNet-B0 (using cached V1 weights if available)...', flush=True)
        weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[0] = nn.Dropout(p=0.40)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    return model


def set_trainable(model, stage: str):
    for p in model.features.parameters():
        p.requires_grad = False
    if stage == 'fine_tune':
        # Only final two EfficientNet feature groups, not entire backbone.
        for block in list(model.features.children())[-2:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def save_json_no_overwrite(path: Path, obj):
    if path.exists():
        raise FileExistsError(f'Existing output: {path}')
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, args):
    ckpt_path = ROOT / 'models' / f'{args.run_name}_best.pth'
    history_path = ROOT / 'data' / 'processed' / f'{args.run_name}_history.json'
    val_path = ROOT / 'data' / 'processed' / f'{args.run_name}_best_val_metrics.json'
    pred_path = ROOT / 'data' / 'processed' / f'{args.run_name}_best_val_predictions.csv'
    for p in (ckpt_path, history_path, val_path, pred_path):
        if p.exists():
            raise FileExistsError(f'Refusing to overwrite {p}; set a new --run-name.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Verify VS Code uses Python from conda bc_ai.')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    print('Device: cuda |', torch.cuda.get_device_name(0), flush=True)
    print('Training:', len(train_df), 'images /', train_df.case_id.nunique(), 'cases', flush=True)
    print('Validation:', len(val_df), 'images /', val_df.case_id.nunique(), 'cases', flush=True)
    train_data = BRADatasetV2(train_df, True)
    val_data = BRADatasetV2(val_df, False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    counts = np.bincount(train_df.birads.map(TO_INDEX).to_numpy(), minlength=4)
    if np.any(counts == 0):
        raise ValueError('Missing class in training data')
    weights = 1.0 / np.sqrt(counts.astype(float))
    weights /= weights.mean()
    print('Class counts:', counts.tolist())
    print('CE weights:', np.round(weights, 4).tolist(), flush=True)
    model = create_model(scratch=args.scratch).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    scaler = torch.amp.GradScaler('cuda')
    best = -1.0
    best_epoch = None
    best_metrics = None
    best_predictions = None
    history = []
    global_epoch = 0
    val_cases = val_df.case_id.astype(str).tolist()
    # Head warmup is always completed. Patience only applies within fine-tuning
    # to avoid stopping before the pretrained features are allowed to adapt.
    stages = [('head_warmup', args.head_epochs), ('fine_tune', args.fine_epochs)]
    for stage, epochs in stages:
        set_trainable(model, stage)
        if stage == 'head_warmup':
            optimizer = torch.optim.AdamW(model.classifier.parameters(),
                                          lr=args.head_lr, weight_decay=args.weight_decay)
        else:
            feature_params = [p for p in model.features.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([
                {'params': feature_params, 'lr': args.backbone_lr},
                {'params': list(model.classifier.parameters()), 'lr': args.finetune_head_lr},
            ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7,
        )
        patience = 0
        print(f'\n--- Stage: {stage} | up to {epochs} epochs ---', flush=True)
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_m = run_epoch(model, train_loader, criterion, device,
                                optimizer=optimizer, scaler=scaler)
            val_m, val_pred, val_prob = run_epoch(
                model, val_loader, criterion, device, case_ids=val_cases,
                return_predictions=True,
            )
            score = val_m['macro_f1']
            scheduler.step(score)
            lrs = [round(float(g['lr']), 9) for g in optimizer.param_groups]
            record = {
                'epoch': global_epoch, 'stage': stage, 'stage_epoch': stage_epoch,
                'train': train_m, 'val': val_m, 'learning_rates': lrs,
            }
            history.append(record)
            if score > best + args.min_delta:
                best = score
                best_epoch = global_epoch
                best_metrics = copy.deepcopy(val_m)
                best_predictions = (list(val_pred), val_prob.copy())
                patience = 0
                ckpt = {
                    'state_dict': model.state_dict(),
                    'architecture': 'torchvision.efficientnet_b0',
                    'experiment': args.run_name,
                    'num_classes': 4,
                    'class_values': list(CLASS_VALUES),
                    'class_names': [f'BI-RADS {v}' for v in CLASS_VALUES],
                    'best_epoch': best_epoch,
                    'best_stage': stage,
                    'best_val_macro_f1': best,
                    'best_val_metrics': best_metrics,
                    'image_size': IMAGE_SIZE,
                    'preprocessing': 'RGB, ImageOps.pad preserve-aspect 224x224, ImageNet normalization',
                    'pretrained_imagenet': not args.scratch,
                    'seed': SEED,
                    'train_csv': 'data/processed/bra_train.csv',
                    'val_csv': 'data/processed/bra_val.csv',
                    'config': vars(args),
                    'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                    'research_only': True,
                }
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, ckpt_path)
                tag = ' [saved]'
            else:
                patience += 1
                tag = f' | patience={patience}/{args.patience}'
            print(f'Epoch {global_epoch:02d} [{stage} {stage_epoch:02d}]: '
                  f'train F1={train_m["macro_f1"]:.4f}; '
                  f'val F1={score:.4f}; '
                  f'val balanced acc={val_m["balanced_accuracy"]:.4f}; '
                  f'case-equal F1={val_m["case_equal_weighted_image_macro_f1"]:.4f}; '
                  f'best={best:.4f}; lrs={lrs}{tag}', flush=True)
            if stage == 'fine_tune' and patience >= args.patience:
                print('Fine-tuning early stopping triggered.', flush=True)
                break
    history_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_no_overwrite(history_path, {
        'experiment': args.run_name,
        'config': vars(args),
        'best_epoch': best_epoch,
        'best_val_macro_f1': best,
        'history': history,
        'research_only': True,
        'test_set_used': False,
    })
    save_json_no_overwrite(val_path, {'best_epoch': best_epoch, **best_metrics})
    saved_pred, saved_prob = best_predictions
    pred_df = val_df[['case_id', 'image_id', 'image_path', 'birads']].copy() if 'image_id' in val_df.columns else val_df[['case_id', 'image_path', 'birads']].copy()
    pred_df['predicted_birads'] = [CLASS_VALUES[x] for x in saved_pred]
    for i, value in enumerate(CLASS_VALUES):
        pred_df[f'prob_birads_{value}'] = saved_prob[:, i]
    if pred_path.exists():
        raise FileExistsError(pred_path)
    pred_df.to_csv(pred_path, index=False)
    print('\nBest checkpoint:', ckpt_path)
    print('Best epoch:', best_epoch, '| best validation macro-F1:', round(best, 5))
    print('History:', history_path)
    print('Validation metrics:', val_path)
    print('Validation predictions:', pred_path)
    print('TEST SPLIT NOT LOADED OR USED FOR MODEL SELECTION.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-only', action='store_true')
    group.add_argument('--train', action='store_true')
    parser.add_argument('--run-name', default='bra_birads_v2')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--head-epochs', type=int, default=4)
    parser.add_argument('--fine-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=9)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--finetune-head-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--scratch', action='store_true', help='Not comparable to ImageNet pretrained baseline')
    args = parser.parse_args()
    if not (1 <= args.batch_size <= 64 and 1 <= args.head_epochs <= 50 and
            1 <= args.fine_epochs <= 100 and 1 <= args.patience <= 100):
        parser.error('Invalid batch-size, epoch count or patience')
    if not args.run_name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in args.run_name):
        parser.error('Invalid --run-name')
    if any(getattr(args, x) <= 0 for x in ('head_lr', 'backbone_lr', 'finetune_head_lr', 'weight_decay')):
        parser.error('LR and weight_decay must be positive')
    train_df = load_split('train')
    val_df = load_split('val')
    if not audit(train_df, val_df):
        sys.exit(2)
    conflicts = val_df.groupby('case_id')['birads'].nunique()
    mixed_ids = conflicts[conflicts > 1].index.tolist()
    print('\nValidation cases with distinct image-level BI-RADS annotations:', mixed_ids)
    print('Image labels preserved; case-equal metrics do NOT majority-vote case 856.')
    if args.audit_only:
        print('PASS: audit-only; no model loaded or trained.')
        return
    train(train_df, val_df, args)


if __name__ == '__main__':
    main()
