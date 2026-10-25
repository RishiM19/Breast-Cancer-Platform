"""Evaluate the validation-selected Lumina BUS-BRA BI-RADS V1 model once on the locked test set.

Copy into C:/BC/Breast-Cancer-Platform/src alongside train_bra_birads_v1.py.

    python src/evaluate_bra_birads_v1_test.py --audit-only
    python src/evaluate_bra_birads_v1_test.py --evaluate

This script NEVER trains or adjusts model weights or thresholds. It refuses to
overwrite prior test outputs. Once the test is viewed, do not tune against it.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from train_bra_birads_v1 import (
    ROOT, CLASS_VALUES, TO_INDEX, IMAGE_SIZE, BRADataset, load_split, metrics,
)

CHECKPOINT = ROOT / 'models' / 'bra_birads_v1_best.pth'
METRICS_PATH = ROOT / 'data' / 'processed' / 'bra_birads_v1_test_metrics.json'
PREDICTIONS_PATH = ROOT / 'data' / 'processed' / 'bra_birads_v1_test_predictions.csv'


def check_split_integrity(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    # Including original case-level checks matters: images of one case can have different BI-RADS labels.
    for name, df in [('train', train), ('val', val), ('test', test)]:
        if df['__image_abs__'].duplicated().any():
            raise ValueError(f'{name} has repeated image paths')
        missing = [str(p) for p in df['__image_abs__'] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'{name}: missing {len(missing)} images; first: {missing[0]}')
        print(f'{name}: {len(df)} images, {df.case_id.nunique()} cases; BI-RADS counts '
              f'{[int((df.birads == x).sum()) for x in CLASS_VALUES]}')

    splits = {'train': train, 'val': val, 'test': test}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        left, right = splits[a], splits[b]
        common_cases = set(left.case_id) & set(right.case_id)
        common_paths = set(left['__image_abs__']) & set(right['__image_abs__'])
        if common_cases or common_paths:
            raise ValueError(f'{a}/{b} leakage: {len(common_cases)} shared cases, '
                             f'{len(common_paths)} shared paths')
        if 'image_sha256' in left.columns and 'image_sha256' in right.columns:
            def hashes(frame):
                return {v.strip().lower() for v in frame.image_sha256.dropna().astype(str)
                        if v.strip() and v.strip().lower() not in {'nan', 'none'}}
            if hashes(left) & hashes(right):
                raise ValueError(f'{a}/{b}: overlapping image hashes')
    print('PASS: case, path, and available image-hash leakage checks.')


def load_selected_model(device: torch.device) -> tuple[nn.Module, dict]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Missing V1 checkpoint: {CHECKPOINT}')
    # Local checkpoint created by the project's V1 trainer, with tensors and simple metadata.
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise ValueError('Unexpected V1 checkpoint format; expected dictionary with state_dict')
    if list(ckpt.get('class_values', [])) != list(CLASS_VALUES):
        raise ValueError(f'Wrong checkpoint label mapping: {ckpt.get("class_values")}')
    if ckpt.get('num_classes') != 4:
        raise ValueError('Checkpoint does not have four classes')
    if ckpt.get('image_size') != IMAGE_SIZE:
        raise ValueError('Checkpoint image size differs from V1 preprocessing')
    if ckpt.get('architecture') != 'torchvision.efficientnet_b0':
        raise ValueError('Unexpected model architecture')

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device).eval()
    return model, ckpt


def run_evaluation(test: pd.DataFrame, batch_size: int):
    for output in (METRICS_PATH, PREDICTIONS_PATH):
        if output.exists():
            raise FileExistsError(f'Existing test result; refusing overwrite: {output}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| GPU:', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none')
    model, ckpt = load_selected_model(device)
    print('Loaded selected V1 checkpoint: epoch', ckpt.get('best_epoch'),
          '| validation macro-F1', ckpt.get('best_val_macro_f1'))
    dataset = BRADataset(test, training=False)  # exact V1 inference preprocessing
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))

    probs = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    pred_prob = np.concatenate(probs, axis=0)
    if pred_prob.shape != (len(test), 4):
        raise AssertionError(f'Unexpected probability array shape: {pred_prob.shape}')
    target = test.birads.map(TO_INDEX).to_numpy()
    predicted = pred_prob.argmax(axis=1)
    report = metrics(target.tolist(), predicted.tolist())
    case_ids = test.case_id.astype(str).tolist()
    case_sizes = test.case_id.astype(str).value_counts().to_dict()
    case_equal_weights = np.asarray([1.0 / case_sizes[c] for c in case_ids])
    report['case_equal_weighted_image_accuracy'] = float(accuracy_score(
        target, predicted, sample_weight=case_equal_weights))
    report['case_equal_weighted_image_macro_f1'] = float(f1_score(
        target, predicted, labels=[0, 1, 2, 3], average='macro',
        sample_weight=case_equal_weights, zero_division=0))
    report['case_count'] = int(test.case_id.nunique())
    report['images'] = len(test)
    report['checkpoint'] = str(CHECKPOINT)
    report['checkpoint_epoch'] = ckpt.get('best_epoch')
    report['checkpoint_val_macro_f1'] = ckpt.get('best_val_macro_f1')
    report['class_values'] = list(CLASS_VALUES)
    report['evaluation_protocol'] = (
        'Single locked BUS-BRA test evaluation, model V1 selected from validation V1/V2 comparison. '
        'No training or threshold tuning on test. Image-level annotations; preserve mixed-label case 856.'
    )
    report['evaluated_at_utc'] = datetime.now(timezone.utc).isoformat()

    per_image = pd.DataFrame({
        'image_id': test.image_id.values if 'image_id' in test.columns else test.index.astype(str),
        'image_path': test.image_path.values,
        'case_id': case_ids,
        'actual_birads': test.birads.values,
        'predicted_birads': [CLASS_VALUES[i] for i in predicted],
        'correct': target == predicted,
        'prob_birads_2': pred_prob[:, 0],
        'prob_birads_3': pred_prob[:, 1],
        'prob_birads_4': pred_prob[:, 2],
        'prob_birads_5': pred_prob[:, 3],
    })
    if 'device' in test.columns:
        per_image['device'] = test.device.values

    # Outputs are new files only. Avoid overwriting checkpoints and previous experiments.
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n===== BI-RADS V1 LOCKED TEST REPORT =====')
    print(json.dumps({
        'checkpoint_epoch': report['checkpoint_epoch'],
        'images': report['images'], 'cases': report['case_count'],
        'accuracy': report['accuracy'], 'balanced_accuracy': report['balanced_accuracy'],
        'macro_f1': report['macro_f1'],
        'case_equal_weighted_image_macro_f1': report['case_equal_weighted_image_macro_f1'],
        'per_class': {k: {'precision': report['per_class'][k]['precision'],
                          'recall': report['per_class'][k]['recall'],
                          'f1': report['per_class'][k]['f1-score']}
                      for k in ('2', '3', '4', '5')},
        'confusion_matrix': report['confusion_matrix'],
    }, indent=2))
    print('Saved:', METRICS_PATH, '\nSaved:', PREDICTIONS_PATH)
    print('Research use only: this test split has now been examined; do not tune against it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--audit-only', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    parser.add_argument('--batch-size', type=int, default=12)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 256:
        parser.error('batch-size must be between 1 and 256')
    train, val, test = load_split('train'), load_split('val'), load_split('test')
    check_split_integrity(train, val, test)
    if args.audit_only:
        print('Audit complete. No model inference or result output performed.')
        return
    run_evaluation(test, args.batch_size)


if __name__ == '__main__':
    main()
