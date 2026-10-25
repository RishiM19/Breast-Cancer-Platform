"""Lumina research-only Grad-CAM for the selected BUS-BRA BI-RADS V1 classifier.

From the repository root in PowerShell:
  python src/gradcam_bra_birads_v1.py --audit-only
  python src/gradcam_bra_birads_v1.py --run --index 0
  python src/gradcam_bra_birads_v1.py --run --index 0 --target-class 5

No training, network, external Grad-CAM package, or synthetic fallback.
The script does not inspect the test split. Visualizations are NOT lesion masks.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import efficientnet_b0
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'models' / 'bra_birads_v1_best.pth'
VAL_CSV = ROOT / 'data' / 'processed' / 'bra_val.csv'
OUTPUT_DIR = ROOT / 'data' / 'processed' / 'gradcam_bra_birads_v1'
LABELS = (2, 3, 4, 5)
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_path_from_manifest(raw: object) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    return ROOT / path


def load_validation_row(index: int) -> tuple[pd.Series, Path]:
    if not VAL_CSV.is_file():
        raise FileNotFoundError(f'Validation manifest not found: {VAL_CSV}')
    df = pd.read_csv(VAL_CSV)
    required = {'case_id', 'image_path', 'birads'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Validation manifest is missing columns: {sorted(missing)}')
    if not 0 <= index < len(df):
        raise IndexError(f'--index {index} invalid: valid range is 0 to {len(df)-1}')
    row = df.iloc[index]
    path = image_path_from_manifest(row['image_path'])
    if not path.is_file():
        raise FileNotFoundError(f'Validation image not found: {path}')
    return row, path


def preprocess(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    # Matches BRADataset(training=False) in train_bra_birads_v1.py exactly.
    with Image.open(image_path) as source:
        rgb = source.convert('RGB')
        padded = ImageOps.pad(
            rgb, (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0), centering=(0.5, 0.5),
        )
    original = np.array(padded, dtype=np.uint8)
    normalized = TF.normalize(TF.to_tensor(padded), MEAN, STD)
    return original, normalized.unsqueeze(0)


def load_model(device: torch.device) -> tuple[torch.nn.Module, dict]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT}')
    # Only load the checkpoint created locally by our V1 training script.
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise ValueError('Expected the V1 checkpoint dict containing state_dict.')
    if tuple(ckpt.get('class_values', ())) != LABELS:
        raise ValueError(f'Class mapping mismatch: {ckpt.get("class_values")} vs {LABELS}')
    if ckpt.get('num_classes') != 4:
        raise ValueError('Expected a four-output BI-RADS model.')
    if ckpt.get('architecture') != 'torchvision.efficientnet_b0':
        raise ValueError(f'Unexpected checkpoint architecture: {ckpt.get("architecture")}')
    if ckpt.get('image_size', IMAGE_SIZE) != IMAGE_SIZE:
        raise ValueError('Checkpoint image size does not match evaluation preprocessing.')
    # weights=None prevents an unnecessary download. V1 weights come from the checkpoint.
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device).eval()
    return model, ckpt


def get_gradcam(
    model: torch.nn.Module, batch: torch.Tensor, class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 224x224 normalized Grad-CAM and four softmax scores.

    Hook to features[5]: a spatial feature map with higher spatial resolution before the final EfficientNet blocks.
    Uses true derivatives of the selected class logit; no synthetic fallback.
    """
    state: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, activation: torch.Tensor):
        state['activation'] = activation.detach()
        activation.register_hook(lambda grad: state.__setitem__('gradient', grad.detach()))

    handle = model.features[5].register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(batch)
            probs = logits.softmax(dim=1).detach().cpu().numpy()[0]
            logits[0, class_index].backward()
        if 'activation' not in state or 'gradient' not in state:
            raise RuntimeError('Gradient hook did not capture activations and gradients.')
        act = state['activation']
        grad = state['gradient']
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)
        cam_np = cam[0, 0].detach().cpu().float().numpy()
        if not np.isfinite(cam_np).all():
            raise RuntimeError('Grad-CAM returned non-finite values.')
        lo, hi = float(cam_np.min()), float(cam_np.max())
        if hi - lo < 1e-12:
            raise RuntimeError('Grad-CAM is constant/zero. No heatmap produced; do not fabricate one.')
        cam_np = np.clip((cam_np - lo) / (hi - lo), 0, 1)
        return cam_np, probs
    finally:
        handle.remove()


def save_visualization(
    image: np.ndarray,
    heat: np.ndarray,
    probs: np.ndarray,
    row: pd.Series,
    index: int,
    target_class: int,
    checkpoint: dict,
    image_path: Path,
) -> Path:
    predicted_class = LABELS[int(np.argmax(probs))]
    gt_class = int(row['birads'])
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', image_path.stem)
    prefix = f'val_{index:03d}_{stem}_target_{target_class}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f'{prefix}.png'
    npy = OUTPUT_DIR / f'{prefix}_cam.npy'
    meta_path = OUTPUT_DIR / f'{prefix}_metadata.json'
    if any(p.exists() for p in (png, npy, meta_path)):
        raise FileExistsError(f'Output already exists for {prefix}; choose another --index or --target-class.')

    img01 = image.astype(np.float32) / 255.0
    colormap = plt.get_cmap('jet')(heat)[..., :3]
    overlay = np.clip(0.65 * img01 + 0.35 * colormap, 0, 1)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    axs[0].imshow(image)
    axs[0].set_title('V1 model input (padded to 224x224)')
    axs[1].imshow(heat, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title(f'Grad-CAM for BI-RADS {target_class}')
    axs[2].imshow(overlay)
    axs[2].set_title('Gradient attribution overlay')
    for ax in axs:
        ax.axis('off')
    fig.suptitle(
        f'Case {row["case_id"]}  |  Annotated: {gt_class}  |  '
        f'Predicted: {predicted_class} ({probs[LABELS.index(predicted_class)]:.1%})',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(png, dpi=155, bbox_inches='tight')
    plt.close(fig)
    np.save(npy, heat.astype(np.float32))
    metadata = {
        'research_only': True,
        'explanation_type': 'Gradient-weighted Class Activation Mapping (Grad-CAM)',
        'target_layer': 'efficientnet_b0.features[5]',
        'heatmap_is_not_segmentation': True,
        'synthetic_fallback_used': False,
        'split': 'validation',
        'validation_row_index': index,
        'image_path': str(image_path),
        'case_id': str(row['case_id']),
        'annotated_birads': gt_class,
        'predicted_birads': predicted_class,
        'target_birads': target_class,
        'softmax_scores': {str(label): float(probs[j]) for j, label in enumerate(LABELS)},
        'checkpoint': str(CHECKPOINT),
        'checkpoint_epoch': checkpoint.get('best_epoch'),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved visualization:', png)
    print('Saved normalized activation map:', npy)
    print('Saved metadata:', meta_path)
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--audit-only', action='store_true')
    action.add_argument('--run', action='store_true')
    ap.add_argument('--index', type=int, default=0,
                    help='Zero-based row index into bra_val.csv (default: 0)')
    ap.add_argument('--target-class', type=int, choices=LABELS, default=None,
                    help='BI-RADS 2/3/4/5; default: predicted class')
    args = ap.parse_args()

    row, image_path = load_validation_row(args.index)
    print('Checkpoint:', CHECKPOINT, '| exists:', CHECKPOINT.is_file())
    print('Validation example:', image_path)
    print('Annotated BI-RADS:', row['birads'], '| Case:', row['case_id'])
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if args.audit_only:
        print('PASS: files found. No checkpoint loading, gradients or output generated.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    model, ckpt = load_model(device)
    image, batch = preprocess(image_path)
    batch = batch.to(device)
    # First prediction, without disrupting the gradient-enabled Grad-CAM pass.
    with torch.no_grad():
        logits = model(batch)
        predicted_index = int(logits.argmax(dim=1).item())
    target_index = predicted_index if args.target_class is None else LABELS.index(args.target_class)
    heat, probs = get_gradcam(model, batch, target_index)
    predicted = LABELS[int(np.argmax(probs))]
    target = LABELS[target_index]
    print('Prediction:', f'BI-RADS {predicted}', '| CAM target:', f'BI-RADS {target}')
    print('Scores:', {str(v): round(float(probs[i]), 4) for i, v in enumerate(LABELS)})
    save_visualization(image, heat, probs, row, args.index, target, ckpt, image_path)
    print('These are model-attribution maps, not lesion outlines or clinical explanations.')


if __name__ == '__main__':
    main()
