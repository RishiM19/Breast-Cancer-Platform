"""Read-only error analysis for existing UCLM Segmentation V2 locked-test outputs.
Place in src/ and run from project root: python src/visualize_uclm_v2_errors.py
Uses the already-selected best checkpoint; does not update checkpoint or test metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from train_uclm_segmentation_v2 import ROOT, UNet
from uclm_segmentation_dataset_v2 import UCLMSegmentationDatasetV2, MEAN, STD

REPORT = ROOT / 'data/processed/uclm_segmentation_v2_test_predictions.csv'
TEST = ROOT / 'data/processed/uclm_test.csv'
BEST = ROOT / 'models/uclm_unet_v2_best.pth'
OUT = ROOT / 'data/processed/uclm_v2_error_analysis'


def choose_cases(df):
    lesion = df[df.has_gt_lesion.astype(str).str.lower().eq('true')].copy()
    normal = df[~df.has_gt_lesion.astype(str).str.lower().eq('true')].copy()
    missed = lesion[lesion.pred_lesion_pixels == 0].head(3)
    poor = lesion[(lesion.pred_lesion_pixels > 0) & (lesion.binary_lesion_dice_if_present < .25)].sort_values('binary_lesion_dice_if_present').head(3)
    good = lesion.sort_values('binary_lesion_dice_if_present', ascending=False).head(3)
    fp = normal[normal.pred_lesion_pixels > 0].sort_values('pred_lesion_pixels', ascending=False).head(3)
    selected = []
    seen = set()
    for title, section in [('Fully missed', missed), ('Poor overlap', poor), ('False positive', fp), ('Best overlap', good)]:
        for index, row in section.iterrows():
            if index not in seen:
                selected.append((title, int(index), row))
                seen.add(index)
    return selected


def color_mask(label_map):
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    out[label_map == 1] = [0, 255, 0]
    out[label_map == 2] = [255, 55, 55]
    return out


def main():
    for path in (REPORT, TEST, BEST):
        if not path.exists():
            raise FileNotFoundError(path)
    df = pd.read_csv(REPORT)
    ds = UCLMSegmentationDatasetV2(TEST, ROOT, augment=False)
    if len(df) != len(ds):
        raise ValueError(f'CSV rows ({len(df)}) do not match test manifest ({len(ds)})')
    if df.image_path.astype(str).str.replace('\\', '/', regex=False).tolist() != ds.data.image_path.astype(str).str.replace('\\', '/', regex=False).tolist():
        raise ValueError('Prediction rows and test manifest order differ. Stop to avoid mismatched visualizations.')
    ckpt = torch.load(BEST, map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(base=int(ckpt.get('base_channels', 32))).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    size = int(ckpt.get('image_size', 256))
    if ds.size != size:
        ds = UCLMSegmentationDatasetV2(TEST, ROOT, size=size, augment=False)
    choices = choose_cases(df)
    if not choices:
        raise ValueError('No cases selected')
    OUT.mkdir(parents=True, exist_ok=True)
    groupings = {}
    with torch.inference_mode():
        for group, i, row in choices:
            tensor, gt_t = ds[i]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                pred = model(tensor.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()
            gt = gt_t.numpy()
            image = ((tensor.numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
            # Overlays distinguish ground truth (green) and predicted lesions (magenta), irrespective of benign/malignant class.
            overlay = image.copy().astype(np.float32)
            gt_bin, pred_bin = gt > 0, pred > 0
            overlay[gt_bin] = .6 * overlay[gt_bin] + .4 * np.array([0, 255, 0])
            overlay[pred_bin] = .6 * overlay[pred_bin] + .4 * np.array([255, 0, 255])
            overlay = overlay.clip(0, 255).astype(np.uint8)
            groupings.setdefault(group, []).append((Path(row.image_path).name, image, gt, pred, overlay, row.binary_lesion_dice_if_present))
    for group, items in groupings.items():
        fig, axes = plt.subplots(len(items), 4, figsize=(16, max(3.4, len(items) * 3.4)), squeeze=False)
        for r, (name, image, gt, pred, overlay, dice) in enumerate(items):
            captions = ['Ultrasound', 'Ground truth (green=benign, red=malignant)', 'Predicted mask', 'Overlay: GT green / prediction magenta']
            images = [image, color_mask(gt), color_mask(pred), overlay]
            for c in range(4):
                axes[r, c].imshow(images[c]); axes[r, c].axis('off'); axes[r, c].set_title(captions[c], fontsize=9)
            axes[r, 0].set_ylabel(f'{name}\nDice: {dice:.3f}' if pd.notna(dice) else name, rotation=90, fontsize=9)
        fig.suptitle(f'UCLM V2: {group} — qualitative error analysis', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .96))
        dest = OUT / (group.lower().replace(' ', '_') + '.png')
        fig.savefig(dest, dpi=135, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', dest)
    print('Analysis images saved to:', OUT)
    print('The existing checkpoint and metric files were not modified.')


if __name__ == '__main__':
    main()
