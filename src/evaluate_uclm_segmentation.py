"""
LUMINA — UCLM Multiclass U-Net Test Evaluation

Evaluates the LOCKED UCLM test set using:
    models/uclm_unet_best.pth

Outputs:
    data/processed/uclm_segmentation_test_metrics.json
    data/processed/uclm_segmentation_test_predictions.csv
    data/processed/uclm_segmentation_visuals/
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT_DIR / "data" / "processed" / "uclm_test.csv"
MODEL_PATH = ROOT_DIR / "models" / "uclm_unet_best.pth"

OUTPUT_JSON = ROOT_DIR / "data" / "processed" / "uclm_segmentation_test_metrics.json"
OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "uclm_segmentation_test_predictions.csv"
VISUAL_DIR = ROOT_DIR / "data" / "processed" / "uclm_segmentation_visuals"

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_CLASSES = 3
BASE_CHANNELS = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UCLMSegmentationDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        required = {"image_path", "mask_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def mask_to_classes(mask_rgb):
        class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

        green = (
            (mask_rgb[:, :, 0] == 0)
            & (mask_rgb[:, :, 1] == 255)
            & (mask_rgb[:, :, 2] == 0)
        )
        red = (
            (mask_rgb[:, :, 0] == 255)
            & (mask_rgb[:, :, 1] == 0)
            & (mask_rgb[:, :, 2] == 0)
        )

        class_mask[green] = 1
        class_mask[red] = 2
        return class_mask

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(ROOT_DIR / row["image_path"]).convert("RGB")
        mask = Image.open(ROOT_DIR / row["mask_path"]).convert("RGB")

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.uint8)
        class_mask = self.mask_to_classes(mask_np)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_np = (image_np - mean) / std

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(class_mask).long()

        return (
            image_tensor,
            mask_tensor,
            str(row["image_path"]),
            str(row["mask_path"]),
            str(row["label"]),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def confusion_from_batch(target, prediction):
    encoded = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return torch.bincount(
        encoded, minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(matrix):
    matrix = matrix.astype(np.float64)
    dice, iou, precision, recall = [], [], [], []

    for class_id in range(NUM_CLASSES):
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp

        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        p_den = tp + fp
        r_den = tp + fn

        dice.append(2 * tp / dice_den if dice_den > 0 else 1.0)
        iou.append(tp / iou_den if iou_den > 0 else 1.0)
        precision.append(tp / p_den if p_den > 0 else 1.0)
        recall.append(tp / r_den if r_den > 0 else 1.0)

    return np.array(dice), np.array(iou), np.array(precision), np.array(recall)


def make_overlay(image_rgb, mask):
    overlay = image_rgb.astype(np.float32).copy()

    benign = mask == 1
    malignant = mask == 2

    if benign.any():
        overlay[benign] = 0.55 * overlay[benign] + 0.45 * np.array([0, 255, 0], dtype=np.float32)

    if malignant.any():
        overlay[malignant] = 0.55 * overlay[malignant] + 0.45 * np.array([255, 0, 0], dtype=np.float32)

    return overlay.astype(np.uint8)


def save_visual(image_rgb, ground_truth, prediction, label, image_path, save_path):
    prediction_overlay = make_overlay(image_rgb, prediction)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original\n{label}")
    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=2)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=2)
    axes[2].set_title("Prediction")
    axes[3].imshow(prediction_overlay)
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 76)
    print("LUMINA — UCLM MULTICLASS U-NET TEST EVALUATION")
    print("=" * 76)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found:\n{MODEL_PATH}")

    dataset = UCLMSegmentationDataset(TEST_CSV)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Test images: {len(dataset)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print()
    print("Evaluating locked test set...")

    global_confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
        device=device,
    )

    rows = []
    visual_candidates = []

    with torch.no_grad():
        for images, masks, image_paths, mask_paths, labels in tqdm(
            loader, desc="Test", dynamic_ncols=True
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)

            predictions = torch.argmax(logits, dim=1)

            global_confusion += confusion_from_batch(
                masks, predictions
            ).to(device)

            for i in range(images.size(0)):
                gt = masks[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()

                class_dices = []
                for class_id in (1, 2):
                    gt_class = gt == class_id
                    pred_class = pred == class_id
                    intersection = np.logical_and(gt_class, pred_class).sum()
                    denom = gt_class.sum() + pred_class.sum()
                    class_dices.append(
                        1.0 if denom == 0 else 2.0 * intersection / denom
                    )

                lesion_dice = float(np.mean(class_dices))

                unique_pred = sorted(np.unique(pred).tolist())
                predicted_lesion_pixels = int(np.sum(pred != 0))
                ground_truth_lesion_pixels = int(np.sum(gt != 0))

                rows.append({
                    "image_path": image_paths[i],
                    "mask_path": mask_paths[i],
                    "label": labels[i],
                    "benign_dice": class_dices[0],
                    "malignant_dice": class_dices[1],
                    "lesion_dice": lesion_dice,
                    "predicted_classes": ",".join(map(str, unique_pred)),
                    "predicted_lesion_pixels": predicted_lesion_pixels,
                    "ground_truth_lesion_pixels": ground_truth_lesion_pixels,
                })

                image_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_display = (
                    (image_tensor * std + mean)
                    .clamp(0, 1)
                    .permute(1, 2, 0)
                    .numpy()
                    * 255
                ).astype(np.uint8)

                visual_candidates.append({
                    "score": lesion_dice,
                    "image_path": image_paths[i],
                    "label": labels[i],
                    "image": image_display,
                    "ground_truth": gt,
                    "prediction": pred,
                })

    confusion = global_confusion.cpu().numpy()
    dice, iou, precision, recall = metrics_from_confusion(confusion)

    mean_lesion_dice = float(np.mean(dice[[1, 2]]))
    mean_lesion_iou = float(np.mean(iou[[1, 2]]))
    pixel_accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))

    class_names = ["Background", "Benign", "Malignant"]

    metrics = {
        "checkpoint": str(MODEL_PATH),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_images": len(dataset),
        "pixel_accuracy": pixel_accuracy,
        "mean_dice_all_classes": float(np.mean(dice)),
        "mean_iou_all_classes": float(np.mean(iou)),
        "mean_lesion_dice": mean_lesion_dice,
        "mean_lesion_iou": mean_lesion_iou,
        "per_class": {},
        "confusion_matrix_rows_ground_truth_cols_prediction": confusion.astype(int).tolist(),
    }

    for class_id, class_name in enumerate(class_names):
        metrics["per_class"][class_name] = {
            "dice": float(dice[class_id]),
            "iou": float(iou[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
        }

    OUTPUT_JSON.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(rows).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    visual_candidates.sort(key=lambda x: x["score"])

    selected = []
    selected.extend(visual_candidates[:3])
    selected.extend(visual_candidates[-3:])

    normal_candidates = [
        item for item in visual_candidates if item["label"] == "Normal"
    ]

    if normal_candidates:
        normal_candidates.sort(
            key=lambda x: int(np.sum(x["prediction"] != 0)),
            reverse=True,
        )
        selected.append(normal_candidates[0])

    seen = set()
    for index, item in enumerate(selected, start=1):
        stem = Path(item["image_path"]).stem
        name = f"{index:02d}_{stem}_{item['label']}.png"
        if name in seen:
            continue
        seen.add(name)

        save_visual(
            item["image"],
            item["ground_truth"],
            item["prediction"],
            item["label"],
            item["image_path"],
            VISUAL_DIR / name,
        )

    print()
    print("=" * 76)
    print("LOCKED TEST RESULTS")
    print("=" * 76)
    print(f"Pixel accuracy:   {pixel_accuracy:.4f}")
    print(f"Mean Dice:        {np.mean(dice):.4f}")
    print(f"Mean IoU:         {np.mean(iou):.4f}")
    print(f"Lesion Dice:      {mean_lesion_dice:.4f}")
    print(f"Lesion IoU:       {mean_lesion_iou:.4f}")
    print()

    for i, name in enumerate(class_names):
        print(
            f"{name:10s} | "
            f"Dice={dice[i]:.4f} | "
            f"IoU={iou[i]:.4f} | "
            f"Precision={precision[i]:.4f} | "
            f"Recall={recall[i]:.4f}"
        )

    print()
    print("Confusion matrix (rows=ground truth, cols=prediction):")
    print(confusion.astype(int))
    print()
    print(f"Metrics JSON: {OUTPUT_JSON}")
    print(f"Per-image CSV: {OUTPUT_CSV}")
    print(f"Visuals: {VISUAL_DIR}")
    print()
    print("TEST EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
