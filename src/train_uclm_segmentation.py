"""
LUMINA — UCLM Multiclass U-Net Segmentation Training (FULL RUN)

Task:
    0 = Background
    1 = Benign lesion
    2 = Malignant lesion

Training behavior:
    - Runs all MAX_EPOCHS. No early stopping.
    - Shows tqdm progress bars for train and validation batches.
    - Saves the best validation checkpoint based on mean lesion Dice.
    - Test set is NOT used during training or model selection.

Files:
    data/processed/uclm_train.csv
    data/processed/uclm_val.csv
    data/processed/uclm_test.csv

Outputs:
    models/uclm_unet_best.pth
    data/processed/uclm_segmentation_history.json
"""

from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT_DIR / "data" / "processed" / "uclm_train.csv"
VAL_CSV = ROOT_DIR / "data" / "processed" / "uclm_val.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "uclm_unet_best.pth"
FINAL_MODEL_PATH = MODEL_DIR / "uclm_unet_final.pth"
HISTORY_PATH = ROOT_DIR / "data" / "processed" / "uclm_segmentation_history.json"

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0

# FULL FIXED TRAINING RUN
MAX_EPOCHS = 40

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 3
BASE_CHANNELS = 32

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class UCLMSegmentationDataset(Dataset):
    """
    UCLM ultrasound segmentation dataset.

    RGB mask encoding:
        black = 0 Background
        green = 1 Benign
        red   = 2 Malignant
    """

    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        required = {"image_path", "mask_path", "label"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def mask_to_classes(mask_rgb: np.ndarray) -> np.ndarray:
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

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = ROOT_DIR / row["image_path"]
        mask_path = ROOT_DIR / row["mask_path"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        # CRITICAL: nearest-neighbor for segmentation masks.
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.uint8)

        class_mask = self.mask_to_classes(mask_np)

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )
        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image_np = (image_np - mean) / std

        image_tensor = torch.from_numpy(
            image_np.transpose(2, 0, 1)
        ).float()

        mask_tensor = torch.from_numpy(
            class_mask
        ).long()

        return image_tensor, mask_tensor


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base: int = 32,
    ):
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

        self.outc = nn.Conv2d(
            base,
            num_classes,
            kernel_size=1,
        )

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


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_pixel_class_weights(dataset: UCLMSegmentationDataset):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for index in tqdm(
        range(len(dataset)),
        desc="Scanning train masks",
        leave=False,
    ):
        _, mask = dataset[index]

        values, frequencies = torch.unique(
            mask,
            return_counts=True,
        )

        for value, frequency in zip(
            values.tolist(),
            frequencies.tolist(),
        ):
            counts[value] += frequency

    frequency = counts / counts.sum()

    # Moderate inverse-square-root weighting.
    weights = 1.0 / np.sqrt(
        np.maximum(frequency, 1e-12)
    )

    weights = weights / weights.mean()

    return counts, frequency, weights


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    target_one_hot = F.one_hot(
        target,
        num_classes=NUM_CLASSES,
    ).permute(
        0, 3, 1, 2
    ).float()

    losses = []

    # Lesion classes only.
    for class_id in [1, 2]:
        target_class = target_one_hot[:, class_id]
        prob_class = probabilities[:, class_id]

        target_area = target_class.sum()

        # This batch has no target pixels of this class.
        if target_area.item() == 0:
            continue

        intersection = (
            prob_class * target_class
        ).sum()

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            prob_class.sum()
            + target_class.sum()
            + 1e-6
        )

        losses.append(1.0 - dice)

    if not losses:
        return torch.tensor(
            0.0,
            device=logits.device,
        )

    return torch.stack(losses).mean()


# ============================================================
# GLOBAL METRIC ACCUMULATION
# ============================================================

def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
):
    """
    Add pixel-level predictions to a [C,C] confusion matrix.

    Rows = ground truth
    Cols = prediction
    """

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    flat_target = target.reshape(-1)
    flat_prediction = prediction.reshape(-1)

    encoded = (
        flat_target * NUM_CLASSES
        + flat_prediction
    )

    batch_confusion = torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    confusion += batch_confusion


def metrics_from_confusion_matrix(
    confusion: torch.Tensor,
):
    dice = []
    iou = []

    for class_id in range(NUM_CLASSES):
        tp = confusion[class_id, class_id].float()

        fp = confusion[:, class_id].sum().float() - tp
        fn = confusion[class_id, :].sum().float() - tp

        denominator_dice = (
            2.0 * tp + fp + fn
        )

        denominator_iou = (
            tp + fp + fn
        )

        if denominator_dice.item() == 0:
            dice_value = 1.0
        else:
            dice_value = (
                2.0 * tp
                / denominator_dice
            ).item()

        if denominator_iou.item() == 0:
            iou_value = 1.0
        else:
            iou_value = (
                tp
                / denominator_iou
            ).item()

        dice.append(dice_value)
        iou.append(iou_value)

    return np.array(dice), np.array(iou)


# ============================================================
# TRAIN / VALIDATE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    training: bool,
    epoch: int,
):
    if training:
        model.train()
        phase = "Train"
    else:
        model.eval()
        phase = "Val"

    total_loss = 0.0
    total_samples = 0

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
        device=device,
    )

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d} {phase}",
        leave=False,
        dynamic_ncols=True,
    )

    for images, masks in progress:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            if device.type == "cuda":
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(images)

                    ce = criterion(
                        logits,
                        masks,
                    )

                    dl = dice_loss(
                        logits,
                        masks,
                    )

                    # Equal contribution from CE and Dice.
                    loss = 0.5 * ce + 0.5 * dl
            else:
                logits = model(images)

                ce = criterion(
                    logits,
                    masks,
                )

                dl = dice_loss(
                    logits,
                    masks,
                )

                loss = 0.5 * ce + 0.5 * dl

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

        with torch.no_grad():
            update_confusion_matrix(
                confusion,
                logits.detach(),
                masks,
            )

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}"
        )

    avg_loss = (
        total_loss
        / max(total_samples, 1)
    )

    dice, iou = metrics_from_confusion_matrix(
        confusion.cpu()
    )

    return (
        avg_loss,
        dice,
        iou,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 76)
    print("LUMINA — UCLM MULTICLASS U-NET SEGMENTATION TRAINING")
    print("=" * 76)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        print(
            f"CUDA: {torch.version.cuda}"
        )

    print()

    train_dataset = UCLMSegmentationDataset(
        TRAIN_CSV
    )

    val_dataset = UCLMSegmentationDataset(
        VAL_CSV
    )

    print(
        f"Train images: {len(train_dataset)}"
    )
    print(
        f"Validation images: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()
    print(
        "Computing training pixel distribution..."
    )

    counts, frequencies, class_weights = (
        compute_pixel_class_weights(
            train_dataset
        )
    )

    class_names = [
        "Background",
        "Benign",
        "Malignant",
    ]

    print()

    for idx, name in enumerate(class_names):
        print(
            f"{name:10s}: "
            f"{counts[idx]:,.0f} pixels "
            f"({frequencies[idx] * 100:.4f}%) "
            f"weight={class_weights[idx]:.4f}"
        )

    print()

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        f"Fixed epochs: {MAX_EPOCHS}"
    )
    print(
        f"Initial learning rate: "
        f"{LEARNING_RATE}"
    )
    print(
        "Early stopping: DISABLED"
    )
    print()

    print("=" * 76)
    print("STARTING FULL TRAINING RUN")
    print("=" * 76)

    history = []

    best_val_lesion_dice = -1.0
    best_epoch = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_loss, train_dice, train_iou = (
            run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                training=True,
                epoch=epoch,
            )
        )

        val_loss, val_dice, val_iou = (
            run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                scaler=None,
                training=False,
                epoch=epoch,
            )
        )

        train_lesion_dice = float(
            np.mean(train_dice[[1, 2]])
        )

        val_lesion_dice = float(
            np.mean(val_dice[[1, 2]])
        )

        val_lesion_iou = float(
            np.mean(val_iou[[1, 2]])
        )

        scheduler.step(
            val_lesion_dice
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}"
        )
        print(
            f"  Train loss:          {train_loss:.4f}"
        )
        print(
            f"  Val loss:            {val_loss:.4f}"
        )
        print(
            f"  Train lesion Dice:   "
            f"{train_lesion_dice:.4f}"
        )
        print(
            f"  Val lesion Dice:     "
            f"{val_lesion_dice:.4f}"
        )
        print(
            f"  Val lesion IoU:      "
            f"{val_lesion_iou:.4f}"
        )
        print(
            f"  Learning rate:       "
            f"{current_lr:.2e}"
        )

        print(
            "  Val Dice:"
            f"  BG={val_dice[0]:.4f}"
            f" | Benign={val_dice[1]:.4f}"
            f" | Malignant={val_dice[2]:.4f}"
        )

        print(
            "  Val IoU :"
            f"  BG={val_iou[0]:.4f}"
            f" | Benign={val_iou[1]:.4f}"
            f" | Malignant={val_iou[2]:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_dice_background": float(
                train_dice[0]
            ),
            "train_dice_benign": float(
                train_dice[1]
            ),
            "train_dice_malignant": float(
                train_dice[2]
            ),
            "train_iou_background": float(
                train_iou[0]
            ),
            "train_iou_benign": float(
                train_iou[1]
            ),
            "train_iou_malignant": float(
                train_iou[2]
            ),
            "train_lesion_dice": (
                train_lesion_dice
            ),
            "val_dice_background": float(
                val_dice[0]
            ),
            "val_dice_benign": float(
                val_dice[1]
            ),
            "val_dice_malignant": float(
                val_dice[2]
            ),
            "val_iou_background": float(
                val_iou[0]
            ),
            "val_iou_benign": float(
                val_iou[1]
            ),
            "val_iou_malignant": float(
                val_iou[2]
            ),
            "val_lesion_dice": (
                val_lesion_dice
            ),
            "val_lesion_iou": (
                val_lesion_iou
            ),
            "learning_rate": float(
                current_lr
            ),
        }

        history.append(row)

        # Keep BEST checkpoint for later evaluation.
        if val_lesion_dice > best_val_lesion_dice:
            best_val_lesion_dice = (
                val_lesion_dice
            )
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "best_val_lesion_dice": (
                        best_val_lesion_dice
                    ),
                    "num_classes": NUM_CLASSES,
                    "base_channels": (
                        BASE_CHANNELS
                    ),
                    "image_size": IMAGE_SIZE,
                    "class_names": class_names,
                    "class_weights": (
                        class_weights.tolist()
                    ),
                },
                BEST_MODEL_PATH,
            )

            print(
                f"  >>> BEST CHECKPOINT UPDATED "
                f"(epoch {best_epoch}, "
                f"lesion Dice="
                f"{best_val_lesion_dice:.4f})"
            )

    # Save final epoch separately.
    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "epoch": MAX_EPOCHS,
            "num_classes": NUM_CLASSES,
            "base_channels": BASE_CHANNELS,
            "image_size": IMAGE_SIZE,
            "class_names": class_names,
            "class_weights": (
                class_weights.tolist()
            ),
        },
        FINAL_MODEL_PATH,
    )

    HISTORY_PATH.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("FULL TRAINING RUN COMPLETE")
    print("=" * 76)
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )
    print(
        f"Best validation lesion Dice: "
        f"{best_val_lesion_dice:.4f}"
    )
    print(
        f"Best model:  {BEST_MODEL_PATH}"
    )
    print(
        f"Final model: {FINAL_MODEL_PATH}"
    )
    print(
        f"History:     {HISTORY_PATH}"
    )
    print()
    print(
        "The UCLM test set was NOT used."
    )
    print(
        "Next step: evaluate the locked test set "
        "using the BEST checkpoint."
    )


if __name__ == "__main__":
    main()
