"""
LUMINA — UCLM Segmentation Visualization V3
============================================

Purpose:
    Create clinically meaningful representative examples from the
    LOCKED UCLM test set using the best U-Net checkpoint.

Selection:
    1. Worst benign case       -> lowest benign Dice
    2. Best benign case        -> highest benign Dice
    3. Worst malignant case    -> lowest malignant Dice
    4. Best malignant case     -> highest malignant Dice
    5. Normal with largest FP  -> largest predicted lesion area
    6. Clean normal case       -> zero predicted lesion pixels, preferably
                                  with the largest clean margin from lesion
                                  classes (all normal cases are eligible)

No NumPy / Pandas / Matplotlib.
Outputs are PNGs created with PIL.

Output:
    data/processed/uclm_segmentation_visuals_v3/
"""

from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT_DIR / "data" / "processed" / "uclm_test.csv"
MODEL_PATH = ROOT_DIR / "models" / "uclm_unet_best.pth"

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "uclm_segmentation_visuals_v3"
)

IMAGE_SIZE = 256
BASE_CHANNELS = 32
NUM_CLASSES = 3


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(self, x):
        return self.conv(
            self.pool(x)
        )


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
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

        return self.conv(
            torch.cat(
                [skip, x],
                dim=1,
            )
        )


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        base=32,
    ):
        super().__init__()

        self.inc = DoubleConv(
            in_channels,
            base,
        )

        self.down1 = DownBlock(
            base,
            base * 2,
        )

        self.down2 = DownBlock(
            base * 2,
            base * 4,
        )

        self.down3 = DownBlock(
            base * 4,
            base * 8,
        )

        self.down4 = DownBlock(
            base * 8,
            base * 16,
        )

        self.up1 = UpBlock(
            base * 16,
            base * 8,
            base * 8,
        )

        self.up2 = UpBlock(
            base * 8,
            base * 4,
            base * 4,
        )

        self.up3 = UpBlock(
            base * 4,
            base * 2,
            base * 2,
        )

        self.up4 = UpBlock(
            base * 2,
            base,
            base,
        )

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
# IMAGE / MASK HELPERS
# ============================================================

def rgb_mask_to_classes(mask_image):
    """
    Expert UCLM mask:
        black -> 0 Background
        green -> 1 Benign
        red   -> 2 Malignant
    """

    mask_image = (
        mask_image
        .convert("RGB")
        .resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )
    )

    values = []

    for r, g, b in mask_image.getdata():
        if r == 0 and g == 255 and b == 0:
            values.append(1)
        elif r == 255 and g == 0 and b == 0:
            values.append(2)
        else:
            values.append(0)

    return torch.tensor(
        values,
        dtype=torch.long,
    ).reshape(
        IMAGE_SIZE,
        IMAGE_SIZE,
    )


def image_to_tensor(image):
    image = (
        image
        .convert("RGB")
        .resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
    )

    values = []

    for pixel in image.getdata():
        values.extend(pixel)

    tensor = torch.tensor(
        values,
        dtype=torch.float32,
    ).reshape(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    )

    tensor = (
        tensor / 255.0
    ).permute(
        2,
        0,
        1,
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    ).view(
        3,
        1,
        1,
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(
        3,
        1,
        1,
    )

    return (
        (tensor - mean) / std
    ).unsqueeze(0)


def class_mask_to_image(mask):
    pixels = []

    for value in mask.reshape(-1).tolist():
        if value == 1:
            pixels.append((0, 255, 0))
        elif value == 2:
            pixels.append((255, 0, 0))
        else:
            pixels.append((0, 0, 0))

    result = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    result.putdata(pixels)

    return result


def overlay_mask(image, mask):
    base = (
        image
        .convert("RGB")
        .resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
    )

    base_pixels = list(
        base.getdata()
    )

    mask_values = (
        mask.reshape(-1).tolist()
    )

    output = []

    for pixel, value in zip(
        base_pixels,
        mask_values,
    ):
        if value == 1:
            color = (0, 255, 0)
            alpha = 0.45
        elif value == 2:
            color = (255, 0, 0)
            alpha = 0.45
        else:
            output.append(pixel)
            continue

        output.append(
            tuple(
                int(
                    (1 - alpha) * pixel[i]
                    + alpha * color[i]
                )
                for i in range(3)
            )
        )

    result = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    result.putdata(output)

    return result


def add_title(image, title):
    canvas = Image.new(
        "RGB",
        (
            IMAGE_SIZE,
            IMAGE_SIZE + 34,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, 34),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            17,
        )
    except Exception:
        font = ImageFont.load_default()

    draw.text(
        (7, 8),
        title,
        fill="black",
        font=font,
    )

    return canvas


def make_panel(
    original,
    gt,
    prediction,
    overlay,
    label,
    details,
):
    panels = [
        add_title(
            original.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            ),
            f"Original — {label}",
        ),
        add_title(
            class_mask_to_image(gt),
            "Ground Truth",
        ),
        add_title(
            class_mask_to_image(prediction),
            "Prediction",
        ),
        add_title(
            overlay,
            "Prediction Overlay",
        ),
    ]

    panel = Image.new(
        "RGB",
        (
            IMAGE_SIZE * 4,
            IMAGE_SIZE + 34,
        ),
        "white",
    )

    for index, item in enumerate(
        panels
    ):
        panel.paste(
            item,
            (
                index * IMAGE_SIZE,
                0,
            ),
        )

    # Add compact metadata at bottom by increasing canvas height.
    final = Image.new(
        "RGB",
        (
            panel.width,
            panel.height + 44,
        ),
        "white",
    )

    final.paste(
        panel,
        (0, 0),
    )

    draw = ImageDraw.Draw(
        final
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            14,
        )
    except Exception:
        font = ImageFont.load_default()

    draw.text(
        (8, panel.height + 12),
        details,
        fill="black",
        font=font,
    )

    return final


# ============================================================
# METRICS
# ============================================================

def dice_class(
    ground_truth,
    prediction,
    class_id,
):
    gt = (
        ground_truth == class_id
    )

    pred = (
        prediction == class_id
    )

    gt_area = gt.sum().item()
    pred_area = pred.sum().item()

    if gt_area + pred_area == 0:
        return 1.0

    intersection = torch.logical_and(
        gt,
        pred,
    ).sum().item()

    return (
        2.0 * intersection
        / (gt_area + pred_area)
    )


def lesion_dice(
    ground_truth,
    prediction,
):
    return (
        dice_class(
            ground_truth,
            prediction,
            1,
        )
        + dice_class(
            ground_truth,
            prediction,
            2,
        )
    ) / 2.0


def lesion_pixel_count(
    mask,
):
    return int(
        torch.sum(
            mask != 0
        ).item()
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print(
        "LUMINA — UCLM SEGMENTATION VISUALIZATION V3"
    )
    print("=" * 72)

    print("MAIN STARTED")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    print("Loading model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    rows = []

    with TEST_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    print(
        f"Test images: {len(rows)}"
    )

    candidates = []

    print()
    print(
        "Running inference on test images..."
    )

    with torch.no_grad():
        for index, row in enumerate(
            rows,
            start=1,
        ):
            image = Image.open(
                ROOT_DIR
                / row["image_path"]
            ).convert("RGB")

            mask = Image.open(
                ROOT_DIR
                / row["mask_path"]
            ).convert("RGB")

            gt = rgb_mask_to_classes(
                mask
            )

            tensor = image_to_tensor(
                image
            ).to(device)

            if device.type == "cuda":
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(tensor)
            else:
                logits = model(tensor)

            prediction = torch.argmax(
                logits,
                dim=1,
            )[0].cpu()

            benign_score = dice_class(
                gt,
                prediction,
                1,
            )

            malignant_score = dice_class(
                gt,
                prediction,
                2,
            )

            lesion_score = (
                benign_score
                + malignant_score
            ) / 2.0

            predicted_lesion_pixels = (
                lesion_pixel_count(
                    prediction
                )
            )

            candidates.append(
                {
                    "path": row["image_path"],
                    "label": row["label"],
                    "original": image,
                    "gt": gt,
                    "prediction": prediction,
                    "benign_dice": benign_score,
                    "malignant_dice": malignant_score,
                    "lesion_dice": lesion_score,
                    "predicted_lesion_pixels": predicted_lesion_pixels,
                }
            )

            if index == 1 or index % 10 == 0:
                print(
                    f"  Processed "
                    f"{index}/{len(rows)}"
                )

    benign_cases = [
        item
        for item in candidates
        if item["label"] == "Benign"
    ]

    malignant_cases = [
        item
        for item in candidates
        if item["label"] == "Malignant"
    ]

    normal_cases = [
        item
        for item in candidates
        if item["label"] == "Normal"
    ]

    if not benign_cases:
        raise RuntimeError(
            "No benign test cases found."
        )

    if not malignant_cases:
        raise RuntimeError(
            "No malignant test cases found."
        )

    if not normal_cases:
        raise RuntimeError(
            "No normal test cases found."
        )

    worst_benign = min(
        benign_cases,
        key=lambda x: x["benign_dice"],
    )

    best_benign = max(
        benign_cases,
        key=lambda x: x["benign_dice"],
    )

    worst_malignant = min(
        malignant_cases,
        key=lambda x: x["malignant_dice"],
    )

    best_malignant = max(
        malignant_cases,
        key=lambda x: x["malignant_dice"],
    )

    largest_normal_false_positive = max(
        normal_cases,
        key=lambda x: x["predicted_lesion_pixels"],
    )

    clean_normals = [
        item
        for item in normal_cases
        if item["predicted_lesion_pixels"] == 0
    ]

    if clean_normals:
        clean_normal = clean_normals[0]
    else:
        clean_normal = min(
            normal_cases,
            key=lambda x: x["predicted_lesion_pixels"],
        )

    selected = [
        (
            "worst_benign",
            worst_benign,
        ),
        (
            "best_benign",
            best_benign,
        ),
        (
            "worst_malignant",
            worst_malignant,
        ),
        (
            "best_malignant",
            best_malignant,
        ),
        (
            "normal_largest_false_positive",
            largest_normal_false_positive,
        ),
        (
            "normal_cleanest",
            clean_normal,
        ),
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Saving selected diagnostic examples..."
    )

    for number, (
        category,
        item,
    ) in enumerate(
        selected,
        start=1,
    ):
        prediction_overlay = overlay_mask(
            item["original"],
            item["prediction"],
        )

        details = (
            f"{category} | "
            f"image={Path(item['path']).name} | "
            f"label={item['label']} | "
            f"Benign Dice={item['benign_dice']:.4f} | "
            f"Malignant Dice={item['malignant_dice']:.4f} | "
            f"Predicted lesion pixels={item['predicted_lesion_pixels']:,}"
        )

        panel = make_panel(
            item["original"],
            item["gt"],
            item["prediction"],
            prediction_overlay,
            item["label"],
            details,
        )

        filename = (
            f"{number:02d}_"
            f"{category}_"
            f"{Path(item['path']).stem}_"
            f"{item['label']}.png"
        )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        panel.save(
            output_path
        )

        print(
            f"  {filename}"
        )

    print()
    print(
        "VISUALIZATION COMPLETE"
    )
    print(
        f"Output folder:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
