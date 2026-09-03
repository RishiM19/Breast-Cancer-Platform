"""
LUMINA — UCLM Segmentation Visualization V2
NO NUMPY / NO PANDAS / NO MATPLOTLIB

Loads the best U-Net checkpoint and creates 7 representative
test-set images:
    Original | Ground Truth | Prediction | Overlay

Output:
    data/processed/uclm_segmentation_visuals_v2/
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
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "uclm_segmentation_visuals_v2"

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


# ============================================================
# MASK / IMAGE HELPERS
# ============================================================

def rgb_mask_to_classes(mask_image):
    mask_image = mask_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST,
    )

    pixels = list(mask_image.getdata())
    values = []

    for r, g, b in pixels:
        if r == 0 and g == 255 and b == 0:
            values.append(1)
        elif r == 255 and g == 0 and b == 0:
            values.append(2)
        else:
            values.append(0)

    return torch.tensor(
        values,
        dtype=torch.long,
    ).reshape(IMAGE_SIZE, IMAGE_SIZE)


def image_to_tensor(image):
    image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    pixels = list(image.getdata())

    values = [
        channel
        for pixel in pixels
        for channel in pixel
    ]

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
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(3, 1, 1)

    return (
        (tensor - mean) / std
    ).unsqueeze(0)


def class_mask_to_image(mask):
    values = mask.reshape(-1).tolist()
    pixels = []

    for value in values:
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
    base = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    base_pixels = list(base.getdata())
    mask_values = mask.reshape(-1).tolist()

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


def dice_class(gt, pred, class_id):
    gt_mask = gt == class_id
    pred_mask = pred == class_id

    gt_area = gt_mask.sum().item()
    pred_area = pred_mask.sum().item()

    if gt_area + pred_area == 0:
        return 1.0

    intersection = torch.logical_and(
        gt_mask,
        pred_mask,
    ).sum().item()

    return (
        2.0
        * intersection
        / (gt_area + pred_area)
    )


def lesion_dice(gt, pred):
    return (
        dice_class(gt, pred, 1)
        + dice_class(gt, pred, 2)
    ) / 2.0


def add_title(image, title):
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE + 32),
        "white",
    )

    canvas.paste(image, (0, 32))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            17,
        )
    except Exception:
        font = ImageFont.load_default()

    draw.text(
        (8, 7),
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
):
    images = [
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
            IMAGE_SIZE + 32,
        ),
        "white",
    )

    for index, image in enumerate(images):
        panel.paste(
            image,
            (index * IMAGE_SIZE, 0),
        )

    return panel


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("LUMINA — UCLM SEGMENTATION VISUALIZATION V2")
    print("=" * 70)
    print("MAIN STARTED")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"Model: {MODEL_PATH}")
    print(f"CSV:   {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV}"
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
        checkpoint["model_state_dict"]
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
        rows = list(csv.DictReader(f))

    print(
        f"Test images: {len(rows)}"
    )

    candidates = []

    print()
    print("Running test inference...")

    with torch.no_grad():
        for index, row in enumerate(
            rows,
            start=1,
        ):
            image = Image.open(
                ROOT_DIR / row["image_path"]
            ).convert("RGB")

            mask = Image.open(
                ROOT_DIR / row["mask_path"]
            ).convert("RGB")

            ground_truth = (
                rgb_mask_to_classes(mask)
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

            score = lesion_dice(
                ground_truth,
                prediction,
            )

            candidates.append(
                {
                    "path": row["image_path"],
                    "label": row["label"],
                    "original": image,
                    "gt": ground_truth,
                    "prediction": prediction,
                    "score": score,
                }
            )

            if index == 1 or index % 10 == 0:
                print(
                    f"  {index}/{len(rows)}"
                )

    # Worst 3 and best 3.
    candidates.sort(
        key=lambda item: item["score"]
    )

    selected = []

    for item in candidates[:3]:
        selected.append(
            ("worst", item)
        )

    for item in candidates[-3:]:
        selected.append(
            ("best", item)
        )

    # Normal case with the largest false-positive
    # predicted lesion area.
    normal_cases = [
        item
        for item in candidates
        if item["label"] == "Normal"
    ]

    if normal_cases:
        normal_cases.sort(
            key=lambda item: (
                item["prediction"] != 0
            ).sum().item(),
            reverse=True,
        )

        selected.append(
            (
                "normal_false_positive",
                normal_cases[0],
            )
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving visualizations...")

    for number, (
        category,
        item,
    ) in enumerate(
        selected,
        start=1,
    ):
        overlay = overlay_mask(
            item["original"],
            item["prediction"],
        )

        panel = make_panel(
            item["original"],
            item["gt"],
            item["prediction"],
            overlay,
            item["label"],
        )

        stem = Path(
            item["path"]
        ).stem

        output_path = (
            OUTPUT_DIR
            / f"{number:02d}_{category}_{stem}_{item['label']}.png"
        )

        panel.save(
            output_path,
        )

        print(
            f"  {output_path.name} | "
            f"{category} | "
            f"lesion Dice="
            f"{item['score']:.4f}"
        )

    print()
    print("VISUALIZATION COMPLETE")
    print(
        f"Output folder:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
