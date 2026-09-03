from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "processed" / "uclm_train.csv"


def load_mask(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # OpenCV loads RGB images as BGR
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Green -> benign
    green = (
        (mask[:, :, 0] == 0)
        & (mask[:, :, 1] == 255)
        & (mask[:, :, 2] == 0)
    )

    # Red -> malignant
    red = (
        (mask[:, :, 0] == 255)
        & (mask[:, :, 1] == 0)
        & (mask[:, :, 2] == 0)
    )

    class_mask[green] = 1
    class_mask[red] = 2

    return class_mask


def create_overlay(image, class_mask):
    overlay = image.copy()

    # Semi-transparent lesion overlay
    benign = class_mask == 1
    malignant = class_mask == 2

    # Green for benign
    overlay[benign] = (
        0.6 * overlay[benign] + 0.4 * np.array([0, 255, 0])
    ).astype(np.uint8)

    # Red for malignant
    overlay[malignant] = (
        0.6 * overlay[malignant] + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return overlay


def show_sample(row):
    image_path = ROOT_DIR / row["image_path"]
    mask_path = ROOT_DIR / row["mask_path"]
    label = row["label"]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_mask = load_mask(mask_path)
    overlay = create_overlay(image, class_mask)

    print(f"\nImage : {image_path.name}")
    print(f"Label : {label}")
    print(f"Mask classes present: {np.unique(class_mask).tolist()}")

    for cls in [0, 1, 2]:
        count = int(np.sum(class_mask == cls))
        print(f"  Class {cls}: {count:,} pixels")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original — {label}")
    axes[0].axis("off")

    axes[1].imshow(class_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 70)
    print("LUMINA — UCLM SEGMENTATION VISUAL CHECK")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    # Pick one malignant and one benign example
    samples = []

    malignant = df[df["label"] == "Malignant"]
    benign = df[df["label"] == "Benign"]

    if not malignant.empty:
        samples.append(malignant.iloc[0])

    if not benign.empty:
        samples.append(benign.iloc[0])

    if not samples:
        raise RuntimeError("No benign or malignant samples found.")

    for row in samples:
        show_sample(row)


if __name__ == "__main__":
    main()