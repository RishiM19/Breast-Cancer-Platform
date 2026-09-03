from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SPLITS = {
    "train": PROCESSED_DIR / "uclm_train.csv",
    "val": PROCESSED_DIR / "uclm_val.csv",
    "test": PROCESSED_DIR / "uclm_test.csv",
}

LESION_RGB = (0, 255, 0)


def inspect_mask(mask_path):
    """Inspect a single mask and return basic information."""

    mask = Image.open(mask_path).convert("RGB")
    arr = np.asarray(mask)

    green_pixels = np.all(
        arr == LESION_RGB,
        axis=-1
    )

    positive_pixels = int(green_pixels.sum())

    # Check whether any colors other than black/green exist.
    unique_colors = np.unique(
        arr.reshape(-1, 3),
        axis=0
    )

    allowed_colors = {
        (0, 0, 0),
        (0, 255, 0),
    }

    unexpected_colors = [
        tuple(color.tolist())
        for color in unique_colors
        if tuple(color.tolist()) not in allowed_colors
    ]

    return (
        positive_pixels,
        unexpected_colors
    )


def analyze_split(split_name, csv_path):

    print("\n" + "=" * 65)
    print(f"BUS-UCLM {split_name.upper()} MASK CHECK")
    print("=" * 65)

    df = pd.read_csv(csv_path)

    total_masks = len(df)
    non_empty = 0
    empty = 0
    total_lesion_pixels = 0

    unexpected_color_masks = []

    lesion_examples = []
    empty_examples = []

    for _, row in df.iterrows():

        mask_path = ROOT_DIR / str(
            row["mask_path"]
        )

        positive_pixels, unexpected_colors = (
            inspect_mask(mask_path)
        )

        total_lesion_pixels += positive_pixels

        if positive_pixels > 0:

            non_empty += 1

            if len(lesion_examples) < 5:
                lesion_examples.append(
                    (
                        row["image"],
                        row["label"],
                        positive_pixels
                    )
                )

        else:

            empty += 1

            if len(empty_examples) < 5:
                empty_examples.append(
                    (
                        row["image"],
                        row["label"]
                    )
                )

        if unexpected_colors:

            unexpected_color_masks.append(
                (
                    row["image"],
                    unexpected_colors
                )
            )

    print(f"Total masks        : {total_masks}")
    print(f"Non-empty masks    : {non_empty}")
    print(f"Empty masks        : {empty}")
    print(f"Total lesion pixels: {total_lesion_pixels:,}")

    print("\nExample lesion-containing masks:")

    for image, label, pixels in lesion_examples:

        print(
            f"  {image:<15} "
            f"{label:<10} "
            f"green pixels={pixels:,}"
        )

    print("\nExample empty masks:")

    for image, label in empty_examples:

        print(
            f"  {image:<15} "
            f"{label:<10}"
        )

    print(
        f"\nMasks with unexpected colors: "
        f"{len(unexpected_color_masks)}"
    )

    if unexpected_color_masks:

        print("\nFirst unexpected-color examples:")

        for image, colors in unexpected_color_masks[:5]:

            print(
                f"  {image}: {colors}"
            )


def main():

    print("=" * 65)
    print("LUMINA — FAST UCLM MASK VALIDATION")
    print("=" * 65)

    for split_name, csv_path in SPLITS.items():

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Missing CSV:\n{csv_path}"
            )

        analyze_split(
            split_name,
            csv_path
        )

    print("\n" + "=" * 65)
    print("MASK VALIDATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()