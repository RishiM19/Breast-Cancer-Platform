from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent

MASK_DIR = (
    ROOT_DIR
    / "data"
    / "BUS_UCLM"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset"
    / "masks"
)


def inspect_mask(path: Path):

    print("\n" + "=" * 70)
    print(f"MASK: {path.name}")
    print("=" * 70)

    image = Image.open(path).convert("RGB")

    array = np.array(image)

    print(f"Size: {image.size}")
    print(f"Mode: {image.mode}")

    pixels = array.reshape(-1, 3)

    unique_colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True,
    )

    print(
        f"\nNumber of unique RGB colors: "
        f"{len(unique_colors)}"
    )

    print("\nMost common colors:")

    order = np.argsort(counts)[::-1]

    for index in order[:20]:

        rgb = tuple(
            unique_colors[index].tolist()
        )

        count = int(counts[index])

        percentage = (
            count / len(pixels) * 100
        )

        print(
            f"  RGB {rgb}: "
            f"{count:,} pixels "
            f"({percentage:.4f}%)"
        )


def main():

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Mask directory not found:\n{MASK_DIR}"
        )

    mask_files = sorted(
        [
            p for p in MASK_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
            }
        ]
    )

    print(
        f"Found {len(mask_files)} mask files."
    )

    # Inspect the first three masks.
    for mask_path in mask_files[:3]:
        inspect_mask(mask_path)

    print("\n" + "=" * 70)
    print("MASK INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()