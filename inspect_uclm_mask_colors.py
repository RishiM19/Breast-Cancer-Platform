from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SPLITS = [
    PROCESSED_DIR / "uclm_train.csv",
    PROCESSED_DIR / "uclm_val.csv",
    PROCESSED_DIR / "uclm_test.csv",
]

GREEN = np.array([0, 255, 0])
RED = np.array([255, 0, 0])


def get_colors(mask_path):

    image = Image.open(mask_path).convert("RGB")
    array = np.asarray(image)

    green = np.all(array == GREEN, axis=-1)
    red = np.all(array == RED, axis=-1)

    green_pixels = int(green.sum())
    red_pixels = int(red.sum())

    return green_pixels, red_pixels


def main():

    print("=" * 70)
    print("LUMINA — UCLM RED/GREEN MASK INSPECTION")
    print("=" * 70)

    green_only = []
    red_only = []
    both = []

    for csv_path in SPLITS:

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():

            mask_path = ROOT_DIR / str(
                row["mask_path"]
            )

            green_pixels, red_pixels = get_colors(
                mask_path
            )

            if green_pixels > 0 and red_pixels == 0:

                green_only.append(
                    (
                        row["image"],
                        row["label"],
                        green_pixels
                    )
                )

            elif red_pixels > 0 and green_pixels == 0:

                red_only.append(
                    (
                        row["image"],
                        row["label"],
                        red_pixels
                    )
                )

            elif red_pixels > 0 and green_pixels > 0:

                both.append(
                    (
                        row["image"],
                        row["label"],
                        green_pixels,
                        red_pixels
                    )
                )

    print("\nGREEN ONLY:")
    print(
        f"Count: {len(green_only)}"
    )

    for item in green_only[:10]:

        print(
            f"  {item[0]:15} "
            f"{item[1]:10} "
            f"green={item[2]:,}"
        )

    print("\nRED ONLY:")
    print(
        f"Count: {len(red_only)}"
    )

    for item in red_only[:10]:

        print(
            f"  {item[0]:15} "
            f"{item[1]:10} "
            f"red={item[2]:,}"
        )

    print("\nBOTH RED + GREEN:")
    print(
        f"Count: {len(both)}"
    )

    for item in both[:10]:

        print(
            f"  {item[0]:15} "
            f"{item[1]:10} "
            f"green={item[2]:,} "
            f"red={item[3]:,}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()