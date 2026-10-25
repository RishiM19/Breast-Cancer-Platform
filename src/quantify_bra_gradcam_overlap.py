"""Lumina BI-RADS V1: read-only annotation/Grad-CAM alignment audit.

Usage from project root:
  python src/quantify_bra_gradcam_overlap.py --index 0 --target 3
  python src/gradcam_bra_birads_v1.py --run --index 0 --target-class 4
  python src/quantify_bra_gradcam_overlap.py --index 0 --target 4

Only validation images are used. Never interprets attribution as diagnosis.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = ROOT / "data/processed/bra_val.csv"
CAM_DIR = ROOT / "data/processed/gradcam_bra_birads_v1"
SIZE = 224


def resolve_path(value: object) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="Zero-based bra_val.csv row")
    parser.add_argument("--target", type=int, default=3, choices=(2, 3, 4, 5), help="Grad-CAM target BI-RADS category")
    args = parser.parse_args()

    df = pd.read_csv(VAL_CSV)
    needed = {"image_path", "mask_path", "birads", "case_id"}
    if missing := needed - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if args.index < 0 or args.index >= len(df):
        raise IndexError(f"Index must be 0..{len(df)-1}")

    row = df.iloc[args.index]
    image_path = resolve_path(row["image_path"])
    mask_path = resolve_path(row["mask_path"])
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", image_path.stem)
    prefix = f"val_{args.index:03d}_{stem}_target_{args.target}"
    cam_path = CAM_DIR / f"{prefix}_cam.npy"
    if not cam_path.is_file():
        raise FileNotFoundError(f"Grad-CAM array missing: {cam_path}\nGenerate it with: python src/gradcam_bra_birads_v1.py --run --index {args.index} --target-class {args.target}")

    cam = np.load(cam_path, allow_pickle=False)
    if cam.shape != (SIZE, SIZE) or not np.isfinite(cam).all():
        raise ValueError(f"Unexpected CAM shape or nonfinite values: {cam.shape}")
    if float(cam.min()) < 0 or float(cam.max()) > 1.0001:
        raise ValueError("Expected a normalized [0,1] CAM")

    with Image.open(image_path) as img_src, Image.open(mask_path) as mask_src:
        img = img_src.convert("RGB")
        mask = mask_src.convert("L")
        if img.size != mask.size:
            raise ValueError(f"Original image/mask sizes differ: {img.size} vs {mask.size}; inspect alignment before analysis")
        vals = np.unique(np.asarray(mask))
        if not set(vals.tolist()).issubset({0, 255}):
            raise ValueError(f"Expected an exact binary 0/255 lesion mask; got values: {vals[:30].tolist()}")
        # Exactly match ImageOps.pad and centered positioning in V1's model input.
        padded_img = ImageOps.pad(img, (SIZE, SIZE), method=Image.Resampling.BILINEAR,
                                  color=(0, 0, 0), centering=(0.5, 0.5))
        padded_mask = ImageOps.pad(mask, (SIZE, SIZE), method=Image.Resampling.NEAREST,
                                   color=0, centering=(0.5, 0.5))
        valid = ImageOps.pad(Image.new("L", img.size, 255), (SIZE, SIZE),
                             method=Image.Resampling.NEAREST, color=0,
                             centering=(0.5, 0.5))

    lesion = np.asarray(padded_mask) > 127
    valid_image = np.asarray(valid) > 127
    if not np.any(lesion):
        raise ValueError("No lesion pixels found in annotation; choose a lesion-positive image")
    if np.any(lesion & ~valid_image):
        raise ValueError("Annotation projects into padding; check preprocessing")

    n_valid = int(valid_image.sum())
    n_lesion = int(lesion.sum())
    area_fraction = n_lesion / n_valid
    mass_total = float(cam[valid_image].sum())
    if mass_total <= 1e-10:
        raise ValueError("CAM activation is zero in image area")
    mass_fraction = float(cam[lesion].sum()) / mass_total
    enrichment = mass_fraction / area_fraction

    valid_indices = np.flatnonzero(valid_image.reshape(-1))
    k = max(1, int(round(0.20 * len(valid_indices))))
    top_indices = valid_indices[np.argpartition(cam.reshape(-1)[valid_indices], -k)[-k:]]
    n_top_in_lesion = int(lesion.reshape(-1)[top_indices].sum())
    top20_precision = n_top_in_lesion / k
    top20_lesion_coverage = n_top_in_lesion / n_lesion

    result = {
        "split": "validation",
        "row_index": args.index,
        "case_id": str(row["case_id"]),
        "annotated_birads": int(row["birads"]),
        "gradcam_target_birads": args.target,
        "lesion_area_fraction_of_imaged_pixels": round(area_fraction, 5),
        "fraction_cam_activation_inside_lesion": round(mass_fraction, 5),
        "lesion_attribution_enrichment_vs_uniform": round(enrichment, 4),
        "fraction_top20percent_cam_pixels_inside_lesion": round(top20_precision, 5),
        "fraction_lesion_pixels_in_top20percent_cam": round(top20_lesion_coverage, 5),
        "notes": "Descriptive spatial-attribution metrics; not lesion Dice, clinical validation or proof of causation. Excludes padded black margins.",
    }
    out_png = CAM_DIR / f"{prefix}_mask_overlap.png"
    out_json = CAM_DIR / f"{prefix}_mask_overlap.json"

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    image_np = np.asarray(padded_img)
    ax[0].imshow(image_np)
    ax[0].contour(lesion.astype(int), levels=[0.5], colors=["lime"], linewidths=1.4)
    ax[0].set_title("Model input + ground-truth border")
    ax[1].imshow(lesion, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Reference lesion mask")
    ax[2].imshow(image_np)
    ax[2].imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    ax[2].contour(lesion.astype(int), levels=[0.5], colors=["lime"], linewidths=1.4)
    ax[2].set_title("Grad-CAM + ground-truth border")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Case {row['case_id']} | Target {args.target} | CAM mass in lesion: {mass_fraction:.1%} | Lesion area: {area_fraction:.1%}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved:", out_png)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
