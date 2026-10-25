"""Read-only BUSI train/validation near-duplicate visual screening.

Usage from repository root:
    python src/busi_phash_visual_review.py

Never accesses test.csv or test images. Creates only review outputs in a new folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn

EXPECTED = {"normal": 0, "benign": 1, "malignant": 2}


def perceptual_hash(path: Path) -> np.ndarray:
    # Exactly matches the earlier BUSI pHash investigation.
    with Image.open(path) as img:
        gray = img.convert("L").resize((32, 32))
        a = np.asarray(gray, dtype=np.float32)
    low = dctn(a, norm="ortho")[:8, :8].copy()
    return (low > np.median(low.ravel()[1:])).ravel()


def read_manifest(root: Path, name: str):
    filename = root / "data" / "processed" / f"{name}.csv"
    df = pd.read_csv(filename)
    required = {"filepath", "label", "label_name"}
    if not required <= set(df.columns):
        raise RuntimeError(f"Missing columns in {filename}: {sorted(required - set(df.columns))}")
    items = []
    for idx, row in df.iterrows():
        label = str(row["label_name"]).strip().lower()
        code = int(row["label"])
        if EXPECTED.get(label) != code:
            raise RuntimeError(f"Inconsistent label in {name}.csv row {idx}: {label}={code}")
        local = Path(str(row["filepath"]).replace("\\", "/"))
        full = (local if local.is_absolute() else root / local).resolve()
        if not full.is_file():
            raise FileNotFoundError(f"Missing {name} image: {full}")
        items.append({"path": full, "name": full.name, "label": label, "hash": perceptual_hash(full)})
    return items


def image_thumbnail(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        thumb = ImageOps.contain(rgb, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (242, 242, 242))
    canvas.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
    return canvas


def make_sheet(rows, path: Path, page_no: int):
    width, height = 1370, 340
    margin = 28
    header = 92
    per_row = 420
    canvas = Image.new("RGB", (width, header + per_row * len(rows) + 30), "white")
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    pen.text((margin, 25), f"BUSI train-vs-validation similarity | page {page_no} | VISUAL SCREENING ONLY", fill="black", font=font)
    pen.text((margin, 52), "pHash similarity is not proof of image duplication or patient identity. Test split not accessed.", fill=(100, 40, 40), font=font)
    thumb_w, thumb_h = 610, 332
    right_x = 725
    for j, item in enumerate(rows):
        y = header + j * per_row
        pen.line((margin, y, width-margin, y), fill=(205, 205, 205), width=2)
        pen.text((margin, y + 10), f"PAIR {item['rank']} | pHash Hamming distance {item['phash_distance']} / 64", fill=(0, 0, 0), font=font)
        pen.text((margin, y + 30), f"VALIDATION: {item['validation_image']} [{item['validation_label']}]", fill=(0, 0, 0), font=font)
        pen.text((right_x, y + 30), f"TRAIN: {item['train_image']} [{item['train_label']}]", fill=(0, 0, 0), font=font)
        canvas.paste(image_thumbnail(Path(item["validation_path"]), thumb_w, thumb_h), (margin, y + 56))
        canvas.paste(image_thumbnail(Path(item["train_path"]), thumb_w, thumb_h), (right_x, y + 56))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description="BUSI validation pHash contact sheets")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--threshold", type=int, default=8, help="Include nearest pairs at/below this pHash distance")
    parser.add_argument("--pairs-per-page", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/busi_phash_visual_review"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 64:
        parser.error("--threshold must be 0..64")
    if args.pairs_per_page < 1 or args.pairs_per_page > 7:
        parser.error("--pairs-per-page must be 1..7")
    root = args.root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    train = read_manifest(root, "train")
    val = read_manifest(root, "val")
    th = np.stack([r["hash"] for r in train])
    ranked = []
    counts = Counter()
    for v in val:
        distances = np.count_nonzero(th != v["hash"], axis=1)
        nearest_idx = int(np.argmin(distances))
        best = train[nearest_idx]
        dist = int(distances[nearest_idx])
        counts[dist] += 1
        ranked.append({
            "phash_distance": dist,
            "validation_image": v["name"], "validation_label": v["label"],
            "validation_path": str(v["path"]),
            "train_image": best["name"], "train_label": best["label"],
            "train_path": str(best["path"]),
            "label_disagreement": (v["label"] != best["label"]),
            "review_category": "", "review_notes": "",
        })
    ranked.sort(key=lambda x: (x["phash_distance"], x["validation_image"].lower()))
    candidates = [r for r in ranked if r["phash_distance"] <= args.threshold]
    for i, row in enumerate(candidates, 1):
        row["rank"] = i
    out.mkdir(parents=True, exist_ok=True)
    # CSV of *all* validation nearest neighbours, to support threshold review.
    all_csv = out / "all_validation_nearest_matches.csv"
    fields = ["phash_distance", "validation_image", "validation_label", "validation_path", "train_image", "train_label", "train_path", "label_disagreement", "review_category", "review_notes"]
    with all_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: r.get(key, "") for key in fields} for r in ranked])
    review_csv = out / "review_candidates.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank"] + fields)
        writer.writeheader()
        writer.writerows([{key: r.get(key, "") for key in ["rank"]+fields} for r in candidates])
    pages = []
    for start in range(0, len(candidates), args.pairs_per_page):
        page = len(pages) + 1
        filename = out / f"busi_near_duplicate_pairs_{page:02d}.png"
        make_sheet(candidates[start:start+args.pairs_per_page], filename, page)
        pages.append(filename.name)
    summary = {
        "purpose": "Candidate near-duplicate VISUAL screening; never infer patient identity from pHash",
        "train_rows": len(train), "validation_rows": len(val),
        "count_nearest_distance_eq_0": sum(r["phash_distance"] == 0 for r in ranked),
        "count_nearest_distance_le_4": sum(r["phash_distance"] <= 4 for r in ranked),
        "count_nearest_distance_le_8": sum(r["phash_distance"] <= 8 for r in ranked),
        "threshold": args.threshold,
        "candidate_count": len(candidates),
        "cross_label_candidate_count": sum(r["label_disagreement"] for r in candidates),
        "page_files": pages,
        "notes": [
            "A distance of zero does not establish byte identity (or even visual identity).",
            "Image similarity is not patient identification.",
            "The nearest pHash match need not be the actual nearest image under other measures.",
            "Screensheets should be reviewed without using model predictions.",
            "No model weights, manifests, or source images modified. Test split never opened."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BUSI pHash candidate visual review (READ-ONLY inputs)")
    for k, v in summary.items():
        if k not in ("notes", "page_files"):
            print(f"{k}: {v}")
    print("Contact sheets:")
    for p in pages:
        print(" ", out / p)
    print("Candidate annotations CSV:", review_csv)
    print("Validation nearest-match CSV:", all_csv)
    print("No original data or model files modified. Test manifest never read.")


if __name__ == "__main__":
    main()
