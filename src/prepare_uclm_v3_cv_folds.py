"""Prepare leakage-checked patient-grouped CV manifests for Lumina UCLM V3.

Run from repository root:
    python src/prepare_uclm_v3_cv_folds.py

This script creates CSV manifests only. It does NOT train a model, alter
original CSVs, read test images or masks, or overwrite checkpoints.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED / "uclm_train.csv"
VAL_CSV = PROCESSED / "uclm_val.csv"
TEST_CSV = PROCESSED / "uclm_test.csv"
OUTPUT = PROCESSED / "uclm_v3_cv_folds"
SEED = 42
FOLDS = 5


def patient_id(image_path: str) -> str:
    stem = Path(str(image_path)).stem
    if "_" not in stem:
        raise ValueError(f"Cannot extract patient ID: {image_path}")
    return stem.split("_")[0]


def resolve_path(path: str) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else ROOT / p


def annotation(mask_path: str) -> str:
    path = resolve_path(mask_path)
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    black = np.all(rgb == (0, 0, 0), axis=-1)
    green = np.all(rgb == (0, 255, 0), axis=-1)
    red = np.all(rgb == (255, 0, 0), axis=-1)
    if not np.all(black | green | red):
        raise ValueError(f"Unexpected annotation colors: {path}")
    if np.any(green) and np.any(red):
        raise ValueError(f"Both lesion labels found: {path}")
    if np.any(green):
        return "Benign"
    if np.any(red):
        return "Malignant"
    return "Empty"


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)  # filenames only; NO test images or labels opened
    required = {"image_path", "mask_path"}
    for name, frame in (("train", train), ("val", val)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} manifest missing columns {required - set(frame.columns)}")

    # Patient HESN has 13 mismatched image/mask pairs. Never relabel or resize them.
    development = pd.concat([train, val], ignore_index=True)
    development["patient_id"] = development["image_path"].map(patient_id)
    excluded = development["patient_id"].eq("HESN")
    print(f"Excluded HESN images: {int(excluded.sum())}")
    development = development.loc[~excluded].copy().reset_index(drop=True)

    if development["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths in development manifests")

    dev_patients = set(development["patient_id"])
    test_patients = set(test["image_path"].map(patient_id))
    overlap = dev_patients & test_patients
    if overlap:
        raise ValueError(f"Test/development patient leakage: {sorted(overlap)}")

    print("Auditing development masks for exact RGB label colors...")
    development["annotation"] = development["mask_path"].map(annotation)
    totals = development["annotation"].value_counts()
    print(f"Development patients: {len(dev_patients)}; images: {len(development)}")
    print("Annotations:", {k: int(totals.get(k, 0)) for k in ("Empty", "Benign", "Malignant")})

    if len(dev_patients) != 31 or len(development) != 565:
        raise ValueError("Unexpected development size; review source manifests before proceeding")
    if any(int(totals.get(k, 0)) != v for k, v in
           (("Empty", 348), ("Benign", 142), ("Malignant", 75))):
        raise ValueError("Class counts differ from the audited data; review before proceeding")

    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(development, development["annotation"], development["patient_id"]))
    summary_rows = []
    patient_rows = []
    validation_seen = set()
    manifest_columns = [c for c in development.columns if c not in ("patient_id", "annotation")]
    planned_files = []
    for n, (tr_indices, va_indices) in enumerate(splits, start=1):
        tr = development.iloc[tr_indices].copy()
        va = development.iloc[va_indices].copy()
        train_patients = set(tr["patient_id"])
        val_patients = set(va["patient_id"])
        if train_patients & val_patients:
            raise AssertionError(f"Patient leakage inside fold {n}")
        if validation_seen & val_patients:
            raise AssertionError(f"Patient validated more than once: fold {n}")
        validation_seen.update(val_patients)
        counts = va["annotation"].value_counts()
        if any(int(counts.get(label, 0)) == 0 for label in ("Empty", "Benign", "Malignant")):
            raise ValueError(f"Fold {n} lacks an annotation class; rebalance folds")
        summary_rows.append({
            "fold": n, "train_patients": len(train_patients), "train_images": len(tr),
            "val_patients": len(val_patients), "val_images": len(va),
            "val_empty": int(counts.get("Empty", 0)),
            "val_benign": int(counts.get("Benign", 0)),
            "val_malignant": int(counts.get("Malignant", 0)),
        })
        patient_rows.extend({"patient_id": p, "validation_fold": n} for p in sorted(val_patients))
        planned_files.extend((
            (OUTPUT / f"fold_{n:02d}_train.csv", tr[manifest_columns]),
            (OUTPUT / f"fold_{n:02d}_val.csv", va[manifest_columns]),
        ))
    if validation_seen != dev_patients:
        raise AssertionError("Not every development patient was assigned exactly one validation fold")
    summary = pd.DataFrame(summary_rows)
    print("\nProposed patient-grouped folds:")
    print(summary.to_string(index=False))
    print("\nEach patient is held out exactly once. No overlap with test patients.")

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, frame in planned_files:
        frame.to_csv(path, index=False)
    summary.to_csv(OUTPUT / "fold_summary.csv", index=False)
    pd.DataFrame(patient_rows).sort_values(["validation_fold", "patient_id"]).to_csv(
        OUTPUT / "patient_assignments.csv", index=False
    )
    print(f"\nSaved fold manifests and audit summaries under:\n{OUTPUT}")
    print("Original manifests, datasets and checkpoints were not modified.")


if __name__ == "__main__":
    main()
