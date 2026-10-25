"""Read-only BUSI checkpoint/validation audit from recovered original scripts.

Usage (from repository root):
  python src/audit_busi_original.py

This script never opens test.csv or the test images, and never writes files.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

# Original label order from process_data.py and evaluate.py
TRUE_CLASSES = ("Normal", "Benign", "Malignant")
# Incorrect existing main.py order, assessed ONLY for comparison
OLD_DISPLAY_CLASSES = ("Benign", "Malignant", "Normal")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def normalize_path(raw: str, repo: Path) -> Path:
    candidate = Path(str(raw).replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = repo / candidate
    return candidate.resolve()


def read_manifest(repo: Path, name: str) -> pd.DataFrame:
    path = repo / "data" / "processed" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    necessary = {"filepath", "label", "label_name"}
    if not necessary.issubset(df.columns):
        raise ValueError(f"{path}: missing columns {necessary - set(df.columns)}")
    df = df.copy()
    df["resolved_path"] = df["filepath"].map(lambda p: normalize_path(p, repo))
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not df["label"].isin(range(3)).all():
        raise ValueError(f"{name}: labels outside [0, 1, 2]")
    for idx, label, raw in zip(df.index, df["label"], df["label_name"]):
        if str(raw).strip().lower() != TRUE_CLASSES[label].lower():
            raise ValueError(f"Manifest label mismatch row {idx}: index {label}, label_name {raw}")
    return df


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    checkpoint = repo / "models" / "best_model.pth"
    print("BUSI ORIGINAL VALIDATION AUDIT (READ-ONLY)")
    print("Repository:", repo)
    print("Original class order:", TRUE_CLASSES)
    print("Existing backend display order:", OLD_DISPLAY_CLASSES)

    train = read_manifest(repo, "train.csv")  # filepaths only: no training images opened
    val = read_manifest(repo, "val.csv")
    train_paths = set(train["resolved_path"].map(str))
    val_paths = set(val["resolved_path"].map(str))
    overlap = train_paths & val_paths
    print(f"Training manifest: {len(train)} rows; validation manifest: {len(val)} rows")
    print(f"Exact train/validation filepath overlap: {len(overlap)}")
    print(f"Duplicate validation filepaths: {val['resolved_path'].duplicated().sum()}")
    if overlap:
        print("WARNING: overlapping paths detected; first 5:", list(sorted(overlap))[:5])
    missing = [str(p) for p in val["resolved_path"] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation images ({len(missing)}): {missing[:5]}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference device:", device)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Expected weights-only state_dict")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    true, predicted = [], []
    examples = []
    with torch.inference_mode():
        for start in range(0, len(val), 16):
            block = val.iloc[start:start + 16]
            tensors = []
            for p in block["resolved_path"]:
                with Image.open(p) as img:
                    tensors.append(transform(img.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            scores = model(batch)
            if not torch.isfinite(scores).all():
                raise ValueError(f"Non-finite logits in batch starting at {start}")
            pred_ids = scores.argmax(dim=1).cpu().tolist()
            predicted.extend(pred_ids)
            true.extend(block["label"].tolist())
            for path, truth, pred in zip(block["resolved_path"], block["label"], pred_ids):
                if truth != pred and len(examples) < 10:
                    examples.append((path.name, TRUE_CLASSES[truth], TRUE_CLASSES[pred]))

    cm = [[sum(1 for y, p in zip(true, predicted) if y == a and p == b)
           for b in range(3)] for a in range(3)]
    correct = sum(y == p for y, p in zip(true, predicted))
    wrong_display = sum(TRUE_CLASSES[y] == OLD_DISPLAY_CLASSES[p]
                        for y, p in zip(true, predicted))
    print("\nLabels: 0=Normal, 1=Benign, 2=Malignant")
    print("Validation label counts:", {TRUE_CLASSES[k]: Counter(true)[k] for k in range(3)})
    print("Predicted index counts:", {TRUE_CLASSES[k]: Counter(predicted)[k] for k in range(3)})
    print(f"CORRECT CLASS MAPPING validation accuracy: {correct}/{len(true)} = {correct / len(true):.2%}")
    print(f"OLD BACKEND DISPLAY MAPPING apparent accuracy: {wrong_display}/{len(true)} = {wrong_display / len(true):.2%}")
    print("\nConfusion matrix (rows = true classes, columns = predicted classes):")
    print("            Normal  Benign  Malignant")
    for name, row in zip(TRUE_CLASSES, cm):
        print(f"{name:<10} {row[0]:>6} {row[1]:>7} {row[2]:>10}")
    print("\nClass-wise recall:")
    for k, name in enumerate(TRUE_CLASSES):
        denom = sum(cm[k]); val_recall = cm[k][k] / denom if denom else float('nan')
        print(f"  {name:<10} {cm[k][k]}/{denom} = {val_recall:.2%}")
    print("\nFirst 10 incorrect validation predictions under correct mapping:")
    for example in examples:
        print("  image={}, true={}, predicted={}".format(*example))
    print("\nNo files written. Test manifest/images were not opened.")
    if overlap:
        print("WARNING: exact path overlap means accuracy needs leakage review before interpretation")
    print("WARNING: patient-level separation and duplicate-image hashes NOT evaluated here")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
