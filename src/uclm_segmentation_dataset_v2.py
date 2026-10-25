"""UCLM V2 segmentation dataset: paired geometry, exact integer labels.

CSV image_path/mask_path are relative to the project root (or absolute).
Class ids: background=0, benign=1 (green), malignant=2 (red).
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["Background", "Benign", "Malignant"]


def rgb_to_labels(rgb: np.ndarray) -> np.ndarray:
    """Strict RGB conversion: unknown annotation colors trigger an error."""
    black = np.all(rgb == (0, 0, 0), axis=-1)
    green = np.all(rgb == (0, 255, 0), axis=-1)
    red = np.all(rgb == (255, 0, 0), axis=-1)
    if not np.all(black | green | red):
        unknown = np.unique(rgb[~(black | green | red)].reshape(-1, 3), axis=0)
        raise ValueError(f"Unexpected mask RGB values (first five): {unknown[:5].tolist()}")
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    labels[green] = 1
    labels[red] = 2
    return labels


class UCLMSegmentationDatasetV2(Dataset):
    def __init__(self, csv_path, root_dir, size=256, augment=False, exclude_patients=None):
        self.csv_path = Path(csv_path)
        self.root = Path(root_dir)
        self.size = int(size)
        self.augment = augment
        self.data = pd.read_csv(self.csv_path).reset_index(drop=True)
        required = {"image_path", "mask_path", "label", "patient_id"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns in {self.csv_path}: {sorted(missing)}")
        self.excluded_patients = tuple(sorted({str(x).strip().upper() for x in (exclude_patients or [])}))
        self.excluded_rows = 0
        if self.excluded_patients:
            is_excluded = self.data["patient_id"].astype(str).str.strip().str.upper().isin(self.excluded_patients)
            self.excluded_rows = int(is_excluded.sum())
            self.data = self.data.loc[~is_excluded].reset_index(drop=True)
        if self.data.empty:
            raise ValueError(f"Empty manifest after exclusions: {self.csv_path}")

    def __len__(self):
        return len(self.data)

    def _path(self, value):
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self._path(row.image_path)
        mask_path = self._path(row.mask_path)
        with Image.open(image_path) as obj:
            image = obj.convert("RGB")
        with Image.open(mask_path) as obj:
            rgb_mask = obj.convert("RGB")
        if image.size != rgb_mask.size:
            raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
        # Decode *before* resizing so mask pixels never become soft labels.
        mask = Image.fromarray(rgb_to_labels(np.asarray(rgb_mask)), mode="L")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            # Safe in-plane geometry: apply identical operations to both.
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < 0.5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            k = random.randrange(4)
            if k:
                angle = 90 * k
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
            # Image-only photometric augmentation; no modification to label mask.
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        labels = np.asarray(mask, dtype=np.uint8).astype(np.int64, copy=True)
        return normalized_rgb_tensor(image), torch.from_numpy(labels)


def normalized_rgb_tensor(image):
    """Shared exact evaluation normalization; caller supplies resized RGB pixels."""
    img = np.asarray(image, dtype=np.float32) / 255.0
    img = ((img - MEAN) / STD).transpose(2, 0, 1).copy()
    return torch.from_numpy(img)


def validate_patient_splits(train_csv, val_csv, test_csv):
    splits = {}
    for name, path in (("train", train_csv), ("val", val_csv), ("test", test_csv)):
        if path is None or not Path(path).exists():
            continue
        df = pd.read_csv(path)
        if "patient_id" not in df:
            raise ValueError(f"patient_id missing from {path}")
        splits[name] = set(df.patient_id.astype(str))
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = splits[a] & splits[b]
            if overlap:
                raise ValueError(f"Patient leakage between {a} and {b}: {sorted(overlap)[:5]}")
    return {name: len(ids) for name, ids in splits.items()}
