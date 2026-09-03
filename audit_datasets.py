"""
Lumina Breast AI
Step 1: BUS-UCLM and BUS-BRA dataset audit + manifest generation

IMPORTANT:
- Read-only with respect to original datasets.
- Does NOT modify BUSI.
- Does NOT move, rename, resize, or overwrite images/masks.
- Creates only:
    data/processed/uclm_manifest.csv
    data/processed/bra_manifest.csv

Run from the Breast-Cancer-Platform project root:

    python src/audit_datasets.py
"""

from __future__ import annotations

import ast
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# BUS-UCLM
UCLM_ROOT = (
    DATA_DIR
    / "BUS_UCLM"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset"
)

UCLM_IMAGES_DIR = UCLM_ROOT / "images"
UCLM_MASKS_DIR = UCLM_ROOT / "masks"
UCLM_INFO_CSV = UCLM_ROOT / "INFO.csv"

# BUS-BRA
BRA_ROOT = DATA_DIR / "BUS_BRA" / "BUSBRA"

BRA_IMAGES_DIR = BRA_ROOT / "Images"
BRA_MASKS_DIR = BRA_ROOT / "Masks"
BRA_CSV = BRA_ROOT / "bus_data.csv"


# ============================================================
# GENERAL HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def normalize_text(value) -> str:
    """Convert a CSV cell into a clean string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def relative_path(path: Optional[Path]) -> str:
    """Return a project-relative path, or empty string."""
    if path is None:
        return ""

    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """
    Calculate a SHA256 hash for duplicate detection.

    Returns None if the file cannot be read.
    """
    try:
        sha = hashlib.sha256()

        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                sha.update(chunk)

        return sha.hexdigest()

    except OSError as exc:
        print(f"WARNING: Could not hash {path}: {exc}")
        return None


def get_image_info(path: Path) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Read image dimensions and mode.

    Returns:
        width, height, mode
    """
    try:
        with Image.open(path) as img:
            return img.width, img.height, img.mode

    except Exception as exc:
        print(f"WARNING: Could not read image {path}: {exc}")
        return None, None, None


def build_file_index(directory: Path) -> dict[str, list[Path]]:
    """
    Recursively index files by lowercase filename stem.

    Example:
        ALWI_001.png -> {"alwi_001": [Path(...)]}
    """
    index: dict[str, list[Path]] = defaultdict(list)

    if not directory.exists():
        print(f"WARNING: Directory does not exist: {directory}")
        return index

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.stem.lower()].append(path)

    return index


def choose_matching_file(
    index: dict[str, list[Path]],
    filename: str,
) -> Optional[Path]:
    """
    Find an image/mask by filename stem.
    """
    stem = Path(filename).stem.lower()

    matches = index.get(stem, [])

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"WARNING: Multiple files found for '{filename}': "
            f"{[str(p) for p in matches]}"
        )

    return matches[0]


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_distribution(series: pd.Series, label: str) -> None:
    print(f"\n{label}:")

    counts = series.fillna("").astype(str).str.strip().value_counts(dropna=False)

    for value, count in counts.items():
        value_display = value if value else "<EMPTY>"
        print(f"  {value_display:<30} {count}")


def print_duplicate_summary(
    records: list[dict],
    hash_field: str = "image_sha256",
) -> None:
    """
    Print duplicate file groups based on content hash.
    """
    groups = defaultdict(list)

    for record in records:
        file_hash = record.get(hash_field)

        if file_hash:
            groups[file_hash].append(record["image_path"])

    duplicate_groups = [
        paths
        for paths in groups.values()
        if len(paths) > 1
    ]

    if duplicate_groups:
        print(f"\nDuplicate image groups: {len(duplicate_groups)}")

        for i, paths in enumerate(duplicate_groups[:20], start=1):
            print(f"\n  Group {i}:")
            for path in paths:
                print(f"    {path}")

        if len(duplicate_groups) > 20:
            print(
                f"\n  ... and {len(duplicate_groups) - 20} more duplicate groups."
            )

    else:
        print("\nDuplicate image groups: 0")


# ============================================================
# BUS-UCLM AUDIT
# ============================================================

def audit_uclm() -> pd.DataFrame:
    print_section("BUS-UCLM AUDIT")

    required_paths = [
        UCLM_ROOT,
        UCLM_IMAGES_DIR,
        UCLM_MASKS_DIR,
        UCLM_INFO_CSV,
    ]

    for path in required_paths:
        print(f"{'FOUND' if path.exists() else 'MISSING'}: {path}")

    if not UCLM_INFO_CSV.exists():
        raise FileNotFoundError(
            f"BUS-UCLM INFO.csv was not found:\n{UCLM_INFO_CSV}"
        )

    print("\nReading INFO.csv...")

    # BUS-UCLM uses semicolon-separated values.
    info = pd.read_csv(
        UCLM_INFO_CSV,
        sep=";",
        dtype=str,
        keep_default_na=False,
    )

    info.columns = [
        normalize_text(column)
        for column in info.columns
    ]

    print(f"Rows in INFO.csv: {len(info)}")
    print(f"Columns: {list(info.columns)}")

    required_columns = {
        "Image",
        "Resolution",
        "Label",
        "Doppler",
        "Marks",
        "Combined",
    }

    missing_columns = required_columns.difference(info.columns)

    if missing_columns:
        raise ValueError(
            "BUS-UCLM INFO.csv is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    image_index = build_file_index(UCLM_IMAGES_DIR)
    mask_index = build_file_index(UCLM_MASKS_DIR)

    print(f"\nIndexed image files: {sum(map(len, image_index.values()))}")
    print(f"Indexed mask files: {sum(map(len, mask_index.values()))}")

    records: list[dict] = []

    for _, row in info.iterrows():

        image_name = normalize_text(row["Image"])

        image_path = choose_matching_file(
            image_index,
            image_name,
        )

        mask_path = choose_matching_file(
            mask_index,
            image_name,
        )

        width = None
        height = None
        actual_mode = None
        image_hash = None

        if image_path is not None:

            width, height, actual_mode = get_image_info(
                image_path
            )

            image_hash = file_sha256(image_path)

        csv_resolution = normalize_text(row["Resolution"])

        records.append(
            {
                "dataset": "BUS-UCLM",
                "image": image_name,
                "image_path": relative_path(image_path),
                "mask_path": relative_path(mask_path),
                "label": normalize_text(row["Label"]),
                "resolution_csv": csv_resolution,
                "actual_width": width,
                "actual_height": height,
                "image_mode": actual_mode or "",
                "doppler": normalize_text(row["Doppler"]),
                "marks": normalize_text(row["Marks"]),
                "combined": normalize_text(row["Combined"]),
                "image_exists": image_path is not None,
                "mask_exists": mask_path is not None,
                "image_sha256": image_hash or "",
                # Patient ID is NOT invented here.
                # We will establish the correct grouping later.
                "patient_id": image_name.split("_")[0],
            }
        )

    manifest = pd.DataFrame(records)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(f"\nManifest rows: {len(manifest)}")

    print_distribution(
        manifest["label"],
        "Class distribution",
    )

    print_distribution(
        manifest["doppler"],
        "Doppler distribution",
    )

    print_distribution(
        manifest["marks"],
        "Marks distribution",
    )

    print_distribution(
        manifest["combined"],
        "Combined distribution",
    )

    missing_images = (~manifest["image_exists"]).sum()
    missing_masks = (~manifest["mask_exists"]).sum()

    print("\nMissing image files:", missing_images)
    print("Missing mask files:", missing_masks)

    # --------------------------------------------------------
    # Resolution consistency
    # --------------------------------------------------------

    actual_dimensions = (
        manifest[
            ["actual_width", "actual_height"]
        ]
        .dropna()
        .value_counts()
    )

    print("\nActual image dimensions:")

    for dimensions, count in actual_dimensions.items():
        print(f"  {dimensions}: {count}")

    # --------------------------------------------------------
    # Hash duplicates
    # --------------------------------------------------------

    print_duplicate_summary(manifest.to_dict("records"))

    return manifest


# ============================================================
# BUS-BRA AUDIT
# ============================================================

def parse_bbox(value: str) -> Optional[list]:
    """
    Safely parse BUS-BRA BBOX values.

    Example:
        "[91,24,103,79]"
    """
    value = normalize_text(value)

    if not value:
        return None

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple)):
            return list(parsed)

    except (ValueError, SyntaxError):
        pass

    return None


def audit_bra() -> pd.DataFrame:
    print_section("BUS-BRA AUDIT")

    required_paths = [
        BRA_ROOT,
        BRA_IMAGES_DIR,
        BRA_MASKS_DIR,
        BRA_CSV,
    ]

    for path in required_paths:
        print(f"{'FOUND' if path.exists() else 'MISSING'}: {path}")

    if not BRA_CSV.exists():
        raise FileNotFoundError(
            f"BUS-BRA bus_data.csv was not found:\n{BRA_CSV}"
        )

    print("\nReading bus_data.csv...")

    # BUS-BRA metadata is comma-separated.
    data = pd.read_csv(
        BRA_CSV,
        dtype=str,
        keep_default_na=False,
    )

    data.columns = [
        normalize_text(column)
        for column in data.columns
    ]

    print(f"Rows in bus_data.csv: {len(data)}")
    print(f"Columns: {list(data.columns)}")

    required_columns = {
        "ID",
        "Case",
        "Histology",
        "Pathology",
        "BIRADS",
        "Device",
        "Width",
        "Height",
        "Side",
        "HOB",
        "K5B",
        "K10B",
        "HOP",
        "K5P",
        "K10P",
        "BBOX",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "BUS-BRA bus_data.csv is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    image_index = build_file_index(BRA_IMAGES_DIR)
    mask_index = build_file_index(BRA_MASKS_DIR)

    print(f"\nIndexed image files: {sum(map(len, image_index.values()))}")
    print(f"Indexed mask files: {sum(map(len, mask_index.values()))}")

    records: list[dict] = []

    for _, row in data.iterrows():

        record_id = normalize_text(row["ID"])

        # Search using the exact ID first.
        image_path = choose_matching_file(
            image_index,
            record_id,
        )

        mask_name = record_id

        if mask_name.lower().startswith("bus_"):
            mask_name = "mask_" + mask_name[4:]

        mask_path = choose_matching_file(
            mask_index,
            mask_name,
)

        # If that doesn't work, try a few common alternatives.
        if image_path is None:

            possible_names = [
                record_id,
                f"{record_id}.png",
                f"{record_id}.jpg",
                f"{record_id}.jpeg",
            ]

            for candidate in possible_names:

                image_path = choose_matching_file(
                    image_index,
                    candidate,
                )

                if image_path is not None:
                    break

        width = None
        height = None
        actual_mode = None
        image_hash = None

        if image_path is not None:

            width, height, actual_mode = get_image_info(
                image_path
            )

            image_hash = file_sha256(image_path)

        bbox = parse_bbox(row["BBOX"])

        records.append(
            {
                "dataset": "BUS-BRA",
                "image_id": record_id,
                "image_path": relative_path(image_path),
                "mask_path": relative_path(mask_path),

                # Case is preserved as the grouping key.
                "case_id": normalize_text(row["Case"]),

                "histology": normalize_text(row["Histology"]),
                "pathology": normalize_text(row["Pathology"]),
                "birads": normalize_text(row["BIRADS"]),

                "device": normalize_text(row["Device"]),

                "csv_width": normalize_text(row["Width"]),
                "csv_height": normalize_text(row["Height"]),

                "actual_width": width,
                "actual_height": height,
                "image_mode": actual_mode or "",

                "side": normalize_text(row["Side"]),

                "HOB": normalize_text(row["HOB"]),
                "K5B": normalize_text(row["K5B"]),
                "K10B": normalize_text(row["K10B"]),
                "HOP": normalize_text(row["HOP"]),
                "K5P": normalize_text(row["K5P"]),
                "K10P": normalize_text(row["K10P"]),

                "bbox": str(bbox) if bbox is not None else "",

                "image_exists": image_path is not None,
                "mask_exists": mask_path is not None,

                "image_sha256": image_hash or "",
            }
        )

    manifest = pd.DataFrame(records)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(f"\nManifest rows: {len(manifest)}")

    print_distribution(
        manifest["pathology"],
        "Pathology distribution",
    )

    print_distribution(
        manifest["histology"],
        "Histology distribution",
    )

    print_distribution(
        manifest["birads"],
        "BI-RADS distribution",
    )

    print_distribution(
        manifest["device"],
        "Ultrasound device distribution",
    )

    print_distribution(
        manifest["side"],
        "Side distribution",
    )

    missing_images = (~manifest["image_exists"]).sum()
    missing_masks = (~manifest["mask_exists"]).sum()

    print("\nMissing image files:", missing_images)
    print("Missing mask files:", missing_masks)

    # --------------------------------------------------------
    # Case-level statistics
    # --------------------------------------------------------

    unique_cases = manifest["case_id"].replace("", pd.NA).nunique()

    print(f"\nUnique case IDs: {unique_cases}")

    images_per_case = (
        manifest
        .groupby("case_id", dropna=False)
        .size()
        .sort_values(ascending=False)
    )

    print("\nImages per case:")
    print(images_per_case.head(20).to_string())

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    actual_dimensions = (
        manifest[
            ["actual_width", "actual_height"]
        ]
        .dropna()
        .value_counts()
    )

    print("\nActual image dimensions:")

    for dimensions, count in actual_dimensions.items():
        print(f"  {dimensions}: {count}")

    # --------------------------------------------------------
    # Hash duplicates
    # --------------------------------------------------------

    print_duplicate_summary(manifest.to_dict("records"))

    # --------------------------------------------------------
    # Pathology + BI-RADS relationship
    # --------------------------------------------------------

    print("\nPathology × BI-RADS:")

    cross_tab = pd.crosstab(
        manifest["pathology"],
        manifest["birads"],
        dropna=False,
    )

    print(cross_tab.to_string())

    return manifest


# ============================================================
# SAVE MANIFESTS
# ============================================================

def save_manifest(
    manifest: pd.DataFrame,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved manifest:")
    print(f"  {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print_section("LUMINA BREAST AI — DATASET AUDIT")

    print("Project root:")
    print(ROOT_DIR)

    print("\nExisting BUSI data/model are NOT modified.")

    try:

        uclm_manifest = audit_uclm()

        bra_manifest = audit_bra()

        save_manifest(
            uclm_manifest,
            PROCESSED_DIR / "uclm_manifest.csv",
        )

        save_manifest(
            bra_manifest,
            PROCESSED_DIR / "bra_manifest.csv",
        )

    except Exception as exc:

        print("\n" + "=" * 80)
        print("AUDIT FAILED")
        print("=" * 80)

        print(f"{type(exc).__name__}: {exc}")

        return 1

    print_section("AUDIT COMPLETE")

    print("Created:")
    print(f"  {PROCESSED_DIR / 'uclm_manifest.csv'}")
    print(f"  {PROCESSED_DIR / 'bra_manifest.csv'}")

    print("\nNo original image, mask, CSV, model, database, or application file")
    print("was modified by this script.")

    return 0


if __name__ == "__main__":
    sys.exit(main())    