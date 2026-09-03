from pathlib import Path
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

UCLM_MANIFEST = PROCESSED_DIR / "uclm_manifest.csv"
BRA_MANIFEST = PROCESSED_DIR / "bra_manifest.csv"


RANDOM_SEED = 42


# ============================================================
# UCLM
# ============================================================

def create_uclm_splits(df: pd.DataFrame):

    print("\n" + "=" * 70)
    print("CREATING BUS-UCLM PATIENT-LEVEL SPLITS")
    print("=" * 70)

    required = {
        "patient_id",
        "image_path",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"UCLM manifest is missing columns: {sorted(missing)}"
        )

    if df["patient_id"].isna().any() or (
        df["patient_id"].astype(str).str.strip() == ""
    ).any():
        raise ValueError(
            "UCLM contains missing patient IDs."
        )

    patients = (
        df["patient_id"]
        .drop_duplicates()
        .tolist()
    )

    print(f"Total patients: {len(patients)}")
    print(f"Total images:   {len(df)}")

    # --------------------------------------------------------
    # Per-patient class counts
    # --------------------------------------------------------

    patient_stats = (
        pd.crosstab(
            df["patient_id"],
            df["label"]
        )
        .fillna(0)
    )

    for cls in ["Normal", "Benign", "Malignant"]:
        if cls not in patient_stats.columns:
            patient_stats[cls] = 0

    patient_stats = patient_stats[
        ["Normal", "Benign", "Malignant"]
    ]

    patient_stats["Total"] = patient_stats.sum(axis=1)

    # --------------------------------------------------------
    # Desired number of patients
    # --------------------------------------------------------

    n_train = 26
    n_val = 6
    n_test = 6

    if n_train + n_val + n_test != len(patients):
        raise ValueError(
            "UCLM patient split counts do not equal total patients."
        )

    target_ratios = {
        "train": n_train / len(patients),
        "val": n_val / len(patients),
        "test": n_test / len(patients),
    }

    # Target image-level class counts
    total_class_counts = (
        df["label"]
        .value_counts()
        .to_dict()
    )

    target_class = {
        split: {
            cls: total_class_counts.get(cls, 0)
            * target_ratios[split]
            for cls in ["Normal", "Benign", "Malignant"]
        }
        for split in ["train", "val", "test"]
    }

    total_images = len(df)

    target_images = {
        "train": total_images * target_ratios["train"],
        "val": total_images * target_ratios["val"],
        "test": total_images * target_ratios["test"],
    }

    # --------------------------------------------------------
    # Randomized search for a good patient allocation
    # --------------------------------------------------------

    rng = random.Random(RANDOM_SEED)

    patient_list = patients.copy()

    best_assignment = None
    best_score = float("inf")

    iterations = 100_000

    for _ in range(iterations):

        rng.shuffle(patient_list)

        train_patients = patient_list[:n_train]
        val_patients = patient_list[
            n_train:n_train + n_val
        ]
        test_patients = patient_list[
            n_train + n_val:
        ]

        splits = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
        }

        score = 0.0

        for split_name, split_patients in splits.items():

            stats = patient_stats.loc[
                split_patients
            ]

            # Image-count deviation
            actual_images = stats["Total"].sum()

            image_error = (
                abs(actual_images - target_images[split_name])
                / total_images
            )

            score += image_error * 5

            # Class-count deviation
            for cls in [
                "Normal",
                "Benign",
                "Malignant",
            ]:

                actual = stats[cls].sum()
                target = target_class[
                    split_name
                ][cls]

                class_error = (
                    abs(actual - target)
                    / max(total_class_counts.get(cls, 1), 1)
                )

                score += class_error * 10

            # Penalize missing classes in validation/test
            if split_name in {"val", "test"}:

                for cls in [
                    "Normal",
                    "Benign",
                    "Malignant",
                ]:

                    if stats[cls].sum() == 0:
                        score += 100

        if score < best_score:

            best_score = score

            best_assignment = {
                key: value.copy()
                for key, value in splits.items()
            }

    if best_assignment is None:
        raise RuntimeError(
            "Failed to find a valid UCLM split."
        )

    # --------------------------------------------------------
    # Expand patients -> images
    # --------------------------------------------------------

    split_frames = {}

    for split_name, split_patients in (
        best_assignment.items()
    ):

        split_frames[split_name] = df[
            df["patient_id"].isin(split_patients)
        ].copy()

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nSelected patient counts:")

    for split_name, split_patients in (
        best_assignment.items()
    ):
        print(
            f"  {split_name:<5}: "
            f"{len(split_patients)} patients"
        )

    print("\nImage counts:")

    for split_name, frame in (
        split_frames.items()
    ):

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"Images: {len(frame)}"
        )

        print(
            frame["label"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    train_ids = set(
        best_assignment["train"]
    )

    val_ids = set(
        best_assignment["val"]
    )

    test_ids = set(
        best_assignment["test"]
    )

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert (
        train_ids
        | val_ids
        | test_ids
    ) == set(patients)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    paths = {
        "train": PROCESSED_DIR / "uclm_train.csv",
        "val": PROCESSED_DIR / "uclm_val.csv",
        "test": PROCESSED_DIR / "uclm_test.csv",
    }

    for split_name, path in paths.items():

        split_frames[split_name].to_csv(
            path,
            index=False
        )

        print(
            f"\nSaved {split_name}:"
            f" {path}"
        )

    return split_frames


# ============================================================
# BUS-BRA
# ============================================================

def create_bra_splits(df: pd.DataFrame):

    print("\n" + "=" * 70)
    print("CREATING BUS-BRA CASE-LEVEL SPLITS")
    print("=" * 70)

    required = {
        "case_id",
        "image_path",
        "pathology",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"BUS-BRA manifest is missing columns: {sorted(missing)}"
        )

    # One pathology label per case
    case_table = (
        df.groupby("case_id")["pathology"]
        .first()
        .reset_index()
    )

    case_ids = case_table["case_id"]

    labels = case_table["pathology"]

    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_cases, temp_cases = train_test_split(
        case_ids,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    temp_labels = (
        case_table
        .set_index("case_id")
        .loc[temp_cases, "pathology"]
    )

    # --------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    val_cases, test_cases = train_test_split(
        temp_cases,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

    train_cases = set(train_cases)
    val_cases = set(val_cases)
    test_cases = set(test_cases)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    assert train_cases.isdisjoint(val_cases)
    assert train_cases.isdisjoint(test_cases)
    assert val_cases.isdisjoint(test_cases)

    assert (
        train_cases
        | val_cases
        | test_cases
    ) == set(case_ids)

    split_frames = {
        "train": df[
            df["case_id"].isin(train_cases)
        ].copy(),

        "val": df[
            df["case_id"].isin(val_cases)
        ].copy(),

        "test": df[
            df["case_id"].isin(test_cases)
        ].copy(),
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nCase counts:")

    for split_name, cases in [
        ("train", train_cases),
        ("val", val_cases),
        ("test", test_cases),
    ]:

        print(
            f"  {split_name:<5}: "
            f"{len(cases)} cases"
        )

    print("\nImage/pathology distribution:")

    for split_name, frame in (
        split_frames.items()
    ):

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"Images: {len(frame)}"
        )

        print(
            frame["pathology"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    paths = {
        "train": PROCESSED_DIR / "bra_train.csv",
        "val": PROCESSED_DIR / "bra_val.csv",
        "test": PROCESSED_DIR / "bra_test.csv",
    }

    for split_name, path in paths.items():

        split_frames[split_name].to_csv(
            path,
            index=False
        )

        print(
            f"\nSaved {split_name}:"
            f" {path}"
        )

    return split_frames


# ============================================================
# MAIN
# ============================================================

def main():

    if not UCLM_MANIFEST.exists():
        raise FileNotFoundError(
            UCLM_MANIFEST
        )

    if not BRA_MANIFEST.exists():
        raise FileNotFoundError(
            BRA_MANIFEST
        )

    print("=" * 70)
    print("LUMINA — PATIENT/CASE LEVEL DATA SPLITTING")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\nOriginal images, masks, BUSI data, models,"
        "\nand application files are not modified."
    )

    uclm = pd.read_csv(
        UCLM_MANIFEST
    )

    bra = pd.read_csv(
        BRA_MANIFEST
    )

    create_uclm_splits(uclm)

    create_bra_splits(bra)

    print("\n" + "=" * 70)
    print("ALL SPLITS CREATED SUCCESSFULLY")
    print("=" * 70)

    print("\nNew files:")

    for filename in [
        "uclm_train.csv",
        "uclm_val.csv",
        "uclm_test.csv",
        "bra_train.csv",
        "bra_val.csv",
        "bra_test.csv",
    ]:
        print(
            f"  data/processed/{filename}"
        )


if __name__ == "__main__":
    main()