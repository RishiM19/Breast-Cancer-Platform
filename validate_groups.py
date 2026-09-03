from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

UCLM_PATH = PROCESSED_DIR / "uclm_manifest.csv"
BRA_PATH = PROCESSED_DIR / "bra_manifest.csv"


def validate_uclm(df):
    print("\n" + "=" * 65)
    print("BUS-UCLM PATIENT SUMMARY")
    print("=" * 65)

    print(f"Total images    : {len(df)}")
    print(f"Unique patients : {df['patient_id'].nunique()}")

    # Patient × diagnostic class
    composition = pd.crosstab(
        df["patient_id"],
        df["label"]
    )

    for col in ["Normal", "Benign", "Malignant"]:
        if col not in composition.columns:
            composition[col] = 0

    composition = composition[
        ["Normal", "Benign", "Malignant"]
    ]

    composition["Total"] = composition.sum(axis=1)

    def profile(row):
        classes = []

        if row["Normal"] > 0:
            classes.append("N")

        if row["Benign"] > 0:
            classes.append("B")

        if row["Malignant"] > 0:
            classes.append("M")

        return "+".join(classes)

    composition["Profile"] = composition.apply(
        profile,
        axis=1
    )

    # Profile counts only
    profile_counts = composition["Profile"].value_counts()

    print("\nPatient profile distribution:")
    for profile_name, count in profile_counts.sort_index().items():
        print(f"  {profile_name:<7} : {count}")

    # Patient-level class presence
    print("\nPatients containing each class:")

    for label in ["Normal", "Benign", "Malignant"]:
        count = (composition[label] > 0).sum()
        print(f"  {label:<10}: {count}")

    # Images per patient
    print("\nImages per patient:")
    print(
        f"  Min    : {composition['Total'].min()}"
    )
    print(
        f"  Median : {composition['Total'].median():.1f}"
    )
    print(
        f"  Mean   : {composition['Total'].mean():.2f}"
    )
    print(
        f"  Max    : {composition['Total'].max()}"
    )

    print("\nUCLM patient composition check complete.")


def validate_bra(df):
    print("\n" + "=" * 65)
    print("BUS-BRA CASE SUMMARY")
    print("=" * 65)

    print(f"Total images : {len(df)}")
    print(f"Unique cases : {df['case_id'].nunique()}")

    # Case × pathology
    composition = pd.crosstab(
        df["case_id"],
        df["pathology"]
    )

    for col in ["benign", "malignant"]:
        if col not in composition.columns:
            composition[col] = 0

    composition = composition[
        ["benign", "malignant"]
    ]

    composition["Total"] = composition.sum(axis=1)

    def profile(row):
        if row["benign"] > 0 and row["malignant"] > 0:
            return "B+M"
        elif row["benign"] > 0:
            return "B"
        elif row["malignant"] > 0:
            return "M"
        return "Unknown"

    composition["Profile"] = composition.apply(
        profile,
        axis=1
    )

    print("\nCase profile distribution:")
    for profile_name, count in (
        composition["Profile"]
        .value_counts()
        .sort_index()
        .items()
    ):
        print(f"  {profile_name:<7} : {count}")

    print("\nCases containing each pathology:")

    for label in ["benign", "malignant"]:
        count = (composition[label] > 0).sum()
        print(f"  {label:<10}: {count}")

    print("\nImages per case:")
    print(
        f"  Min    : {composition['Total'].min()}"
    )
    print(
        f"  Median : {composition['Total'].median():.1f}"
    )
    print(
        f"  Mean   : {composition['Total'].mean():.2f}"
    )
    print(
        f"  Max    : {composition['Total'].max()}"
    )

    print("\nBUS-BRA case composition check complete.")


def main():

    if not UCLM_PATH.exists():
        raise FileNotFoundError(UCLM_PATH)

    if not BRA_PATH.exists():
        raise FileNotFoundError(BRA_PATH)

    uclm = pd.read_csv(UCLM_PATH)
    bra = pd.read_csv(BRA_PATH)

    validate_uclm(uclm)
    validate_bra(bra)

    print("\n" + "=" * 65)
    print("GROUP ANALYSIS COMPLETE")
    print("=" * 65)
    print("No datasets were modified.")


if __name__ == "__main__":
    main()