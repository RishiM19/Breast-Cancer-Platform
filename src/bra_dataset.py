import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 0  # Safe for Windows


# BUS-BRA is a binary classification task for this stage.
CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}

INDEX_TO_CLASS = {
    0: "Benign",
    1: "Malignant",
}


# ============================================================
# DATASET
# ============================================================

class BRADataset(Dataset):

    def __init__(
        self,
        csv_file,
        root_dir=".",
        transform=None,
    ):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        required_columns = {
            "image_path",
            "pathology",
            "case_id",
        }

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                f"BUS-BRA CSV is missing columns: {sorted(missing)}"
            )

        # Normalize pathology labels.
        self.data["pathology"] = (
            self.data["pathology"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        invalid_labels = (
            set(self.data["pathology"].unique())
            - set(CLASS_TO_INDEX.keys())
        )

        if invalid_labels:
            raise ValueError(
                f"Unknown BUS-BRA pathology labels: "
                f"{sorted(invalid_labels)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = str(row["image_path"])

        if not os.path.isabs(image_path):
            image_path = os.path.join(
                self.root_dir,
                image_path,
            )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:
            raise RuntimeError(
                f"Could not open image:\n{image_path}"
            ) from exc

        pathology = (
            str(row["pathology"])
            .strip()
            .lower()
        )

        label = CLASS_TO_INDEX[pathology]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

def get_bra_transforms():

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return train_transform, eval_transform


# ============================================================
# DATALOADERS
# ============================================================

def get_bra_dataloaders(
    data_dir="data/processed",
    batch_size=BATCH_SIZE,
):

    train_transform, eval_transform = (
        get_bra_transforms()
    )

    train_dataset = BRADataset(
        csv_file=os.path.join(
            data_dir,
            "bra_train.csv",
        ),
        root_dir=".",
        transform=train_transform,
    )

    val_dataset = BRADataset(
        csv_file=os.path.join(
            data_dir,
            "bra_val.csv",
        ),
        root_dir=".",
        transform=eval_transform,
    )

    test_dataset = BRADataset(
        csv_file=os.path.join(
            data_dir,
            "bra_test.csv",
        ),
        root_dir=".",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("LUMINA — BUS-BRA DATALOADER TEST")
    print("=" * 65)

    try:

        (
            train_loader,
            val_loader,
            test_loader,
        ) = get_bra_dataloaders()

        print(
            f"\nTrain images: "
            f"{len(train_loader.dataset)}"
        )

        print(
            f"Validation images: "
            f"{len(val_loader.dataset)}"
        )

        print(
            f"Test images: "
            f"{len(test_loader.dataset)}"
        )

        images, labels = next(
            iter(train_loader)
        )

        print(
            "\nSUCCESS!"
        )

        print(
            f"Image batch shape: "
            f"{images.shape}"
        )

        print(
            f"Label batch shape: "
            f"{labels.shape}"
        )

        print(
            f"Labels: "
            f"{labels.tolist()}"
        )

        print(
            "\nClass mapping:"
        )

        for index, name in INDEX_TO_CLASS.items():

            print(
                f"  {index} -> {name}"
            )

    except Exception as exc:

        print(
            "\nERROR:"
        )

        print(exc)