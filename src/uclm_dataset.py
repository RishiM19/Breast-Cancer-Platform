import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 0

CLASS_TO_INDEX = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2,
}

INDEX_TO_CLASS = {
    0: "Normal",
    1: "Benign",
    2: "Malignant",
}


class UCLMDataset(Dataset):
    def __init__(self, csv_file, root_dir=".", transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        required_columns = {
            "image_path",
            "label",
            "patient_id",
        }

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                f"UCLM CSV is missing columns: {sorted(missing)}"
            )

        invalid_labels = (
            set(self.data["label"].unique())
            - set(CLASS_TO_INDEX.keys())
        )

        if invalid_labels:
            raise ValueError(
                f"Unknown UCLM labels: {sorted(invalid_labels)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = str(row["image_path"])

        if not os.path.isabs(image_path):
            image_path = os.path.join(
                self.root_dir,
                image_path
            )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        label_name = str(row["label"]).strip()

        label = CLASS_TO_INDEX[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_uclm_transforms():

    train_transform = transforms.Compose([
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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def get_uclm_dataloaders(
    data_dir="data/processed",
    batch_size=BATCH_SIZE,
):

    train_transform, eval_transform = (
        get_uclm_transforms()
    )

    train_dataset = UCLMDataset(
        csv_file=os.path.join(
            data_dir,
            "uclm_train.csv"
        ),
        root_dir=".",
        transform=train_transform,
    )

    val_dataset = UCLMDataset(
        csv_file=os.path.join(
            data_dir,
            "uclm_val.csv"
        ),
        root_dir=".",
        transform=eval_transform,
    )

    test_dataset = UCLMDataset(
        csv_file=os.path.join(
            data_dir,
            "uclm_test.csv"
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


if __name__ == "__main__":

    print("Testing BUS-UCLM DataLoader...")

    try:

        train_loader, val_loader, test_loader = (
            get_uclm_dataloaders()
        )

        images, labels = next(
            iter(train_loader)
        )

        print("\nSUCCESS!")
        print(
            f"Image batch shape: {images.shape}"
        )
        print(
            f"Label batch shape: {labels.shape}"
        )
        print(
            f"Labels: {labels.tolist()}"
        )

        print(
            "\nClass mapping:"
        )

        for index, name in INDEX_TO_CLASS.items():
            print(
                f"  {index} -> {name}"
            )

    except Exception as exc:

        print("\nERROR:")
        print(exc)