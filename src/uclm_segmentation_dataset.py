import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0

# UCLM annotation encoding discovered from the dataset:
#
# Black  -> background
# Green  -> benign lesion
# Red    -> malignant lesion

BACKGROUND_RGB = (0, 0, 0)
BENIGN_RGB = (0, 255, 0)
MALIGNANT_RGB = (255, 0, 0)


CLASS_NAMES = [
    "Background",
    "Benign",
    "Malignant",
]


# ============================================================
# DATASET
# ============================================================

class UCLMSegmentationDataset(Dataset):

    def __init__(
        self,
        csv_file,
        root_dir=".",
        image_transform=None,
        mask_transform=None,
    ):

        self.data = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        required_columns = {
            "image_path",
            "mask_path",
            "label",
            "patient_id",
        }

        missing = (
            required_columns
            - set(self.data.columns)
        )

        if missing:

            raise ValueError(
                "UCLM segmentation CSV is missing "
                f"columns: {sorted(missing)}"
            )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = str(
            row["image_path"]
        )

        mask_path = str(
            row["mask_path"]
        )

        if not os.path.isabs(image_path):

            image_path = os.path.join(
                self.root_dir,
                image_path
            )

        if not os.path.isabs(mask_path):

            mask_path = os.path.join(
                self.root_dir,
                mask_path
            )

        if not os.path.isfile(image_path):

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        if not os.path.isfile(mask_path):

            raise FileNotFoundError(
                f"Mask not found:\n{mask_path}"
            )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask_rgb = Image.open(
            mask_path
        ).convert("RGB")

        mask_array = np.asarray(
            mask_rgb
        )

        # Create integer class map.
        #
        # 0 = Background
        # 1 = Benign
        # 2 = Malignant

        mask = np.zeros(
            mask_array.shape[:2],
            dtype=np.uint8
        )

        benign_pixels = np.all(
            mask_array == BENIGN_RGB,
            axis=-1
        )

        malignant_pixels = np.all(
            mask_array == MALIGNANT_RGB,
            axis=-1
        )

        mask[benign_pixels] = 1
        mask[malignant_pixels] = 2

        mask = Image.fromarray(
            mask,
            mode="L"
        )

        # ----------------------------------------------------
        # TRANSFORMS
        # ----------------------------------------------------

        if self.image_transform is not None:

            image = self.image_transform(
                image
            )

        if self.mask_transform is not None:

            mask = self.mask_transform(
                mask
            )

        # Remove channel dimension.
        #
        # Result:
        # [256, 256]
        #
        # Pixel values:
        # 0, 1, 2

        mask = mask.squeeze(0).long()

        return image, mask


# ============================================================
# TRANSFORMS
# ============================================================

def get_segmentation_transforms():

    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
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

    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=(
                    transforms
                    .InterpolationMode
                    .NEAREST
                ),
            ),

            transforms.ToTensor(),
        ]
    )

    return (
        image_transform,
        mask_transform,
    )


# ============================================================
# DATALOADERS
# ============================================================

def get_uclm_segmentation_dataloaders(
    data_dir="data/processed",
    batch_size=BATCH_SIZE,
):

    (
        image_transform,
        mask_transform,
    ) = get_segmentation_transforms()

    train_dataset = UCLMSegmentationDataset(
        csv_file=os.path.join(
            data_dir,
            "uclm_train.csv",
        ),
        root_dir=".",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    val_dataset = UCLMSegmentationDataset(
        csv_file=os.path.join(
            data_dir,
            "uclm_val.csv",
        ),
        root_dir=".",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    test_dataset = UCLMSegmentationDataset(
        csv_file=os.path.join(
            data_dir,
            "uclm_test.csv",
        ),
        root_dir=".",
        image_transform=image_transform,
        mask_transform=mask_transform,
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

    print("=" * 70)
    print(
        "LUMINA — UCLM MULTICLASS SEGMENTATION "
        "DATALOADER TEST"
    )
    print("=" * 70)

    print(
        "\nMask classes:"
    )

    print(
        "  0 -> Background"
    )

    print(
        "  1 -> Benign"
    )

    print(
        "  2 -> Malignant"
    )

    try:

        (
            train_loader,
            val_loader,
            test_loader,
        ) = get_uclm_segmentation_dataloaders(
            batch_size=BATCH_SIZE
        )

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

        images, masks = next(
            iter(train_loader)
        )

        print("\nSUCCESS!")

        print(
            f"Image batch shape: "
            f"{images.shape}"
        )

        print(
            f"Mask batch shape: "
            f"{masks.shape}"
        )

        print(
            f"Mask dtype: "
            f"{masks.dtype}"
        )

        unique_values = torch.unique(
            masks
        ).cpu().tolist()

        print(
            f"Unique mask values in batch: "
            f"{unique_values}"
        )

        print(
            f"Total background pixels: "
            f"{(masks == 0).sum().item():,}"
        )

        print(
            f"Total benign pixels: "
            f"{(masks == 1).sum().item():,}"
        )

        print(
            f"Total malignant pixels: "
            f"{(masks == 2).sum().item():,}"
        )

    except Exception as exc:

        print("\nERROR:")
        print(exc)