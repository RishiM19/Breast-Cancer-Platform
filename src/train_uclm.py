import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from uclm_dataset import get_uclm_dataloaders


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/uclm_best_model.pth"

NUM_CLASSES = 3


# ============================================================
# MODEL
# ============================================================

def build_uclm_model(num_classes=NUM_CLASSES):
    from torchvision.models import (
        efficientnet_b0,
        EfficientNet_B0_Weights,
    )

    print("\nLoading ImageNet-pretrained EfficientNet-B0...")

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        num_classes
    )

    return model.to(DEVICE)


# ============================================================
# TRAINING
# ============================================================

def train_uclm_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=NUM_EPOCHS,
):

    start_time = time.time()

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
    }

    # Modern AMP API
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    for epoch in range(num_epochs):

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}"
        )

        print("-" * 60)

        # ====================================================
        # TRAIN + VALIDATION
        # ====================================================

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                loader = train_loader

            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_correct = 0

            all_labels = []
            all_predictions = []

            for images, labels in tqdm(
                loader,
                desc=f"{phase.capitalize()} phase",
            ):

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.set_grad_enabled(
                    phase == "train"
                ):

                    with torch.amp.autocast(
                        device_type=DEVICE.type,
                        enabled=(DEVICE.type == "cuda"),
                    ):

                        outputs = model(images)

                        loss = criterion(
                            outputs,
                            labels
                        )

                        predictions = torch.argmax(
                            outputs,
                            dim=1
                        )

                    if phase == "train":

                        scaler.scale(
                            loss
                        ).backward()

                        scaler.step(
                            optimizer
                        )

                        scaler.update()

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

                running_correct += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

                all_labels.extend(
                    labels.detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                all_predictions.extend(
                    predictions.detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

            epoch_loss = (
                running_loss
                / len(loader.dataset)
            )

            epoch_accuracy = (
                running_correct
                / len(loader.dataset)
            )

            # ------------------------------------------------
            # Balanced accuracy
            # ------------------------------------------------

            if phase == "val":

                balanced_acc = (
                    balanced_accuracy_score(
                        all_labels,
                        all_predictions,
                    )
                )

            else:

                balanced_acc = None

            print(
                f"{phase.upper()} "
                f"Loss: {epoch_loss:.4f} "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

            if balanced_acc is not None:

                print(
                    f"Validation Balanced Accuracy: "
                    f"{balanced_acc:.4f}"
                )

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            if phase == "train":

                history[
                    "train_loss"
                ].append(epoch_loss)

                history[
                    "train_accuracy"
                ].append(epoch_accuracy)

            else:

                history[
                    "val_loss"
                ].append(epoch_loss)

                history[
                    "val_accuracy"
                ].append(epoch_accuracy)

                history[
                    "val_balanced_accuracy"
                ].append(balanced_acc)

                # ------------------------------------------------
                # Save best model
                # ------------------------------------------------

                if epoch_accuracy > best_val_accuracy:

                    best_val_accuracy = epoch_accuracy

                    best_model_weights = (
                        copy.deepcopy(
                            model.state_dict()
                        )
                    )

                    os.makedirs(
                        "models",
                        exist_ok=True
                    )

                    torch.save(
                        model.state_dict(),
                        MODEL_PATH,
                    )

                    print(
                        "\n>>> NEW BEST UCLM MODEL SAVED"
                    )

                    print(
                        f"Validation accuracy: "
                        f"{best_val_accuracy:.4f}"
                    )

                    print(
                        f"Saved to: {MODEL_PATH}"
                    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)

    print(
        f"Training complete in "
        f"{elapsed // 60:.0f}m "
        f"{elapsed % 60:.0f}s"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    # Restore best weights
    model.load_state_dict(
        best_model_weights
    )

    return model, history


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LUMINA — BUS-UCLM EFFICIENTNET-B0 TRAINING")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    print(
        "\nExisting BUSI model will NOT be modified."
    )

    print(
        f"UCLM checkpoint: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print(
        "\nInitializing UCLM DataLoaders..."
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_uclm_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(
        f"Train images: "
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nBuilding model..."
    )

    model = build_uclm_model()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nStarting UCLM training..."
    )

    model, history = train_uclm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
    )

    print(
        "\nUCLM training finished."
    )

    print(
        f"Best model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()