import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from bra_dataset import get_bra_dataloaders


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50

PATIENCE = 8
MIN_DELTA = 0.001

NUM_CLASSES = 2

CLASS_NAMES = [
    "Benign",
    "Malignant",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/bra_best_model.pth"


# ============================================================
# MODEL
# ============================================================

def build_bra_model(num_classes=NUM_CLASSES):

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
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_loader):

    print("\nCalculating class weights...")

    class_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    for _, labels in train_loader:

        labels_np = labels.numpy()

        for class_idx in range(NUM_CLASSES):

            class_counts[class_idx] += np.sum(
                labels_np == class_idx
            )

    print("\nTraining class counts:")

    for idx, name in enumerate(CLASS_NAMES):

        print(
            f"  {name:<10}: "
            f"{class_counts[idx]}"
        )

    total = class_counts.sum()

    weights = total / (
        NUM_CLASSES * class_counts
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print("\nCalculated class weights:")

    for idx, name in enumerate(CLASS_NAMES):

        print(
            f"  {name:<10}: "
            f"{weights[idx].item():.4f}"
        )

    return weights


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
):

    start_time = time.time()

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    best_balanced_accuracy = 0.0

    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
    }

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    for epoch in range(MAX_EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{MAX_EPOCHS}"
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
                desc=f"{phase.capitalize()} phase"
            ):

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.set_grad_enabled(
                    phase == "train"
                ):

                    with torch.amp.autocast(
                        device_type=DEVICE.type,
                        enabled=(DEVICE.type == "cuda")
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

            if phase == "val":

                balanced_accuracy = (
                    balanced_accuracy_score(
                        all_labels,
                        all_predictions
                    )
                )

            else:

                balanced_accuracy = None

            print(
                f"{phase.upper()} "
                f"Loss: {epoch_loss:.4f} "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

            if balanced_accuracy is not None:

                print(
                    "Validation Balanced Accuracy: "
                    f"{balanced_accuracy:.4f}"
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
                ].append(balanced_accuracy)

                # ------------------------------------------------
                # BEST MODEL
                # ------------------------------------------------

                if (
                    balanced_accuracy
                    > best_balanced_accuracy
                    + MIN_DELTA
                ):

                    best_balanced_accuracy = (
                        balanced_accuracy
                    )

                    best_model_weights = (
                        copy.deepcopy(
                            model.state_dict()
                        )
                    )

                    epochs_without_improvement = 0

                    os.makedirs(
                        "models",
                        exist_ok=True
                    )

                    torch.save(
                        model.state_dict(),
                        MODEL_PATH
                    )

                    print(
                        "\n>>> NEW BEST BUS-BRA MODEL"
                    )

                    print(
                        f"Balanced accuracy: "
                        f"{balanced_accuracy:.4f}"
                    )

                    print(
                        f"Saved to: "
                        f"{MODEL_PATH}"
                    )

                else:

                    epochs_without_improvement += 1

                    print(
                        f"No improvement "
                        f"({epochs_without_improvement}/"
                        f"{PATIENCE})"
                    )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if epochs_without_improvement >= PATIENCE:

            print(
                "\n>>> EARLY STOPPING"
            )

            print(
                "Validation balanced accuracy "
                f"did not improve for {PATIENCE} epochs."
            )

            break

    elapsed = time.time() - start_time

    model.load_state_dict(
        best_model_weights
    )

    print("\n" + "=" * 60)

    print(
        f"Training complete in "
        f"{elapsed // 60:.0f}m "
        f"{elapsed % 60:.0f}s"
    )

    print(
        f"Best validation balanced accuracy: "
        f"{best_balanced_accuracy:.4f}"
    )

    print(
        f"Best model saved at: "
        f"{MODEL_PATH}"
    )

    return model, history


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LUMINA — BUS-BRA EFFICIENTNET-B0 TRAINING")
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
        "\nBUSI and UCLM models will NOT be modified."
    )

    print(
        f"Checkpoint: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "\nInitializing BUS-BRA DataLoaders..."
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_bra_dataloaders(
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
    # WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nBuilding BUS-BRA model..."
    )

    model = build_bra_model()

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nStarting BUS-BRA training..."
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    print(
        "\nBUS-BRA training finished."
    )

    print(
        f"Best model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()