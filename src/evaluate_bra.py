import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from bra_dataset import get_bra_dataloaders
from train_bra import build_bra_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/bra_best_model.pth"

CLASS_NAMES = [
    "Benign",
    "Malignant",
]

BATCH_SIZE = 16


def evaluate_bra():

    print("=" * 70)
    print("LUMINA — BUS-BRA TEST EVALUATION")
    print("=" * 70)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"\nDevice: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    # --------------------------------------------------------
    # TEST DATA ONLY
    # --------------------------------------------------------

    _, _, test_loader = get_bra_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(
        f"Test images: {len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading BUS-BRA model...")

    model = build_bra_model(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    print(
        "\nRunning predictions on untouched BUS-BRA test set..."
    )

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # --------------------------------------------------------
    # OVERALL PERFORMANCE
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions,
        )
    )

    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy:          "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f} "
        f"({balanced_accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\nRows = Actual"
        "\nColumns = Predicted\n"
    )

    print(
        f"{'':15}"
        f"{CLASS_NAMES[0]:>12}"
        f"{CLASS_NAMES[1]:>14}"
    )

    print("-" * 41)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:15}"
            f"{cm[i, 0]:>12}"
            f"{cm[i, 1]:>14}"
        )

    # --------------------------------------------------------
    # PER-CLASS RECALL / SENSITIVITY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS RECALL / SENSITIVITY")
    print("=" * 70)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        actual_count = cm[i].sum()

        correct = cm[i, i]

        recall = (
            correct / actual_count
            if actual_count > 0
            else 0.0
        )

        print(
            f"{class_name:<12}: "
            f"{recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # BINARY CLINICAL-STYLE METRICS
    # --------------------------------------------------------

    # Class index:
    # 0 = Benign
    # 1 = Malignant

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("MALIGNANT VS BENIGN METRICS")
    print("=" * 70)

    print(
        f"Sensitivity (Malignant): "
        f"{sensitivity:.4f} "
        f"({sensitivity * 100:.2f}%)"
    )

    print(
        f"Specificity (Benign):    "
        f"{specificity:.4f} "
        f"({specificity * 100:.2f}%)"
    )

    print(
        f"PPV:                     "
        f"{ppv:.4f} "
        f"({ppv * 100:.2f}%)"
    )

    print(
        f"NPV:                     "
        f"{npv:.4f} "
        f"({npv * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUS-BRA EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nThe test set was not used during training."
    )


if __name__ == "__main__":
    evaluate_bra()