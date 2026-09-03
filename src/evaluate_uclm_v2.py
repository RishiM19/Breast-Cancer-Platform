import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from uclm_dataset import get_uclm_dataloaders
from train_uclm_v2 import build_uclm_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/uclm_v2_best_model.pth"

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant",
]

BATCH_SIZE = 16


def evaluate_uclm_v2():

    print("=" * 70)
    print("LUMINA — BUS-UCLM V2 TEST EVALUATION")
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

    _, _, test_loader = get_uclm_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(
        f"Test images: {len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading UCLM V2 model...")

    model = build_uclm_model(
        num_classes=3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    print(
        "\nRunning predictions on untouched UCLM test set..."
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
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    # --------------------------------------------------------
    # OVERALL METRICS
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
        f"{CLASS_NAMES[1]:>12}"
        f"{CLASS_NAMES[2]:>14}"
    )

    print("-" * 53)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:15}"
            f"{cm[i, 0]:>12}"
            f"{cm[i, 1]:>12}"
            f"{cm[i, 2]:>14}"
        )

    # --------------------------------------------------------
    # SENSITIVITY / RECALL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS RECALL / SENSITIVITY")
    print("=" * 70)

    recalls = {}

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

        recalls[class_name] = recall

        print(
            f"{class_name:<12}: "
            f"{recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # SPECIFIC MALIGNANT METRICS
    # --------------------------------------------------------

    malignant_idx = 2

    malignant_tp = cm[
        malignant_idx,
        malignant_idx
    ]

    malignant_fn = (
        cm[malignant_idx].sum()
        - malignant_tp
    )

    malignant_recall = (
        malignant_tp
        / (malignant_tp + malignant_fn)
        if (malignant_tp + malignant_fn) > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("MALIGNANT DETECTION")
    print("=" * 70)

    print(
        f"Malignant sensitivity: "
        f"{malignant_recall:.4f} "
        f"({malignant_recall * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V2 EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nTest set was not used during model selection."
    )


if __name__ == "__main__":
    evaluate_uclm_v2()