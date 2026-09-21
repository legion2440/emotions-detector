from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow import keras

from common import EMOTIONS, MODEL_DIR, load_fer_csv, resolve_labeled_test_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optional confusion matrix and per-class metrics."
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_DIR / "final_emotion_model.keras"
    )
    parser.add_argument("--data", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = resolve_labeled_test_csv(args.data)
    x, y = load_fer_csv(test_path, require_labels=True)
    assert y is not None

    model = keras.models.load_model(args.model)
    probabilities = model.predict(x, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y,
        predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        digits=4,
        zero_division=0,
    )
    report_path = MODEL_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y, predictions, labels=list(range(len(EMOTIONS))))
    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=EMOTIONS)
    display.plot(ax=ax, xticks_rotation=35, values_format="d", colorbar=False)
    fig.tight_layout()
    matrix_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(matrix_path, dpi=160)
    plt.close(fig)

    wrong = np.flatnonzero(predictions != y)
    if len(wrong):
        wrong_confidence = probabilities[wrong, predictions[wrong]]
        top_wrong = wrong[np.argsort(wrong_confidence)[-12:][::-1]]
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        for ax in axes.flat:
            ax.axis("off")
        for ax, sample_index in zip(axes.flat, top_wrong):
            predicted = int(predictions[sample_index])
            truth = int(y[sample_index])
            confidence = float(probabilities[sample_index, predicted])
            ax.imshow(x[sample_index].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"GT: {EMOTIONS[truth]}\n"
                f"Pred: {EMOTIONS[predicted]} {confidence:.0%}",
                fontsize=9,
            )
            ax.axis("off")
        fig.tight_layout()
        mistakes_path = MODEL_DIR / "confident_mistakes.png"
        fig.savefig(mistakes_path, dpi=160)
        plt.close(fig)
        print(f"Saved: {mistakes_path}")

    print(report)
    print(f"Saved: {report_path}")
    print(f"Saved: {matrix_path}")


if __name__ == "__main__":
    main()
