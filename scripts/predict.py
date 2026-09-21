from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from common import (
    MODEL_DIR,
    benchmark_predict,
    configure_tensorflow_memory_growth,
    load_fer_csv,
    resolve_labeled_test_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the final emotion model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "final_emotion_model.keras",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Labeled FER test CSV. Defaults to data/test_with_emotions.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_tensorflow_memory_growth()

    test_path = resolve_labeled_test_csv(args.data)
    x_test, y_test = load_fer_csv(test_path, require_labels=True)
    assert y_test is not None

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train it with python ./scripts/train.py"
        )

    model = keras.models.load_model(args.model)
    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == y_test))

    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    if args.benchmark:
        batch = x_test[: min(32, len(x_test))]
        metrics = benchmark_predict(model, batch)
        print(
            "Inference: "
            f"{metrics['image_ms']:.3f} ms/image, "
            f"{metrics['images_per_second']:.1f} images/s "
            f"(batch={int(metrics['batch_size'])})"
        )


if __name__ == "__main__":
    main()
