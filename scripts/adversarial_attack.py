from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import (
    EMOTION_TO_INDEX,
    EMOTIONS,
    MODEL_DIR,
    load_fer_csv,
    resolve_labeled_test_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional targeted adversarial attack: Happy -> Sad."
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_DIR / "final_emotion_model.keras"
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--happy-threshold", type=float, default=0.90)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=10.0 / 255.0,
        help="Maximum absolute pixel change in normalized [0,1] units.",
    )
    parser.add_argument("--step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "adversarial",
    )
    return parser.parse_args()


def choose_happy_example(model, x: np.ndarray, y: np.ndarray, threshold: float):
    happy = EMOTION_TO_INDEX["Happy"]
    indices = np.flatnonzero(y == happy)
    if len(indices) == 0:
        raise RuntimeError("The labeled test set contains no Happy examples.")

    probabilities = model.predict(x[indices], batch_size=256, verbose=0)
    happy_scores = probabilities[:, happy]
    best_local = int(np.argmax(happy_scores))
    best_index = int(indices[best_local])
    best_score = float(happy_scores[best_local])
    if best_score < threshold:
        raise RuntimeError(
            f"No correctly labeled Happy image reached {threshold:.0%}. "
            f"Best Happy confidence was {best_score:.2%}."
        )
    return best_index, probabilities[best_local]


def targeted_pgd(
    model,
    image: np.ndarray,
    *,
    target_class: int,
    epsilon: float,
    step_size: float,
    steps: int,
):
    original = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    adversarial = tf.identity(original)
    target = tf.constant([target_class], dtype=tf.int64)

    final_probs = model(adversarial, training=False)[0]
    completed_steps = 0

    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            predictions = model(adversarial, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions
            )
        gradient = tape.gradient(loss, adversarial)
        adversarial = adversarial - step_size * tf.sign(gradient)
        delta = tf.clip_by_value(adversarial - original, -epsilon, epsilon)
        adversarial = tf.clip_by_value(original + delta, 0.0, 1.0)
        final_probs = model(adversarial, training=False)[0]
        completed_steps = step + 1
        if int(tf.argmax(final_probs)) == target_class:
            break

    return adversarial.numpy()[0], final_probs.numpy(), completed_steps


def save_gray(path: Path, image: np.ndarray) -> None:
    array = np.clip(image.squeeze() * 255.0, 0, 255).astype(np.uint8)
    import cv2

    if not cv2.imwrite(str(path), array):
        raise OSError(f"Unable to save {path}")


def main() -> None:
    args = parse_args()
    if args.epsilon <= 0 or args.step_size <= 0 or args.steps <= 0:
        raise SystemExit("epsilon, step-size and steps must be positive.")

    test_path = resolve_labeled_test_csv(args.data)
    x, y = load_fer_csv(test_path, require_labels=True)
    assert y is not None
    model = keras.models.load_model(args.model)

    index, original_probs = choose_happy_example(
        model, x, y, args.happy_threshold
    )
    original = x[index]
    target = EMOTION_TO_INDEX["Sad"]
    adversarial, adversarial_probs, steps_used = targeted_pgd(
        model,
        original,
        target_class=target,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_gray(args.output_dir / "original.png", original)
    save_gray(args.output_dir / "adversarial.png", adversarial)

    delta = adversarial - original
    max_delta = float(np.max(np.abs(delta)))
    mean_delta = float(np.mean(np.abs(delta)))
    l2 = float(np.linalg.norm(delta.ravel()))

    perturbation_visual = delta.squeeze()
    scale = np.max(np.abs(perturbation_visual))
    if scale > 0:
        perturbation_visual = perturbation_visual / (2.0 * scale) + 0.5
    else:
        perturbation_visual = np.full_like(perturbation_visual, 0.5)
    plt.imsave(
        args.output_dir / "perturbation_amplified.png",
        perturbation_visual,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    original_class = int(np.argmax(original_probs))
    adversarial_class = int(np.argmax(adversarial_probs))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(original.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Original\n{EMOTIONS[original_class]} "
        f"{original_probs[original_class]:.1%}"
    )
    axes[1].imshow(perturbation_visual, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Perturbation\n(amplified)")
    axes[2].imshow(adversarial.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Adversarial\n{EMOTIONS[adversarial_class]} "
        f"{adversarial_probs[adversarial_class]:.1%}"
    )
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)

    metrics = (
        f"Selected test index: {index}\n"
        f"Original: {EMOTIONS[original_class]} "
        f"{float(original_probs[original_class]):.4%}\n"
        f"Target: Sad\n"
        f"Adversarial: {EMOTIONS[adversarial_class]} "
        f"{float(adversarial_probs[adversarial_class]):.4%}\n"
        f"Steps used: {steps_used}\n"
        f"Epsilon: {args.epsilon:.8f} ({args.epsilon * 255:.2f}/255)\n"
        f"Max |pixel delta|: {max_delta:.8f} ({max_delta * 255:.2f}/255)\n"
        f"Mean |pixel delta|: {mean_delta:.8f}\n"
        f"L2 distance: {l2:.8f}\n"
        f"Success: {adversarial_class == target}\n"
    )
    (args.output_dir / "metrics.txt").write_text(metrics, encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
