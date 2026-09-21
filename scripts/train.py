from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    DATA_DIR,
    MODEL_DIR,
    TENSORBOARD_DIR,
    append_training_run,
    build_baseline_model,
    build_final_model,
    build_training_callbacks,
    compute_class_weights,
    configure_tensorflow_memory_growth,
    ensure_output_dirs,
    history_to_dict,
    load_fer_csv,
    plot_learning_curves,
    save_history,
    set_global_seed,
    stratified_train_validation_split,
    timestamp_tag,
    write_architecture_report,
)


FINAL_NOTES = """
The mandatory model is trained from scratch: no pre-trained weights are used.

Iteration 1 is available through --profile baseline and intentionally uses a
small three-block CNN. Its purpose is to validate the FER CSV loader, training
loop, validation split, checkpointing and evaluation pipeline.

The final profile increases representational capacity with four two-convolution
blocks (64/128/256/256 filters), Batch Normalization, L2 regularization and
progressively stronger Dropout. Training-only horizontal flip, small rotation,
translation and zoom augmentations improve robustness while preserving the
48x48 grayscale input expected by FER. A 256-unit dense head feeds the seven
class softmax output.

EarlyStopping monitors validation loss and restores the best weights.
ReduceLROnPlateau lowers the learning rate when validation loss stalls.
TensorBoard records every run. The test set is never loaded by this script.
"""


BASELINE_NOTES = """
This is the first, deliberately small training iteration. It validates the
end-to-end training pipeline before increasing model capacity. It uses three
Conv2D + MaxPool stages and one dense layer with Dropout. No pre-trained weights
are used and the test set is never loaded during training.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mandatory from-scratch FER emotion classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "train.csv",
        help="FER training CSV (default: data/train.csv)",
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "final"),
        default="final",
        help="CNN architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use balanced class weights. Off by default because overall accuracy is the audit metric.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    ensure_output_dirs()
    set_global_seed(args.seed)
    devices = configure_tensorflow_memory_growth()
    print("TensorFlow GPUs:", ", ".join(devices) if devices else "none detected")

    print(f"Loading training set: {args.data}")
    x, y = load_fer_csv(args.data, require_labels=True)
    assert y is not None
    x_train, x_val, y_train, y_val = stratified_train_validation_split(
        x,
        y,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    print(
        f"Train: {len(x_train)} samples | Validation: {len(x_val)} samples | "
        f"Input: {x_train.shape[1:]}"
    )

    if args.profile == "baseline":
        model = build_baseline_model()
        model_path = MODEL_DIR / "baseline_emotion_model.keras"
        history_path = MODEL_DIR / "baseline_training_history.json"
        curves_path = MODEL_DIR / "baseline_learning_curves.png"
        arch_path = MODEL_DIR / "baseline_emotion_model_arch.txt"
        notes = BASELINE_NOTES
    else:
        model = build_final_model()
        model_path = MODEL_DIR / "final_emotion_model.keras"
        history_path = MODEL_DIR / "training_history.json"
        curves_path = MODEL_DIR / "learning_curves.png"
        arch_path = MODEL_DIR / "final_emotion_model_arch.txt"
        notes = FINAL_NOTES

    log_dir = TENSORBOARD_DIR / f"{args.profile}_{timestamp_tag()}"
    callbacks = build_training_callbacks(
        model_path=model_path,
        log_dir=log_dir,
        early_stopping_patience=args.early_stopping_patience,
    )
    class_weight = compute_class_weights(y_train) if args.class_weights else None
    if class_weight:
        print("Class weights:", class_weight)

    model.summary()
    history_obj = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1,
    )

    history = history_to_dict(history_obj)

    from tensorflow import keras

    model = keras.models.load_model(model_path)
    save_history(history, history_path)
    plot_learning_curves(history, curves_path)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    record = {
        "timestamp": timestamp_tag(),
        "profile": args.profile,
        "epochs_ran": len(history["loss"]),
        "best_epoch_by_val_loss": best_epoch,
        "best_val_loss": float(np.min(history["val_loss"])),
        "best_val_accuracy": float(np.max(history["val_accuracy"])),
        "batch_size": args.batch_size,
        "validation_size": args.validation_size,
        "class_weights": bool(args.class_weights),
        "model_path": str(model_path.relative_to(model_path.parents[2])),
        "tensorboard_log_dir": str(log_dir),
    }
    append_training_run(record)
    write_architecture_report(
        model,
        profile=args.profile,
        history=history,
        output_path=arch_path,
        notes=notes,
    )

    print()
    print(f"Saved model: {model_path}")
    print(f"Saved learning curves: {curves_path}")
    print(f"Saved architecture report: {arch_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("TensorBoard command: tensorboard --logdir results/tensorboard")
    if args.profile == "final":
        print(
            "After training, run: python ./scripts/predict.py "
            "and save a TensorBoard screenshot as results/model/tensorboard.png"
        )


if __name__ == "__main__":
    main()
