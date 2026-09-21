from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from common import (
    DATA_DIR,
    MODEL_DIR,
    TENSORBOARD_DIR,
    build_training_callbacks,
    configure_tensorflow_memory_growth,
    ensure_output_dirs,
    history_to_dict,
    load_fer_csv,
    merge_histories,
    model_summary_text,
    plot_learning_curves,
    save_history,
    set_global_seed,
    stratified_train_validation_split,
    timestamp_tag,
)


def build_pretrained_model():
    inputs = keras.Input(shape=(48, 48, 1), name="image")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05, fill_mode="nearest")(x)
    x = layers.Resizing(96, 96)(x)
    x = layers.Concatenate(axis=-1)([x, x, x])
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(7, activation="softmax", name="emotion")(x)
    model = keras.Model(inputs, outputs, name="fer_mobilenetv2_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional transfer-learning emotion classifier."
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    set_global_seed(args.seed)
    configure_tensorflow_memory_growth()

    x, y = load_fer_csv(args.data, require_labels=True)
    assert y is not None
    x_train, x_val, y_train, y_val = stratified_train_validation_split(
        x, y, validation_size=args.validation_size, seed=args.seed
    )

    model, base = build_pretrained_model()
    stage1_path = MODEL_DIR / "pretrained_stage1.keras"
    stage2_path = MODEL_DIR / "pretrained_stage2.keras"

    history1 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=build_training_callbacks(
            model_path=stage1_path,
            log_dir=TENSORBOARD_DIR / f"pretrained_head_{timestamp_tag()}",
            early_stopping_patience=7,
        ),
        verbose=1,
    )
    h1 = history_to_dict(history1)

    h2: dict[str, list[float]] = {}
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history2 = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            callbacks=build_training_callbacks(
                model_path=stage2_path,
                log_dir=TENSORBOARD_DIR / f"pretrained_finetune_{timestamp_tag()}",
                early_stopping_patience=6,
            ),
            verbose=1,
        )
        h2 = history_to_dict(history2)

    best_stage1 = float(np.min(h1["val_loss"]))
    best_stage2 = float(np.min(h2["val_loss"])) if h2 else float("inf")
    selected_path = stage2_path if best_stage2 < best_stage1 else stage1_path
    selected = keras.models.load_model(selected_path)
    final_path = MODEL_DIR / "pre_trained_model.keras"
    selected.save(final_path)

    merged = merge_histories(h1, h2)
    save_history(merged, MODEL_DIR / "pre_trained_training_history.json")
    plot_learning_curves(merged, MODEL_DIR / "pre_trained_learning_curves.png")

    report = f"""Optional pre-trained CNN
========================

Backbone: MobileNetV2 with ImageNet weights
Input adaptation: 48x48x1 -> resize 96x96 -> repeat grayscale channel to RGB
Stage 1: frozen backbone, train classifier head
Stage 2: optional fine-tuning of the last 30 backbone layers
Selected checkpoint: {selected_path.name}
Best stage-1 validation loss: {best_stage1:.6f}
Best stage-2 validation loss: {best_stage2:.6f}

This model is optional and does not replace final_emotion_model.keras,
which is trained from scratch as required by the project.

model.summary()
---------------
{model_summary_text(selected)}
"""
    (MODEL_DIR / "pre_trained_model_architecture.txt").write_text(
        report, encoding="utf-8"
    )
    print(f"Saved optional pre-trained model: {final_path}")


if __name__ == "__main__":
    main()
