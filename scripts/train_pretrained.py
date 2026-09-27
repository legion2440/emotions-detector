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


INPUT_SIZE = 128
BACKBONE_NAME = "resnet50v2_backbone"


def build_transfer_callbacks(
    *,
    model_path: Path,
    log_dir: Path,
    patience: int,
):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def build_pretrained_model():
    inputs = keras.Input(shape=(48, 48, 1), name="image")

    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.04, fill_mode="nearest", name="aug_rotate")(x)
    x = layers.RandomTranslation(
        0.04, 0.04, fill_mode="nearest", name="aug_translate"
    )(x)
    x = layers.RandomZoom(0.06, fill_mode="nearest", name="aug_zoom")(x)

    x = layers.Resizing(INPUT_SIZE, INPUT_SIZE, name="resize_for_backbone")(x)
    x = layers.Concatenate(axis=-1, name="grayscale_to_rgb")([x, x, x])
    # ResNet50V2 preprocess_input maps [0, 255] RGB to [-1, 1].
    # FER loader produces [0, 1], so scale to [0, 255] first.
    x = layers.Rescaling(255.0, name="to_255")(x)
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="resnet50v2_preprocess",
    )(x)

    base = keras.applications.ResNet50V2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base._name = BACKBONE_NAME
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.45, name="head_dropout")(x)
    outputs = layers.Dense(7, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="fer_resnet50v2_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_backbone_tail(model: keras.Model, trainable_tail: int) -> keras.Model:
    base = model.get_layer(BACKBONE_NAME)
    base.trainable = True

    for layer in base.layers:
        layer.trainable = False

    candidates = [
        layer
        for layer in base.layers
        if not isinstance(layer, layers.BatchNormalization)
    ]
    for layer in candidates[-trainable_tail:]:
        layer.trainable = True

    # Keep every BatchNorm frozen. With FER's small grayscale domain and a modest
    # batch size, updating ImageNet BN statistics is a common source of collapse.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional ImageNet transfer-learning emotion classifier."
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument(
        "--epochs",
        type=int,
        default=24,
        help="Maximum frozen-backbone epochs.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=20,
        help="Maximum fine-tuning epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=35,
        help="Number of non-BatchNorm backbone layers to unfreeze from the tail.",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.fine_tune_epochs < 0:
        raise SystemExit("--fine-tune-epochs must be >= 0.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.fine_tune_layers <= 0:
        raise SystemExit("--fine-tune-layers must be positive.")
    if args.fine_tune_lr <= 0:
        raise SystemExit("--fine-tune-lr must be positive.")

    ensure_output_dirs()
    set_global_seed(args.seed)
    devices = configure_tensorflow_memory_growth()
    print("TensorFlow GPUs:", ", ".join(devices) if devices else "none detected")

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
        f"Backbone input: {INPUT_SIZE}x{INPUT_SIZE} RGB"
    )

    stage1_path = MODEL_DIR / "pretrained_stage1.keras"
    stage2_path = MODEL_DIR / "pretrained_stage2.keras"

    model, _ = build_pretrained_model()
    print("\nStage 1: frozen ImageNet backbone")
    history1 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=build_transfer_callbacks(
            model_path=stage1_path,
            log_dir=TENSORBOARD_DIR / f"pretrained_head_{timestamp_tag()}",
            patience=7,
        ),
        shuffle=True,
        verbose=1,
    )
    h1 = history_to_dict(history1)

    # Fine-tuning must start from the best frozen-backbone checkpoint rather
    # than whatever weights happen to remain at the end of stage 1.
    model = keras.models.load_model(stage1_path)

    h2: dict[str, list[float]] = {}
    if args.fine_tune_epochs > 0:
        print(
            "\nStage 2: fine-tuning "
            f"{args.fine_tune_layers} non-BatchNorm backbone layers"
        )
        model = unfreeze_backbone_tail(model, args.fine_tune_layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history2 = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            callbacks=build_transfer_callbacks(
                model_path=stage2_path,
                log_dir=TENSORBOARD_DIR / f"pretrained_finetune_{timestamp_tag()}",
                patience=7,
            ),
            shuffle=True,
            verbose=1,
        )
        h2 = history_to_dict(history2)

    best_stage1_acc = float(np.max(h1["val_accuracy"]))
    best_stage1_loss = float(np.min(h1["val_loss"]))
    best_stage2_acc = (
        float(np.max(h2["val_accuracy"])) if h2 else float("-inf")
    )
    best_stage2_loss = float(np.min(h2["val_loss"])) if h2 else float("inf")

    # Accuracy is the project criterion. Never replace a better stage-1 model
    # merely because fine-tuning happened to obtain a slightly smaller loss.
    selected_path = (
        stage2_path
        if h2 and best_stage2_acc > best_stage1_acc
        else stage1_path
    )
    selected = keras.models.load_model(selected_path)

    final_path = MODEL_DIR / "pre_trained_model.keras"
    selected.save(final_path)

    merged = merge_histories(h1, h2)
    save_history(merged, MODEL_DIR / "pre_trained_training_history.json")
    plot_learning_curves(
        merged,
        MODEL_DIR / "pre_trained_learning_curves.png",
    )

    selected_stage = "fine-tuned" if selected_path == stage2_path else "frozen-head"
    report = f"""Optional pre-trained CNN
========================

Backbone: ResNet50V2 with ImageNet weights
Backbone input: {INPUT_SIZE}x{INPUT_SIZE} RGB
Input adaptation: 48x48x1 -> resize {INPUT_SIZE}x{INPUT_SIZE} -> repeat grayscale to RGB
Stage 1: frozen ImageNet backbone, train a BatchNorm + Dense(256) classifier head
Stage 2: reload the best stage-1 checkpoint, then fine-tune the last
         {args.fine_tune_layers} non-BatchNorm backbone layers at LR={args.fine_tune_lr:g}
BatchNorm policy: all backbone BatchNorm layers remain frozen during fine-tuning
Checkpoint criterion: maximum validation accuracy
Selected checkpoint: {selected_path.name} ({selected_stage})

Best stage-1 validation accuracy: {best_stage1_acc:.4%}
Best stage-1 validation loss: {best_stage1_loss:.6f}
Best stage-2 validation accuracy: {best_stage2_acc:.4%}
Best stage-2 validation loss: {best_stage2_loss:.6f}

This model is optional and does not replace final_emotion_model.keras,
which is trained from scratch as required by the project.

model.summary()
---------------
{model_summary_text(selected)}
"""
    (MODEL_DIR / "pre_trained_model_architecture.txt").write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(f"Best stage-1 val_accuracy: {best_stage1_acc:.2%}")
    if h2:
        print(f"Best stage-2 val_accuracy: {best_stage2_acc:.2%}")
    print(f"Selected checkpoint: {selected_path.name}")
    print(f"Saved optional pre-trained model: {final_path}")
    print(
        "Evaluate it with: python ./scripts/predict.py "
        "--model results/model/pre_trained_model.keras"
    )


if __name__ == "__main__":
    main()
