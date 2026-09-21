from __future__ import annotations

import io
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "model"
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"
PREPROCESSING_DIR = RESULTS_DIR / "preprocessing_test"

IMAGE_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)
EMOTION_TO_INDEX = {name: index for index, name in enumerate(EMOTIONS)}
DEFAULT_SEED = 42
DEFAULT_FPS = 30.0


def ensure_output_dirs() -> None:
    for path in (MODEL_DIR, TENSORBOARD_DIR, PREPROCESSING_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def configure_tensorflow_memory_growth() -> list[str]:
    import tensorflow as tf

    device_names: list[str] = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        device_names.append(gpu.name)
    return device_names


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def _parse_pixels(series: pd.Series) -> np.ndarray:
    rows: list[np.ndarray] = []
    expected = IMAGE_SIZE * IMAGE_SIZE
    for row_index, raw in enumerate(series.astype(str)):
        pixels = np.fromstring(raw, sep=" ", dtype=np.float32)
        if pixels.size != expected:
            raise ValueError(
                f"Row {row_index} contains {pixels.size} pixels; expected {expected}."
            )
        rows.append(pixels)
    if not rows:
        raise ValueError("Dataset contains no image rows.")
    array = np.stack(rows, axis=0).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    array /= 255.0
    return array.astype(np.float32, copy=False)


def load_fer_csv(
    path: str | Path,
    *,
    require_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    pixel_column = _find_column(frame.columns, ("pixels", "pixel"))
    if pixel_column is None:
        raise ValueError(
            f"{path.name} has no 'pixels' column. Columns: {list(frame.columns)}"
        )
    x = _parse_pixels(frame[pixel_column])

    label_column = _find_column(frame.columns, ("emotion", "label", "target"))
    if label_column is None:
        if require_labels:
            raise ValueError(
                f"{path.name} has no emotion labels; expected 'emotion' or 'label'."
            )
        return x, None

    y = frame[label_column].to_numpy(dtype=np.int64, copy=True)
    invalid = y[(y < 0) | (y >= NUM_CLASSES)]
    if invalid.size:
        raise ValueError(
            f"{path.name} contains labels outside 0..{NUM_CLASSES - 1}: "
            f"{np.unique(invalid).tolist()}"
        )
    if len(y) != len(x):
        raise ValueError("Image and label counts do not match.")
    return x, y


def stratified_train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    validation_size: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.05 <= validation_size <= 0.4:
        raise ValueError("validation_size must be between 0.05 and 0.4.")
    return train_test_split(
        x,
        y,
        test_size=validation_size,
        random_state=seed,
        stratify=y,
    )


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(index): float(weight) for index, weight in zip(classes, weights)}


def build_baseline_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="fer_baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def _conv_block(x, filters: int, dropout: float, *, regularizer):
    from tensorflow.keras import layers

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_final_model():
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.05, fill_mode="nearest", name="aug_rotate")(x)
    x = layers.RandomTranslation(
        0.05, 0.05, fill_mode="nearest", name="aug_translate"
    )(x)
    x = layers.RandomZoom(0.08, fill_mode="nearest", name="aug_zoom")(x)

    l2 = regularizers.l2(1e-4)
    x = _conv_block(x, 64, 0.20, regularizer=l2)
    x = _conv_block(x, 128, 0.25, regularizer=l2)
    x = _conv_block(x, 256, 0.30, regularizer=l2)
    x = _conv_block(x, 256, 0.35, regularizer=l2)

    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="fer_from_scratch_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_training_callbacks(
    *,
    model_path: Path,
    log_dir: Path,
    early_stopping_patience: int = 12,
):
    from tensorflow import keras

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def history_to_dict(history: Any) -> dict[str, list[float]]:
    data = history.history if hasattr(history, "history") else history
    return {
        str(key): [float(value) for value in values]
        for key, values in data.items()
    }


def merge_histories(*histories: dict[str, list[float]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def save_history(history: dict[str, list[float]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_history(path: str | Path) -> dict[str, list[float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_learning_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    required = {"loss", "val_loss", "accuracy", "val_accuracy"}
    missing = sorted(required - set(history))
    if missing:
        raise ValueError(f"Training history is missing keys: {missing}")

    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["accuracy"], label="train")
    axes[0].plot(epochs, history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def model_summary_text(model: Any) -> str:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    return buffer.getvalue()


def append_training_run(record: dict[str, Any]) -> None:
    ensure_output_dirs()
    path = MODEL_DIR / "training_runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_training_runs() -> list[dict[str, Any]]:
    path = MODEL_DIR / "training_runs.jsonl"
    if not path.is_file():
        return []
    runs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def write_architecture_report(
    model: Any,
    *,
    profile: str,
    history: dict[str, list[float]],
    output_path: str | Path,
    notes: str,
) -> None:
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = float(np.min(history["val_loss"]))
    best_val_accuracy = float(np.max(history["val_accuracy"]))
    runs = read_training_runs()

    lines = [
        "Emotions Detector - CNN Architecture Report",
        "=" * 44,
        "",
        f"Profile: {profile}",
        f"Best validation epoch (by val_loss): {best_epoch}",
        f"Best validation loss: {best_val_loss:.6f}",
        f"Best validation accuracy: {best_val_accuracy:.4%}",
        "",
        "Architecture rationale",
        "----------------------",
        notes.strip(),
        "",
        "Iteration history",
        "-----------------",
    ]
    if runs:
        for run in runs[-10:]:
            lines.append(
                "- {timestamp} | {profile} | best val_acc={best_val_accuracy:.4%} "
                "| best val_loss={best_val_loss:.6f}".format(
                    timestamp=run.get("timestamp", "?"),
                    profile=run.get("profile", "?"),
                    best_val_accuracy=float(run.get("best_val_accuracy", 0.0)),
                    best_val_loss=float(run.get("best_val_loss", 0.0)),
                )
            )
    else:
        lines.append("- No earlier recorded runs.")

    lines.extend(["", "model.summary()", "---------------", model_summary_text(model)])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_labeled_test_csv(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates = (DATA_DIR / "test_with_emotions.csv", DATA_DIR / "test.csv")
    for candidate in candidates:
        if candidate.is_file():
            try:
                frame = pd.read_csv(candidate, nrows=1)
            except Exception:
                continue
            if _find_column(frame.columns, ("emotion", "label", "target")) is not None:
                return candidate
    return DATA_DIR / "test_with_emotions.csv"


@dataclass(frozen=True)
class OpenedCapture:
    capture: Any
    first_frame: np.ndarray | None
    backend: int | None


def webcam_backend_preferences(platform: str | None = None) -> tuple[int, ...]:
    current_platform = platform or sys.platform
    if current_platform == "win32":
        return cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY
    return (cv2.CAP_ANY,)


def open_capture(source: int | str | Path) -> OpenedCapture:
    if isinstance(source, int):
        for backend in webcam_backend_preferences():
            capture = cv2.VideoCapture(source, backend)
            if not capture.isOpened():
                capture.release()
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                return OpenedCapture(capture, frame, backend)
            capture.release()
        raise OSError(f"Unable to open webcam device {source}")

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Unable to open video file: {path}")
    return OpenedCapture(capture, None, None)


def parse_source(value: str) -> int | str:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def safe_capture_fps(capture: Any) -> float:
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    except Exception:
        return DEFAULT_FPS
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        return DEFAULT_FPS
    return fps


@dataclass(frozen=True)
class FaceDetection:
    x: int
    y: int
    width: int
    height: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector:
    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise RuntimeError(f"Unable to load OpenCV face cascade: {cascade_path}")
        self.classifier = classifier
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame is None or frame.size == 0:
            return []
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        detections = [
            FaceDetection(int(x), int(y), int(w), int(h))
            for x, y, w, h in boxes
        ]
        detections.sort(key=lambda item: item.area, reverse=True)
        return detections


def crop_face(
    frame: np.ndarray,
    face: FaceDetection,
    *,
    margin: float = 0.12,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = face.xyxy
    pad_x = int(face.width * margin)
    pad_y = int(face.height * margin)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def preprocess_face(
    frame: np.ndarray,
    face: FaceDetection,
) -> tuple[np.ndarray, np.ndarray]:
    crop = crop_face(frame, face)
    if crop.size == 0:
        raise ValueError("Detected face crop is empty.")
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = tensor[..., np.newaxis]
    return resized, tensor


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (70, 210, 70),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = origin
    y = max(y, text_height + baseline + 4)
    x = max(x, 0)
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 5),
        (x + text_width + 6, y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, y - baseline - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_predictions(
    frame: np.ndarray,
    faces: list[FaceDetection],
    probabilities: np.ndarray,
    *,
    fps: float | None = None,
    show_top3: bool = True,
) -> np.ndarray:
    output = frame.copy()
    if len(faces) != len(probabilities):
        raise ValueError("faces and probabilities must have the same length.")

    for index, (face, probs) in enumerate(zip(faces, probabilities)):
        x1, y1, x2, y2 = face.xyxy
        emotion_index = int(np.argmax(probs))
        confidence = float(probs[emotion_index])
        color = (70, 210, 70)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        _draw_label(
            output,
            f"{EMOTIONS[emotion_index]} {confidence:.1%}",
            (x1, y1),
            color,
        )

        if index == 0 and show_top3:
            top3 = np.argsort(probs)[-3:][::-1]
            panel_x = 12
            panel_y = 16
            bar_width = 150
            row_height = 28
            for rank, class_index in enumerate(top3):
                class_index = int(class_index)
                probability = float(probs[class_index])
                y = panel_y + rank * row_height
                cv2.rectangle(
                    output,
                    (panel_x, y),
                    (panel_x + bar_width, y + 18),
                    (35, 35, 35),
                    -1,
                )
                cv2.rectangle(
                    output,
                    (panel_x, y),
                    (panel_x + int(bar_width * probability), y + 18),
                    (90, 160, 90),
                    -1,
                )
                cv2.putText(
                    output,
                    f"{EMOTIONS[class_index]} {probability:.0%}",
                    (panel_x + 4, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    if fps is not None:
        _draw_label(
            output,
            f"FPS {fps:.1f}",
            (12, output.shape[0] - 12),
            (35, 35, 35),
        )
    return output


def benchmark_predict(model: Any, batch: np.ndarray, repeats: int = 30) -> dict[str, float]:
    model.predict(batch, verbose=0)
    started = perf_counter()
    for _ in range(repeats):
        model.predict(batch, verbose=0)
    elapsed = perf_counter() - started
    per_batch_ms = elapsed * 1000.0 / repeats
    per_image_ms = per_batch_ms / len(batch)
    return {
        "batch_size": float(len(batch)),
        "repeats": float(repeats),
        "batch_ms": per_batch_ms,
        "image_ms": per_image_ms,
        "images_per_second": 1000.0 / per_image_ms if per_image_ms else float("inf"),
    }
