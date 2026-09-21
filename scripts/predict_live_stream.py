from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from tensorflow import keras

from common import (
    EMOTIONS,
    MODEL_DIR,
    FaceDetection,
    FaceDetector,
    configure_tensorflow_memory_growth,
    open_capture,
    parse_source,
    preprocess_face,
    render_predictions,
)


@dataclass
class SmoothedFace:
    box: FaceDetection
    probabilities: np.ndarray


def intersection_over_union(a: FaceDetection, b: FaceDetection) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def smooth_predictions(
    previous: list[SmoothedFace],
    faces: list[FaceDetection],
    probabilities: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, list[SmoothedFace]]:
    if not faces:
        return probabilities, []

    output = probabilities.astype(np.float32, copy=True)
    used_previous: set[int] = set()
    new_state: list[SmoothedFace] = []

    for index, face in enumerate(faces):
        best_index = -1
        best_iou = 0.0
        for old_index, old in enumerate(previous):
            if old_index in used_previous:
                continue
            score = intersection_over_union(face, old.box)
            if score > best_iou:
                best_iou = score
                best_index = old_index

        if best_index >= 0 and best_iou >= 0.25:
            used_previous.add(best_index)
            old_probs = previous[best_index].probabilities
            output[index] = alpha * output[index] + (1.0 - alpha) * old_probs
            total = float(output[index].sum())
            if total > 0:
                output[index] /= total

        new_state.append(SmoothedFace(face, output[index].copy()))

    return output, new_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces and classify facial emotions from webcam/video."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0 by default) or path to a recorded video.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "final_emotion_model.keras",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="EMA weight of the newest prediction (0 < alpha <= 1).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help="Seconds between terminal emotion reports.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after N wall-clock seconds; 0 means no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise SystemExit("--smoothing-alpha must be in (0, 1].")
    if args.print_interval <= 0:
        raise SystemExit("--print-interval must be positive.")

    configure_tensorflow_memory_growth()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train it with python ./scripts/train.py"
        )

    print("Reading video stream ...")
    model = keras.models.load_model(args.model)
    detector = FaceDetector()
    opened = open_capture(parse_source(args.source))
    capture = opened.capture
    pending = opened.first_frame

    smoother: list[SmoothedFace] = []
    started = perf_counter()
    last_frame_at = started
    fps_ema: float | None = None
    last_print = started - args.print_interval

    try:
        while True:
            if pending is not None:
                frame = pending
                pending = None
                ok = True
            else:
                ok, frame = capture.read()
            if not ok or frame is None:
                break

            now = perf_counter()
            frame_delta = max(now - last_frame_at, 1e-9)
            instant_fps = 1.0 / frame_delta
            fps_ema = (
                instant_fps
                if fps_ema is None
                else 0.10 * instant_fps + 0.90 * fps_ema
            )
            last_frame_at = now

            faces = detector.detect(frame)
            if faces:
                batch = np.stack(
                    [preprocess_face(frame, face)[1] for face in faces],
                    axis=0,
                )
                probabilities = model.predict(batch, verbose=0)
                if not args.no_smoothing:
                    probabilities, smoother = smooth_predictions(
                        smoother,
                        faces,
                        probabilities,
                        alpha=args.smoothing_alpha,
                    )
                else:
                    smoother = [
                        SmoothedFace(face, probs.copy())
                        for face, probs in zip(faces, probabilities)
                    ]
            else:
                probabilities = np.empty((0, len(EMOTIONS)), dtype=np.float32)
                smoother = []

            if now - last_print >= args.print_interval:
                print("Preprocessing ...")
                if faces:
                    primary = probabilities[0]
                    emotion_index = int(np.argmax(primary))
                    print(
                        f"{datetime.now():%H:%M:%S}s : "
                        f"{EMOTIONS[emotion_index]} , "
                        f"{float(primary[emotion_index]) * 100:.0f}%"
                    )
                else:
                    print(f"{datetime.now():%H:%M:%S}s : No face")
                last_print = now

            if not args.headless:
                rendered = render_predictions(
                    frame,
                    faces,
                    probabilities,
                    fps=fps_ema,
                    show_top3=True,
                )
                cv2.imshow("Emotions detector", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_seconds > 0 and now - started >= args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
