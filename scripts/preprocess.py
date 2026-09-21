from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from time import perf_counter

import cv2

from common import (
    PREPROCESSING_DIR,
    FaceDetector,
    open_capture,
    parse_source,
    preprocess_face,
    safe_capture_fps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture/scan a video stream and save 48x48 grayscale face crops "
            "for the audit preprocessing test."
        )
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (for example 0) or path to a video file.",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.9)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=PREPROCESSING_DIR)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the stream while collecting audit crops.",
    )
    parser.add_argument(
        "--no-record-input",
        action="store_true",
        help="Do not create results/preprocessing_test/input_video.mp4.",
    )
    return parser.parse_args()


def _open_writer(path: Path, frame, fps: float) -> cv2.VideoWriter | None:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        print("Warning: OpenCV could not create input_video.mp4 with mp4v.")
        return None
    return writer


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive.")
    if args.interval <= 0:
        raise SystemExit("--interval must be positive.")
    if args.target_count <= 0:
        raise SystemExit("--target-count must be positive.")

    source = parse_source(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    for old_image in args.output.glob("image*.png"):
        old_image.unlink()

    detector = FaceDetector()
    opened = open_capture(source)
    capture = opened.capture
    fps = safe_capture_fps(capture)
    first_frame = opened.first_frame
    writer: cv2.VideoWriter | None = None

    copy_source = (
        not args.no_record_input
        and not isinstance(source, int)
        and Path(source).suffix.lower() == ".mp4"
    )
    if copy_source:
        source_path = Path(source).resolve()
        destination = (args.output / "input_video.mp4").resolve()
        if source_path != destination:
            shutil.copy2(source_path, destination)

    started = perf_counter()
    frame_index = 0
    next_sample_time = 0.0
    saved = 0

    try:
        pending = first_frame
        while True:
            if pending is not None:
                frame = pending
                pending = None
                ok = True
            else:
                ok, frame = capture.read()

            if not ok or frame is None:
                break

            frame_index += 1
            if isinstance(source, int):
                elapsed = perf_counter() - started
            else:
                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                elapsed = (
                    timestamp_ms / 1000.0
                    if timestamp_ms > 0
                    else frame_index / fps
                )

            if not args.no_record_input and not copy_source:
                if writer is None:
                    writer = _open_writer(args.output / "input_video.mp4", frame, fps)
                if writer is not None:
                    writer.write(frame)

            faces = detector.detect(frame)
            if elapsed >= next_sample_time and faces:
                gray48, _ = preprocess_face(frame, faces[0])
                output_path = args.output / f"image{saved}.png"
                if not cv2.imwrite(str(output_path), gray48):
                    raise OSError(f"Unable to save {output_path}")
                saved += 1
                next_sample_time = elapsed + args.interval
                print(
                    f"Saved {output_path.name}: {gray48.shape[1]}x{gray48.shape[0]} "
                    f"grayscale, t={elapsed:.2f}s"
                )

            if args.preview:
                preview = frame.copy()
                for face in faces:
                    x1, y1, x2, y2 = face.xyxy
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (70, 210, 70), 2)
                cv2.putText(
                    preview,
                    f"saved {saved}/{args.target_count}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Emotions detector - preprocessing audit", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if saved >= args.target_count:
                break
            if elapsed >= args.duration:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"Preprocessed face crops saved: {saved}")
    if saved < args.target_count:
        raise SystemExit(
            f"Only {saved}/{args.target_count} face crops were collected. "
            "Use a longer video/duration or keep the face visible to the camera."
        )


if __name__ == "__main__":
    main()
