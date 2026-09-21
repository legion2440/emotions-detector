from __future__ import annotations

import argparse
from pathlib import Path

from common import MODEL_DIR, load_history, plot_learning_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate validation loss/accuracy learning curves."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=MODEL_DIR / "training_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "learning_curves.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.history)
    plot_learning_curves(history, args.output)
    print(f"Saved learning curves: {args.output}")


if __name__ == "__main__":
    main()
