import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib.pyplot as plt


def _to_history_dict(history_like: Any) -> Dict[str, Any]:
    """Normalize a Keras History object or dict into a plain dictionary."""
    if hasattr(history_like, "history"):
        history_like = history_like.history

    if not isinstance(history_like, Mapping):
        raise TypeError(
            "history_like must be a keras.callbacks.History object or a dictionary-like object."
        )

    return dict(history_like)


def _get_metric_pair(history: Mapping[str, Any], metric: str) -> Tuple[Any, Any]:
    train_key = metric
    val_key = f"val_{metric}"

    if train_key not in history or val_key not in history:
        missing = [k for k in (train_key, val_key) if k not in history]
        raise KeyError(f"Missing metric keys in history: {missing}")

    return history[train_key], history[val_key]


def save_training_metric_plots(history_like: Any, output_dir: str = "metric_images") -> None:
    """
    Save training and validation curves for accuracy, loss, and auc as PNG images.

    Output files:
    - accuracy.png
    - loss.png
    - auc.png
    """
    history = _to_history_dict(history_like)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("auc", "AUC"),
    ]

    for metric_key, title in metric_specs:
        train_values, val_values = _get_metric_pair(history, metric_key)

        plt.figure(figsize=(8, 5))
        plt.plot(train_values, label=f"Train {title}")
        plt.plot(val_values, label=f"Validation {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"Training and Validation {title}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{metric_key}.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate accuracy/loss/auc plot images from a training history JSON file."
    )
    parser.add_argument(
        "--history-json",
        type=str,
        default="training_history.json",
        help="Path to a JSON file containing the model.fit() history dictionary.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metric_images",
        help="Directory where PNG files will be saved.",
    )

    args = parser.parse_args()

    history_json_path = Path(args.history_json)
    if not history_json_path.exists():
        parser.error(
            f"History JSON not found: {history_json_path}. "
            "Train the model first and save history.history to a JSON file, "
            "or pass --history-json with a valid file path."
        )

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    save_training_metric_plots(history, args.output_dir)
    print(f"Saved metric images in: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
