"""Command-line entry point for the reproducible credit-risk baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from feature_engineering.pipeline import build_pipeline, load_dataset, prepare_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-safe credit-risk baseline."
    )
    parser.add_argument("--data", type=Path, default=Path("dataset.txt"))
    parser.add_argument("--report", type=Path, help="Optional JSON output path.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def evaluate(data_path: Path, test_size: float, random_state: int) -> dict[str, float | int]:
    data = load_dataset(data_path)
    features, target = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    return {
        "rows": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "positive_rate": round(float(target.mean()), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions)), 6),
        "recall": round(float(recall_score(y_test, predictions)), 6),
        "f1": round(float(f1_score(y_test, predictions)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 6),
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.data, args.test_size, args.random_state)
    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(output)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{output}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
