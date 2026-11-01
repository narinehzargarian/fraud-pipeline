"""Checks the training method on real, public data: the ULB credit card dataset.

This model isn't used by the live pipeline, because the dataset has no card IDs. It answers one
question: does this approach hold up on real fraud, rather than only on data I generated?

usage: python -m fraud.benchmark --out artifacts
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import average_precision_score

from fraud.train import chronological_split, precision_recall, threshold_for_precision, train_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    frame = data.frame
    label_column = data.target_names[0]
    labels = frame[label_column].astype(str).str.strip("'").astype(int).to_numpy()
    # The file is already in time order, so splitting by position is a split by time.
    values = frame.drop(columns=[label_column]).to_numpy(dtype=np.float64)

    train_slice, validation_slice, test_slice = chronological_split(len(labels))
    model = train_classifier(values[train_slice], labels[train_slice])
    threshold = threshold_for_precision(model.predict_proba(values[validation_slice])[:, 1], labels[validation_slice], 0.80)
    scores = model.predict_proba(values[test_slice])[:, 1]
    test_labels = labels[test_slice]
    metrics = {
        "data": "ULB credit card fraud (OpenML 1597)",
        "transactions": len(labels),
        "test_fraud_percent": round(100 * float(test_labels.mean()), 3),
        "test_fraud_count": int(test_labels.sum()),
        "pr_auc": round(float(average_precision_score(test_labels, scores)), 3),
        "random_baseline": round(float(test_labels.mean()), 4),
        "at_threshold_chosen_on_validation": precision_recall(scores >= threshold, test_labels),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(metrics, indent=2))
    print("BENCHMARK " + json.dumps(metrics))


if __name__ == "__main__":
    main()
