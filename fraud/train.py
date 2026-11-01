"""Trains the model on simulated transactions and evaluates it on a later period.

The split is by time, not random. A random split would let the model learn from the future, and
would report better results than it could ever achieve in production.

usage: python -m fraud.train --out artifacts
"""
import argparse
import json
import pathlib
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud import features as feature_code
from fraud import rules, simulator
from fraud.scorer import to_dict


def build_matrix(transactions):
    """Computes features in time order, so each row sees only its own past."""
    store = feature_code.MemoryStore()
    rows = []
    for transaction in transactions:
        txn = to_dict(transaction)
        rows.append(feature_code.compute(txn, store.history(txn["card_id"], txn["ts"])))
        store.add(txn)
    matrix = np.array([[row[name] for name in feature_code.FEATURE_NAMES] for row in rows], dtype=np.float64)
    labels = np.array([t.is_fraud for t in transactions], dtype=np.int8)
    return matrix, labels, rows


def precision_recall(predicted, labels):
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = int((predicted & (labels == 1)).sum())
    return {
        "precision": round(true_positive / max(int(predicted.sum()), 1), 3),
        "recall": round(true_positive / max(int(labels.sum()), 1), 3),
        "flagged_percent": round(100 * float(predicted.mean()), 2),
    }


def threshold_for_precision(scores, labels, target):
    """The lowest score threshold that still reaches the target precision on validation data."""
    order = np.argsort(-scores)
    hits = np.cumsum(labels[order])
    precision = hits / np.arange(1, len(scores) + 1)
    good = np.nonzero(precision >= target)[0]
    if len(good) == 0:
        return float(scores.max())
    return float(scores[order[good[-1]]])


def train_classifier(train_x, train_y, seed=7):
    # Fraud is rare, so fraud rows count more. Otherwise predicting "never fraud" looks great.
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced", random_state=seed)
    return model.fit(train_x, train_y)


def chronological_split(count, train=0.6, validation=0.2):
    train_end, validation_end = int(count * train), int(count * (train + validation))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--cards", type=int, default=2000)
    args = parser.parse_args()

    transactions = simulator.generate(days=args.days, cards=args.cards)
    x, y, rows = build_matrix(transactions)
    train_slice, validation_slice, test_slice = chronological_split(len(y))

    start = time.monotonic()
    model = train_classifier(x[train_slice], y[train_slice])
    train_seconds = time.monotonic() - start

    validation_scores = model.predict_proba(x[validation_slice])[:, 1]
    decline_threshold = threshold_for_precision(validation_scores, y[validation_slice], 0.90)
    review_threshold = threshold_for_precision(validation_scores, y[validation_slice], 0.30)

    test_y = y[test_slice]
    test_scores = model.predict_proba(x[test_slice])[:, 1]
    test_rows = rows[test_slice]
    test_transactions = transactions[test_slice.start:test_slice.stop]
    rule_hits = np.array([bool(rules.triggered_rules(row)) for row in test_rows])
    actions = [
        rules.decide(score, rules.triggered_rules(row), review_threshold, decline_threshold)
        for score, row in zip(test_scores, test_rows)
    ]
    flagged = np.array([action != "approve" for action in actions])

    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    linear.fit(x[train_slice], y[train_slice])

    per_pattern = {}
    for pattern in ("card_testing", "burst", "big_purchase", "quiet"):
        members = np.array([t.pattern == pattern for t in test_transactions])
        per_pattern[pattern] = round(float(flagged[members].mean()), 3) if members.any() else None

    metrics = {
        "data": "simulated",
        "transactions": len(y),
        "test_transactions": int(len(test_y)),
        "test_fraud_percent": round(100 * float(test_y.mean()), 3),
        "pr_auc": {
            "model": round(float(average_precision_score(test_y, test_scores)), 3),
            "logistic_regression_baseline": round(float(average_precision_score(test_y, linear.predict_proba(x[test_slice])[:, 1])), 3),
            "random_baseline": round(float(test_y.mean()), 3),
        },
        "thresholds": {"review": review_threshold, "decline": decline_threshold},
        "rules_only": precision_recall(rule_hits, test_y),
        "model_only_at_decline": precision_recall(test_scores >= decline_threshold, test_y),
        "model_only_at_review": precision_recall(test_scores >= review_threshold, test_y),
        "combined_flagged": precision_recall(flagged, test_y),
        "recall_by_pattern": per_pattern,
        "train_seconds": round(train_seconds, 1),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thresholds": metrics["thresholds"], "features": feature_code.FEATURE_NAMES}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main()
