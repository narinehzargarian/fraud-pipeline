"""Scores one transaction: reads its card's history, computes features, applies model and rules."""
import dataclasses
import pathlib
import time

import joblib
import numpy as np
import pandas as pd  # noqa: F401  (joblib-loaded models may expect it to be importable)

from fraud import features as feature_code
from fraud import rules


def to_dict(transaction):
    return dataclasses.asdict(transaction) if dataclasses.is_dataclass(transaction) else dict(transaction)


class Scorer:
    def __init__(self, store, artifact_path="artifacts/model.joblib"):
        artifact = joblib.load(pathlib.Path(artifact_path))
        self.model = artifact["model"]
        self.thresholds = artifact["thresholds"]
        self.names = artifact["features"]
        self.store = store

    def score(self, transaction):
        start = time.perf_counter()
        txn = to_dict(transaction)
        computed = feature_code.compute(txn, self.store.history(txn["card_id"], txn["ts"]))
        row = np.array([[computed[name] for name in self.names]], dtype=np.float64)
        score = float(self.model.predict_proba(row)[0, 1])
        triggered = rules.triggered_rules(computed)
        action = rules.decide(score, triggered, self.thresholds["review"], self.thresholds["decline"])
        self.store.add(txn)
        return {
            "txn_id": txn["id"],
            "score": round(score, 4),
            "action": action,
            "rules": triggered,
            "features": computed,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
