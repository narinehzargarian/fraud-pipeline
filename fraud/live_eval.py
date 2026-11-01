"""Compares what the live pipeline decided with the simulator's answers, which it never saw.

Run this after a replay. It shows whether the live system performs like the offline evaluation.

usage: python -m fraud.live_eval
"""
import json

import numpy as np

from fraud import db, simulator
from fraud.train import precision_recall


def main():
    labels = {t.id: t for t in simulator.generate()}
    rows = db.connect().execute("SELECT txn_id, action FROM decisions WHERE txn_id LIKE 'txn-%'").fetchall()
    truth = np.array([labels[txn_id].is_fraud for txn_id, _ in rows], dtype=np.int8)
    declined = [action == "decline" for _, action in rows]
    flagged = [action != "approve" for _, action in rows]
    patterns = {}
    for pattern in ("card_testing", "burst", "big_purchase", "quiet"):
        members = [(txn_id, action) for txn_id, action in rows if labels[txn_id].pattern == pattern]
        patterns[pattern] = round(sum(action != "approve" for _, action in members) / len(members), 3) if members else None
    print("LIVE " + json.dumps({
        "scored": len(rows), "fraud": int(truth.sum()),
        "declined": precision_recall(declined, truth),
        "flagged": precision_recall(flagged, truth),
        "recall_by_pattern": patterns,
    }))


if __name__ == "__main__":
    main()
