"""Fast tests with no external services."""
import numpy as np
import pytest

from fraud import features, rules, simulator
from fraud.schemas import ReviewIn, TransactionIn
from fraud.scorer import to_dict
from fraud.train import build_matrix, chronological_split, precision_recall, threshold_for_precision


def txn(id="t1", ts=1_000_000.0, amount=20.0, merchant="m1", online=False, lat=34.0, lon=-118.0, category="retail", card="c1"):
    return {"id": id, "card_id": card, "ts": ts, "amount": amount, "merchant_id": merchant, "category": category,
            "lat": lat, "lon": lon, "online": online}


def test_simulator_is_repeatable_and_ordered():
    first = simulator.generate(days=10, cards=50, seed=3)
    second = simulator.generate(days=10, cards=50, seed=3)
    assert first == second
    assert first != simulator.generate(days=10, cards=50, seed=4)
    assert all(a.ts <= b.ts for a, b in zip(first, first[1:]))
    assert len({t.id for t in first}) == len(first)


def test_simulated_fraud_is_rare_and_includes_every_pattern():
    data = simulator.generate(days=60, cards=500)
    rate = sum(t.is_fraud for t in data) / len(data)
    assert 0.0005 < rate < 0.02
    assert {t.pattern for t in data if t.is_fraud} == {"card_testing", "burst", "big_purchase", "quiet"}


def test_a_transaction_never_counts_itself():
    current = txn()
    assert features.compute(current, [current])["count_1h"] == 0


def test_future_events_are_ignored():
    # This is what prevents leaking future data, and what makes redelivered messages safe.
    later = txn(id="later", ts=1_000_100.0, amount=999.0)
    computed = features.compute(txn(), [later])
    assert computed["count_1h"] == 0
    assert computed["average_amount_7d"] == 0


def test_rolling_windows():
    now = 1_000_000.0
    history = [txn(id="a", ts=now - 100, amount=2.0, merchant="m1"), txn(id="b", ts=now - 1800, amount=4.0, merchant="m2"),
               txn(id="c", ts=now - 7200, amount=100.0, merchant="m3"), txn(id="d", ts=now - 90_000, amount=50.0, merchant="m4")]
    computed = features.compute(txn(ts=now, merchant="m9"), history)
    assert computed["count_1h"] == 2
    assert computed["count_24h"] == 3
    assert computed["amount_1h"] == 6
    assert computed["amount_24h"] == 106
    assert computed["distinct_merchants_1h"] == 2
    assert computed["small_count_1h"] == 2
    assert computed["history_count_7d"] == 4
    assert computed["new_merchant_7d"] == 1
    assert features.compute(txn(ts=now, merchant="m1"), history)["new_merchant_7d"] == 0


def test_distance_matches_a_known_value():
    assert features.km_between(34.05, -118.24, 40.71, -74.01) == pytest.approx(3936, abs=30)  # LA to New York.


def test_travel_speed_ignores_online_purchases():
    past = txn(id="a", ts=1_000_000.0, lat=34.05, lon=-118.24)
    far = txn(id="b", ts=1_000_600.0, lat=51.5, lon=-0.13)
    assert features.compute(far, [past])["speed_kmh"] > 10_000
    online = {**far, "online": True}
    assert features.compute(online, [past])["speed_kmh"] == 0


def test_memory_store_ignores_repeats_and_old_events():
    store = features.MemoryStore()
    store.add(txn())
    store.add(txn())
    assert len(store.history("c1", 1_000_000.0)) == 1
    assert store.history("c1", 1_000_000.0 + features.HISTORY_SECONDS + 1) == []


def test_each_rule_fires_on_its_pattern():
    base = {"km_from_last": 0, "speed_kmh": 0, "small_count_1h": 0, "count_1h": 0, "history_count_7d": 0,
            "amount_ratio_7d": 0, "risky_category": 0, "new_merchant_7d": 0, "amount": 10}
    assert rules.triggered_rules(base) == []
    assert rules.triggered_rules({**base, "km_from_last": 5000, "speed_kmh": 9000}) == ["impossible_travel"]
    assert rules.triggered_rules({**base, "small_count_1h": 3}) == ["card_testing"]
    assert rules.triggered_rules({**base, "count_1h": 5}) == ["high_velocity"]
    assert rules.triggered_rules({**base, "history_count_7d": 5, "amount_ratio_7d": 10}) == ["amount_spike"]
    assert rules.triggered_rules({**base, "risky_category": 1, "new_merchant_7d": 1, "amount": 200}) == ["risky_new_merchant"]


def test_a_rule_can_only_send_to_review():
    assert rules.decide(0.01, ["amount_spike"], 0.5, 0.9) == "review"
    assert rules.decide(0.95, [], 0.5, 0.9) == "decline"
    assert rules.decide(0.6, [], 0.5, 0.9) == "review"
    assert rules.decide(0.1, [], 0.5, 0.9) == "approve"


def test_matrix_rows_see_only_the_past():
    data = simulator.generate(days=5, cards=20)
    x, y, rows = build_matrix(data)
    assert x.shape == (len(data), len(features.FEATURE_NAMES))
    assert y.sum() == sum(t.is_fraud for t in data)
    # Adding later transactions must not change any earlier row.
    _, _, shorter = build_matrix(data[:len(data) // 2])
    assert rows[:len(shorter)] == shorter


def test_split_is_chronological_and_complete():
    train, validation, test = chronological_split(100)
    assert (train.stop, validation.start, validation.stop, test.start, test.stop) == (60, 60, 80, 80, 100)


def test_precision_recall():
    labels = np.array([1, 1, 0, 0])
    assert precision_recall([True, False, True, False], labels) == {"precision": 0.5, "recall": 0.5, "flagged_percent": 50.0}
    assert precision_recall([False] * 4, labels)["precision"] == 0


def test_threshold_reaches_target_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([1, 1, 0, 1, 0])
    assert threshold_for_precision(scores, labels, 1.0) == 0.8
    assert threshold_for_precision(scores, labels, 0.75) == 0.6
    assert threshold_for_precision(scores, np.zeros(5, dtype=int), 0.9) == 0.9


def test_input_validation():
    good = {**txn()}
    assert TransactionIn(**good).amount == 20
    for bad in ({"amount": -5}, {"amount": 0}, {"lat": 200}, {"id": ""}, {"ts": -1}):
        with pytest.raises(ValueError):
            TransactionIn(**{**good, **bad})
    with pytest.raises(ValueError):
        ReviewIn(verdict="maybe", analyst="a")


def test_to_dict_handles_both_forms():
    item = simulator.generate(days=1, cards=5)[0]
    assert to_dict(item)["id"] == item.id
    assert to_dict({"id": "x"}) == {"id": "x"}
