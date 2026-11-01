"""Per-card rolling features. Training and live scoring run this same code.

A common way fraud models fail in production is computing features differently in the two places.
Here the only difference is where history is stored: in memory for training, Redis when live.
"""
import json
import math
from collections import defaultdict

HISTORY_SECONDS = 7 * 86_400
RISKY_CATEGORIES = {"electronics", "gift_cards"}
SMALL_AMOUNT = 5.0

FEATURE_NAMES = [
    "amount", "log_amount", "online", "hour", "risky_category",
    "count_1h", "count_24h", "amount_1h", "amount_24h", "distinct_merchants_1h", "small_count_1h",
    "history_count_7d", "average_amount_7d", "amount_ratio_7d", "new_merchant_7d",
    "seconds_since_last", "km_from_last", "speed_kmh",
]


def km_between(lat_a, lon_a, lat_b, lon_b):
    """Great-circle distance, using the haversine formula."""
    lat_a, lon_a, lat_b, lon_b = map(math.radians, (lat_a, lon_a, lat_b, lon_b))
    h = math.sin((lat_b - lat_a) / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin((lon_b - lon_a) / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def compute(txn, history):
    """Features for one transaction, using only events from before it.

    Filtering here also makes redelivered messages safe: a transaction scored a second time sees
    the same past and gets the same features, even if later events have arrived since.
    """
    past = sorted((e for e in history if e["ts"] <= txn["ts"] and e["id"] != txn["id"]), key=lambda e: e["ts"])
    last_hour = [e for e in past if e["ts"] >= txn["ts"] - 3600]
    last_day = [e for e in past if e["ts"] >= txn["ts"] - 86_400]
    average = sum(e["amount"] for e in past) / len(past) if past else 0.0
    seconds_since_last, km_from_last, speed = -1.0, -1.0, 0.0
    in_person = [e for e in past if not e["online"]]
    if past:
        seconds_since_last = txn["ts"] - past[-1]["ts"]
    if in_person and not txn["online"]:
        last = in_person[-1]
        km_from_last = km_between(last["lat"], last["lon"], txn["lat"], txn["lon"])
        speed = km_from_last / (max(txn["ts"] - last["ts"], 60.0) / 3600)
    return {
        "amount": txn["amount"],
        "log_amount": math.log1p(txn["amount"]),
        "online": float(txn["online"]),
        "hour": (txn["ts"] % 86_400) / 3600,
        "risky_category": float(txn["category"] in RISKY_CATEGORIES),
        "count_1h": float(len(last_hour)),
        "count_24h": float(len(last_day)),
        "amount_1h": sum(e["amount"] for e in last_hour),
        "amount_24h": sum(e["amount"] for e in last_day),
        "distinct_merchants_1h": float(len({e["merchant_id"] for e in last_hour})),
        "small_count_1h": float(sum(e["amount"] < SMALL_AMOUNT for e in last_hour)),
        "history_count_7d": float(len(past)),
        "average_amount_7d": average,
        "amount_ratio_7d": txn["amount"] / average if average > 0 else 0.0,
        "new_merchant_7d": float(txn["merchant_id"] not in {e["merchant_id"] for e in past}),
        "seconds_since_last": seconds_since_last,
        "km_from_last": km_from_last,
        "speed_kmh": speed,
    }


def _event(txn):
    return {k: txn[k] for k in ("id", "ts", "amount", "merchant_id", "lat", "lon", "online")}


class MemoryStore:
    def __init__(self):
        self.events = defaultdict(dict)

    def history(self, card_id, ts):
        return [e for e in self.events[card_id].values() if e["ts"] >= ts - HISTORY_SECONDS]

    def add(self, txn):
        # Keyed by transaction ID, so adding the same one twice changes nothing.
        self.events[txn["card_id"]][txn["id"]] = _event(txn)
        cutoff = txn["ts"] - HISTORY_SECONDS
        if len(self.events[txn["card_id"]]) > 256:
            self.events[txn["card_id"]] = {k: e for k, e in self.events[txn["card_id"]].items() if e["ts"] >= cutoff}


class RedisStore:
    """One sorted set per card, ordered by time. Each entry is unique, so retries are harmless."""

    def __init__(self, client):
        self.client = client

    def history(self, card_id, ts):
        raw = self.client.zrangebyscore(f"card:{card_id}", ts - HISTORY_SECONDS, "+inf")
        return [json.loads(item) for item in raw]

    def add(self, txn):
        key = f"card:{txn['card_id']}"
        pipe = self.client.pipeline()
        pipe.zadd(key, {json.dumps(_event(txn), sort_keys=True): txn["ts"]})
        pipe.zremrangebyscore(key, "-inf", txn["ts"] - HISTORY_SECONDS)
        pipe.expire(key, HISTORY_SECONDS)
        pipe.execute()
