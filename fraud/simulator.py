"""Generates card transactions with labeled fraud patterns.

Public fraud datasets anonymize away card IDs, so they can't support per-card features like
"uses in the last hour". This simulator provides that history. To keep the problem honest, some
legitimate behavior looks suspicious (travel, big purchases, shopping bursts), and some fraud looks
normal ("quiet" fraud), so a perfect score isn't possible.
"""
import dataclasses
import math

import numpy as np

DAY = 86_400
CATEGORIES = ["groceries", "restaurants", "gas", "retail", "travel", "electronics", "gift_cards", "subscriptions"]
COMMON = ["groceries", "restaurants", "gas", "retail", "subscriptions"]
ONLINE_CATEGORIES = {"subscriptions", "electronics", "gift_cards", "retail"}
CITIES = [(34.05, -118.24), (40.71, -74.01), (41.88, -87.63), (29.76, -95.37), (47.61, -122.33), (25.76, -80.19)]
FAR_CITIES = [(51.51, -0.13), (35.68, 139.69), (-23.55, -46.63), (19.08, 72.88), (6.52, 3.38)]


@dataclasses.dataclass
class Transaction:
    id: str
    card_id: str
    ts: float
    amount: float
    merchant_id: str
    category: str
    lat: float
    lon: float
    online: bool
    is_fraud: bool = False
    pattern: str = "legit"


def _merchant(rng, category):
    return f"{category}-{rng.integers(0, 40)}"


def _jitter(rng, place, spread=0.15):
    return float(place[0] + rng.normal(0, spread)), float(place[1] + rng.normal(0, spread))


def _legit_day(rng, card_id, home, spend_scale, start):
    """One ordinary day for a card, including some behavior that resembles fraud."""
    events = []
    traveling = rng.random() < 0.01
    place = FAR_CITIES[rng.integers(0, len(FAR_CITIES))] if traveling else home
    count = rng.poisson(1.6)
    if rng.random() < 0.01:
        count += rng.integers(5, 10)  # A real shopping trip with many purchases.
    for _ in range(count):
        category = str(rng.choice(COMMON))
        amount = float(rng.lognormal(math.log(spend_scale), 0.7))
        if rng.random() < 0.004:
            category = str(rng.choice(["electronics", "travel"]))  # A real big purchase.
            amount *= float(rng.uniform(8, 30))
        online = category in ONLINE_CATEGORIES and rng.random() < 0.4
        lat, lon = _jitter(rng, home if online else place)
        events.append(Transaction(
            "", card_id, start + float(rng.uniform(6 * 3600, 23 * 3600)), round(amount, 2),
            _merchant(rng, category), category, lat, lon, online,
        ))
    return events


def _fraud_episode(rng, card_id, home, spend_scale, start):
    """One fraud event on a card, using one of four patterns."""
    pattern = str(rng.choice(["card_testing", "burst", "big_purchase", "quiet"], p=[0.25, 0.25, 0.25, 0.25]))
    when = start + float(rng.uniform(0, DAY - 7200))
    far = FAR_CITIES[rng.integers(0, len(FAR_CITIES))]
    events = []

    def add(amount, category, place, online, offset):
        lat, lon = _jitter(rng, place)
        events.append(Transaction(
            "", card_id, when + offset, round(float(amount), 2), _merchant(rng, category), category,
            lat, lon, online, True, pattern,
        ))

    if pattern == "card_testing":
        offset = 0.0
        for _ in range(rng.integers(3, 8)):
            add(rng.uniform(0.5, 3.0), "subscriptions", home, True, offset)
            offset += float(rng.uniform(30, 400))
        add(rng.uniform(200, 900), "gift_cards", home, True, offset)
    elif pattern == "burst":
        offset = 0.0
        for _ in range(rng.integers(3, 7)):
            add(rng.uniform(80, 500), str(rng.choice(["electronics", "gift_cards"])), far, False, offset)
            offset += float(rng.uniform(120, 900))
    elif pattern == "big_purchase":
        add(spend_scale * rng.uniform(15, 60), "electronics", far, False, 0.0)
    else:
        category = str(rng.choice(COMMON))
        add(rng.lognormal(math.log(spend_scale), 0.5), category, home, category in ONLINE_CATEGORIES, 0.0)
    return events


def generate(days=90, cards=2000, seed=7, fraud_episodes_per_card_day=0.0012, start_ts=1_750_000_000):
    """Returns transactions in time order. The same seed always gives the same data."""
    rng = np.random.default_rng(seed)
    transactions = []
    for index in range(cards):
        card_id = f"card-{index:05d}"
        home = CITIES[rng.integers(0, len(CITIES))]
        spend_scale = float(rng.lognormal(math.log(35), 0.5))
        for day in range(days):
            start = start_ts + day * DAY
            transactions += _legit_day(rng, card_id, home, spend_scale, start)
            if rng.random() < fraud_episodes_per_card_day:
                transactions += _fraud_episode(rng, card_id, home, spend_scale, start)
    transactions.sort(key=lambda t: t.ts)
    for number, transaction in enumerate(transactions):
        transaction.id = f"txn-{number:08d}"
    return transactions
