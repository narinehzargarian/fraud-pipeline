"""Explainable rules, and how they combine with the model score into a decision.

A rule can send a transaction to review, but can't decline it alone. Legitimate travel and big
purchases trip these rules, and declining a real customer's card has a real cost.
"""

RULES = {
    "impossible_travel": lambda f: f["km_from_last"] > 500 and f["speed_kmh"] > 1000,
    "card_testing": lambda f: f["small_count_1h"] >= 3,
    "high_velocity": lambda f: f["count_1h"] >= 5,
    "amount_spike": lambda f: f["history_count_7d"] >= 5 and f["amount_ratio_7d"] >= 10,
    "risky_new_merchant": lambda f: f["risky_category"] == 1 and f["new_merchant_7d"] == 1 and f["amount"] >= 200,
}


def triggered_rules(features):
    return [name for name, rule in RULES.items() if rule(features)]


def decide(score, rules, review_threshold, decline_threshold):
    """Returns 'decline', 'review', or 'approve'."""
    if score >= decline_threshold:
        return "decline"
    if score >= review_threshold or rules:
        return "review"
    return "approve"
