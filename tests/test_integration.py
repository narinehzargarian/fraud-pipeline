"""Runs against real Redis, PostgreSQL, and Kafka. Skipped unless RUN_INTEGRATION is set."""
import json
import os
import pathlib
import subprocess
import sys
import uuid

import pytest
import redis
from fastapi.testclient import TestClient

from fraud import db, features, simulator
from fraud.features import MemoryStore, RedisStore
from fraud.scorer import Scorer, to_dict
from fraud.stream import handle

pytestmark = pytest.mark.skipif("RUN_INTEGRATION" not in os.environ, reason="needs the services")

MODEL = pathlib.Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))


@pytest.fixture(scope="module")
def client():
    return redis.Redis.from_url("redis://localhost:6380/0", decode_responses=True)


@pytest.fixture(scope="module")
def conn():
    connection = db.connect()
    db.init_schema(connection)
    return connection


def unique_txn(card=None, ts=1_750_000_000.0, **changes):
    item = {"id": f"it-{uuid.uuid4()}", "card_id": card or f"it-{uuid.uuid4()}", "ts": ts, "amount": 25.0,
            "merchant_id": "retail-1", "category": "retail", "lat": 34.0, "lon": -118.0, "online": False}
    return {**item, **changes}


def test_redis_and_memory_give_identical_features(client):
    """The guard against training and live scoring drifting apart."""
    data = simulator.generate(days=20, cards=30, seed=11)
    prefix = uuid.uuid4().hex
    memory, live = MemoryStore(), RedisStore(client)
    compared = 0
    for transaction in data:
        txn = to_dict(transaction)
        txn["card_id"] = prefix + txn["card_id"]
        assert features.compute(txn, memory.history(txn["card_id"], txn["ts"])) == \
            features.compute(txn, live.history(txn["card_id"], txn["ts"]))
        memory.add(txn)
        live.add(txn)
        compared += 1
    assert compared > 500


def test_redis_ignores_repeats(client):
    store = RedisStore(client)
    txn = unique_txn()
    store.add(txn)
    store.add(txn)
    assert len(store.history(txn["card_id"], txn["ts"])) == 1


def test_redelivered_message_is_stored_once_and_scored_the_same(client, conn):
    scorer = Scorer(RedisStore(client), MODEL)
    card = f"it-{uuid.uuid4()}"
    first = unique_txn(card=card)
    assert handle(json.dumps(first), scorer, conn, lambda _: None) == "stored"
    assert handle(json.dumps(first), scorer, conn, lambda _: None) == "duplicate"
    decision_one = scorer.score(first)
    second = unique_txn(card=card, ts=first["ts"] + 60)
    scorer.score(second)
    # A late repeat of the first message must not see the event that came after it.
    assert scorer.score(first)["features"] == decision_one["features"]


def test_bad_message_goes_to_dead_letters(client, conn):
    scorer = Scorer(RedisStore(client), MODEL)
    dead = []
    for bad in (b"not json", json.dumps({"id": "x"}).encode(), json.dumps(unique_txn(amount=-1)).encode()):
        assert handle(bad, scorer, conn, dead.append) == "rejected"
    assert len(dead) == 3


def test_api_score_review_and_stats(conn):
    from fraud.api import app

    api = TestClient(app)
    card = f"it-{uuid.uuid4()}"
    decision = None
    for offset in range(8):  # Small repeated charges, the card-testing pattern.
        decision = api.post("/score", json=unique_txn(card=card, ts=1_750_000_000.0 + offset * 60, amount=1.5,
                                                       category="subscriptions", online=True)).json()
    assert "card_testing" in decision["rules"]
    assert decision["action"] in ("review", "decline")

    open_ids = [row["txn_id"] for row in api.get("/flagged", params={"limit": 200}).json()]
    assert decision["txn_id"] in open_ids

    assert api.post(f"/flagged/{decision['txn_id']}/review", json={"verdict": "fraud", "analyst": "test"}).status_code == 200
    assert decision["txn_id"] not in [row["txn_id"] for row in api.get("/flagged", params={"limit": 200}).json()]
    reviewed = {row["txn_id"]: row for row in api.get("/flagged", params={"status": "reviewed", "limit": 200}).json()}
    assert reviewed[decision["txn_id"]]["verdict"] == "fraud"

    # An analyst can change their mind, and the latest answer wins.
    assert api.post(f"/flagged/{decision['txn_id']}/review", json={"verdict": "legit", "analyst": "test"}).status_code == 200
    assert api.post("/flagged/does-not-exist/review", json={"verdict": "fraud", "analyst": "test"}).status_code == 404
    assert api.post(f"/flagged/{decision['txn_id']}/review", json={"verdict": "maybe", "analyst": "test"}).status_code == 422
    assert api.post("/score", json=unique_txn(amount=-5)).status_code == 422
    assert api.get("/flagged", params={"status": "nonsense"}).status_code == 422
    assert api.get("/stats").json()["reviewed"] >= 1


def test_kafka_pipeline_end_to_end(conn):
    """Real producer and consumer processes, connected through a real Kafka broker."""
    group = f"it-{uuid.uuid4().hex}"
    env = {**os.environ, "KAFKA_GROUP": group}
    before = conn.execute("SELECT count(*) FROM decisions").fetchone()[0]
    count = 300
    produced = subprocess.run([sys.executable, "-m", "fraud.stream", "produce", "--count", str(count), "--seed", str(uuid.uuid4().int % 100_000)],
                              env=env, capture_output=True, text=True, timeout=180, check=True)
    assert json.loads(produced.stdout.splitlines()[-1])["produced"] == count
    consumed = subprocess.run([sys.executable, "-m", "fraud.stream", "consume", "--exit-when-idle", "6"],
                              env=env, capture_output=True, text=True, timeout=300, check=True)
    summary = json.loads(consumed.stdout.splitlines()[-1])
    assert summary["stored"] + summary["duplicate"] >= count
    assert summary["rejected"] == 0
    assert conn.execute("SELECT count(*) FROM decisions").fetchone()[0] > before
