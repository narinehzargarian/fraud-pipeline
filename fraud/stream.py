"""Kafka pipeline: a producer replays transactions, and a consumer scores and stores them.

Delivery is at-least-once: the offset is committed only after the decision is stored. A crash can
redeliver a message, but the card history and the decisions table both ignore repeats.

usage: python -m fraud.stream produce --count 20000 --rate 2000
       python -m fraud.stream consume
"""
import argparse
import dataclasses
import json
import os
import signal
import time

import pydantic
import redis
from confluent_kafka import Consumer, Producer

from fraud import db, simulator
from fraud.features import RedisStore
from fraud.schemas import TransactionIn
from fraud.scorer import Scorer

TOPIC = "transactions"
DEAD_LETTERS = "transactions-dead-letter"


def brokers():
    return os.environ.get("KAFKA_BROKERS", "localhost:9092")


def produce(count, rate, seed):
    # Rows are taken from the end of the data, the period the model never trained on.
    transactions = simulator.generate(seed=seed)[-count:]
    producer = Producer({"bootstrap.servers": brokers(), "enable.idempotence": True})
    start = time.monotonic()
    for number, transaction in enumerate(transactions):
        payload = dataclasses.asdict(transaction)
        # Sent without the answer. Scoring must not be able to see the label.
        payload.pop("is_fraud")
        payload.pop("pattern")
        # Keyed by card, so one card's transactions stay in order within a partition.
        producer.produce(TOPIC, key=transaction.card_id, value=json.dumps(payload))
        producer.poll(0)
        if rate and number % 100 == 0:
            ahead = number / rate - (time.monotonic() - start)
            if ahead > 0:
                time.sleep(ahead)
    producer.flush()
    print(json.dumps({"produced": len(transactions), "seconds": round(time.monotonic() - start, 1)}))


def handle(message_value, scorer, conn, dead_letter):
    """Processes one message. Returns 'stored', 'duplicate', or 'rejected'."""
    try:
        txn = TransactionIn.model_validate_json(message_value).model_dump()
    except pydantic.ValidationError:
        dead_letter(message_value)
        return "rejected"
    decision = scorer.score(txn)
    return "stored" if db.save_decision(conn, txn, decision) else "duplicate"


def consume(idle_exit_seconds=0.0):
    client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"), decode_responses=True)
    scorer = Scorer(RedisStore(client), os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    conn = db.connect()
    db.init_schema(conn)
    consumer = Consumer({
        "bootstrap.servers": brokers(), "group.id": os.environ.get("KAFKA_GROUP", "fraud-scorer"),
        "auto.offset.reset": "earliest", "enable.auto.commit": False,
    })
    dead_producer = Producer({"bootstrap.servers": brokers()})
    consumer.subscribe([TOPIC])

    stopping = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stopping.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))

    counts = {"stored": 0, "duplicate": 0, "rejected": 0}
    start = last_message = time.monotonic()
    while not stopping["now"]:
        message = consumer.poll(0.5)
        if message is None:
            if idle_exit_seconds and sum(counts.values()) and time.monotonic() - last_message > idle_exit_seconds:
                break
            continue
        if message.error():
            print(json.dumps({"kafka_error": str(message.error())}), flush=True)
            continue
        counts[handle(message.value(), scorer, conn, lambda value: dead_producer.produce(DEAD_LETTERS, value=value))] += 1
        consumer.commit(message=message, asynchronous=True)
        last_message = time.monotonic()
    consumer.close()
    dead_producer.flush()
    elapsed = max(last_message - start, 0.001)
    print(json.dumps({**counts, "seconds": round(elapsed, 1), "per_second": round(sum(counts.values()) / elapsed)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["produce", "consume"])
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exit-when-idle", type=float, default=0.0)
    args = parser.parse_args()
    if args.mode == "produce":
        produce(args.count, args.rate, args.seed)
    else:
        consume(args.exit_when_idle)


if __name__ == "__main__":
    main()
