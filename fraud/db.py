"""Stores decisions and analyst reviews in PostgreSQL."""
import os

import psycopg
from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    txn_id      TEXT PRIMARY KEY,
    card_id     TEXT NOT NULL,
    ts          DOUBLE PRECISION NOT NULL,
    amount      DOUBLE PRECISION NOT NULL,
    merchant_id TEXT NOT NULL,
    category    TEXT NOT NULL,
    score       DOUBLE PRECISION NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('approve', 'review', 'decline')),
    rules       JSONB NOT NULL,
    features    JSONB NOT NULL,
    latency_ms  DOUBLE PRECISION NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS decisions_flagged ON decisions (score DESC) WHERE action <> 'approve';
CREATE TABLE IF NOT EXISTS reviews (
    txn_id     TEXT PRIMARY KEY REFERENCES decisions (txn_id),
    verdict    TEXT NOT NULL CHECK (verdict IN ('fraud', 'legit')),
    analyst    TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def connect(url=None):
    return psycopg.connect(url or os.environ.get("DATABASE_URL", "postgresql://fraud:localdevpassword@localhost:5433/fraud"), autocommit=True)


def init_schema(conn):
    conn.execute(SCHEMA)


def save_decision(conn, txn, decision):
    """Returns False if this transaction was already stored, so redelivery is harmless."""
    result = conn.execute(
        "INSERT INTO decisions (txn_id, card_id, ts, amount, merchant_id, category, score, action, rules, features, latency_ms) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (txn_id) DO NOTHING",
        (txn["id"], txn["card_id"], txn["ts"], txn["amount"], txn["merchant_id"], txn["category"], decision["score"],
         decision["action"], Jsonb(decision["rules"]), Jsonb(decision["features"]), decision["latency_ms"]),
    )
    return result.rowcount == 1


def flagged(conn, status="open", limit=50):
    condition = {"open": "r.txn_id IS NULL", "reviewed": "r.txn_id IS NOT NULL", "all": "TRUE"}[status]
    cursor = conn.execute(
        "SELECT d.txn_id, d.card_id, d.ts, d.amount, d.merchant_id, d.category, d.score, d.action, d.rules, d.features, "
        "r.verdict, r.analyst, r.note FROM decisions d LEFT JOIN reviews r USING (txn_id) "
        f"WHERE d.action <> 'approve' AND {condition} ORDER BY d.score DESC, d.txn_id LIMIT %s",
        (limit,),
    )
    columns = [c.name for c in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def save_review(conn, txn_id, verdict, analyst, note=""):
    """Returns False if the transaction doesn't exist. A later review replaces an earlier one."""
    exists = conn.execute("SELECT 1 FROM decisions WHERE txn_id = %s", (txn_id,)).fetchone()
    if not exists:
        return False
    conn.execute(
        "INSERT INTO reviews (txn_id, verdict, analyst, note) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (txn_id) DO UPDATE SET verdict = EXCLUDED.verdict, analyst = EXCLUDED.analyst, "
        "note = EXCLUDED.note, decided_at = now()",
        (txn_id, verdict, analyst, note),
    )
    return True


def stats(conn):
    row = conn.execute(
        "SELECT count(*), "
        "count(*) FILTER (WHERE action = 'approve'), count(*) FILTER (WHERE action = 'review'), "
        "count(*) FILTER (WHERE action = 'decline'), "
        "COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), 0), "
        "COALESCE(percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms), 0) FROM decisions"
    ).fetchone()
    reviewed = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE verdict = 'fraud') FROM reviews"
    ).fetchone()
    return {
        "total": row[0], "approved": row[1], "in_review": row[2], "declined": row[3],
        "latency_p50_ms": round(row[4], 2), "latency_p99_ms": round(row[5], 2),
        "reviewed": reviewed[0], "confirmed_fraud": reviewed[1],
        "review_precision": round(reviewed[1] / reviewed[0], 3) if reviewed[0] else None,
    }
