"""REST API: synchronous scoring and the analyst review queue. Run with: uvicorn fraud.api:app"""
import os
from typing import Literal

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fraud import db
from fraud.features import RedisStore
from fraud.schemas import ReviewIn, TransactionIn
from fraud.scorer import Scorer

app = FastAPI(title="Fraud Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("DASHBOARD_ORIGIN", "http://localhost:5173")],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)
state = {}


def parts():
    if not state:
        client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"), decode_responses=True)
        state["scorer"] = Scorer(RedisStore(client), os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
        state["conn"] = db.connect()
        db.init_schema(state["conn"])
    return state


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(body: TransactionIn):
    context = parts()
    txn = body.model_dump()
    decision = context["scorer"].score(txn)
    db.save_decision(context["conn"], txn, decision)
    return decision


@app.get("/flagged")
def list_flagged(status: Literal["open", "reviewed", "all"] = "open", limit: int = 50):
    return db.flagged(parts()["conn"], status, min(max(limit, 1), 200))


@app.post("/flagged/{txn_id}/review")
def review(txn_id: str, body: ReviewIn):
    if not db.save_review(parts()["conn"], txn_id, body.verdict, body.analyst, body.note):
        raise HTTPException(status_code=404, detail="transaction not found")
    return {"txn_id": txn_id, "verdict": body.verdict}


@app.get("/stats")
def get_stats():
    return db.stats(parts()["conn"])
