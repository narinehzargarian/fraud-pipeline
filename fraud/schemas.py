"""Validates transactions at the edges of the system, so bad input can't reach the model."""
from typing import Literal

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    card_id: str = Field(min_length=1, max_length=64)
    ts: float = Field(gt=0)
    amount: float = Field(gt=0, le=1_000_000)
    merchant_id: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=32)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    online: bool


class ReviewIn(BaseModel):
    verdict: Literal["fraud", "legit"]
    analyst: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)
