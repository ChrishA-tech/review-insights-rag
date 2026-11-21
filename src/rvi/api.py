"""FastAPI service. Run: uvicorn rvi.api:app --port 8080"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import tools

app = FastAPI(title="Review insights", version="1.0.0")


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health() -> dict:
    d, _, s = tools._corpus()
    return {"status": "ok", "reviews": len(d), "retriever": s["retriever"]}


@app.get("/search")
def search(q: str, department: str | None = None, max_rating: int | None = None, k: int = 6) -> dict:
    r = tools.search_reviews(q, department, max_rating, k)
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/aspects")
def aspects(department: str | None = None) -> dict:
    r = tools.get_aspect_stats(department)
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/odds-ratios")
def odds() -> dict:
    return tools.get_odds_ratios()


class Batch(BaseModel):
    instances: list[dict]


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
def predict(b: Batch) -> dict:
    return {"predictions": [tools.search_reviews(i.get("query", ""), i.get("department"), i.get("max_rating"))
                            for i in b.instances]}
