"""Tools for the review analyst agent. The index is built once from the cleaned reviews."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import data, retrieval

STATS = Path("artifacts/stats.json")


@lru_cache(maxsize=1)
def _corpus():
    d, _ = data.load()
    stats = json.loads(STATS.read_text())
    return d, retrieval.Index(d["text"].tolist()), stats


def search_reviews(query: str, department: str | None = None, max_rating: int | None = None, k: int = 6) -> dict:
    d, idx, stats = _corpus()
    mask = np.ones(len(d), bool)
    if department:
        mask &= d["department"].str.lower().to_numpy() == department.strip().lower()
    if max_rating:
        mask &= d["rating"].to_numpy() <= int(max_rating)
    if not mask.any():
        return {"error": f"no reviews match department={department} max_rating={max_rating}"}
    hits = idx.search(query, k=int(min(k, 10)), method=stats["retriever"], mask=mask)
    return {"query": query, "retriever": stats["retriever"], "reviews": [
        {"review_id": int(d.at[i, "review_id"]), "rating": int(d.at[i, "rating"]), "department": d.at[i, "department"],
         "recommended": bool(d.at[i, "recommended"]), "text": d.at[i, "text"][:600]} for i in hits]}


def get_aspect_stats(department: str | None = None) -> dict:
    _, _, stats = _corpus()
    if not department:
        return {"scope": "all departments", "aspect_rates": stats["aspect_rates_all"], "reviews": stats["reviews"],
                "not_recommended_rate": stats["not_recommended_rate"]}
    for dep, v in stats["by_department"].items():
        if dep.lower() == department.strip().lower():
            return {"department": dep, **v}
    return {"error": f"unknown department {department}; options: {sorted(stats['by_department'])}"}


def get_odds_ratios() -> dict:
    _, _, stats = _corpus()
    return {"model": "logistic regression: not recommended ~ aspects + department + age band",
            "note": "associations, not causes", "odds_ratios": stats["odds_ratios"]}


REGISTRY = {"search_reviews": search_reviews, "get_aspect_stats": get_aspect_stats, "get_odds_ratios": get_odds_ratios}
SCHEMAS = [
    {"type": "function", "function": {"name": "search_reviews", "description": "Search customer reviews. Optional department (Tops, Dresses, Bottoms, Intimate, Jackets, Trend) and max_rating filters.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "department": {"type": "string"}, "max_rating": {"type": "integer"}, "k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_aspect_stats", "description": "Share of reviews mentioning each aspect (runs_small, runs_large, fabric_quality, length, colour_vs_photo, price, comfort, returning), overall or for one department.",
     "parameters": {"type": "object", "properties": {"department": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_odds_ratios", "description": "How much each aspect raises or lowers the odds of a customer not recommending the item.",
     "parameters": {"type": "object", "properties": {}}}},
]
