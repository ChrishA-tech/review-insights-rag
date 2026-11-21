"""Women's E-Commerce Clothing Reviews (23,486 real, anonymised reviews; Nick Brooks ("nicapotato"), 2018, released CC0 on Kaggle)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MIRROR = ("https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/"
          "Womens%20Clothing%20E-Commerce%20Reviews.csv")


def load(path: str | Path = "data/reviews.csv") -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(path)
    log = {"raw_rows": len(d)}
    d = d.rename(columns={"Unnamed: 0": "review_id", "Clothing ID": "clothing_id", "Age": "age", "Title": "title",
                          "Review Text": "text", "Rating": "rating", "Recommended IND": "recommended",
                          "Positive Feedback Count": "helpful_votes", "Division Name": "division",
                          "Department Name": "department", "Class Name": "class_name"})
    log["no_text"] = int(d["text"].isna().sum())
    d = d[d["text"].notna()]
    dup = d.duplicated(subset=["text"])
    log["duplicate_text"] = int(dup.sum())
    d = d[~dup]
    log["no_department"] = int(d["department"].isna().sum())
    d = d[d["department"].notna()].copy()
    d["text"] = d["text"].str.strip()
    d["title"] = d["title"].fillna("").str.strip()
    d["not_recommended"] = 1 - d["recommended"]
    log["clean_rows"] = len(d)
    return d.reset_index(drop=True), log
