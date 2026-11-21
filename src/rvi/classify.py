"""Predict 'not recommended' from review text. Products are split by clothing ID so no product appears in both
train and test (reviews of the same item share vocabulary and would leak)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    g = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(g.split(df, groups=df["clothing_id"]))
    return df.iloc[tr], df.iloc[te]


def fit(train: pd.DataFrame):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    X = vec.fit_transform(train["title"] + ". " + train["text"])
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, train["not_recommended"])
    return vec, clf


def evaluate(vec, clf, test: pd.DataFrame) -> dict:
    p = clf.predict_proba(vec.transform(test["title"] + ". " + test["text"]))[:, 1]
    y = test["not_recommended"].to_numpy()
    lo, hi = _boot_auc(y, p)
    terms = np.array(vec.get_feature_names_out()); w = clf.coef_[0]
    return {"roc_auc": round(roc_auc_score(y, p), 4), "roc_auc_ci95": [lo, hi], "pr_auc": round(average_precision_score(y, p), 4),
            "base_rate": round(float(y.mean()), 4), "f1_at_0.5": round(f1_score(y, p >= 0.5), 4),
            "top_not_recommended_terms": terms[np.argsort(w)[-15:][::-1]].tolist(),
            "top_recommended_terms": terms[np.argsort(w)[:15]].tolist()}


def _boot_auc(y, p, n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            a.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def rating_disagreement(df: pd.DataFrame) -> dict:
    """How often the star rating and the recommend flag disagree (a label-noise check)."""
    return {"rated_4_or_5_but_not_recommended": int(((df["rating"] >= 4) & (df["recommended"] == 0)).sum()),
            "rated_1_or_2_but_recommended": int(((df["rating"] <= 2) & (df["recommended"] == 1)).sum()),
            "rated_3_recommended_share": round(float(df.loc[df["rating"] == 3, "recommended"].mean()), 4)}
