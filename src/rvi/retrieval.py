"""Three retrievers over review text: BM25 (lexical), LSA (TF-IDF + truncated SVD, dense) and a hybrid fused with
reciprocal rank fusion (RRF, k=60). Evaluated label-free as known-item search: a review's own title is the query and the
review text (title removed) is the target."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TOKEN = re.compile(r"[a-z0-9']+")
STOP = set("a an the and or but if of to in on for with at by from is are was were be been it this that i me my we you "
           "she her they them so very just".split())


def tokens(s: str) -> list[str]:
    return [t for t in TOKEN.findall(s.lower()) if t not in STOP]


class Index:
    def __init__(self, texts: list[str], dims: int = 200, seed: int = 0, min_df: int = 2):
        self.bm25 = BM25Okapi([tokens(t) for t in texts])
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, stop_words="english")
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.dense = normalize(self.svd.fit_transform(X))
        self.n = len(texts)

    def bm25_scores(self, q: str) -> np.ndarray:
        return self.bm25.get_scores(tokens(q))

    def dense_scores(self, q: str) -> np.ndarray:
        v = normalize(self.svd.transform(self.tfidf.transform([q])))
        return self.dense @ v[0]

    def search(self, q: str, k: int = 10, method: str = "hybrid", mask: np.ndarray | None = None) -> list[int]:
        def rank(s):
            s = s.copy()
            if mask is not None:
                s[~mask] = -np.inf
            return np.argsort(-s)
        if method == "bm25":
            return rank(self.bm25_scores(q))[:k].tolist()
        if method == "lsa":
            return rank(self.dense_scores(q))[:k].tolist()
        rb, rd = rank(self.bm25_scores(q))[:200], rank(self.dense_scores(q))[:200]
        fused: dict[int, float] = {}
        for r in (rb, rd):
            for pos, i in enumerate(r):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + pos + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:k]


def query_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Titles of at least 4 words that are unique in the corpus."""
    t = df["title"].str.strip()
    return df[(t.str.split().str.len() >= 4) & ~t.str.lower().duplicated(keep=False)]


def known_item_eval(df: pd.DataFrame, idx: Index, sample: pd.DataFrame, ks=(1, 5, 10)) -> dict:
    """Query = a review's title; target = that review's text (titles are not indexed)."""
    out = {}
    for m in ["bm25", "lsa", "hybrid"]:
        ranks = []
        for i, row in sample.iterrows():
            res = idx.search(row["title"], k=100, method=m)
            ranks.append(res.index(i) + 1 if i in res else None)
        r = np.array([x if x else 10**9 for x in ranks])
        out[m] = {**{f"recall@{k}": round(float((r <= k).mean()), 4) for k in ks},
                  "mrr@100": round(float(np.mean([1 / x if x <= 100 else 0 for x in r])), 4)}
    out["queries"] = len(sample)
    return out
