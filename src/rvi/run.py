"""End-to-end: clean -> aspects and odds ratios -> classifier -> retrieval benchmark -> artifacts and report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import aspects, classify, data, retrieval


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/reviews.csv")
    ap.add_argument("--val-queries", type=int, default=300)
    ap.add_argument("--test-queries", type=int, default=1000)
    a = ap.parse_args(argv)
    R, A = Path("reports"), Path("artifacts"); R.mkdir(exist_ok=True); A.mkdir(exist_ok=True)
    d, log = data.load(a.data)
    tags = aspects.tag(d)
    orr = aspects.odds_ratios(d, tags)
    deps = aspects.by_department(d, tags)
    tr, te = classify.split(d)
    vec, clf = classify.fit(tr)
    cls = classify.evaluate(vec, clf, te)
    cls.update({"train_reviews": len(tr), "test_reviews": len(te), "train_products": int(tr["clothing_id"].nunique()),
                "test_products": int(te["clothing_id"].nunique()),
                "products_in_both": len(set(tr["clothing_id"]) & set(te["clothing_id"]))})

    idx = retrieval.Index(d["text"].tolist())
    pool = retrieval.query_pool(d)
    val = pool.sample(min(a.val_queries, len(pool)), random_state=1)
    test = pool.drop(val.index).sample(min(a.test_queries, len(pool) - len(val)), random_state=0)
    r_val = retrieval.known_item_eval(d, idx, val)
    champ = max(["bm25", "lsa", "hybrid"], key=lambda m: r_val[m]["mrr@100"])  # chosen on validation queries only
    r_test = retrieval.known_item_eval(d, idx, test)

    # audit sheet: a random sample per aspect for a human to mark correct / incorrect
    rows = []
    for k in tags.columns:
        hit = d[tags[k] == 1]
        for _, r in hit.sample(min(20, len(hit)), random_state=0).iterrows():
            rows.append({"aspect": k, "review_id": int(r["review_id"]), "text": r["text"][:300], "correct (y/n)": ""})
    import pandas as pd
    pd.DataFrame(rows).to_csv("eval/aspect_audit.csv", index=False)

    stats = {"reviews": len(d), "not_recommended_rate": round(float(d["not_recommended"].mean()), 4),
             "aspect_rates_all": {k: round(float(tags[k].mean()), 4) for k in tags.columns},
             "by_department": deps, "odds_ratios": orr.drop(columns=["p_value"]).to_dict("index"), "retriever": champ}
    (A / "stats.json").write_text(json.dumps(stats, indent=1))
    res = {"data": {"source": "Women's E-Commerce Clothing Reviews (Nick Brooks, 'nicapotato' on Kaggle, 2018; CC0)", **log,
                    "products": int(d["clothing_id"].nunique()), "departments": sorted(deps)},
           "label_check": classify.rating_disagreement(d), "aspects": {"rates": stats["aspect_rates_all"],
           "odds_ratios": orr.to_dict("index")}, "classifier": cls,
           "retrieval": {"protocol": "known-item: review title as query, review text as target; disjoint validation and test queries",
                         "validation": r_val, "champion_chosen_on_validation": champ, "test": r_test}}
    (R / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ["data", "classifier", "retrieval"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
