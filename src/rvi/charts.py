import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("reports")


def main():
    r = json.loads((R / "results.json").read_text())
    o = r["aspects"]["odds_ratios"]
    keys = sorted(o, key=lambda k: o[k]["odds_ratio"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, k in enumerate(keys):
        v = o[k]
        ax.errorbar(v["odds_ratio"], i, xerr=[[v["odds_ratio"] - v["ci_low"]], [v["ci_high"] - v["odds_ratio"]]], fmt="ko", capsize=3)
        ax.text(v["ci_high"] * 1.05, i, f'{v["odds_ratio"]:.2f}  ({100 * v["share_of_reviews"]:.1f}% of reviews)', va="center", fontsize=8)
    ax.set_xscale("log"); ax.axvline(1, color="0.6", lw=0.8)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=9)
    ax.set_xlabel("Odds ratio for 'not recommended' (95% CI, log scale)"); ax.set_xlim(0.2, 10)
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_title("What a review mentions vs whether the customer recommends")
    fig.tight_layout(); fig.savefig(R / "01_odds_ratios.png", dpi=150); plt.close(fig)

    t = r["retrieval"]["test"]; ms = ["bm25", "hybrid", "lsa"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["BM25", "Hybrid (RRF)", "LSA dense"], [100 * t[m]["recall@10"] for m in ms], color=["k", "0.55", "0.8"])
    for i, m in enumerate(ms):
        ax.text(i, 100 * t[m]["recall@10"] + 0.4, f'{100 * t[m]["recall@10"]:.1f}%', ha="center", fontsize=9)
    ax.set_ylabel("Recall@10 (%)"); ax.set_title(f"Known-item search, {t['queries']:,} held-out title queries")
    fig.tight_layout(); fig.savefig(R / "02_retrieval.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
