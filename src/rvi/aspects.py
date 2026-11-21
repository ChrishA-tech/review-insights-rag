"""Rule-based aspect tagging (transparent and auditable) and an odds-ratio model linking aspects to not recommending."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ASPECTS = {
    "runs_small": r"\b(runs?|ran|fits?) (very |really |a bit |a little |way )?small\b|\btoo (small|tight|snug)\b|\bsize(d)? up\b|\border(ed)? (a|one) size up\b",
    "runs_large": r"\b(runs?|ran|fits?) (very |really |a bit |a little |way )?(large|big)\b|\btoo (big|large|loose|baggy)\b|\bsize(d)? down\b|\bhuge\b|\bboxy\b",
    "fabric_quality": r"\b(thin|cheap(ly)?|flimsy|see[- ]?through|sheer|itchy|scratchy|pill(s|ed|ing)?|snag(s|ged)?|ripped|tore|unravel\w*|poor(ly)? (made|quality))\b",
    "length": r"\btoo (short|long)\b|\b(short|long)er than (expected|pictured|i expected)\b|\bhit(s)? (me )?(at|below|above) (the )?(knee|ankle|calf)\b",
    "colour_vs_photo": r"\b(colou?r|shade)s? (is|was|were|are) (different|off|not)\b|\b(not|nothing) like (the )?(picture|photo|image)\b|\b(looks?|looked) different\b|\bin person\b",
    "price": r"\b(expensive|overpriced|pricey|price ?tag|full price|on sale|worth the (price|money))\b",
    "comfort": r"\b(comfortable|comfy|soft|cozy|cosy)\b",
    "returning": r"\b(return(ed|ing)?( it| this| them)?|send(ing)? (it|this|them) back|sent (it|this|them) back)\b",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in ASPECTS.items()}


def tag(df: pd.DataFrame) -> pd.DataFrame:
    text = df["title"].fillna("") + ". " + df["text"]
    return pd.DataFrame({k: text.map(lambda t, p=p: int(bool(p.search(t)))) for k, p in _COMPILED.items()}, index=df.index)


def odds_ratios(df: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression: not_recommended ~ aspects + department + age band. Odds ratios with 95% CIs.
    'returning' is excluded: it is a consequence of not liking the item, not a cause."""
    cols = [c for c in tags.columns if c != "returning"]
    d = pd.concat([df[["not_recommended", "department", "age"]], tags[cols]], axis=1)
    d["age_band"] = pd.cut(d["age"], [0, 29, 39, 49, 59, 200], labels=["<30", "30s", "40s", "50s", "60+"]).astype(str)
    m = smf.logit("not_recommended ~ " + " + ".join(cols) + " + C(department) + C(age_band)", data=d).fit(disp=0)
    ci = m.conf_int()
    out = pd.DataFrame({"odds_ratio": np.exp(m.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]), "p_value": m.pvalues})
    out = out.loc[cols].copy()
    out["share_of_reviews"] = tags[cols].mean()
    out["not_recommended_rate_when_present"] = [df.loc[tags[c] == 1, "not_recommended"].mean() for c in cols]
    return out.round(4)


def by_department(df: pd.DataFrame, tags: pd.DataFrame) -> dict:
    out = {}
    for dep, g in df.groupby("department"):
        t = tags.loc[g.index]
        out[dep] = {"reviews": int(len(g)), "not_recommended_rate": round(float(g["not_recommended"].mean()), 4),
                    "avg_rating": round(float(g["rating"].mean()), 3),
                    "aspect_rates": {k: round(float(t[k].mean()), 4) for k in tags.columns}}
    return out
