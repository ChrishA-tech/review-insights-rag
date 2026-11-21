"""Agent evaluation set: statistics questions with known answers, retrieval questions scored on citation validity,
and questions the data cannot answer."""
import json
from pathlib import Path

s = json.loads(Path("artifacts/stats.json").read_text())
qs = []
for dep in ["Dresses", "Tops", "Bottoms", "Jackets"]:
    v = s["by_department"][dep]
    qs.append({"id": f"nr_{dep}", "type": "stat", "question": f"What share of {dep.lower()} reviews do not recommend the item? Give the not recommended rate.", "expected": [v["not_recommended_rate"]]})
    qs.append({"id": f"small_{dep}", "type": "stat", "question": f"In {dep.lower()}, what share of reviews say the item runs small (runs_small rate)?", "expected": [v["aspect_rates"]["runs_small"]]})
for a in ["fabric_quality", "runs_large", "comfort", "length"]:
    qs.append({"id": f"or_{a}", "type": "stat", "question": f"What is the odds ratio for {a.replace('_', ' ')} on not recommending?", "expected": [s["odds_ratios"][a]["odds_ratio"]]})
for i, q in enumerate(["What do customers say about dresses that run small?", "Why do people return jackets?",
                       "What complaints are there about thin or see-through tops?", "What do reviewers like about comfortable bottoms?",
                       "What do customers say when the colour looks different from the photo?", "Are there complaints about pilling sweaters in tops?"]):
    qs.append({"id": f"rag_{i}", "type": "grounded", "question": q, "expected": "citations"})
for i, q in enumerate(["Which brand makes the dresses with the most complaints?", "How much did returns cost the company in 2023?",
                       "What does the retailer's CEO think about sizing?", "Which competitor has better reviews?"]):
    qs.append({"id": f"abstain_{i}", "type": "abstain", "question": q, "expected": None})
Path("eval/questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
print(len(qs), "questions")
