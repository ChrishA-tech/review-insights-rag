"""Score the agent on eval/questions.jsonl.

  python eval/run_eval.py --backend ollama --model qwen2.5:7b      # real LLM (run on your machine)
  python eval/run_eval.py --backend scripted                       # plumbing test only, not a model result
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
from rvi.agent import CITE, OllamaAgent, ScriptedAgent, bad_citations, numbers  # noqa: E402


def correct(q, r) -> bool:
    if q["expected"] is None:
        return r.abstained
    if q["expected"] == "citations":  # grounded answer: must cite at least 2 retrieved reviews and pass the checks
        return (not r.abstained) and r.verified and len(set(CITE.findall(r.answer))) >= 2
    if r.abstained:
        return False
    got = numbers(r.answer)
    return all(any(abs(g - e) <= max(0.051, 0.01 * abs(e)) for g in got) for e in q["expected"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "scripted"], default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--no-verifier", action="store_true", help="ablation: skip the number check")
    a = ap.parse_args()
    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text().splitlines() if l.strip()]
    agent = ScriptedAgent() if a.backend == "scripted" else OllamaAgent(a.model)
    if a.no_verifier and a.backend == "ollama":
        import rvi.agent as ag
        ag.unsupported = lambda *x, **k: []
        ag.bad_citations = lambda *x, **k: []
    rows = []
    for q in qs:
        r = agent.ask(q["question"])
        rows.append({"id": q["id"], "type": q["type"], "correct": correct(q, r), "abstained": r.abstained,
                     "retried": r.retried, "verified": r.verified, "tools": [t["name"] for t in r.tool_calls],
                     "latency_s": r.latency_s, "answer": r.answer})
        print(f"{'OK ' if rows[-1]['correct'] else 'BAD'} {q['id']:<22} {r.answer[:90]}")
    ans = [x for x in rows if x["type"] != "abstain"]
    un = [x for x in rows if x["type"] == "abstain"]
    summary = {"backend": a.backend, "model": a.model if a.backend == "ollama" else "scripted (not an LLM)",
               "verifier": not a.no_verifier, "questions": len(rows),
               "accuracy_answerable": round(sum(x["correct"] for x in ans) / len(ans), 3),
               "correct_abstention": round(sum(x["correct"] for x in un) / len(un), 3),
               "false_abstention": round(sum(x["abstained"] for x in ans) / len(ans), 3),
               "retries": sum(x["retried"] for x in rows), "blocked_by_verifier": sum(not x["verified"] for x in rows),
               "median_latency_s": statistics.median(x["latency_s"] for x in rows),
               "by_type": {t: round(sum(x["correct"] for x in rows if x["type"] == t) / sum(1 for x in rows if x["type"] == t), 3)
                           for t in sorted({x["type"] for x in rows})}}
    tag = "scripted" if a.backend == "scripted" else a.model.replace(":", "_").replace("/", "_") + ("_noverifier" if a.no_verifier else "")
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/agent_eval_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
