"""Tool-calling planning agent with a number verifier.

Backends:
  ollama   - any OpenAI-compatible chat endpoint (default http://localhost:11434/v1, e.g. Ollama with qwen2.5:7b)
  scripted - a deterministic keyword router used only to test the plumbing in CI. It is NOT an LLM and its
             scores must never be reported as model results.

Guardrail: every number in the final answer must match a number returned by a tool (or written in the question).
If not, the agent gets one retry with the unsupported numbers listed, then abstains.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

from . import tools

SYSTEM = ("You are a customer-insights analyst for an online clothing retailer. Use the tools. Quote numbers only from "
          "tool results. When you describe what customers say, cite review IDs like [1234] from search results, and only "
          "put text in double quotes if it is copied exactly from a review. If the tools cannot answer (brands, dates, "
          "costs, anything not in the reviews), reply exactly: 'I can't answer that from the available data.' "
          "Keep answers to 2-4 sentences.")
ABSTAIN = "I can't answer that from the available data."
NUM = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _flatten(obj) -> list[float]:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _flatten(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _flatten(v)]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return [float(obj)]
    if isinstance(obj, str):
        return numbers(obj)
    return []


CITE = re.compile(r"\[(\d{1,6}(?:\s*,\s*\d{1,6})*)\]")            # [12] or [12, 345]
CITE_WORDS = re.compile(r"\breviews?\s+(?:IDs?\s*)?#?(\d{2,6}(?:\s*(?:,|and)\s*#?\d{2,6})*)", re.I)  # review ID 12 / reviews 12 and 34
QUOTE = re.compile(r'"([^"]{8,})"')


def cited_ids(answer: str) -> list[int]:
    ids = []
    for m in CITE.findall(answer or "") + CITE_WORDS.findall(answer or ""):
        ids += [int(x) for x in re.findall(r"\d+", m)]
    return ids


def strip_citations(answer: str) -> str:
    return CITE_WORDS.sub(" ", CITE.sub(" ", answer or ""))


def bad_citations(answer: str, evidence: list) -> list[str]:
    """Cited review IDs must come from a search result; quoted text must appear in one of the retrieved reviews."""
    seen = {r["review_id"]: r["text"].lower() for e in evidence if isinstance(e, dict)
            for r in (e.get("reviews") if isinstance(e.get("reviews"), list) else [])}
    bad = [f"[{c}]" for c in cited_ids(answer) if c not in seen]
    # only check real quotations (3+ words); quoted labels such as "runs_large" are not claims about review text
    bad += [f'"{q[:40]}"' for q in QUOTE.findall(answer or "")
            if len(q.split()) >= 3 and not any(q.lower().strip(" .,") in t for t in seen.values())]
    return bad


def unsupported(answer: str, evidence: list, question: str) -> list[float]:
    allowed = [x for e in evidence for x in _flatten(e)] + numbers(question)
    allowed += [a * 100 for a in allowed if abs(a) <= 1]  # 0.875 may be written as 87.5%
    if re.search(r"confidence|\bCI\b", answer or "", re.I):
        allowed += [90.0, 95.0, 99.0]  # the confidence level of a reported interval is not a data value
    bad = []
    for n in numbers(strip_citations(answer)):
        if not any(abs(n - a) <= max(0.051, 0.01 * abs(a)) for a in allowed):
            bad.append(n)
    return bad


@dataclass
class Result:
    question: str
    answer: str
    tool_calls: list = field(default_factory=list)
    verified: bool = True
    retried: bool = False
    abstained: bool = False
    latency_s: float = 0.0


def _call_tool(name: str, args: dict) -> dict:
    fn = tools.REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": str(e)}


class OllamaAgent:
    def __init__(self, model: str = "qwen2.5:7b", base_url: str | None = None, max_steps: int = 5, timeout: float = 120):
        self.model, self.max_steps = model, max_steps
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")).rstrip("/") + "/chat/completions"
        self.key = os.environ.get("OPENAI_API_KEY", "ollama")
        self.http = httpx.Client(timeout=timeout)

    def _chat(self, messages):
        r = self.http.post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                           json={"model": self.model, "messages": messages, "tools": tools.SCHEMAS, "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    def _loop(self, messages, res: Result, evidence: list) -> str:
        for _ in range(self.max_steps):
            msg = self._chat(messages)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                raw = c["function"].get("arguments") or "{}"
                args = json.loads(raw) if isinstance(raw, str) else raw
                out = _call_tool(name, args)
                res.tool_calls.append({"name": name, "args": args})
                evidence.append(out)
                messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(out)})
        return ABSTAIN

    def ask(self, question: str) -> Result:
        t = time.time()
        res, evidence = Result(question, ""), []
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        ans = self._loop(messages, res, evidence)
        bad = unsupported(ans, evidence, question) + bad_citations(ans, evidence)
        if bad:
            res.retried = True
            messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": f"These numbers are not in any tool result: {bad}. "
                             f"Answer again using only tool results, or say: {ABSTAIN}"})
            ans = self._loop(messages, res, evidence)
            if unsupported(ans, evidence, question) or bad_citations(ans, evidence):
                res.verified, ans = False, ABSTAIN
        res.answer, res.abstained = ans, ABSTAIN.lower()[:20] in ans.lower()
        res.latency_s = round(time.time() - t, 2)
        return res

class ScriptedAgent:
    """Keyword router for CI only (not an LLM)."""
    DEPTS = ["tops", "dresses", "bottoms", "intimate", "jackets", "trend"]

    def ask(self, question: str) -> Result:
        t, q = time.time(), question.lower()
        res, ev = Result(question, ""), []
        dep = next((x for x in self.DEPTS if x in q), None)
        if any(w in q for w in ["brand", "ceo", "2023", "cost the company", "profit", "competitor"]):
            res.answer, res.abstained = ABSTAIN, True
        elif "odds" in q:
            out = tools.get_odds_ratios(); res.tool_calls.append({"name": "get_odds_ratios", "args": {}})
            a = next((k for k in out["odds_ratios"] if k.replace("_", " ") in q or k in q), "fabric_quality")
            res.answer = f"{a} has an odds ratio of {out['odds_ratios'][a]['odds_ratio']} for not recommending."
        elif "share" in q or "how often" in q or "rate" in q:
            out = tools.get_aspect_stats(dep.title() if dep else None); res.tool_calls.append({"name": "get_aspect_stats", "args": {"department": dep}})
            a = next((k for k in out["aspect_rates"] if k.replace("_", " ") in q or k in q), None)
            if a is None and "not recommend" in q:
                res.answer = f"The not-recommended rate is {out['not_recommended_rate']}."
            elif a is None:
                res.answer, res.abstained = ABSTAIN, True
            else:
                res.answer = f"{a} appears in {out['aspect_rates'][a]} of reviews."
        else:
            out = tools.search_reviews(question, department=dep.title() if dep else None); res.tool_calls.append({"name": "search_reviews", "args": {}})
            ids = [r["review_id"] for r in out["reviews"][:3]]
            res.answer = "Relevant reviews: " + " ".join(f"[{i}]" for i in ids) + "."
            ev.append(out)
            res.verified = not bad_citations(res.answer, ev)
        res.latency_s = round(time.time() - t, 4)
        return res
