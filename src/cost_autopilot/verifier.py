"""Phase 3.1-3.2: quality checks and the verifier.

The verifier sends the same prompt to the reference (top-tier) model and
scores how well the cheap answer agrees with it, using a check that fits the
task:
  extraction      -> field recall against the reference's fields
  classification  -> does the label match the reference's label?
  summarization   -> LLM-as-judge score (1-5) vs the reference
  general         -> LLM-as-judge score (1-5) vs the reference
All scores are normalized to 0..1 so they can be compared and charted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .providers import ProviderError, Response, send_request
from .registry import Registry
from .tasks import Message, as_text

JUDGE_SYSTEM = ("You are grading an AI answer against a trusted reference answer for the same "
                "request. Judge whether the candidate would serve the user as well as the reference: "
                "correctness, completeness and following the instructions. Style differences do not "
                "matter. Reply with JSON only: {\"score\": <1-5>, \"reason\": \"<one sentence>\"}. "
                "5 = as good as the reference, 4 = minor omissions, 3 = noticeably worse, "
                "2 = largely wrong or incomplete, 1 = wrong.")


@dataclass
class QualityResult:
    score: float               # normalized 0..1
    threshold: float           # normalized 0..1
    passed: bool
    method: str
    details: dict = field(default_factory=dict)
    judge_cost: float = 0.0

    @property
    def gap(self) -> float:
        return round(max(0.0, self.threshold - self.score), 4)


def _first_json(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower().rstrip(".")


def _salient_tokens(text: str) -> set[str]:
    """Numbers, emails, ids and capitalized words: the things extraction is about."""
    pat = r"[\w.+-]+@[\w-]+\.[\w.]+|\$?\d[\d,.:/-]*|[A-Z][A-Za-z0-9-]{1,}|#\w+"
    return {t.lower().strip(".,") for t in re.findall(pat, text)}


def field_recall(candidate: str, reference: str) -> tuple[float, dict]:
    ref, cand = _first_json(reference), _first_json(candidate)
    if isinstance(ref, dict) and ref:
        cand = cand if isinstance(cand, dict) else {}
        keys = [k for k, v in ref.items() if v not in (None, "", [])]
        if not keys:
            return 1.0, {"fields": 0}
        hit = [k for k in keys if k in cand and _norm(cand[k]) == _norm(ref[k])]
        missing = [k for k in keys if k not in hit]
        return len(hit) / len(keys), {"fields": len(keys), "matched": len(hit), "missing": missing[:10]}
    rt, ct = _salient_tokens(reference), _salient_tokens(candidate)
    if not rt:
        rt, ct = set(_norm(reference).split()), set(_norm(candidate).split())
    if not rt:
        return 1.0, {"tokens": 0}
    return len(rt & ct) / len(rt), {"reference_items": len(rt), "matched": len(rt & ct)}


def _label(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    first = re.sub(r"^(label|category|answer|sentiment|classification)\s*[:\-]\s*", "", first, flags=re.I)
    return re.sub(r"[^a-z0-9 _-]", "", first.lower()).strip().split(" ")[0] if first else ""


def label_match(candidate: str, reference: str) -> tuple[bool, dict]:
    a, b = _label(candidate), _label(reference)
    ok = bool(b) and (a == b or re.search(rf"\b{re.escape(b)}\b", candidate.lower()[:200]) is not None)
    return ok, {"candidate_label": a, "reference_label": b}


def parse_judge(text: str) -> tuple[int | None, str]:
    j = _first_json(text)
    if isinstance(j, dict) and "score" in j:
        try:
            return max(1, min(5, int(round(float(j["score"]))))), str(j.get("reason", ""))
        except (TypeError, ValueError):
            pass
    m = re.search(r"\b([1-5])\b", text)
    return (int(m.group(1)), text[:200]) if m else (None, text[:200])


class Verifier:
    def __init__(self, registry: Registry, verification_cfg: dict):
        self.registry = registry
        self.cfg = verification_cfg

    @property
    def reference(self):
        return self.registry[self.cfg.get("reference_model", self.registry.baseline_name)]

    @property
    def judge(self):
        return self.registry[self.cfg.get("judge_model", self.cfg.get("reference_model", self.registry.baseline_name))]

    def _threshold(self, task: str) -> tuple[str, float]:
        t = (self.cfg.get("thresholds") or {}).get(task) or {}
        if task == "extraction":
            return "field_recall", float(t.get("min_field_recall", 0.8))
        if task == "classification":
            return "label_match", 1.0
        s = float(t.get("min_judge_score", 4))
        return "llm_judge", (s - 1) / 4

    async def score(self, prompt: str | list[Message], task: str, candidate: str,
                    reference: str) -> QualityResult:
        method, thr = self._threshold(task)
        if method == "field_recall":
            s, d = field_recall(candidate, reference)
            return QualityResult(round(s, 4), thr, s >= thr, method, d)
        if method == "label_match":
            ok, d = label_match(candidate, reference)
            return QualityResult(1.0 if ok else 0.0, thr, ok, method, d)
        user = as_text(prompt)
        judge_prompt = (f"<request>\n{user[:6000]}\n</request>\n\n<reference>\n{reference[:6000]}\n</reference>"
                        f"\n\n<candidate>\n{candidate[:6000]}\n</candidate>\n\nGrade the candidate.")
        try:
            jr = await send_request(judge_prompt, self.judge, system=JUDGE_SYSTEM, max_tokens=150)
        except ProviderError as e:
            return QualityResult(0.0, thr, False, "llm_judge_error", {"error": str(e)[:200]})
        raw, why = parse_judge(jr.text)
        if raw is None:
            return QualityResult(0.0, thr, False, "llm_judge_unparsed", {"raw": jr.text[:200]}, jr.cost)
        s = (raw - 1) / 4
        return QualityResult(s, thr, s >= thr - 1e-9, method, {"judge_score": raw, "reason": why}, jr.cost)

    async def verify(self, prompt: str | list[Message], task: str, candidate: Response,
                     max_tokens: int = 1024) -> tuple[QualityResult, Response]:
        ref = await send_request(prompt, self.reference, max_tokens=max_tokens)
        q = await self.score(prompt, task, candidate.text, ref.text)
        return q, ref
