"""Deterministic offline simulator (AUTOPILOT_MOCK=1).

It lets the whole system - routing, verification, escalation, retraining,
dashboard, load test - run without API keys or spend. It is NOT a quality
measurement: savings and parity numbers from a mock run demonstrate the
plumbing, and must be re-measured with real providers before you quote them.

How it fakes quality: every prompt has a hidden difficulty (1-3). If a model's
quality rank >= difficulty it returns the "correct" answer (the same answer the
top model gives, with small wording noise). Otherwise it returns a degraded
answer: wrong label, missing fields, or off-target prose. Difficulty comes from
the human labels in data/*.jsonl when the prompt is known, else a keyword
heuristic, and ~8% of prompts are secretly one tier harder than they look -
which is exactly the kind of case the verifier exists to catch.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from functools import lru_cache

from . import settings
from .registry import ModelConfig
from .tasks import as_text, task_type

_WORDS = ("the system revenue customer risk model latency cost quality report team data "
          "pipeline policy region growth margin forecast incident service contract budget "
          "user signal trend market reservoir pressure output input result plan option "
          "tradeoff evidence assumption impact priority schedule").split()
# Degraded answers draw from a disjoint vocabulary so the judge can tell them apart.
_VAGUE = ("generally it depends various factors overall things somewhat broadly typically "
          "perhaps stuff aspects considerations matters elements basically certain several").split()
_LABELS = ["positive", "negative", "neutral", "urgent", "billing", "technical", "spam", "other"]


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


@lru_cache(maxsize=1)
def _truth() -> dict[int, int]:
    out = {}
    for f in settings.DATA_DIR.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "prompt" in row and "tier" in row:
                    out[_h(row["prompt"].strip())] = int(row["tier"])
    return out


def difficulty(prompt: str) -> int:
    key = _h(prompt.strip())
    d = _truth().get(key)
    if d is None:
        from .features import heuristic_tier
        d = heuristic_tier(prompt)
    if key % 100 < 8:          # hidden hardness
        d = min(3, d + 1)
    return d


def _prose(rng: random.Random, n: int, vocab=_WORDS) -> str:
    return " ".join(rng.choice(vocab) for _ in range(n)).capitalize() + "."


def _answer(prompt: str, d: int, correct: bool, model_name: str) -> str:
    t = task_type(prompt)
    seed = _h(prompt)
    rng = random.Random(seed if correct else seed ^ _h(model_name))
    if t == "classification":
        label = _LABELS[seed % len(_LABELS)]
        if not correct:
            label = _LABELS[(seed + 1 + _h(model_name) % 5) % len(_LABELS)]
        return label
    if t == "extraction":
        fields = {f"field_{i}": f"value_{(seed >> i) % 997}" for i in range(4 + seed % 3)}
        if not correct:
            for k in list(fields)[: len(fields) // 2]:
                fields[k] = None
        return json.dumps(fields)
    n = {1: 40, 2: 110, 3: 260}[d] + seed % 30
    if not correct:
        # right length class but mostly vague filler: plausible-looking, off-target
        good = _prose(random.Random(seed), n // 4)
        return good + " " + _prose(rng, n - n // 4, _VAGUE)
    text = _prose(rng, n)
    if correct and model_name and _h(model_name) % 3:
        # small wording noise so non-reference models are not byte-identical
        words = text.split()
        words[_h(model_name) % len(words)] = "notably"
        text = " ".join(words)
    return text


def _judge(prompt: str) -> str:
    cand = re.search(r"<candidate>(.*?)</candidate>", prompt, re.S)
    ref = re.search(r"<reference>(.*?)</reference>", prompt, re.S)
    a = set((cand.group(1) if cand else "").lower().split())
    b = set((ref.group(1) if ref else "").lower().split())
    jac = len(a & b) / max(1, len(a | b))
    score = max(1, min(5, round(1 + 4 * jac)))
    return json.dumps({"score": score, "reason": f"mock judge: lexical overlap {jac:.2f}"})


def simulate(msgs: list[dict], cfg: ModelConfig):
    from .providers import Response

    user = as_text([m for m in msgs if m["role"] != "system"])
    system = as_text([m for m in msgs if m["role"] == "system"])
    if "<candidate>" in user and "grading" in (system + user).lower():
        text = _judge(user)
    else:
        d = difficulty(user)
        text = _answer(user, d, cfg.rank >= d, cfg.name)
    tin = max(1, len(user + system) // 4)
    tout = max(1, len(text) // 4)
    rng = random.Random(_h(user + cfg.name))
    latency = cfg.avg_latency_ms * rng.uniform(0.7, 1.3) * (0.6 + min(tout, 600) / 500)
    if os.environ.get("AUTOPILOT_MOCK_SLEEP") == "1":
        time.sleep(latency / 1000 / 50)
    return Response(text=text, input_tokens=tin, output_tokens=tout, latency_ms=round(latency, 1),
                    cost=cfg.cost(tin, tout), model=cfg.name, model_id=cfg.model_id,
                    provider=f"mock:{cfg.provider}")
