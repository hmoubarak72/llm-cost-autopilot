"""Phase 6.1: realistic load test (500-1,000 diverse prompts).

Prompts come from two pools:
  - the hand-labeled training set (data/labeled_prompts.jsonl), and
  - a templated pool (data/loadtest_prompts.jsonl) of prompts the classifier
    has never seen, built from task x topic x constraint combinations.
The default mix skews simple (55/30/15), as real traffic usually does. Change
it with --mix.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from pathlib import Path

import httpx

from . import settings
from .engine import Autopilot
from .worker import drain

TOPICS = ["a solar farm permit", "a cloud cost overrun", "a customer churn spike", "a seismic survey",
          "a hospital scheduling system", "a supply-chain delay", "a CO2 injection pilot",
          "a fintech onboarding flow", "a university course redesign", "a mobile app outage",
          "a hiring plan for a data team", "a water utility tariff change", "a warehouse robot rollout",
          "a drilling program in a frontier basin", "an LLM support chatbot"]
ENTITIES = [("Acme Corp", "acme.com", "$12,400", "2026-03-14"), ("Blue Mesa LLC", "bluemesa.io", "$980", "2026-07-02"),
            ("Harbor Health", "harborhealth.org", "$55,000", "2026-01-30"), ("Qasr Energy", "qasr-energy.ly", "$3,215.75", "2026-09-19"),
            ("Northwind", "northwind.io", "$4,812.50", "2026-10-03"), ("Delta Grid", "deltagrid.com", "$71,900", "2026-05-21")]

T1 = [
    "Extract the company name, amount and date as JSON from: '{co} was invoiced {amt} on {date}.'",
    "What is the amount in this sentence: '{co} paid {amt} on {date}'?",
    "Reformat the date {date} as Month Day, Year.",
    "Pull out the domain name from: 'contact us at billing@{dom}'.",
    "Convert to uppercase: {co} quarterly update",
    "List the words in this phrase in alphabetical order: {topic}",
    "Return only the number from: 'Invoice total {amt}'.",
    "Translate '{co} is closed today' into French.",
]
T2 = [
    "Summarize the typical risks of {topic} in four bullets.",
    "Classify this note as risk, decision, or update: 'We approved the budget for {topic}.'",
    "Explain {topic} to a new team member in one paragraph.",
    "Compare two common approaches to {topic} in a short table.",
    "Outline the stakeholders involved in {topic} and what each needs.",
    "Draft a short status update email about {topic} for {co}.",
    "Categorize these concerns about {topic} as technical, financial or regulatory: delays, cost, permits.",
]
T3 = [
    "Analyze the tradeoffs in {topic} for {co} and recommend a plan with risks and mitigations.",
    "Design a step-by-step evaluation strategy for {topic}, justifying each step.",
    "Reason step by step about the second-order effects of a 30% budget cut on {topic}.",
    "Critique this plan for {topic}: 'move fast, fix later'. Steelman it, then evaluate it.",
    "Write a short story set during {topic} with a surprising ending.",
    "Forecast what could go wrong with {topic} over 18 months and how {co} should prepare. State assumptions.",
]


def build_template_pool(path: Path | None = None) -> Path:
    path = Path(path or settings.DATA_DIR / "loadtest_prompts.jsonl")
    rows = []
    for tier, temps in ((1, T1), (2, T2), (3, T3)):
        for t, topic, (co, dom, amt, date) in itertools.product(temps, TOPICS, ENTITIES):
            rows.append({"prompt": t.format(topic=topic, co=co, dom=dom, amt=amt, date=date), "tier": tier})
    uniq = {r["prompt"]: r for r in rows}
    path.write_text("\n".join(json.dumps(r) for r in uniq.values()) + "\n", encoding="utf-8")
    return path


def sample_prompts(n: int, mix=(0.55, 0.30, 0.15), seed: int = 7) -> list[dict]:
    pool_path = settings.DATA_DIR / "loadtest_prompts.jsonl"
    if not pool_path.exists():
        build_template_pool(pool_path)
    pools = {1: [], 2: [], 3: []}
    for f, src in ((settings.LABELED_PROMPTS, "training_set"), (pool_path, "unseen_templates")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line) | {"source": src}
                pools[int(r["tier"])].append(r)
    rng = random.Random(seed)
    out = []
    for tier, share in zip((1, 2, 3), mix):
        k = round(n * share)
        pool = pools[tier]
        out += rng.sample(pool, k) if k <= len(pool) else [rng.choice(pool) for _ in range(k)]
    rng.shuffle(out)
    return out[:n]


async def run_inprocess(prompts: list[dict], concurrency: int = 16, spread_days: float = 0) -> dict:
    ap = Autopilot()
    sem = asyncio.Semaphore(concurrency)
    tally: dict[str, list[int]] = {}   # source -> [n, correct, under-routed]

    async def one(r):
        async with sem:
            res = await ap.complete(r["prompt"], max_tokens=800)
            t = tally.setdefault(r.get("source", "unknown"), [0, 0, 0])
            t[0] += 1
            t[1] += res.decision.predicted_tier == r["tier"]
            t[2] += res.decision.tier < r["tier"]

    t0 = time.time()
    await asyncio.gather(*(one(r) for r in prompts))
    jobs = await drain(ap)
    if spread_days:
        _spread(ap, spread_days)
    return {"sent": len(prompts), "seconds": round(time.time() - t0, 1), "verified": jobs,
            "classifier_vs_labels": {src: {"n": n, "accuracy": round(c / n, 4),
                                           "under_routed": round(u / n, 4)}
                                     for src, (n, c, u) in tally.items()}}


async def run_http(prompts: list[dict], url: str, concurrency: int = 16) -> dict:
    sem = asyncio.Semaphore(concurrency)
    ok = err = 0
    async with httpx.AsyncClient(base_url=url, timeout=120) as c:
        async def one(r):
            nonlocal ok, err
            async with sem:
                resp = await c.post("/v1/completions", json={"prompt": r["prompt"], "max_tokens": 800})
                ok, err = (ok + 1, err) if resp.status_code == 200 else (ok, err + 1)
        t0 = time.time()
        await asyncio.gather(*(one(r) for r in prompts))
    return {"sent": len(prompts), "ok": ok, "errors": err, "seconds": round(time.time() - t0, 1),
            "note": "verification runs in the worker container; check /v1/stats once the backlog drains"}


def _spread(ap, days: float) -> None:
    """Demo only: back-date this run's requests evenly over the past `days` so
    the dashboard's over-time charts have a shape. Never use on real traffic."""
    now = time.time()
    with ap.db.conn() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM requests ORDER BY ts").fetchall()]
        n = len(ids)
        for i, rid in enumerate(ids):
            ts = now - days * 86400 * (1 - i / max(1, n - 1))
            c.execute("UPDATE requests SET ts=? WHERE id=?", (ts, rid))
            c.execute("UPDATE escalations SET ts=? WHERE request_id=?", (ts, rid))
