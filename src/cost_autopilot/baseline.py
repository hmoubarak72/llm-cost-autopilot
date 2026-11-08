"""Phase 1.3: send the same 10 prompts to every model; log outputs, cost, latency."""
from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

from . import settings
from .providers import ProviderError, send_request
from .registry import Registry

PROMPTS = [
    "Convert this list to a JSON array: red, green, blue.",
    "Extract the email address from: 'Questions? Write to support@example.org any time.'",
    "What is 15% of 240?",
    "Summarize the main differences between TCP and UDP in a short table.",
    "Classify the sentiment of this review as positive, negative or mixed: 'Battery life is excellent "
    "but the keyboard broke after three weeks.'",
    "Explain what a confusion matrix shows, with a 2x2 example.",
    "A train leaves at 3pm going 80 km/h; another leaves the same station at 4pm going 100 km/h on the "
    "same track. When does the second catch up? Show your reasoning.",
    "Write a four-line poem about a desert at night.",
    "Our API p99 latency jumped from 300ms to 4s after a deploy that only changed logging. List the "
    "three most likely causes and how to test each.",
    "Is it better for a 6-engineer team to adopt microservices or a modular monolith? Recommend one.",
]


async def run(registry: Registry | None = None, models: list[str] | None = None,
              out_dir: Path | None = None) -> dict:
    reg = registry or Registry.load()
    names = models or [m.name for m in reg]
    out_dir = Path(out_dir or settings.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        cfg = reg[name]
        for i, p in enumerate(PROMPTS):
            try:
                r = await send_request(p, cfg, max_tokens=600)
                rows.append({"model": name, "prompt_idx": i, "ok": True, "input_tokens": r.input_tokens,
                             "output_tokens": r.output_tokens, "latency_ms": r.latency_ms,
                             "cost_usd": r.cost, "output": r.text})
            except ProviderError as e:
                rows.append({"model": name, "prompt_idx": i, "ok": False, "error": str(e)[:200]})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"baseline_{stamp}.json").write_text(json.dumps({"prompts": PROMPTS, "rows": rows}, indent=2))
    with (out_dir / f"baseline_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "prompt_idx", "ok", "input_tokens", "output_tokens",
                                          "latency_ms", "cost_usd", "error"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for name in names:
        ok = [r for r in rows if r["model"] == name and r["ok"]]
        summary[name] = {
            "ok": f"{len(ok)}/{len(PROMPTS)}",
            "total_cost_usd": round(sum(r["cost_usd"] for r in ok), 6),
            "median_latency_ms": round(statistics.median(r["latency_ms"] for r in ok), 0) if ok else None,
            "errors": sorted({r.get("error", "")[:80] for r in rows if r["model"] == name and not r["ok"]}),
        }
    return {"file": str(out_dir / f"baseline_{stamp}.json"), "summary": summary}


def main(models=None):
    res = asyncio.run(run(models=models))
    print(f"wrote {res['file']}")
    print(f"{'model':<16}{'ok':>7}{'cost $':>12}{'p50 ms':>10}")
    for n, s in res["summary"].items():
        print(f"{n:<16}{s['ok']:>7}{s['total_cost_usd']:>12.6f}{str(s['median_latency_ms']):>10}")
        for e in s["errors"]:
            print(f"   ! {e}")
    print("\nPaste measured p50 latencies into config/models.yaml (avg_latency_ms) if they differ.")
