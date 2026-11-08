"""`autopilot` command line."""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from . import settings


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="autopilot", description="LLM Cost Autopilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the complexity classifier on data/labeled_prompts.jsonl")
    t.add_argument("--kind", choices=["logreg", "rf"], default="logreg")
    t.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("retrain", help="retrain with accumulated routing-failure feedback")
    r.add_argument("--force", action="store_true", help="ignore the interval / minimum-examples gate")

    b = sub.add_parser("baseline", help="Phase 1.3: 10 prompts x every model")
    b.add_argument("--models", nargs="*")

    lt = sub.add_parser("loadtest", help="Phase 6.1: send N prompts and write the report")
    lt.add_argument("-n", type=int, default=800)
    lt.add_argument("--mix", default="0.55,0.30,0.15", help="tier 1,2,3 shares")
    lt.add_argument("--url", help="hit a running API instead of running in-process")
    lt.add_argument("--concurrency", type=int, default=16)
    lt.add_argument("--no-retrain", action="store_true")
    lt.add_argument("--spread-days", type=float, default=0,
                    help="demo only (mock mode): back-date requests over N days for the charts")

    sub.add_parser("report", help="regenerate results/cost_savings_report.md from the database")
    sub.add_parser("drain", help="process the verification queue once and exit")
    sub.add_parser("worker", help="run the background verification worker")
    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("stats", help="print cost savings summary")

    a = p.parse_args(argv)
    settings.ensure_dirs()

    if a.cmd == "train":
        from .classifier import train
        rep = train(kind=a.kind, seed=a.seed)
        print(json.dumps({k: rep[k] for k in ("kind", "n_train", "n_test", "accuracy", "cv5_accuracy_mean",
                                              "cv5_accuracy_std", "under_routing_rate", "confusion_matrix")}, indent=2))
        print(f"saved -> {settings.CLASSIFIER_PATH}")
    elif a.cmd == "retrain":
        from .engine import Autopilot
        rep = Autopilot().retrain_if_due(force=a.force)
        print(json.dumps(rep, indent=2) if rep else "not due (use --force)")
    elif a.cmd == "baseline":
        from .baseline import main as bmain
        bmain(a.models)
    elif a.cmd == "loadtest":
        from . import loadtest, report
        from .engine import Autopilot
        mix = tuple(float(x) for x in a.mix.split(","))
        prompts = loadtest.sample_prompts(a.n, mix)
        if a.url:
            res = asyncio.run(loadtest.run_http(prompts, a.url, a.concurrency))
            print(json.dumps(res, indent=2))
            return
        if a.spread_days and not settings.MOCK:
            p.error("--spread-days rewrites timestamps; it is only allowed with AUTOPILOT_MOCK=1")
        res = asyncio.run(loadtest.run_inprocess(prompts, a.concurrency, a.spread_days))
        print(json.dumps(res, indent=2))
        rep = None if a.no_retrain else Autopilot().retrain_if_due(force=True)
        path = report.generate(loadtest=res, retrain=rep)
        print(f"report -> {path}")
    elif a.cmd == "report":
        from .report import generate
        print(f"report -> {generate()}")
    elif a.cmd == "drain":
        from .engine import Autopilot
        from .worker import drain
        print(f"processed {asyncio.run(drain(Autopilot()))} jobs")
    elif a.cmd == "worker":
        from .worker import main as wmain
        wmain()
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("cost_autopilot.api:app", host=a.host, port=a.port,
                    log_level=os.environ.get("LOG_LEVEL", "info").lower())
    elif a.cmd == "stats":
        from .db import DB
        print(json.dumps(DB().stats(), indent=2))


if __name__ == "__main__":
    main()
