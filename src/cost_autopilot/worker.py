"""Background worker: drains the verification queue and runs the weekly retrain."""
from __future__ import annotations

import asyncio
import logging
import os
import time

from .engine import Autopilot

log = logging.getLogger("autopilot.worker")
MAX_ATTEMPTS = 3


async def drain(ap: Autopilot, batch: int = 20, concurrency: int = 8) -> int:
    """Process pending jobs until the queue is empty. Returns jobs processed."""
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(job):
        nonlocal done
        async with sem:
            try:
                await ap.process_job(job)
                ap.db.finish_job(job["id"])
            except Exception as e:  # noqa: BLE001 - a bad job must not kill the worker
                retry = job["attempts"] + 1 < MAX_ATTEMPTS
                ap.db.finish_job(job["id"], error=str(e)[:500], retry=retry)
                if not retry:
                    ap.db.update_request(job["request_id"], verify_status="error")
                log.warning("job %s failed (attempt %s): %s", job["id"], job["attempts"] + 1, e)
            done += 1

    while jobs := ap.db.claim_jobs(batch):
        await asyncio.gather(*(one(j) for j in jobs))
    return done


async def run_forever(poll_s: float = 1.0, retrain_check_s: float = 3600) -> None:
    ap = Autopilot()
    last_check = 0.0
    log.info("worker started; db=%s", ap.db.path)
    while True:
        n = await drain(ap)
        if n:
            log.info("verified %d requests", n)
        if time.time() - last_check > retrain_check_s:
            last_check = time.time()
            rep = ap.retrain_if_due()
            if rep:
                log.info("retrained classifier: acc=%.3f cv=%.3f feedback=%d",
                         rep["accuracy"], rep["cv5_accuracy_mean"], rep["n_feedback"])
        await asyncio.sleep(poll_s)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run_forever(float(os.environ.get("WORKER_POLL_S", "1"))))


if __name__ == "__main__":
    main()
