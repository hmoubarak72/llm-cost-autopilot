"""Phase 4.1: SQLite audit trail + the verification job queue.

SQLite in WAL mode is enough for one API process and one worker sharing a
volume. Every request also gets a structured JSON line in var/requests.jsonl.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT,
    task_type TEXT,
    tier INTEGER,               -- tier actually used for routing
    predicted_tier INTEGER,     -- classifier's raw prediction
    confidence REAL,
    classifier_source TEXT,
    routed_model TEXT,
    reason TEXT,
    fallback_from TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost REAL,                  -- what the routed call cost
    baseline_cost REAL,         -- what the same tokens would cost on the baseline model
    latency_ms REAL,
    verify_status TEXT DEFAULT 'skipped',  -- skipped | queued | done | error
    quality_score REAL,         -- 0..1, normalized
    passed INTEGER,
    verify_cost REAL DEFAULT 0, -- reference + judge calls
    escalated INTEGER DEFAULT 0,
    escalated_model TEXT,
    escalated_text TEXT,
    final_model TEXT
);
CREATE INDEX IF NOT EXISTS ix_requests_ts ON requests(ts);

CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT, ts REAL, mode TEXT,
    original_model TEXT, escalated_model TEXT,
    cost_delta REAL,            -- escalated model cost - original model cost, this request
    quality_score REAL, threshold REAL, quality_gap REAL, task_type TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT, payload TEXT, status TEXT DEFAULT 'pending',
    attempts INTEGER DEFAULT 0, error TEXT, created REAL, updated REAL
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, request_id TEXT, prompt TEXT,
    predicted_tier INTEGER, corrected_tier INTEGER,
    used_in_training INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS retrains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, n_feedback INTEGER, accuracy REAL, cv_accuracy REAL, promoted INTEGER, report TEXT
);
"""

_lock = threading.Lock()


class DB:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---- requests -------------------------------------------------------
    def insert_request(self, row: dict) -> None:
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        with _lock, self.conn() as c:
            c.execute(f"INSERT INTO requests ({cols}) VALUES ({qs})", list(row.values()))
        _jsonl({"event": "request", **row})

    def update_request(self, rid: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        with _lock, self.conn() as c:
            c.execute(f"UPDATE requests SET {sets} WHERE id=?", [*fields.values(), rid])

    def get_request(self, rid: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM requests WHERE id=?", (rid,)).fetchone()
        return dict(r) if r else None

    def add_escalation(self, **row) -> None:
        row.setdefault("ts", time.time())
        cols = ",".join(row)
        with _lock, self.conn() as c:
            c.execute(f"INSERT INTO escalations ({cols}) VALUES ({','.join('?' * len(row))})",
                      list(row.values()))
        _jsonl({"event": "escalation", **row})

    # ---- job queue ------------------------------------------------------
    def enqueue(self, request_id: str, payload: dict) -> None:
        now = time.time()
        with _lock, self.conn() as c:
            c.execute("INSERT INTO jobs (request_id, payload, created, updated) VALUES (?,?,?,?)",
                      (request_id, json.dumps(payload), now, now))

    def claim_jobs(self, n: int = 10) -> list[dict]:
        with _lock, self.conn() as c:
            rows = c.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id LIMIT ?",
                             (n,)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                c.execute(f"UPDATE jobs SET status='running', attempts=attempts+1, updated=? "
                          f"WHERE id IN ({','.join('?' * len(ids))})", [time.time(), *ids])
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def finish_job(self, job_id: int, error: str | None = None, retry: bool = False) -> None:
        status = "pending" if retry else ("error" if error else "done")
        with _lock, self.conn() as c:
            # done jobs drop their payload: it holds the full prompt
            c.execute("UPDATE jobs SET status=?, error=?, updated=?, payload=CASE WHEN ?='done' "
                      "THEN '{}' ELSE payload END WHERE id=?",
                      (status, error, time.time(), status, job_id))

    def pending_jobs(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')").fetchone()[0]

    # ---- feedback / retrain --------------------------------------------
    def add_feedback(self, request_id, prompt, predicted_tier, corrected_tier) -> None:
        with _lock, self.conn() as c:
            c.execute("INSERT INTO feedback (ts, request_id, prompt, predicted_tier, corrected_tier) "
                      "VALUES (?,?,?,?,?)", (time.time(), request_id, prompt, predicted_tier, corrected_tier))

    def feedback_examples(self, only_unused: bool = False) -> list[dict]:
        q = "SELECT * FROM feedback" + (" WHERE used_in_training=0" if only_unused else "") + " ORDER BY id"
        with self.conn() as c:
            return [dict(r) for r in c.execute(q).fetchall()]

    def mark_feedback_used(self) -> None:
        with _lock, self.conn() as c:
            c.execute("UPDATE feedback SET used_in_training=1")

    def add_retrain(self, n_feedback, report: dict) -> None:
        with _lock, self.conn() as c:
            c.execute("INSERT INTO retrains (ts, n_feedback, accuracy, cv_accuracy, promoted, report) "
                      "VALUES (?,?,?,?,?,?)",
                      (time.time(), n_feedback, report.get("accuracy"), report.get("cv5_accuracy_mean"),
                       int(report.get("promoted", True)), json.dumps(report)))

    def last_retrain_ts(self) -> float | None:
        with self.conn() as c:
            r = c.execute("SELECT MAX(ts) FROM retrains").fetchone()[0]  # attempted, promoted or not
        return r

    # ---- stats ----------------------------------------------------------
    def stats(self, since: float | None = None) -> dict:
        where, args = ("WHERE ts>=?", [since]) if since else ("", [])
        with self.conn() as c:
            r = c.execute(f"""
                SELECT COUNT(*) n, COALESCE(SUM(cost),0) cost, COALESCE(SUM(baseline_cost),0) base,
                       COALESCE(SUM(verify_cost),0) vcost,
                       SUM(verify_status='done') verified, SUM(passed=1) passed,
                       SUM(escalated) escalated, AVG(latency_ms) lat,
                       SUM(verify_status IN ('queued')) queued
                FROM requests {where}""", args).fetchone()
            dist = c.execute(f"SELECT routed_model, COUNT(*) n, SUM(cost) cost FROM requests {where} "
                             f"GROUP BY routed_model ORDER BY n DESC", args).fetchall()
            tiers = c.execute(f"SELECT tier, COUNT(*) n FROM requests {where} GROUP BY tier", args).fetchall()
        n = r["n"] or 0
        spend = r["cost"] + r["vcost"]
        verified = r["verified"] or 0
        return {
            "requests": n,
            "routed_cost_usd": round(r["cost"], 6),
            "verification_cost_usd": round(r["vcost"], 6),
            "total_spend_usd": round(spend, 6),
            "baseline_cost_usd": round(r["base"], 6),
            "gross_saved_usd": round(r["base"] - r["cost"], 6),
            "net_saved_usd": round(r["base"] - spend, 6),
            "gross_cost_reduction_pct": round(100 * (1 - r["cost"] / r["base"]), 2) if r["base"] else None,
            "net_cost_reduction_pct": round(100 * (1 - spend / r["base"]), 2) if r["base"] else None,
            "verified_requests": verified,
            "quality_parity_pct": round(100 * (r["passed"] or 0) / verified, 2) if verified else None,
            "escalations": r["escalated"] or 0,
            "escalation_rate_of_verified_pct": round(100 * (r["escalated"] or 0) / verified, 2) if verified else None,
            "verification_backlog": r["queued"] or 0,
            "avg_latency_ms": round(r["lat"], 1) if r["lat"] else None,
            "routing_distribution": [{"model": d["routed_model"], "requests": d["n"],
                                      "share_pct": round(100 * d["n"] / n, 2), "cost_usd": round(d["cost"] or 0, 6)}
                                     for d in dist] if n else [],
            "tier_distribution": {str(t["tier"]): t["n"] for t in tiers},
        }


def _jsonl(obj: dict) -> None:
    try:
        settings.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, settings.LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
    except OSError:
        pass
