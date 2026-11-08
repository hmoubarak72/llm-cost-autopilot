"""Shared queries for the dashboard and the report."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import settings


def load(db_path: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(db_path or settings.DB_PATH)
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(p) as c:
        req = pd.read_sql("SELECT * FROM requests", c)
        esc = pd.read_sql("SELECT * FROM escalations", c)
        rt = pd.read_sql("SELECT ts, n_feedback, accuracy, cv_accuracy, promoted FROM retrains", c)
    for df in (req, esc, rt):
        if not df.empty:
            df["time"] = pd.to_datetime(df["ts"], unit="s")
    return req, esc, rt


def daily(req: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if req.empty:
        return pd.DataFrame()
    g = req.set_index("time").groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({
        "requests": g.size(),
        "routed_cost": g["cost"].sum(),
        "verification_cost": g["verify_cost"].sum(),
        "baseline_cost": g["baseline_cost"].sum(),
        "verified": g["verify_status"].apply(lambda s: (s == "done").sum()),
        "escalated": g["escalated"].sum(),
    })
    out = out[out["requests"] > 0]
    out["total_cost"] = out["routed_cost"] + out["verification_cost"]
    out["saved"] = out["baseline_cost"] - out["total_cost"]
    out["escalation_rate"] = (out["escalated"] / out["verified"].where(out["verified"] > 0)).fillna(0)
    return out
