"""Phase 6: the cost-savings report, static charts and a case-study draft."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import settings
from .analytics import load
from .db import DB


def _charts(req, esc, out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    blue, grey, orange = "#2E6FD8", "#9AA5B1", "#E8833A"

    # 1. cumulative cost vs baseline
    r = req.sort_values("ts").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(r.index + 1, r["baseline_cost"].cumsum(), color=grey, lw=2, label="Everything on baseline")
    ax.plot(r.index + 1, (r["cost"] + r["verify_cost"]).cumsum(), color=blue, lw=2,
            label="Autopilot (incl. verification)")
    ax.set_xlabel("requests")
    ax.set_ylabel("cumulative cost (USD)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "chart_cumulative_cost.png", dpi=160)
    files.append("chart_cumulative_cost.png")
    plt.close(fig)

    # 2. routing distribution
    dist = req["routed_model"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(dist.index[::-1], dist.values[::-1], color=blue)
    for i, v in enumerate(dist.values[::-1]):
        ax.text(v, i, f" {100 * v / dist.sum():.0f}%", va="center")
    ax.set_xlabel("requests")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "chart_routing_distribution.png", dpi=160)
    files.append("chart_routing_distribution.png")
    plt.close(fig)

    # 3. quality score distribution (verified only)
    v = req[req["verify_status"] == "done"]
    if not v.empty:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(v["quality_score"], bins=10, range=(0, 1), color=blue, edgecolor="white")
        ax.axvline(0.75, color=orange, ls="--", lw=1.5, label="judge threshold (4/5)")
        ax.set_xlabel("quality score vs reference (0-1)")
        ax.set_ylabel("verified requests")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "chart_quality_distribution.png", dpi=160)
        files.append("chart_quality_distribution.png")
        plt.close(fig)
    return files


def _sensitivity(req, reference_model: str) -> str:
    if req.empty:
        return ""
    v = req[req["verify_status"] == "done"]
    eligible = req[req["routed_model"] != reference_model]   # requests that could be verified
    if v.empty or eligible.empty:
        return ""
    per = v["verify_cost"].mean()            # overhead per verified request
    base, routed = req["baseline_cost"].sum(), req["cost"].sum()
    rows = ["| sample rate | projected verification cost | projected net reduction |", "|---:|---:|---:|"]
    for rate in (0.05, 0.10, 0.25, 0.50, 1.00):
        over = per * len(eligible) * rate
        rows.append(f"| {int(rate * 100)}% | ${over:.4f} | {100 * (1 - (routed + over) / base):.1f}% |")
    return "\n".join(rows)


def generate(out_dir: Path | None = None, loadtest: dict | None = None, retrain: dict | None = None) -> Path:
    out = Path(out_dir or settings.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    s = DB().stats()
    req, esc, _ = load()
    clf = json.loads(settings.CLASSIFIER_REPORT.read_text()) if settings.CLASSIFIER_REPORT.exists() else {}
    charts = _charts(req, esc, out) if not req.empty else []
    mock = settings.MOCK
    banner = ("> **Simulated run (AUTOPILOT_MOCK=1).** These numbers come from the offline simulator. "
              "They prove the routing, verification and feedback plumbing work end to end; they are not "
              "a measurement of real model quality or real savings. Re-run with real API keys before "
              "quoting any of them.\n\n") if mock else ""

    by_task = ""
    if not req.empty:
        v = req[req["verify_status"] == "done"]
        if not v.empty:
            t = v.groupby("task_type").agg(verified=("id", "size"), parity=("passed", "mean"),
                                           mean_score=("quality_score", "mean"))
            by_task = "| task | verified | quality parity | mean score |\n|---|---:|---:|---:|\n" + "\n".join(
                f"| {k} | {int(r.verified)} | {100 * r.parity:.1f}% | {r.mean_score:.2f} |" for k, r in t.iterrows())
    from .registry import Registry
    from .routing import RoutingConfig
    reg = Registry.load()
    ref = RoutingConfig.load(reg).verification.get("reference_model", reg.baseline_name)
    sens = _sensitivity(req, ref)
    esc_tbl = ""
    if not esc.empty:
        e = esc.groupby(["original_model", "escalated_model"]).agg(
            n=("id", "size"), cost_delta=("cost_delta", "sum"), gap=("quality_gap", "mean"))
        esc_tbl = "| from | to | escalations | total cost delta | mean quality gap |\n|---|---|---:|---:|---:|\n" + "\n".join(
            f"| {a} | {b} | {int(r.n)} | ${r.cost_delta:.4f} | {r.gap:.2f} |" for (a, b), r in e.iterrows())
    dist = "\n".join(f"| {d['model']} | {d['requests']} | {d['share_pct']}% | ${d['cost_usd']:.4f} |"
                     for d in s["routing_distribution"])
    cm = clf.get("confusion_matrix", {}).get("rows_true_cols_pred")
    cm_txt = ("| true \\ pred | 1 | 2 | 3 |\n|---|---:|---:|---:|\n" +
              "\n".join(f"| {i + 1} | " + " | ".join(map(str, row)) + " |" for i, row in enumerate(cm))) if cm else "n/a"

    md = f"""# LLM Cost Autopilot - cost savings report

Generated {time.strftime('%Y-%m-%d %H:%M')}{' (simulated)' if mock else ''}.

{banner}## Headline

**{s['net_cost_reduction_pct']}% lower cost than sending everything to the baseline model, with
{s['quality_parity_pct']}% quality parity on verified requests.**

| metric | value |
|---|---:|
| requests | {s['requests']} |
| baseline cost (all on baseline model) | ${s['baseline_cost_usd']:.4f} |
| routed cost | ${s['routed_cost_usd']:.4f} |
| verification overhead | ${s['verification_cost_usd']:.4f} |
| **total spend** | **${s['total_spend_usd']:.4f}** |
| gross reduction (before verification) | {s['gross_cost_reduction_pct']}% |
| **net reduction (after verification)** | **{s['net_cost_reduction_pct']}%** |
| verified requests | {s['verified_requests']} |
| quality parity | {s['quality_parity_pct']}% |
| escalation rate (of verified) | {s['escalation_rate_of_verified_pct']}% |

Quality parity = share of verified cheap-model answers that met the task's threshold against the
reference model's answer. It is measured on a sample, not on every request.

## What verification costs

Verifying a request means paying for the reference model's answer again, plus a judge call for
open-ended tasks. Projected from this run's measured per-request overhead:

{sens or 'n/a'}

This is the main tuning knob: verify enough to catch routing drift, not so much that the check
eats the savings.

## Routing distribution

| model | requests | share | cost |
|---|---:|---:|---:|
{dist}

## Quality by task type

{by_task or 'No verified requests yet.'}

## Escalations

{esc_tbl or 'No escalations.'}

## Classifier

Held-out accuracy {clf.get('accuracy')} (n={clf.get('n_test')}), 5-fold CV {clf.get('cv5_accuracy_mean')}
± {clf.get('cv5_accuracy_std')}, under-routing rate {clf.get('under_routing_rate')} (true tier above predicted
tier: the mistake that costs quality). Trained with {clf.get('n_feedback', 0)} feedback examples.

{cm_txt}
"""
    if loadtest:
        md += f"\n## Load test\n\n```json\n{json.dumps(loadtest, indent=2)}\n```\n"
    if retrain:
        verdict = ("promoted to live" if retrain.get("promoted")
                   else "rejected by the promotion gate; the previous model stays live")
        md += (f"\n## Feedback loop\n\nAfter the load test a candidate classifier was trained with "
               f"{retrain.get('n_feedback')} routing-failure examples: held-out accuracy "
               f"{retrain.get('accuracy')}, 5-fold CV {retrain.get('cv5_accuracy_mean')} "
               f"(live model: {retrain.get('previous_cv5_accuracy_mean')}). Candidate {verdict}.\n")
    for c in charts:
        md += f"\n![{c}]({c})\n"
    (out / "cost_savings_report.md").write_text(md, encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(s, indent=2))
    _case_study(out, s, clf, mock)
    return out / "cost_savings_report.md"


def _case_study(out: Path, s: dict, clf: dict, mock: bool) -> None:
    x, y = s["net_cost_reduction_pct"], s["quality_parity_pct"]
    note = ("<!-- Numbers below are from a SIMULATED run. Replace them with a real run before publishing. -->\n"
            if mock else "")
    txt = f"""{note}# I built a system that cut LLM API costs by {x}% while keeping {y}% quality parity

**The problem.** Most teams send every request to their best model. Most requests don't need it:
reformatting a date or pulling an invoice number out of an email does not need the same model as
reasoning through an outage. That gap is pure spend.

**What I built.** A routing layer in front of OpenAI, Anthropic and a local Llama (Ollama). Callers
hit one OpenAI-compatible endpoint and never choose a model. For each request:

1. A complexity classifier (scikit-learn: hand-built features + TF-IDF, {clf.get('cv5_accuracy_mean')} 5-fold
   CV accuracy on {clf.get('n_train', 0) + clf.get('n_test', 0)} labeled prompts, including routing-failure feedback) assigns tier 1, 2 or 3.
2. A YAML routing map sends the tier to the cheapest model allowed to handle it. Low-confidence
   predictions are bumped up a tier, because under-routing is the expensive mistake.
3. A sample of answers is re-run on the top model in a background worker and scored with a check
   that fits the task: field recall for extraction, label match for classification, LLM-as-judge for
   summaries and open-ended work.
4. A failed check is logged as a routing failure, the better answer is stored (or returned inline
   when the caller's latency budget allows), and the prompt becomes a training example. The
   classifier retrains weekly on those failures, so it gets better where it was wrong.

**The number.** Over {s['requests']} requests the router spent ${s['total_spend_usd']:.2f} against a
${s['baseline_cost_usd']:.2f} all-baseline bill: a {x}% net reduction *after* paying for verification
(gross {s['gross_cost_reduction_pct']}%). {y}% of verified answers matched the reference model's quality bar;
the rest were escalated.

**What I'd tell a VP of Engineering.** The savings come from the long tail of easy requests, and they
only hold if you measure quality continuously. Verification is the cost of trusting the router, so
it's sampled, and its cost is subtracted from the headline number rather than hidden.
"""
    (out / "case_study.md").write_text(txt, encoding="utf-8")
