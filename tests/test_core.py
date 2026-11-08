import asyncio
import json
import random

import pytest

from cost_autopilot.classifier import ComplexityClassifier, load_labeled
from cost_autopilot.db import DB
from cost_autopilot.engine import Autopilot
from cost_autopilot.features import extract_features
from cost_autopilot.providers import Response, send_request
from cost_autopilot.registry import Registry
from cost_autopilot.routing import RoutingConfig
from cost_autopilot.tasks import task_type
from cost_autopilot.verifier import field_recall, label_match, parse_judge


def run(c):
    return asyncio.run(c)


# ---- Phase 1 ---------------------------------------------------------------
def test_registry_and_cost_math():
    reg = Registry.load()
    assert {"gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "llama-local"} <= set(reg.models)
    g = reg["gpt-4o"]
    assert g.cost(1_000_000, 0) == pytest.approx(2.50)
    assert g.cost(1000, 500) == pytest.approx(0.0025 + 0.005)
    assert reg["llama-local"].cost(10_000, 10_000) == 0
    assert reg.baseline.name == "gpt-4o"


def test_send_request_returns_standard_response():
    reg = Registry.load()
    for m in reg:
        r = run(send_request("Convert 72F to Celsius.", m))
        assert isinstance(r, Response) and r.text
        assert r.input_tokens > 0 and r.output_tokens > 0 and r.latency_ms > 0
        assert r.cost == pytest.approx(m.cost(r.input_tokens, r.output_tokens))
        assert r.model == m.name


# ---- Phase 2 ---------------------------------------------------------------
def test_dataset_size_and_balance():
    X, y = load_labeled()
    assert len(X) >= 200
    assert min(y.count(t) for t in (1, 2, 3)) >= 60


def test_features_respond_to_complexity():
    easy = extract_features("Uppercase this sentence: hello world")
    hard = extract_features("Analyze the tradeoffs and recommend a strategy, step by step, justifying each.")
    assert hard["n_complex"] > easy["n_complex"]
    assert extract_features("Who sent it?\n\nEmail: Hi team, please refund invoice 42. Thanks, Dana")["has_context"] == 1


def test_classifier_meets_v1_bar(trained):
    assert trained["accuracy"] >= 0.8
    assert trained["cv5_accuracy_mean"] >= 0.8
    clf = ComplexityClassifier()
    assert clf.predict("Convert this list to JSON: a, b, c").tier == 1
    assert clf.predict("Design a migration strategy for a 2TB database with rollback, "
                       "and justify the tradeoffs.").tier == 3


def test_routing_map_and_confidence_bump(trained):
    reg = Registry.load()
    cfg = RoutingConfig.load(reg)
    ap = Autopilot(reg, cfg)
    d = ap.router.decide("Convert this list to JSON: a, b, c")
    assert d.model == cfg.model_for_tier(d.tier)
    cfg.raw["min_confidence"] = 1.01   # force a bump
    d2 = ap.router.decide("Convert this list to JSON: a, b, c")
    assert d2.tier == min(3, d2.predicted_tier + 1) and "bumped" in d2.reason


def test_routing_config_validation():
    reg = Registry.load()
    cfg = RoutingConfig.load(reg)
    with pytest.raises(ValueError):
        cfg.update({"tiers": {1: "does-not-exist"}})
    cfg.update({"tiers": {1: "claude-haiku"}})
    assert RoutingConfig.load(reg).model_for_tier(1) == "claude-haiku"   # persisted


# ---- Phase 3 ---------------------------------------------------------------
def test_task_types():
    assert task_type("Classify this ticket as billing or technical") == "classification"
    assert task_type("Extract the invoice number as JSON") == "extraction"
    assert task_type("Summarize this article in 3 sentences") == "summarization"
    assert task_type("Write a poem about rain") == "general"


def test_quality_checks():
    ref = json.dumps({"invoice": "INV-1", "amount": "$5", "email": "a@b.co"})
    assert field_recall(ref, ref)[0] == 1.0
    assert field_recall(json.dumps({"invoice": "INV-1"}), ref)[0] == pytest.approx(1 / 3)
    assert field_recall("invoice INV-1, $5", "INV-1 for $5 from a@b.co")[0] == pytest.approx(2 / 3)
    assert label_match("Billing", "billing")[0]
    assert label_match("Category: billing - the user was charged twice", "billing")[0]
    assert not label_match("technical", "billing")[0]
    assert parse_judge('{"score": 4, "reason": "ok"}')[0] == 4
    assert parse_judge("Score: 2 because...")[0] == 2


def test_cheap_model_failure_is_escalated_and_fed_back(trained):
    from cost_autopilot.classifier import Prediction
    ap = Autopilot(rng=random.Random(0))
    # simulate a classifier mistake: a hard prompt predicted as tier 1 -> llama-local
    ap.router.classifier.predict = lambda p: Prediction(1, 0.95, {1: 0.95}, "classifier")
    prompt = "Analyze the tradeoffs of build vs buy for a subsurface ML platform and recommend one."
    r = run(ap.complete(prompt, verify="sync", latency_budget_ms=60_000))
    assert r.escalated and r.response.model == "gpt-4o" and r.original.model == "llama-local"
    row = ap.db.get_request(r.request_id)
    assert row["escalated"] == 1 and row["verify_status"] == "done" and row["verify_cost"] > 0
    fb = ap.db.feedback_examples()
    assert fb and fb[0]["corrected_tier"] == fb[0]["predicted_tier"] + 1


def test_sync_falls_back_to_async_when_latency_does_not_permit(trained):
    ap = Autopilot()
    r = run(ap.complete("Explain what a hash map is.", verify="sync", latency_budget_ms=1))
    if r.response.model != "gpt-4o":
        assert r.verification == "queued"
        assert ap.db.pending_jobs() == 1


def test_async_queue_and_worker(trained):
    from cost_autopilot.worker import drain
    ap = Autopilot()
    ids = [run(ap.complete(p, verify="always")).request_id
           for p in ["Uppercase: hello", "Summarize the benefits of Docker.", "Translate 'hi' to French."]]
    n = run(drain(ap))
    assert n >= 1 and ap.db.pending_jobs() == 0
    statuses = {ap.db.get_request(i)["verify_status"] for i in ids}
    assert statuses <= {"done", "skipped"}


def test_retrain_uses_feedback(trained):
    ap = Autopilot()
    for i in range(6):
        ap.db.add_feedback(f"r{i}", f"Tricky prompt number {i} that looks simple", 1, 2)
    assert ap.retrain_if_due() is not None   # first retrain, enough examples
    assert ap.retrain_if_due() is None       # not due again for 7 days
    assert ap.db.last_retrain_ts() is not None
    assert all(r["used_in_training"] for r in ap.db.feedback_examples())


# ---- Phase 4 ---------------------------------------------------------------
def test_stats_savings_math(trained):
    ap = Autopilot()
    ap.config.raw["verification"]["sample_rate"] = 0
    for p in ["Uppercase: hello", "Convert 3 km to miles.", "What is 15% of 240?"]:
        run(ap.complete(p))
    s = DB().stats()
    assert s["requests"] == 3
    assert s["baseline_cost_usd"] >= s["routed_cost_usd"]
    assert s["net_saved_usd"] == pytest.approx(s["baseline_cost_usd"] - s["total_spend_usd"], abs=1e-6)
    assert sum(d["requests"] for d in s["routing_distribution"]) == 3


def test_retrain_promotion_gate_rejects_worse_model(trained):
    import json
    from cost_autopilot import settings
    ap = Autopilot()
    # pretend the live model is near-perfect so any candidate looks worse
    rep = json.loads(settings.CLASSIFIER_REPORT.read_text())
    rep["cv5_accuracy_mean"] = 0.999
    settings.CLASSIFIER_REPORT.write_text(json.dumps(rep))
    before = settings.CLASSIFIER_PATH.stat().st_mtime
    for i in range(6):
        ap.db.add_feedback(f"r{i}", f"Uppercase this word {i}", 1, 3)   # deliberately wrong labels
    out = ap.retrain_if_due(force=True)
    assert out["promoted"] is False
    assert settings.CLASSIFIER_PATH.stat().st_mtime == before          # live model untouched
    assert json.loads(settings.CLASSIFIER_REPORT.read_text())["cv5_accuracy_mean"] == 0.999
