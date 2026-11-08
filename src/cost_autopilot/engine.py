"""The autopilot: route -> call -> log -> (verify -> escalate -> feed back).

Phase 3.3 (auto-escalation) and 3.4 (feedback + weekly retrain) live here so the
API and the background worker share exactly one implementation.
"""
from __future__ import annotations

import hashlib
import random
import time
import uuid
from dataclasses import dataclass

from . import classifier as clf_mod
from .db import DB
from .providers import ProviderError, Response, send_request
from .registry import Registry
from .routing import RouteDecision, Router, RoutingConfig
from .tasks import Message, as_text, task_type
from .verifier import QualityResult, Verifier


@dataclass
class CompletionResult:
    request_id: str
    response: Response
    decision: RouteDecision
    task_type: str
    baseline_cost: float
    verification: str            # skipped | queued | sync
    escalated: bool = False
    quality: QualityResult | None = None
    original: Response | None = None   # set when escalation replaced the answer
    fallback_from: str | None = None


class Autopilot:
    def __init__(self, registry: Registry | None = None, config: RoutingConfig | None = None,
                 db: DB | None = None, classifier: clf_mod.ComplexityClassifier | None = None,
                 rng: random.Random | None = None):
        self.registry = registry or Registry.load()
        self.config = config or RoutingConfig.load(self.registry)
        self.db = db or DB()
        self.router = Router(self.registry, self.config, classifier)
        self.rng = rng or random.Random()

    @property
    def verifier(self) -> Verifier:
        return Verifier(self.registry, self.config.verification)

    # ------------------------------------------------------------------
    async def _call_with_fallback(self, prompt, model: str, max_tokens: int, temperature: float):
        try:
            return await send_request(prompt, self.registry[model], max_tokens=max_tokens,
                                      temperature=temperature), None
        except ProviderError:
            fb = self.config.fallback(model)
            if not fb:
                raise
            return await send_request(prompt, self.registry[fb], max_tokens=max_tokens,
                                      temperature=temperature), model

    def _should_verify(self, model: str) -> bool:
        if model == self.verifier.reference.name:
            return False      # nothing to compare against
        return self.rng.random() < float(self.config.verification.get("sample_rate", 0))

    async def complete(self, prompt: str | list[Message], *, max_tokens: int = 1024,
                       temperature: float = 0.0, verify: str = "auto",
                       latency_budget_ms: float | None = None) -> CompletionResult:
        """verify: "auto" (sampled, async), "always" (async), "sync" (inline if the
        latency budget allows, else async), "off"."""
        text = as_text(prompt)
        decision = self.router.decide(text)
        task = task_type(text)
        resp, fallback_from = await self._call_with_fallback(prompt, decision.model, max_tokens, temperature)
        baseline_cost = self.registry.baseline.cost(resp.input_tokens, resp.output_tokens)
        rid = uuid.uuid4().hex[:16]

        eligible = resp.model != self.verifier.reference.name
        if verify == "off" or not eligible:
            mode = "skipped"
        elif verify in ("always", "sync"):
            mode = verify
        else:
            mode = "queued" if self._should_verify(resp.model) else "skipped"

        if mode == "sync":
            budget = latency_budget_ms or float(self.config.verification.get("default_latency_budget_ms", 8000))
            if resp.latency_ms + self.verifier.reference.avg_latency_ms > budget:
                mode = "queued"   # latency doesn't permit inline escalation
        if mode == "always":
            mode = "queued"

        self.db.insert_request({
            "id": rid, "ts": time.time(),
            "prompt_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "prompt_preview": text[:160],
            "task_type": task, "tier": decision.tier, "predicted_tier": decision.predicted_tier,
            "confidence": decision.confidence, "classifier_source": decision.classifier_source,
            "routed_model": resp.model, "reason": decision.reason, "fallback_from": fallback_from,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "cost": resp.cost, "baseline_cost": baseline_cost, "latency_ms": resp.latency_ms,
            "verify_status": "queued" if mode == "queued" else ("skipped" if mode == "skipped" else "running"),
            "final_model": resp.model,
        })
        result = CompletionResult(rid, resp, decision, task, baseline_cost,
                                  "sync" if mode == "sync" else mode, fallback_from=fallback_from)

        if mode == "queued":
            self.db.enqueue(rid, {"prompt": prompt, "task": task, "candidate": resp.to_dict(),
                                  "tier": decision.tier, "max_tokens": max_tokens})
        elif mode == "sync":
            q, ref, escalated = await self._verify_and_escalate(rid, prompt, task, resp, decision.tier,
                                                                max_tokens, "sync")
            result.quality = q
            if escalated:
                result.original, result.response, result.escalated = resp, ref, True
        return result

    # ------------------------------------------------------------------
    async def _verify_and_escalate(self, rid, prompt, task, cand: Response, tier: int,
                                   max_tokens: int, mode: str):
        q, ref = await self.verifier.verify(prompt, task, cand, max_tokens=max_tokens)
        verify_cost = ref.cost + q.judge_cost
        escalated = not q.passed
        fields = dict(verify_status="done", quality_score=q.score, passed=int(q.passed),
                      verify_cost=verify_cost)
        if escalated:
            fields.update(escalated=1, escalated_model=ref.model, escalated_text=ref.text,
                          final_model=ref.model)
            self.db.add_escalation(request_id=rid, mode=mode, original_model=cand.model,
                                   escalated_model=ref.model, cost_delta=round(ref.cost - cand.cost, 8),
                                   quality_score=q.score, threshold=q.threshold, quality_gap=q.gap,
                                   task_type=task)
            # Phase 3.4: every routing failure becomes a training example.
            # Corrected label = one tier up; repeated failures keep climbing.
            if tier < 3:
                self.db.add_feedback(rid, as_text(prompt), tier, tier + 1)
        self.db.update_request(rid, **fields)
        return q, ref, escalated

    async def process_job(self, job: dict) -> None:
        p = job["payload"]
        cand = Response(**p["candidate"])
        await self._verify_and_escalate(job["request_id"], p["prompt"], p["task"], cand, p["tier"],
                                        p.get("max_tokens", 1024), "async")

    # ------------------------------------------------------------------
    def retrain_if_due(self, force: bool = False) -> dict | None:
        fb = self.config.feedback
        new = self.db.feedback_examples(only_unused=True)
        if not force:
            if len(new) < int(fb.get("min_new_examples", 5)):
                return None
            last = self.db.last_retrain_ts()
            interval = float(fb.get("retrain_interval_days", 7)) * 86400
            if last and time.time() - last < interval:
                return None
        # all feedback ever collected, latest label per prompt wins (and the max tier)
        latest: dict[str, int] = {}
        for r in self.db.feedback_examples():
            latest[r["prompt"]] = max(latest.get(r["prompt"], 0), r["corrected_tier"])
        # Promotion gate: train a candidate next to the live model and only swap it
        # in if cross-validated accuracy didn't drop by more than max_cv_drop.
        # Some failures are hard for reasons the features can't see; learning
        # them as labels can make the classifier worse everywhere else.
        import json
        import os
        from . import settings
        live = self.router.classifier.path
        cand = live.with_suffix(".candidate.joblib")
        prev = json.loads(settings.CLASSIFIER_REPORT.read_text()) if settings.CLASSIFIER_REPORT.exists() else {}
        report = clf_mod.train(extra=list(latest.items()), out=cand)
        max_drop = float(fb.get("max_cv_drop", 0.05))
        before = prev.get("cv5_accuracy_mean")
        promoted = before is None or report["cv5_accuracy_mean"] >= before - max_drop
        report["promoted"] = promoted
        report["previous_cv5_accuracy_mean"] = before
        if promoted:
            os.replace(cand, live)
        else:
            cand.unlink(missing_ok=True)
            settings.CLASSIFIER_REPORT.write_text(json.dumps(prev, indent=2))  # keep live report
        self.db.mark_feedback_used()
        self.db.add_retrain(len(latest), report)
        return report
