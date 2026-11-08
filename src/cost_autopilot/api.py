"""Phase 5: the FastAPI service. The caller never picks the model; the router does."""
from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from . import settings
from .engine import Autopilot
from .providers import ProviderError

app = FastAPI(title="LLM Cost Autopilot", version="0.1.0",
              description="Routes each request to the cheapest model that can handle it, "
                          "and verifies that it was right.")
_ap: Autopilot | None = None


def ap() -> Autopilot:
    global _ap
    if _ap is None:
        settings.ensure_dirs()
        _ap = Autopilot()
    return _ap


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class RoutingOptions(BaseModel):
    verify: Literal["auto", "always", "sync", "off"] = "auto"
    latency_budget_ms: float | None = Field(None, gt=0)


class CompletionRequest(BaseModel):
    messages: list[ChatMessage] | None = None
    prompt: str | None = None
    max_tokens: int = Field(1024, ge=1, le=16000)
    temperature: float = Field(0.0, ge=0, le=2)
    model: str | None = Field(None, description="Ignored: the router chooses. Accepted for "
                                                "drop-in compatibility with OpenAI clients.")
    routing: RoutingOptions = RoutingOptions()

    @model_validator(mode="after")
    def _one_input(self):
        if not self.messages and not self.prompt:
            raise ValueError("provide messages or prompt")
        return self


@app.get("/health")
def health():
    a = ap()
    return {"ok": True, "classifier": "trained" if a.router.classifier.ready else "heuristic",
            "mock": settings.MOCK, "verification_backlog": a.db.pending_jobs()}


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    a = ap()
    prompt = [m.model_dump() for m in req.messages] if req.messages else req.prompt
    try:
        r = await a.complete(prompt, max_tokens=req.max_tokens, temperature=req.temperature,
                             verify=req.routing.verify, latency_budget_ms=req.routing.latency_budget_ms)
    except ProviderError as e:
        raise HTTPException(502, f"all providers failed: {e}") from e
    resp = r.response
    total_cost = resp.cost + (r.original.cost if r.original else 0) + (r.quality.judge_cost if r.quality else 0)
    return {
        "id": f"cmpl-{r.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model_id,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": resp.text}}],
        "usage": {"prompt_tokens": resp.input_tokens, "completion_tokens": resp.output_tokens,
                  "total_tokens": resp.total_tokens},
        "routing": {
            "request_id": r.request_id,
            "selected_model": r.decision.model,   # what the router chose

            "served_by": resp.model,
            "tier": r.decision.tier,
            "predicted_tier": r.decision.predicted_tier,
            "confidence": r.decision.confidence,
            "tier_probabilities": r.decision.probabilities,
            "task_type": r.task_type,
            "reason": r.decision.reason,
            "fallback_from": r.fallback_from,
            "cost_usd": round(total_cost, 8),
            "baseline_model": a.registry.baseline_name,
            "baseline_cost_usd": round(r.baseline_cost, 8),
            "saved_usd": round(r.baseline_cost - total_cost, 8),
            "latency_ms": resp.latency_ms,
            "verification": r.verification,
            "escalated": r.escalated,
            "escalated_from": r.original.model if r.original else None,
            "quality_score": r.quality.score if r.quality else None,
        },
    }


@app.get("/v1/requests/{request_id}")
def get_request(request_id: str):
    """Audit row for one request. After async verification, `escalated_text`
    holds the better answer if the cheap model was judged not good enough."""
    row = ap().db.get_request(request_id)
    if not row:
        raise HTTPException(404, "unknown request id")
    return row


@app.get("/v1/models")
def models():
    a = ap()
    tiers = {int(k): v for k, v in a.config.raw["tiers"].items()}
    return {"baseline_model": a.registry.baseline_name,
            "routing": tiers,
            "models": [m.to_dict() | {"routed_tiers": [t for t, n in tiers.items() if n == m.name]}
                       for m in a.registry]}


@app.get("/v1/stats")
def stats(days: float | None = None):
    since = time.time() - days * 86400 if days else None
    return ap().db.stats(since)


class RoutingConfigPatch(BaseModel):
    tiers: dict[int, str] | None = None
    fallback: dict[str, str] | None = None
    min_confidence: float | None = Field(None, ge=0, le=1)
    verification: dict | None = None
    feedback: dict | None = None


@app.get("/v1/routing-config")
def get_routing_config():
    return ap().config.raw


@app.put("/v1/routing-config")
def put_routing_config(patch: RoutingConfigPatch):
    a = ap()
    try:
        a.config.update(patch.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return a.config.raw


@app.post("/v1/retrain")
def retrain(force: bool = False):
    rep = ap().retrain_if_due(force=force)
    return {"retrained": bool(rep), "report": rep}
