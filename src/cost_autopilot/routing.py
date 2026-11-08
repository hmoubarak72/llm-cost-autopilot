"""Phase 2.4: the tier -> model routing map (YAML, hot-swappable)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import settings
from .classifier import ComplexityClassifier, Prediction
from .registry import Registry

_lock = threading.Lock()


class RoutingConfig:
    def __init__(self, raw: dict, registry: Registry, path: Path | None = None):
        self.path = Path(path or settings.ROUTING_YAML)
        self.registry = registry
        self.raw = raw
        self.validate(raw, registry)

    @staticmethod
    def validate(raw: dict, registry: Registry) -> None:
        tiers = {int(k): v for k, v in raw.get("tiers", {}).items()}
        if set(tiers) != {1, 2, 3}:
            raise ValueError("routing.tiers must map exactly tiers 1, 2 and 3")
        for t, m in tiers.items():
            if m not in registry:
                raise ValueError(f"tier {t}: unknown model {m!r}")
        for a, b in (raw.get("fallback") or {}).items():
            if a not in registry or b not in registry:
                raise ValueError(f"fallback {a}->{b}: unknown model")
        v = raw.get("verification", {})
        for k in ("reference_model", "judge_model"):
            if v.get(k) and v[k] not in registry:
                raise ValueError(f"verification.{k}: unknown model {v[k]!r}")
        sr = v.get("sample_rate", 0)
        if not 0 <= sr <= 1:
            raise ValueError("verification.sample_rate must be in [0, 1]")

    @classmethod
    def load(cls, registry: Registry, path: Path | None = None) -> "RoutingConfig":
        p = Path(path or settings.ROUTING_YAML)
        return cls(yaml.safe_load(p.read_text()), registry, p)

    def save(self) -> None:
        with _lock:
            self.path.write_text(yaml.safe_dump(self.raw, sort_keys=False))

    def update(self, patch: dict) -> None:
        """Merge a partial config (e.g. {"tiers": {"1": "claude-haiku"}}), validate, persist."""
        merged = {**self.raw}
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        if "tiers" in merged:
            merged["tiers"] = {int(k): v for k, v in merged["tiers"].items()}
        self.validate(merged, self.registry)
        self.raw = merged
        self.save()

    # convenience accessors
    def model_for_tier(self, tier: int) -> str:
        return {int(k): v for k, v in self.raw["tiers"].items()}[tier]

    def fallback(self, model: str) -> str | None:
        return (self.raw.get("fallback") or {}).get(model)

    @property
    def min_confidence(self) -> float:
        return float(self.raw.get("min_confidence", 0))

    @property
    def verification(self) -> dict:
        return self.raw.get("verification", {})

    @property
    def feedback(self) -> dict:
        return self.raw.get("feedback", {})


@dataclass
class RouteDecision:
    tier: int
    predicted_tier: int
    model: str
    confidence: float
    probabilities: dict
    classifier_source: str
    reason: str


class Router:
    def __init__(self, registry: Registry, config: RoutingConfig,
                 classifier: ComplexityClassifier | None = None):
        self.registry = registry
        self.config = config
        self.classifier = classifier or ComplexityClassifier()

    def decide(self, prompt: str) -> RouteDecision:
        p: Prediction = self.classifier.predict(prompt)
        tier = p.tier
        names = {1: "simple", 2: "moderate", 3: "complex"}
        reason = f"classified tier {p.tier} ({names[p.tier]}) with confidence {p.confidence:.2f}"
        if p.source == "heuristic":
            reason += " [no trained classifier; keyword heuristic]"
        if p.confidence < self.config.min_confidence and tier < 3:
            tier += 1
            reason += f"; below min_confidence {self.config.min_confidence:.2f}, bumped to tier {tier}"
        model = self.config.model_for_tier(tier)
        reason += f"; tier {tier} -> {model}"
        return RouteDecision(tier, p.tier, model, p.confidence, p.probabilities, p.source, reason)
