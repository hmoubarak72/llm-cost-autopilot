"""Phase 1.1: the model registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

from . import settings

QualityTier = Literal["high", "medium", "low"]
TIER_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class ModelConfig:
    name: str                       # registry key, e.g. "gpt-4o-mini"
    provider: str                   # openai | anthropic | ollama | mock
    model_id: str                   # provider's model string
    input_cost_per_mtok: float      # USD per 1M input tokens
    output_cost_per_mtok: float     # USD per 1M output tokens
    avg_latency_ms: float
    quality_tier: QualityTier

    @property
    def rank(self) -> int:
        return TIER_RANK[self.quality_tier]

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_cost_per_mtok
                + output_tokens * self.output_cost_per_mtok) / 1_000_000

    def to_dict(self) -> dict:
        return asdict(self)


class Registry:
    def __init__(self, models: dict[str, ModelConfig], baseline_model: str):
        if baseline_model not in models:
            raise ValueError(f"baseline_model {baseline_model!r} not in registry")
        self.models = models
        self.baseline_name = baseline_model

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        raw = yaml.safe_load(Path(path or settings.MODELS_YAML).read_text())
        models = {}
        for name, m in raw["models"].items():
            if m["quality_tier"] not in TIER_RANK:
                raise ValueError(f"{name}: quality_tier must be high/medium/low")
            models[name] = ModelConfig(name=name, **m)
        return cls(models, raw["baseline_model"])

    def __getitem__(self, name: str) -> ModelConfig:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(f"unknown model {name!r}; known: {sorted(self.models)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self.models

    def __iter__(self):
        return iter(self.models.values())

    @property
    def baseline(self) -> ModelConfig:
        return self.models[self.baseline_name]
