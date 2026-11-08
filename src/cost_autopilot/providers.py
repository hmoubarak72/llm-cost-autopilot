"""Phase 1.2: one send_request() in front of every provider.

All providers are called over plain HTTP with httpx, so there is no SDK
version drift and the request/response shapes are visible in one file.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import httpx

from . import settings
from .registry import ModelConfig
from .tasks import Message


class ProviderError(RuntimeError):
    pass


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    model: str          # registry name
    model_id: str       # provider model string
    provider: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return asdict(self)


def _messages(prompt: str | list[Message], system: str | None) -> list[Message]:
    msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if system and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs


_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_S)
    return _client


def _key(var: str) -> str:
    k = os.environ.get(var)
    if not k:
        raise ProviderError(f"{var} is not set")
    return k


async def _openai(cfg: ModelConfig, msgs, max_tokens, temperature):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    r = await _http().post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"},
        json={"model": cfg.model_id, "messages": msgs,
              "max_tokens": max_tokens, "temperature": temperature},
    )
    if r.status_code >= 400:
        raise ProviderError(f"openai {r.status_code}: {r.text[:300]}")
    j = r.json()
    u = j.get("usage", {})
    return j["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


async def _anthropic(cfg: ModelConfig, msgs, max_tokens, temperature):
    system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    convo = [m for m in msgs if m["role"] != "system"]
    body = {"model": cfg.model_id, "messages": convo,
            "max_tokens": max_tokens, "temperature": temperature}
    if system:
        body["system"] = system
    r = await _http().post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
        json=body,
    )
    if r.status_code >= 400:
        raise ProviderError(f"anthropic {r.status_code}: {r.text[:300]}")
    j = r.json()
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    u = j.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


async def _ollama(cfg: ModelConfig, msgs, max_tokens, temperature):
    try:
        r = await _http().post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={"model": cfg.model_id, "messages": msgs, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens}},
        )
    except httpx.TransportError as e:
        raise ProviderError(f"ollama unreachable at {settings.OLLAMA_URL}: {e}") from e
    if r.status_code >= 400:
        raise ProviderError(f"ollama {r.status_code}: {r.text[:300]}")
    j = r.json()
    return j["message"]["content"], j.get("prompt_eval_count", 0), j.get("eval_count", 0)


_DISPATCH = {"openai": _openai, "anthropic": _anthropic, "ollama": _ollama}


async def send_request(prompt: str | list[Message], model_config: ModelConfig, *,
                       system: str | None = None, max_tokens: int = 1024,
                       temperature: float = 0.0) -> Response:
    """Send one request to any registered model and get a standardized Response."""
    msgs = _messages(prompt, system)
    if settings.MOCK or model_config.provider == "mock":
        from .mock import simulate
        return simulate(msgs, model_config)

    fn = _DISPATCH.get(model_config.provider)
    if fn is None:
        raise ProviderError(f"unknown provider {model_config.provider!r}")
    t0 = time.perf_counter()
    text, tin, tout = await fn(model_config, msgs, max_tokens, temperature)
    latency = (time.perf_counter() - t0) * 1000
    return Response(text=text, input_tokens=tin, output_tokens=tout, latency_ms=round(latency, 1),
                    cost=model_config.cost(tin, tout), model=model_config.name,
                    model_id=model_config.model_id, provider=model_config.provider)
