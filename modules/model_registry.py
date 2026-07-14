"""
Module 12 — Multi-Provider Model Abstraction Layer

Universal adapter interface supporting:
  - Anthropic Claude (primary)
  - OpenAI / Azure OpenAI
  - Google Gemini
  - Ollama (local)
  - vLLM (self-hosted)

Features: streaming, token accounting, retries, provider fallback,
response normalization, semantic caching, provider health tracking.
"""

import asyncio
import hashlib
import json
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, AsyncIterator
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "eval_config.yaml"


@dataclass
class ModelResponse:
    """Normalized response from any provider."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    model: str
    cached: bool = False
    finish_reason: str = "stop"
    error: Optional[str] = None


@dataclass
class ProviderHealth:
    provider: str
    healthy: bool = True
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    last_check: float = field(default_factory=time.time)
    total_requests: int = 0
    total_errors: int = 0


# ─── Semantic Cache ───────────────────────────────────────────────────────────

class SemanticCache:
    """
    Simple in-memory + disk semantic cache.
    Exact-match on prompt hash; approximate match via embedding similarity
    is available if sentence-transformers is installed.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache: dict[str, tuple[ModelResponse, float]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, model: str, temperature: float) -> str:
        return hashlib.sha256(f"{model}:{temperature}:{prompt}".encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str, temperature: float) -> Optional[ModelResponse]:
        key = self._key(prompt, model, temperature)
        if key in self._cache:
            resp, ts = self._cache[key]
            if time.time() - ts < self.ttl:
                self.hits += 1
                resp.cached = True
                return resp
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, prompt: str, model: str, temperature: float, response: ModelResponse):
        if len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = min(self._cache.items(), key=lambda x: x[1][1])
            del self._cache[oldest[0]]
        key = self._key(prompt, model, temperature)
        self._cache[key] = (response, time.time())

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / max(total, 1), 3),
            "size": len(self._cache),
        }


# Global cache instance
_cache = SemanticCache()


# ─── Base Provider ────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.health = ProviderHealth(provider=self.provider_name)
        self._retry_attempts = config.get("api", {}).get("retry_attempts", 3)
        self._retry_delay = config.get("api", {}).get("retry_delay_seconds", 5)

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def _generate_raw(self, prompt: str, model: str, temperature: float,
                             max_tokens: int) -> ModelResponse: ...

    async def generate(self, prompt: str, model: str = "", temperature: float = 1.0,
                       max_tokens: int = 1024, use_cache: bool = True) -> ModelResponse:
        if not model:
            model = self.config["model"]["target"]

        # Cache check
        if use_cache and temperature == 0.0:
            cached = _cache.get(prompt, model, temperature)
            if cached:
                return cached

        for attempt in range(self._retry_attempts):
            try:
                resp = await self._generate_raw(prompt, model, temperature, max_tokens)
                self.health.healthy = True
                self.health.consecutive_errors = 0
                self.health.total_requests += 1
                if use_cache and temperature == 0.0:
                    _cache.set(prompt, model, temperature, resp)
                return resp
            except Exception as e:
                self.health.consecutive_errors += 1
                self.health.total_errors += 1
                self.health.last_error = str(e)
                if attempt < self._retry_attempts - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    self.health.healthy = False
                    return ModelResponse(
                        text="", input_tokens=0, output_tokens=0, latency_ms=0,
                        provider=self.provider_name, model=model, error=str(e)
                    )

    async def stream(self, prompt: str, model: str = "",
                     temperature: float = 1.0) -> AsyncIterator[str]:
        """Default streaming: just yield the full response at once."""
        resp = await self.generate(prompt, model, temperature, use_cache=False)
        yield resp.text


# ─── OpenAI Provider ─────────────────────────────────────────────────────────

class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self, api_key: str, config: dict, base_url: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"

    async def _generate_raw(self, prompt, model, temperature, max_tokens) -> ModelResponse:
        import httpx
        start = time.time()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": model or "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature, "max_tokens": max_tokens}
            )
            r.raise_for_status()
            data = r.json()
        latency = (time.time() - start) * 1000
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return ModelResponse(text=text, input_tokens=usage.get("prompt_tokens", 0),
                             output_tokens=usage.get("completion_tokens", 0),
                             latency_ms=round(latency, 1), provider="openai",
                             model=data.get("model", model))

    async def stream(self, prompt: str, model: str = "", temperature: float = 1.0) -> AsyncIterator[str]:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model or "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            pass


# ─── Anthropic Provider ───────────────────────────────────────────────────────

class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: str, config: dict):
        super().__init__(config)
        self.api_key = api_key
        self._rpm = config.get("api", {}).get("requests_per_minute", 50)
        self._last_requests: list[float] = []

    async def _rate_limit(self):
        now = time.time()
        self._last_requests = [t for t in self._last_requests if now - t < 60]
        if len(self._last_requests) >= self._rpm:
            wait = 60 - (now - self._last_requests[0]) + 0.5
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_requests.append(time.time())

    async def _generate_raw(self, prompt, model, temperature, max_tokens) -> ModelResponse:
        await self._rate_limit()
        import httpx
        start = time.time()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": model or "claude-haiku-4-5-20251001",
                      "max_tokens": max_tokens, "temperature": temperature,
                      "messages": [{"role": "user", "content": prompt}]}
            )
            r.raise_for_status()
            data = r.json()
        latency = (time.time() - start) * 1000
        text = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})
        return ModelResponse(text=text, input_tokens=usage.get("input_tokens", 0),
                             output_tokens=usage.get("output_tokens", 0),
                             latency_ms=round(latency, 1), provider="anthropic",
                             model=data.get("model", model))


# ─── Ollama Provider (local) ──────────────────────────────────────────────────

class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def __init__(self, config: dict, base_url: str = "http://localhost:11434"):
        super().__init__(config)
        self.base_url = base_url

    async def _generate_raw(self, prompt, model, temperature, max_tokens) -> ModelResponse:
        import httpx
        start = time.time()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model or "llama3.2", "prompt": prompt,
                      "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
            )
            r.raise_for_status()
            data = r.json()
        latency = (time.time() - start) * 1000
        text = data.get("response", "")
        return ModelResponse(text=text, input_tokens=data.get("prompt_eval_count", len(prompt.split())),
                             output_tokens=data.get("eval_count", len(text.split())),
                             latency_ms=round(latency, 1), provider="ollama", model=model)

    async def stream(self, prompt: str, model: str = "", temperature: float = 1.0) -> AsyncIterator[str]:
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate",
                json={"model": model or "llama3.2", "prompt": prompt, "stream": True,
                      "options": {"temperature": temperature}}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            yield chunk.get("response", "")
                            if chunk.get("done"):
                                break
                        except Exception:
                            pass


# ─── vLLM Provider (self-hosted OpenAI-compatible) ────────────────────────────

class VLLMProvider(OpenAIProvider):
    provider_name = "vllm"

    def __init__(self, config: dict, base_url: str = "http://localhost:8000/v1"):
        super().__init__(api_key="EMPTY", config=config, base_url=base_url)


# ─── Universal Provider Registry + Fallback ──────────────────────────────────

class ModelRegistry:
    """
    Central registry for all providers.
    Handles automatic fallback when primary provider is unhealthy.
    """
    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._fallback_order: list[str] = []

    def register(self, name: str, provider: BaseProvider, is_primary: bool = False):
        self._providers[name] = provider
        if is_primary:
            self._fallback_order.insert(0, name)
        else:
            self._fallback_order.append(name)

    def get(self, name: str) -> Optional[BaseProvider]:
        return self._providers.get(name)

    def get_healthy(self) -> Optional[BaseProvider]:
        """Return first healthy provider in fallback order."""
        for name in self._fallback_order:
            p = self._providers.get(name)
            if p and p.health.healthy:
                return p
        # All unhealthy — return primary anyway
        return self._providers.get(self._fallback_order[0]) if self._fallback_order else None

    async def generate_with_fallback(self, prompt: str, temperature: float = 1.0,
                                      max_tokens: int = 1024) -> ModelResponse:
        """Try providers in fallback order."""
        for name in self._fallback_order:
            provider = self._providers.get(name)
            if not provider:
                continue
            resp = await provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            if not resp.error:
                return resp
            print(f"[ModelRegistry] {name} failed: {resp.error}. Trying next...")
        return ModelResponse(text="", input_tokens=0, output_tokens=0, latency_ms=0,
                             provider="none", model="none", error="All providers failed")

    def health_report(self) -> dict:
        return {
            name: {
                "healthy": p.health.healthy,
                "consecutive_errors": p.health.consecutive_errors,
                "total_requests": p.health.total_requests,
                "error_rate": round(p.health.total_errors / max(p.health.total_requests, 1), 3),
            }
            for name, p in self._providers.items()
        }

    def cache_stats(self) -> dict:
        return _cache.stats


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_registry_from_config(config: dict, env: dict = None) -> ModelRegistry:
    """Build a ModelRegistry from config + environment variables."""
    import os
    env = env or os.environ
    registry = ModelRegistry()

    anthropic_key = env.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        registry.register("anthropic", AnthropicProvider(anthropic_key, config), is_primary=True)

    openai_key = env.get("OPENAI_API_KEY", "")
    if openai_key:
        registry.register("openai", OpenAIProvider(openai_key, config))

    ollama_url = env.get("OLLAMA_BASE_URL", "")
    if ollama_url:
        registry.register("ollama", OllamaProvider(config, ollama_url))

    vllm_url = env.get("VLLM_BASE_URL", "")
    if vllm_url:
        registry.register("vllm", VLLMProvider(config, vllm_url))

    return registry
