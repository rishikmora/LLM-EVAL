"""
providers/base.py — Universal Model Abstraction Layer

Supports: Anthropic Claude, OpenAI, Gemini, Ollama, vLLM
Features: streaming, token accounting, retries, provider fallback,
          response normalization, semantic caching, circuit breaker
"""
from __future__ import annotations
import asyncio, hashlib, json, os, time, re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "eval_config.yaml"


class ProviderType(str, Enum):
    GEMINI    = "gemini"
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA    = "ollama"
    VLLM      = "vllm"


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    model: str
    cached: bool = False
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self): return self.input_tokens + self.output_tokens
    @property
    def token_efficiency(self): return self.output_tokens / max(self.input_tokens, 1)


@dataclass
class ProviderConfig:
    provider: ProviderType
    model: str
    api_key: str = ""
    base_url: str = ""
    temperature: float = 1.0
    max_tokens: int = 1024
    timeout: int = 60
    rpm: int = 10
    daily_token_budget: int = 900_000
    retry_attempts: int = 3
    retry_delay: float = 5.0
    enabled: bool = True


# ─── Semantic Cache ───────────────────────────────────────────────────────────

class SemanticCache:
    """In-memory + optional Redis semantic cache with TTL."""
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._cache: dict[str, tuple[ModelResponse, float]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, model: str, temperature: float) -> str:
        if temperature == 0.0:  # deterministic — cache aggressively
            return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        return ""  # non-deterministic — don't cache

    def get(self, prompt: str, model: str, temperature: float) -> Optional[ModelResponse]:
        key = self._key(prompt, model, temperature)
        if not key: return None
        entry = self._cache.get(key)
        if entry:
            resp, ts = entry
            if time.time() - ts < self.ttl:
                self.hits += 1
                cached = ModelResponse(**{**resp.__dict__, "cached": True})
                return cached
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, prompt: str, model: str, temperature: float, response: ModelResponse):
        key = self._key(prompt, model, temperature)
        if not key: return
        if len(self._cache) >= self.max_size:
            oldest = min(self._cache.items(), key=lambda x: x[1][1])
            del self._cache[oldest[0]]
        self._cache[key] = (response, time.time())

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 3),
                "size": len(self._cache), "max_size": self.max_size}


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """Prevents cascade failures when a provider is down."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time = 0.0
        self._state = "closed"  # closed=ok, open=tripped, half_open=testing

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                return False
            return True
        return False

    def record_success(self):
        self._failures = 0
        self._state = "closed"

    def record_failure(self):
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "open"

    def state(self) -> str: return self._state


# ─── Base Provider ────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.cache = SemanticCache()
        self.circuit_breaker = CircuitBreaker()
        self._request_times: list[float] = []

    async def _rate_limit(self):
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self.config.rpm:
            wait = 60 - (now - self._request_times[0]) + 0.1
            if wait > 0:
                await asyncio.sleep(wait)
        self._request_times.append(time.time())

    async def generate(self, prompt: str, temperature: Optional[float] = None,
                       stream: bool = False) -> ModelResponse:
        if self.circuit_breaker.is_open:
            raise RuntimeError(f"[{self.config.provider}] Circuit breaker OPEN — provider unavailable")

        temp = temperature if temperature is not None else self.config.temperature
        cached = self.cache.get(prompt, self.config.model, temp)
        if cached:
            return cached

        await self._rate_limit()
        for attempt in range(self.config.retry_attempts):
            try:
                resp = await self._generate_impl(prompt, temp, stream)
                self.cache.set(prompt, self.config.model, temp, resp)
                self.circuit_breaker.record_success()
                return resp
            except Exception as e:
                self.circuit_breaker.record_failure()
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"[{self.config.provider}] Failed after {self.config.retry_attempts} attempts: {e}")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Streaming generation — yields chunks."""
        async for chunk in self._stream_impl(prompt):
            yield chunk

    @abstractmethod
    async def _generate_impl(self, prompt: str, temperature: float, stream: bool) -> ModelResponse: ...

    async def _stream_impl(self, prompt: str) -> AsyncIterator[str]:
        resp = await self.generate(prompt)
        for word in resp.text.split():
            yield word + " "
            await asyncio.sleep(0.01)

    def health_check(self) -> dict:
        return {"provider": self.config.provider, "model": self.config.model,
                "circuit_breaker": self.circuit_breaker.state(),
                "cache": self.cache.stats(), "enabled": self.config.enabled}


# ─── Gemini Provider ──────────────────────────────────────────────────────────

class GeminiProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        try:
            from google import genai
            from google.genai import types as genai_types
            self._client = genai.Client(api_key=config.api_key)
            self._types = genai_types
        except ImportError:
            self._client = None
            self._types = None

    async def _generate_impl(self, prompt: str, temperature: float, stream: bool) -> ModelResponse:
        if not self._client:
            raise ImportError("google-genai not installed: pip install google-genai")
        start = time.time()
        resp = self._client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=self.config.max_tokens))
        latency = (time.time() - start) * 1000
        text = resp.text if hasattr(resp, "text") and resp.text else ""
        try:
            in_tok = resp.usage_metadata.prompt_token_count or 0
            out_tok = resp.usage_metadata.candidates_token_count or 0
        except Exception:
            in_tok, out_tok = len(prompt.split()), len(text.split())
        return ModelResponse(text=text, input_tokens=in_tok, output_tokens=out_tok,
                             latency_ms=round(latency, 1), provider="gemini",
                             model=self.config.model)


# ─── OpenAI Provider ─────────────────────────────────────────────────────────

class OpenAIProvider(BaseProvider):
    async def _generate_impl(self, prompt: str, temperature: float, stream: bool) -> ModelResponse:
        try:
            import openai
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        start = time.time()
        client = openai.AsyncOpenAI(api_key=self.config.api_key,
                                     base_url=self.config.base_url or None)
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        latency = (time.time() - start) * 1000
        text = resp.choices[0].message.content or ""
        return ModelResponse(text=text, input_tokens=resp.usage.prompt_tokens,
                             output_tokens=resp.usage.completion_tokens,
                             latency_ms=round(latency, 1), provider="openai",
                             model=self.config.model,
                             finish_reason=resp.choices[0].finish_reason)

    async def _stream_impl(self, prompt: str) -> AsyncIterator[str]:
        try:
            import openai
        except ImportError:
            raise ImportError("openai not installed")
        client = openai.AsyncOpenAI(api_key=self.config.api_key)
        async with client.chat.completions.stream(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


# ─── Anthropic Provider ───────────────────────────────────────────────────────

class AnthropicProvider(BaseProvider):
    async def _generate_impl(self, prompt: str, temperature: float, stream: bool) -> ModelResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        start = time.time()
        client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        resp = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.time() - start) * 1000
        text = resp.content[0].text if resp.content else ""
        return ModelResponse(text=text, input_tokens=resp.usage.input_tokens,
                             output_tokens=resp.usage.output_tokens,
                             latency_ms=round(latency, 1), provider="anthropic",
                             model=self.config.model,
                             finish_reason=resp.stop_reason or "stop")


# ─── Ollama Provider (local) ──────────────────────────────────────────────────

class OllamaProvider(BaseProvider):
    async def _generate_impl(self, prompt: str, temperature: float, stream: bool) -> ModelResponse:
        import httpx
        base_url = self.config.base_url or "http://localhost:11434"
        start = time.time()
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(f"{base_url}/api/generate",
                json={"model": self.config.model, "prompt": prompt,
                      "stream": False, "options": {"temperature": temperature,
                                                    "num_predict": self.config.max_tokens}})
            resp.raise_for_status()
            data = resp.json()
        latency = (time.time() - start) * 1000
        text = data.get("response", "")
        in_tok = data.get("prompt_eval_count", len(prompt.split()))
        out_tok = data.get("eval_count", len(text.split()))
        return ModelResponse(text=text, input_tokens=in_tok, output_tokens=out_tok,
                             latency_ms=round(latency, 1), provider="ollama",
                             model=self.config.model)


# ─── vLLM Provider (OpenAI-compatible) ───────────────────────────────────────

class VLLMProvider(OpenAIProvider):
    """vLLM exposes an OpenAI-compatible API."""
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.base_url:
            config.base_url = "http://localhost:8000/v1"

    async def _generate_impl(self, prompt: str, temperature: float, stream: bool) -> ModelResponse:
        resp = await super()._generate_impl(prompt, temperature, stream)
        return ModelResponse(**{**resp.__dict__, "provider": "vllm"})


# ─── Provider Registry & Fallback ────────────────────────────────────────────

PROVIDER_CLASSES = {
    ProviderType.GEMINI:    GeminiProvider,
    ProviderType.OPENAI:    OpenAIProvider,
    ProviderType.ANTHROPIC: AnthropicProvider,
    ProviderType.OLLAMA:    OllamaProvider,
    ProviderType.VLLM:      VLLMProvider,
}


class ProviderRegistry:
    """
    Universal provider manager with fallback chain.
    Usage:
        registry = ProviderRegistry.from_config(cfg)
        response = await registry.generate("hello")
        response = await registry.generate("hello", provider="openai")
    """
    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._fallback_chain: list[str] = []

    def register(self, name: str, provider: BaseProvider, primary: bool = False):
        self._providers[name] = provider
        if primary:
            self._fallback_chain.insert(0, name)
        else:
            self._fallback_chain.append(name)

    async def generate(self, prompt: str, provider: Optional[str] = None,
                       temperature: Optional[float] = None, stream: bool = False) -> ModelResponse:
        targets = [provider] if provider else self._fallback_chain
        last_error = None
        for name in targets:
            p = self._providers.get(name)
            if not p or not p.config.enabled:
                continue
            if p.circuit_breaker.is_open:
                continue
            try:
                return await p.generate(prompt, temperature, stream)
            except Exception as e:
                last_error = e
                print(f"[Registry] Provider '{name}' failed: {e}. Trying fallback...")
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def health(self) -> dict:
        return {name: p.health_check() for name, p in self._providers.items()}

    @classmethod
    def from_config(cls, config: dict) -> "ProviderRegistry":
        registry = cls()
        providers_cfg = config.get("providers", {})

        # Anthropic Claude — primary provider
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            cfg = ProviderConfig(
                provider=ProviderType.ANTHROPIC,
                model=providers_cfg.get("anthropic", {}).get("model", "claude-haiku-4-5-20251001"),
                api_key=anthropic_key,
                temperature=config["model"]["target_temperature"],
                max_tokens=config["model"]["max_tokens"],
                rpm=providers_cfg.get("anthropic", {}).get("rpm", 50),
            )
            registry.register("anthropic", AnthropicProvider(cfg), primary=True)

        # OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            cfg = ProviderConfig(
                provider=ProviderType.OPENAI,
                model=providers_cfg.get("openai", {}).get("model", "gpt-4o-mini"),
                api_key=openai_key,
            )
            registry.register("openai", OpenAIProvider(cfg))

        # Ollama (local — no key needed)
        if providers_cfg.get("ollama", {}).get("enabled", False):
            cfg = ProviderConfig(
                provider=ProviderType.OLLAMA,
                model=providers_cfg.get("ollama", {}).get("model", "llama3"),
                base_url=providers_cfg.get("ollama", {}).get("base_url", "http://localhost:11434"),
            )
            registry.register("ollama", OllamaProvider(cfg))

        # vLLM (local — no key needed)
        if providers_cfg.get("vllm", {}).get("enabled", False):
            cfg = ProviderConfig(
                provider=ProviderType.VLLM,
                model=providers_cfg.get("vllm", {}).get("model", "meta-llama/Llama-3-8b"),
                base_url=providers_cfg.get("vllm", {}).get("base_url", "http://localhost:8000/v1"),
            )
            registry.register("vllm", VLLMProvider(cfg))

        return registry
