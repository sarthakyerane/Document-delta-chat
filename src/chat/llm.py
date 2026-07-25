"""
delta-chat · src/chat/llm.py
══════════════════════════════════════════════════════════════════════════════
Provider-agnostic LLM client — 3-tier fallback pattern.

Tier 1: Groq (llama-3.3-70b-versatile) — fastest, primary
Tier 2: Gemini (gemini-2.5-flash) — vision capable, fallback
Tier 3: Ollama (local) — zero cost, last resort

Pattern mirrors CodeSage: zero-downtime, no hard API key dependency.
Credentials read from env vars — never hardcoded.

LLM non-determinism is isolated here.  Callers that need determinism
should use temperature=0 and note in comments when they do so.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    estimated_cost_usd: float


class AllProvidersFailedError(Exception):
    """Raised when all configured LLM providers fail."""


# ─────────────────────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────────────────────

class GroqProvider:
    name = "groq"

    def available(self) -> bool:
        return bool(settings.groq_api_key)

    def complete(
        self, messages: list[dict], temperature: float = 0.1, **kwargs
    ) -> LLMResponse:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        model = settings.groq_model

        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        duration_ms = (time.time() - t0) * 1000

        content = response.choices[0].message.content or ""
        in_t = response.usage.prompt_tokens if response.usage else 0
        out_t = response.usage.completion_tokens if response.usage else 0
        cost = settings.estimate_cost(model, in_t, out_t)

        return LLMResponse(
            content=content, model=model, provider=self.name,
            input_tokens=in_t, output_tokens=out_t,
            duration_ms=duration_ms, estimated_cost_usd=cost,
        )


class GeminiProvider:
    name = "gemini"

    def available(self) -> bool:
        return bool(settings.google_api_key)

    def complete(
        self, messages: list[dict], temperature: float = 0.1, **kwargs
    ) -> LLMResponse:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=settings.google_api_key)
        model_name = settings.gemini_model
        model = genai.GenerativeModel(model_name)

        # Convert OpenAI-format messages to Gemini format
        gemini_msgs = []
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            gemini_msgs.append({"role": role, "parts": [{"text": m["content"]}]})

        t0 = time.time()
        response = model.generate_content(
            gemini_msgs,
            generation_config={"temperature": temperature,
                               "max_output_tokens": kwargs.get("max_tokens", 4096)},
        )
        duration_ms = (time.time() - t0) * 1000
        content = response.text or ""

        # Approximate token count (Gemini doesn't always return usage)
        in_t = sum(len(m.get("content", "").split()) * 4 // 3 for m in messages)
        out_t = len(content.split()) * 4 // 3
        cost = settings.estimate_cost(model_name, in_t, out_t)

        return LLMResponse(
            content=content, model=model_name, provider=self.name,
            input_tokens=in_t, output_tokens=out_t,
            duration_ms=duration_ms, estimated_cost_usd=cost,
        )


class OllamaProvider:
    name = "ollama"

    def available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def complete(
        self, messages: list[dict], temperature: float = 0.1, **kwargs
    ) -> LLMResponse:
        import httpx

        model = settings.ollama_model
        t0 = time.time()
        r = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json={"model": model, "messages": messages,
                  "options": {"temperature": temperature}, "stream": False},
            timeout=120.0,
        )
        r.raise_for_status()
        duration_ms = (time.time() - t0) * 1000

        data = r.json()
        content = data.get("message", {}).get("content", "")
        in_t = data.get("prompt_eval_count", 0)
        out_t = data.get("eval_count", 0)

        return LLMResponse(
            content=content, model=model, provider=self.name,
            input_tokens=in_t, output_tokens=out_t,
            duration_ms=duration_ms, estimated_cost_usd=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM client — provider-agnostic with automatic fallback
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER_MAP = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


class LLMClient:
    """
    Provider-agnostic LLM client.
    Iterates through settings.provider_order, returns first successful response.
    Records telemetry for every call (success or failure) in the tracer.
    """

    def __init__(self, tracer: Optional[RequestTracer] = None):
        self.tracer = tracer
        self._providers: list[Any] = []
        for name in settings.provider_order:
            cls = _PROVIDER_MAP.get(name)
            if cls:
                self._providers.append(cls())

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(min=1, max=4))
    def complete_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        """
        Synchronous LLM call with automatic provider fallback.
        """
        errors: list[str] = []
        for provider in self._providers:
            if not provider.available():
                log.debug("llm.provider_unavailable", provider=provider.name)
                continue
            try:
                with (self.tracer.span(f"llm.call.{provider.name}")
                      if self.tracer else _null_ctx()) as span:
                    response = provider.complete(messages, temperature, **kwargs)
                    if self.tracer:
                        self.tracer.record_llm_call(
                            model=response.model,
                            provider=response.provider,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            duration_ms=response.duration_ms,
                        )
                    log.info(
                        "llm.call.success",
                        provider=provider.name,
                        model=response.model,
                        in_tokens=response.input_tokens,
                        out_tokens=response.output_tokens,
                        cost_usd=round(response.estimated_cost_usd, 6),
                        duration_ms=round(response.duration_ms, 1),
                    )
                    return response
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                if self.tracer:
                    self.tracer.record_llm_call(
                        model=getattr(provider, "model_name", provider.name),
                        provider=provider.name,
                        input_tokens=0, output_tokens=0, duration_ms=0.0,
                        status="ERROR", error=str(exc),
                    )
                log.warning("llm.provider_failed",
                            provider=provider.name, error=str(exc))
                continue

        raise AllProvidersFailedError(
            f"All LLM providers exhausted. Errors: {'; '.join(errors)}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using Google's embedding model."""
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY required for embeddings")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=settings.google_api_key)
        result = genai.embed_content(
            model=f"models/{settings.gemini_embedding_model}",
            content=texts,
            task_type="retrieval_document",
        )
        return result["embedding"] if isinstance(texts, str) else result["embedding"]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY required for embeddings")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=settings.google_api_key)
        result = genai.embed_content(
            model=f"models/{settings.gemini_embedding_model}",
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]


# Null context manager
from contextlib import contextmanager


@contextmanager
def _null_ctx():
    yield None
