"""
delta-chat · src/config.py
Centralised, type-safe configuration via pydantic-settings.
All values read from environment variables (or .env file).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider_order: str = "groq,gemini,ollama"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    @computed_field  # type: ignore[misc]
    @property
    def provider_order(self) -> list[str]:
        return [p.strip() for p in self.llm_provider_order.split(",") if p.strip()]

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./delta_chat.db"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True
    redis_semantic_cache_threshold: float = 0.90
    redis_cache_ttl: int = 3600

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str | None = None
    chroma_port: int | None = None

    # ── Delta alignment thresholds ────────────────────────────────────────────
    alignment_text_sim_high: float = 0.85
    alignment_text_sim_medium: float = 0.70
    alignment_text_sim_llm_min: float = 0.40
    alignment_bbox_iou_high: float = 0.70
    alignment_use_llm: bool = True
    alignment_llm_batch_size: int = 10

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_provider: Literal["gemini", "tesseract"] = "gemini"
    ocr_dpi: int = 200
    ocr_confidence_threshold: float = 0.60

    # ── Chat / retrieval ──────────────────────────────────────────────────────
    retrieval_top_k: int = 6
    retrieval_similarity_threshold: float = 0.35
    chat_max_context_tokens: int = 6000
    chat_insufficient_grounding_msg: bool = True

    # ── Observability ─────────────────────────────────────────────────────────
    traces_dir: str = "./traces"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    otel_service_name: str = "delta-chat"
    otel_service_version: str = "0.1.0"

    # ── Markup ────────────────────────────────────────────────────────────────
    markup_output_dir: str = "./data/markup"
    markup_box_color_add: str = "00CC44"
    markup_box_color_remove: str = "CC2200"
    markup_box_color_modify: str = "CCAA00"
    markup_line_width: int = 2

    # ── DWG ───────────────────────────────────────────────────────────────────
    dwg_oda_converter_path: str = ""

    # ── Cost estimation (USD per 1K tokens) ───────────────────────────────────
    cost_groq_input: float = 0.00059
    cost_groq_output: float = 0.00079
    cost_gemini_input: float = 0.000075
    cost_gemini_output: float = 0.0003
    cost_ollama_input: float = 0.0
    cost_ollama_output: float = 0.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    samples_dir: str = "./data/samples"

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Settings":
        """Create required directories on startup."""
        import os

        for d in [self.traces_dir, self.chroma_persist_dir, self.markup_output_dir]:
            os.makedirs(d, exist_ok=True)
        return self

    def cost_for_model(self, model_name: str) -> dict[str, float]:
        """Return input/output cost per 1K tokens for a model name."""
        if "groq" in model_name or "llama" in model_name.lower():
            return {"input": self.cost_groq_input, "output": self.cost_groq_output}
        if "gemini" in model_name:
            return {"input": self.cost_gemini_input, "output": self.cost_gemini_output}
        # Ollama / local
        return {"input": self.cost_ollama_input, "output": self.cost_ollama_output}

    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for a single LLM call."""
        rates = self.cost_for_model(model_name)
        return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance — cached after first call."""
    return Settings()
