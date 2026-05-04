"""
Configuration models and helpers for IGRIS_GPT.

Configurations are loaded from environment variables with sensible defaults.  A
configuration file under `config/config.sample.json` documents the available
fields but is not loaded at runtime to avoid leaking secrets.  Instead, the
environment variables override the defaults defined here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# Model alias normalization (common misspellings / shorthand)
MODEL_ALIASES: dict[str, str] = {
    "phi4mini": "phi4-mini",
    "phi4_mini": "phi4-mini",
    "phi-4-mini": "phi4-mini",
    "phi4": "phi4-mini",
    "llama3": "llama3.2",
    "llama32": "llama3.2",
    "llama3_2": "llama3.2",
}


def normalize_model_name(model: str) -> str:
    """Normalize model name via alias table."""
    return MODEL_ALIASES.get(model.strip().lower(), model.strip())


class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class VastAIConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "deepseek-r1:32b"
    fallback_model: str = "qwen2.5-coder:7b"
    auto_provision: bool = False
    require_approval: bool = True
    max_hourly_cost: float = 0.50
    mode: str = "on_demand"  # on_demand | always_on | disabled


class Config(BaseModel):
    local_llm: LLMConfig
    fallback_llm: LLMConfig
    vastai: VastAIConfig = VastAIConfig()
    auto_commit: bool = False
    auto_push: bool = False
    workspace_root: Path = Field(default_factory=lambda: Path(os.getenv("WORKSPACE_ROOT", "./workspace")))
    project_root: Path = Field(default_factory=lambda: Path(os.getenv("PROJECT_ROOT", ".")))

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables with defaults."""
        local_llm = LLMConfig(
            provider=os.getenv("LOCAL_LLM_PROVIDER", "ollama"),
            model=normalize_model_name(os.getenv("LOCAL_LLM_MODEL", "phi4-mini")),
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434"),
        )
        fallback_llm = LLMConfig(
            provider=os.getenv("FALLBACK_LLM_PROVIDER", "openai"),
            model=normalize_model_name(os.getenv("FALLBACK_LLM_MODEL", "gpt-4o-mini")),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        vastai = VastAIConfig(
            api_key=os.getenv("VASTAI_API_KEY"),
            model=os.getenv("VASTAI_MODEL", "deepseek-r1:32b"),
            fallback_model=os.getenv("VASTAI_FALLBACK_MODEL", "qwen2.5-coder:7b"),
            auto_provision=os.getenv("VASTAI_AUTO_PROVISION", "false").lower() == "true",
            require_approval=os.getenv("VASTAI_REQUIRE_APPROVAL", "true").lower() != "false",
            max_hourly_cost=float(os.getenv("VASTAI_MAX_HOURLY_COST", "0.50")),
            mode=os.getenv("VASTAI_MODE", "on_demand"),
        )
        auto_commit = os.getenv("AUTO_COMMIT", "false").lower() == "true"
        auto_push = os.getenv("AUTO_PUSH", "false").lower() == "true"
        return cls(
            local_llm=local_llm,
            fallback_llm=fallback_llm,
            vastai=vastai,
            auto_commit=auto_commit,
            auto_push=auto_push,
        )

    def safe_dict(self) -> dict:
        """Return a representation of the configuration without secrets."""
        data = self.model_dump()
        if data["fallback_llm"].get("api_key"):
            data["fallback_llm"]["api_key"] = None
        if data["vastai"].get("api_key"):
            data["vastai"]["api_key"] = None
        return data


CONFIG = Config.load()