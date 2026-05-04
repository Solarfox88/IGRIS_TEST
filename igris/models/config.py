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


class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class VastAIConfig(BaseModel):
    api_key: Optional[str] = None


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
            model=os.getenv("LOCAL_LLM_MODEL", "phi4-mini"),
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434"),
        )
        fallback_llm = LLMConfig(
            provider=os.getenv("FALLBACK_LLM_PROVIDER", "openai"),
            model=os.getenv("FALLBACK_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        vastai = VastAIConfig(api_key=os.getenv("VASTAI_API_KEY"))
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