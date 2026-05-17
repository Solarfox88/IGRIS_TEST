"""Tests for phi4-mini local LLM hardening (Sprint 19)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from igris.models.config import MODEL_ALIASES, normalize_model_name


class TestModelAliases:
    def test_phi4mini_alias(self):
        assert normalize_model_name("phi4mini") == "phi4-mini"

    def test_phi4_mini_underscore(self):
        assert normalize_model_name("phi4_mini") == "phi4-mini"

    def test_phi_4_mini_dash(self):
        assert normalize_model_name("phi-4-mini") == "phi4-mini"

    def test_phi4_shorthand(self):
        assert normalize_model_name("phi4") == "phi4-mini"

    def test_phi4_mini_unchanged(self):
        assert normalize_model_name("phi4-mini") == "phi4-mini"

    def test_llama3_alias(self):
        assert normalize_model_name("llama3") == "llama3.2"

    def test_llama32_alias(self):
        assert normalize_model_name("llama32") == "llama3.2"

    def test_unknown_model_unchanged(self):
        assert normalize_model_name("mistral") == "mistral"

    def test_whitespace_stripped(self):
        assert normalize_model_name("  phi4mini  ") == "phi4-mini"

    def test_case_insensitive(self):
        assert normalize_model_name("PHI4MINI") == "phi4-mini"


class TestConfigDefaults:
    def test_default_provider_ollama(self):
        from igris.models.config import CONFIG
        assert CONFIG.local_llm.provider == "ollama"

    def test_default_model_phi4_mini(self):
        from igris.models.config import CONFIG
        assert CONFIG.local_llm.model == "phi4-mini"

    def test_default_base_url(self):
        from igris.models.config import CONFIG
        assert "11434" in (CONFIG.local_llm.base_url or "")


class TestChatWithoutOllama:
    def test_chat_no_crash(self):
        from igris.core.chat_engine import chat
        result = chat("hello")
        assert "text" in result
        assert "provider" in result
        assert "fallback_used" in result

    def test_chat_returns_deterministic_fallback(self):
        from igris.core.chat_engine import chat
        result = chat("help me with tests")
        assert len(result["text"]) > 0

    def test_chat_shows_provider_metadata(self):
        from igris.core.chat_engine import chat
        result = chat("status")
        assert "latency_ms" in result
        assert "routing_reason" in result
        assert "model" in result

    @pytest.mark.slow
    def test_streaming_no_crash(self):
        from igris.core.chat_streaming import chat_stream_sync
        chunks = chat_stream_sync("hello")
        assert len(chunks) >= 1
        done = chunks[-1]
        assert done.type == "done"
        assert "provider" in done.metadata


class TestReadinessEndpoint:
    @pytest.fixture
    def client(self, tmp_path):
        from igris.models.config import CONFIG
        root = tmp_path / "project"
        root.mkdir(exist_ok=True)
        for d in [".igris/tasks", ".igris/timeline", ".igris/memory", ".igris/reports/decisions"]:
            (root / d).mkdir(parents=True, exist_ok=True)
        os.environ["PROJECT_ROOT"] = str(root)
        os.environ["WORKSPACE_ROOT"] = str(root)
        CONFIG.project_root = Path(str(root))
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        return TestClient(create_app())

    def test_readiness_has_model_configured(self, client):
        r = client.get("/api/readiness")
        assert r.status_code == 200
        d = r.json()
        assert "local_model_configured" in d
        assert d["local_model_configured"] == "phi4-mini"

    def test_readiness_has_model_available(self, client):
        r = client.get("/api/readiness")
        d = r.json()
        assert "local_model_available" in d

    def test_readiness_has_fallback_info(self, client):
        r = client.get("/api/readiness")
        d = r.json()
        assert "fallback_active" in d
        assert "fallback_reason" in d

    def test_readiness_no_secrets(self, client):
        r = client.get("/api/readiness")
        text = json.dumps(r.json())
        assert "sk-" not in text
        assert "ghp_" not in text


class TestRoutingAvailability:
    @pytest.fixture
    def client(self, tmp_path):
        from igris.models.config import CONFIG
        root = tmp_path / "project"
        root.mkdir(exist_ok=True)
        for d in [".igris/tasks", ".igris/timeline", ".igris/memory", ".igris/reports/decisions"]:
            (root / d).mkdir(parents=True, exist_ok=True)
        os.environ["PROJECT_ROOT"] = str(root)
        os.environ["WORKSPACE_ROOT"] = str(root)
        CONFIG.project_root = Path(str(root))
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        return TestClient(create_app())

    def test_availability_has_model_info(self, client):
        r = client.get("/api/routing/availability")
        d = r.json()
        assert "model_configured" in d["ollama"]
        assert "model_available" in d["ollama"]
        assert "status" in d["ollama"]
        assert "reachable" in d["ollama"]

    def test_availability_has_fallback_chain(self, client):
        r = client.get("/api/routing/availability")
        d = r.json()
        assert "fallback_chain" in d

    def test_availability_has_status_fields(self, client):
        r = client.get("/api/routing/availability")
        d = r.json()
        assert "status" in d["openai"]
        assert "status" in d["vastai"]

    def test_availability_no_secrets(self, client):
        r = client.get("/api/routing/availability")
        text = json.dumps(r.json())
        assert "sk-" not in text
        assert "ghp_" not in text
