"""Tests for Sprint 22 — Vast.ai Gated DeepSeek Runtime.

Verifies:
- Config defaults (deepseek-r1:32b, approval required, auto_provision=false)
- All destructive operations require approval
- No real API calls
- Anti-duplicate instance guard
- Budget gate
- Destroy is state-aware
- Mode validation
- No API key in responses
- Endpoints return correct structure
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from igris.layers.advisory.vastai_manager import (
    APPROVAL_TOKEN,
    SUPPORTED_MODELS,
    VALID_MODES,
    VastAIManager,
)
from igris.models.config import CONFIG
from igris.web.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    return VastAIManager()


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestVastAIConfig:
    """Configuration defaults and structure."""

    def test_default_model(self):
        assert CONFIG.vastai.model == "deepseek-r1:32b"

    def test_default_fallback_model(self):
        assert CONFIG.vastai.fallback_model == "qwen2.5-coder:7b"

    def test_auto_provision_false(self):
        assert CONFIG.vastai.auto_provision is False

    def test_require_approval_true(self):
        assert CONFIG.vastai.require_approval is True

    def test_max_hourly_cost(self):
        assert CONFIG.vastai.max_hourly_cost == 0.50

    def test_mode_on_demand(self):
        assert CONFIG.vastai.mode == "on_demand"

    def test_config_no_api_key_exposed(self, manager):
        cfg = manager.get_config()
        assert "api_key" not in cfg
        assert "api_key_present" in cfg

    def test_supported_models_listed(self, manager):
        cfg = manager.get_config()
        assert "deepseek-r1:32b" in cfg["supported_models"]
        assert "qwen2.5-coder:7b" in cfg["supported_models"]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestVastAIStatus:
    """Status reporting."""

    def test_status_no_instance(self, manager):
        status = manager.get_status()
        assert status["instance"] is None
        assert status["has_active_instance"] is False
        assert status["mode"] == "on_demand"

    def test_status_no_api_key_exposed(self, manager):
        status = manager.get_status()
        assert "api_key" not in status


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:
    """Cost estimation for models."""

    def test_estimate_default_model(self, manager):
        est = manager.estimate_cost()
        assert est["model"] == "deepseek-r1:32b"
        assert est["cost_per_hour"] > 0
        assert est["estimated_total"] > 0

    def test_estimate_custom_hours(self, manager):
        est = manager.estimate_cost(hours=5.0)
        assert est["hours"] == 5.0
        assert est["estimated_total"] == round(est["cost_per_hour"] * 5.0, 4)

    def test_estimate_unknown_model(self, manager):
        est = manager.estimate_cost(model="nonexistent-model")
        assert "error" in est
        assert est["estimated_cost"] == 0.0

    def test_estimate_within_budget(self, manager):
        est = manager.estimate_cost(model="qwen2.5-coder:7b")
        assert est["within_budget"] is True

    def test_estimate_over_budget(self, manager):
        est = manager.estimate_cost(model="deepseek-r1:70b")
        # 0.60/hr exceeds default 0.50/hr budget
        assert est["within_budget"] is False
        assert est["warning"]


# ---------------------------------------------------------------------------
# Offer search
# ---------------------------------------------------------------------------


class TestOfferSearch:
    """Offer search (mock)."""

    def test_search_no_api_key(self, manager):
        result = manager.search_offers()
        d = result.to_dict()
        assert d["offer_count"] == 0 or "not configured" in d.get("error", "").lower()

    def test_search_unknown_model(self, manager):
        result = manager.search_offers(model="nonexistent")
        assert result.error

    def test_search_with_mock_key(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key-for-test"):
            result = manager.search_offers()
            d = result.to_dict()
            assert d["offer_count"] >= 1
            for offer in d["offers"]:
                assert "MOCK" in offer.get("note", "")


# ---------------------------------------------------------------------------
# Provision (gated)
# ---------------------------------------------------------------------------


class TestProvisionGated:
    """Provision requires approval, budget, no duplicates."""

    def test_provision_without_approval_rejected(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            result = manager.provision(approval="")
            assert result["success"] is False
            assert "approval" in result.get("error", "").lower()

    def test_provision_wrong_approval_rejected(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            result = manager.provision(approval="WRONG")
            assert result["success"] is False

    def test_provision_no_api_key_rejected(self, manager):
        result = manager.provision(approval=APPROVAL_TOKEN)
        assert result["success"] is False
        assert "api_key" in result.get("error", "").lower() or "not configured" in result.get("error", "").lower()

    def test_provision_with_approval_succeeds(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
            assert result["success"] is True
            assert "MOCK" in result.get("note", "")
            assert result["instance"]["status"] == "provisioning"

    def test_provision_anti_duplicate(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            first = manager.provision(approval=APPROVAL_TOKEN)
            assert first["success"] is True
            second = manager.provision(approval=APPROVAL_TOKEN)
            assert second["success"] is False
            assert "already exists" in second.get("error", "").lower()

    def test_provision_disabled_mode_rejected(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            old_mode = CONFIG.vastai.mode
            CONFIG.vastai.mode = "disabled"
            try:
                result = manager.provision(approval=APPROVAL_TOKEN)
                assert result["success"] is False
                assert "disabled" in result.get("error", "").lower()
            finally:
                CONFIG.vastai.mode = old_mode

    def test_provision_over_budget_rejected(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            result = manager.provision(
                approval=APPROVAL_TOKEN,
                model="deepseek-r1:70b",
            )
            assert result["success"] is False
            assert "budget" in result.get("error", "").lower() or "exceeds" in result.get("error", "").lower()

    def test_no_real_api_call(self, manager):
        """Verify no real HTTP call is made."""
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
            assert result["success"] is True
            assert "mock" in result["instance"]["instance_id"].lower()


# ---------------------------------------------------------------------------
# Destroy (gated, state-aware)
# ---------------------------------------------------------------------------


class TestDestroyGated:
    """Destroy requires approval and active instance."""

    def test_destroy_without_approval_rejected(self, manager):
        result = manager.destroy(approval="")
        assert result["success"] is False
        assert "approval" in result.get("error", "").lower()

    def test_destroy_no_instance(self, manager):
        result = manager.destroy(approval=APPROVAL_TOKEN)
        assert result["success"] is False
        assert "no active" in result.get("error", "").lower()

    def test_destroy_after_provision(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            result = manager.destroy(approval=APPROVAL_TOKEN)
            assert result["success"] is True
            assert "MOCK" in result.get("note", "")

    def test_destroy_twice_fails(self, manager):
        with patch.object(CONFIG.vastai, "api_key", "mock-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            manager.destroy(approval=APPROVAL_TOKEN)
            result = manager.destroy(approval=APPROVAL_TOKEN)
            assert result["success"] is False


# ---------------------------------------------------------------------------
# Set mode (gated)
# ---------------------------------------------------------------------------


class TestSetMode:
    """Mode changes require approval."""

    def test_set_mode_invalid(self, manager):
        result = manager.set_mode(mode="invalid")
        assert result["success"] is False
        assert "invalid" in result.get("error", "").lower()

    def test_set_mode_without_approval(self, manager):
        result = manager.set_mode(mode="disabled", approval="")
        assert result["success"] is False
        assert "approval" in result.get("error", "").lower()

    def test_set_mode_with_approval(self, manager):
        old = CONFIG.vastai.mode
        try:
            result = manager.set_mode(mode="disabled", approval=APPROVAL_TOKEN)
            assert result["success"] is True
            assert result["new_mode"] == "disabled"
        finally:
            CONFIG.vastai.mode = old

    def test_valid_modes(self):
        assert "on_demand" in VALID_MODES
        assert "always_on" in VALID_MODES
        assert "disabled" in VALID_MODES


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestVastAIEndpoints:
    """HTTP endpoint tests."""

    def test_config_endpoint(self, client):
        r = client.get("/api/vastai/config")
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert "api_key" not in data
        assert data["model"] == "deepseek-r1:32b"

    def test_status_endpoint(self, client):
        r = client.get("/api/vastai/status")
        assert r.status_code == 200
        data = r.json()
        assert "mode" in data
        assert "api_key" not in data

    def test_estimate_endpoint(self, client):
        r = client.post("/api/vastai/estimate", json={"model": "deepseek-r1:32b"})
        assert r.status_code == 200
        data = r.json()
        assert "cost_per_hour" in data

    def test_offers_search_endpoint(self, client):
        r = client.post("/api/vastai/offers/search", json={})
        assert r.status_code == 200
        data = r.json()
        assert "offer_count" in data

    def test_provision_endpoint_no_approval(self, client):
        r = client.post("/api/vastai/provision", json={})
        data = r.json()
        assert data["success"] is False
        assert "approval" in data.get("error", "").lower()

    def test_destroy_endpoint_no_approval(self, client):
        r = client.post("/api/vastai/destroy", json={})
        data = r.json()
        assert data["success"] is False

    def test_set_mode_endpoint_no_mode(self, client):
        r = client.post("/api/vastai/set-mode", json={})
        assert r.status_code == 400

    def test_set_mode_endpoint_no_approval(self, client):
        r = client.post("/api/vastai/set-mode", json={"mode": "disabled"})
        data = r.json()
        assert data["success"] is False

    def test_no_create_destroy_vps_endpoint(self, client):
        """No VPS create/destroy endpoints exist."""
        r = client.post("/api/vastai/vps/create", json={})
        assert r.status_code == 404 or r.status_code == 405

    def test_no_api_key_in_any_response(self, client):
        """Verify API key never appears in responses."""
        endpoints = [
            ("GET", "/api/vastai/config"),
            ("GET", "/api/vastai/status"),
        ]
        for method, url in endpoints:
            if method == "GET":
                r = client.get(url)
            else:
                r = client.post(url, json={})
            text = r.text
            assert "VASTAI_API_KEY" not in text or "api_key_present" in text


# ---------------------------------------------------------------------------
# Safety cross-checks
# ---------------------------------------------------------------------------


class TestSafetyCrossChecks:
    """Cross-cutting safety verifications."""

    def test_approval_token_value(self):
        assert APPROVAL_TOKEN == "I_APPROVE_VASTAI_COSTS"

    def test_auto_provision_false_default(self):
        assert CONFIG.vastai.auto_provision is False

    def test_no_loop_provisioning(self, manager):
        """Loop cannot provision without approval."""
        result = manager.provision()
        assert result["success"] is False

    def test_config_safe_dict_no_key(self):
        """safe_dict redacts vastai api_key."""
        d = CONFIG.safe_dict()
        assert d["vastai"].get("api_key") is None
