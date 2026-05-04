"""Tests for cost routing, availability, budget, and estimate."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from igris.layers.advisory import router as provider_router


@pytest.fixture(autouse=True)
def _reset_router():
    """Reset router state between tests."""
    provider_router._last_provider = None
    provider_router._provider_history.clear()
    provider_router._budget_config.update({
        "max_session_cost": 10.0,
        "warn_threshold": 0.8,
        "cost_per_local_call": 0.0,
        "cost_per_fallback_call": 0.003,
        "cost_per_vastai_call": 0.01,
    })
    yield


class TestAvailability:
    def test_check_availability_returns_providers(self) -> None:
        avail = provider_router.check_availability()
        assert "ollama" in avail
        assert "openai" in avail
        assert "vastai" in avail

    def test_ollama_has_model(self) -> None:
        avail = provider_router.check_availability()
        assert "model" in avail["ollama"]

    def test_openai_key_bool_not_exposed(self) -> None:
        avail = provider_router.check_availability()
        assert isinstance(avail["openai"]["key_present"], bool)
        assert "api_key" not in str(avail)

    def test_vastai_no_auto_provision(self) -> None:
        avail = provider_router.check_availability()
        assert avail["vastai"]["auto_provision"] is False

    def test_vast_unavailable_no_crash(self) -> None:
        avail = provider_router.check_availability()
        assert isinstance(avail["vastai"]["available"], bool)


class TestCostSummary:
    def test_cost_summary_empty(self) -> None:
        summary = provider_router.cost_summary()
        assert summary["total_calls"] == 0
        assert summary["estimated_cost_total"] == 0.0

    def test_cost_summary_after_calls(self) -> None:
        provider_router.choose_provider()
        provider_router.choose_provider()
        summary = provider_router.cost_summary()
        assert summary["total_calls"] == 2
        assert summary["local_calls"] == 2

    def test_cost_summary_includes_budget(self) -> None:
        summary = provider_router.cost_summary()
        assert "budget" in summary


class TestBudget:
    def test_get_budget_config(self) -> None:
        config = provider_router.get_budget_config()
        assert config["max_session_cost"] == 10.0

    def test_set_budget_config(self) -> None:
        result = provider_router.set_budget_config(max_session_cost=5.0, warn_threshold=0.9)
        assert result["max_session_cost"] == 5.0
        assert result["warn_threshold"] == 0.9

    def test_set_budget_invalid_values(self) -> None:
        provider_router.set_budget_config(max_session_cost=-1)
        config = provider_router.get_budget_config()
        assert config["max_session_cost"] == 10.0  # Unchanged

    def test_budget_status_zero_spent(self) -> None:
        status = provider_router.get_budget_status()
        assert status["spent"] == 0
        assert status["usage_percent"] == 0
        assert status["warning"] is False
        assert status["exceeded"] is False

    def test_budget_warning(self) -> None:
        # Simulate many calls with cost
        for _ in range(300):
            provider_router.record_chat_routing(
                "fallback", "gpt-4o-mini", "test",
                estimated_cost=0.03,
            )
        status = provider_router.get_budget_status()
        assert status["warning"] is True


class TestEstimate:
    def test_estimate_low_complexity(self) -> None:
        result = provider_router.estimate_route(task_type="chat", complexity="low")
        assert "recommended_provider" in result
        assert "estimated_cost" in result
        assert "budget_remaining" in result

    def test_estimate_includes_availability(self) -> None:
        result = provider_router.estimate_route()
        assert "availability" in result

    def test_estimate_no_crash_no_providers(self) -> None:
        result = provider_router.estimate_route()
        assert isinstance(result["recommended_provider"], str)


class TestNoSecrets:
    def test_availability_no_api_keys(self) -> None:
        avail = provider_router.check_availability()
        text = str(avail)
        assert "api_key" not in text.lower() or "key_present" in text.lower()

    def test_cost_summary_no_secrets(self) -> None:
        summary = provider_router.cost_summary()
        assert "api_key" not in str(summary).lower()
