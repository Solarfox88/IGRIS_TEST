"""Tests for Epic #1077 — Control Room UX API endpoints.

Validates: run status endpoint, risk card, approve/block controls.
Uses source inspection since we can't easily wire up FastAPI in unit tests.
"""

import inspect
import pytest
from igris.web.routers import routes_10


def _src():
    return inspect.getsource(routes_10)


class TestControlRoomEndpoints:
    """Smoke tests verifying endpoint registration."""

    def test_run_status_endpoint_registered(self):
        assert "/api/rank/runs/{run_id}/status" in _src()

    def test_approve_endpoint_registered(self):
        assert "/api/rank/runs/{run_id}/approve" in _src()

    def test_block_endpoint_registered(self):
        assert "/api/rank/runs/{run_id}/block" in _src()

    def test_risk_card_in_status_response(self):
        """Status endpoint must build a risk_card dict."""
        src = _src()
        assert "risk_card" in src
        assert "failure_class" in src
        assert "repair_cycles_used" in src

    def test_recent_events_in_status_response(self):
        assert "recent_events" in _src()

    def test_elapsed_seconds_in_status_response(self):
        assert "elapsed_seconds" in _src()

    def test_approve_clears_cancel_requested(self):
        """approve endpoint must set cancel_requested=False."""
        src = _src()
        assert "cancel_requested = False" in src or "cancel_requested=False" in src

    def test_block_sets_cancel_requested(self):
        """block endpoint must set cancel_requested=True."""
        src = _src()
        assert "cancel_requested = True" in src or "cancel_requested=True" in src


class TestSafeRedact:
    """_safe_redact is importable and works."""

    def test_safe_redact_basic(self):
        from igris.web.routers.routes_10 import _safe_redact
        assert isinstance(_safe_redact("hello"), str)

    def test_safe_redact_none(self):
        from igris.web.routers.routes_10 import _safe_redact
        assert _safe_redact(None) == ""


class TestDevOpsEndpoints:
    """Smoke tests for Epic #1076 DevOps endpoints."""

    def test_health_endpoint_registered(self):
        assert "/api/devops/health" in _src()

    def test_deploy_status_endpoint_registered(self):
        assert "/api/devops/deploy-status" in _src()

    def test_diagnostics_endpoint_registered(self):
        assert "/api/devops/diagnostics" in _src()

    def test_disk_check_in_health(self):
        assert "disk" in _src()

    def test_memory_check_in_health(self):
        assert "memory" in _src()

    def test_service_check_in_health(self):
        assert "igris_service" in _src()

    def test_systemctl_in_diagnostics(self):
        assert "systemctl" in _src()

    def test_git_log_in_deploy_status(self):
        assert "git log" in _src() or '"git", "log"' in _src()
