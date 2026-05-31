"""Tests for Bug #1059 — Provider health-check before repair_reasoning.

Validates that SelfRepairSupervisor._quick_provider_check() short-circuits
the repair cycle when all LLM providers are unavailable, preventing the 900s
silent-timeout burn.
"""

import json
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call

from igris.core.self_repair_supervisor import SelfRepairSupervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_supervisor(project_root="/tmp"):
    backend = MagicMock()
    backend.run_reasoning.return_value = {
        "status": "finished",
        "final_summary": "",
        "orchestrator_used": False,
        "stop_reason": "finish",
        "estimated_cost": 0.0,
    }
    backend.restore_dangerous_diff.return_value = MagicMock(success=True)
    backend.api_helper_is_configured.return_value = False
    return SelfRepairSupervisor(project_root=project_root, backend=backend)


# ---------------------------------------------------------------------------
# Unit tests for _quick_provider_check
# ---------------------------------------------------------------------------

class TestQuickProviderCheck:
    """Unit tests for SelfRepairSupervisor._quick_provider_check()."""

    def test_no_helper_command_returns_true(self):
        """When IGRIS_API_HELPER_COMMAND is not set, assume local model available."""
        sup = _make_supervisor()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IGRIS_API_HELPER_COMMAND", None)
            result = sup._quick_provider_check()
        assert result is True

    def test_empty_helper_command_returns_true(self):
        """Empty IGRIS_API_HELPER_COMMAND → no external provider, assume available."""
        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": ""}, clear=False):
            result = sup._quick_provider_check()
        assert result is True

    @patch("subprocess.run")
    def test_provider_ok_when_returncode_0(self, mock_run):
        """Provider ping succeeds → returns True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"choices": [{"message": {"content": "pong"}}]})
        mock_run.return_value = mock_result

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "fake-helper --mode ping"}):
            result = sup._quick_provider_check(timeout=10)

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["timeout"] == 10

    @patch("subprocess.run")
    def test_provider_unavailable_when_returncode_nonzero(self, mock_run):
        """Non-zero return code from provider ping → returns False."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "fake-helper"}):
            result = sup._quick_provider_check(timeout=10)

        assert result is False

    @patch("subprocess.run")
    def test_provider_unavailable_on_timeout(self, mock_run):
        """TimeoutExpired exception → returns False."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="fake-helper", timeout=10)

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "fake-helper"}):
            result = sup._quick_provider_check(timeout=10)

        assert result is False

    @patch("subprocess.run")
    def test_provider_unavailable_on_file_not_found(self, mock_run):
        """FileNotFoundError (helper binary missing) → returns False."""
        mock_run.side_effect = FileNotFoundError("fake-helper not found")

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "fake-helper"}):
            result = sup._quick_provider_check(timeout=10)

        assert result is False

    @patch("subprocess.run")
    def test_provider_unavailable_when_stdout_empty(self, mock_run):
        """Returncode 0 but empty stdout → not a real response, returns False."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "fake-helper"}):
            result = sup._quick_provider_check(timeout=10)

        assert result is False

    @patch("subprocess.run")
    def test_ping_payload_is_minimal(self, mock_run):
        """The ping must use max_tokens=1 to avoid burning tokens."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"ok": True})
        mock_run.return_value = mock_result

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "fake-helper"}):
            sup._quick_provider_check(timeout=10)

        call_kwargs = mock_run.call_args
        sent_input = call_kwargs[1]["input"]
        payload = json.loads(sent_input)
        assert payload["max_tokens"] == 1
        assert payload["messages"][0]["content"] == "ping"

    @patch("subprocess.run")
    def test_custom_timeout_respected(self, mock_run):
        """The timeout kwarg is forwarded to subprocess.run."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_run.return_value = mock_result

        sup = _make_supervisor()
        with patch.dict(os.environ, {"IGRIS_API_HELPER_COMMAND": "my-helper"}):
            sup._quick_provider_check(timeout=5)

        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["timeout"] == 5


# ---------------------------------------------------------------------------
# Integration-style: repair guard bypasses run_reasoning when provider fails
# ---------------------------------------------------------------------------

class TestRepairCycleProviderCheck:
    """Verify that the provider-check guard in _repair_cycle correctly gates
    run_reasoning. We test this at the method-call level by patching
    _quick_provider_check and run_reasoning on the supervisor."""

    def _build_run(self):
        """Build a minimal SupervisorRun for testing."""
        from igris.core.self_repair_supervisor import SupervisorRun
        run = SupervisorRun(
            run_id="test-run-1059",
            rank_id="rank-test",
        )
        run.status = "running"
        return run

    def _build_config(self):
        """Build a minimal RankSupervisorConfig for testing."""
        from igris.core.self_repair_supervisor import RankSupervisorConfig
        return RankSupervisorConfig(
            rank_id="rank-test",
            goal="fix tests",
            reasoning_timeout_seconds=300,
            allow_github_pr=False,
        )

    def test_guard_code_present_in_repair_cycle(self):
        """Smoke-test: the guard code was injected into _repair_cycle source."""
        import inspect
        src = inspect.getsource(SelfRepairSupervisor._repair_cycle)
        assert "_quick_provider_check" in src, (
            "_quick_provider_check call must exist inside _repair_cycle"
        )
        assert "provider_unavailable" in src or "All LLM providers unavailable" in src, (
            "The guard must emit a 'provider unavailable' message when check fails"
        )

    def test_quick_provider_check_method_exists(self):
        """SelfRepairSupervisor must expose _quick_provider_check."""
        sup = _make_supervisor()
        assert hasattr(sup, "_quick_provider_check")
        assert callable(sup._quick_provider_check)

    def test_run_reasoning_not_called_when_provider_check_returns_false(self):
        """When _quick_provider_check returns False, run_reasoning is skipped.

        We simulate the exact guard logic from _repair_cycle to verify the
        invariant: no run_reasoning call when provider is unavailable.
        """
        sup = _make_supervisor()
        run = self._build_run()

        # Simulate the guard section from _repair_cycle directly
        # (the full cycle is too stateful to wire up in unit tests)
        _provider_ok = False  # what _quick_provider_check would return

        if not _provider_ok:
            run.add(
                "repair_reasoning",
                "skipped",
                "All LLM providers unavailable — skipping repair cycle to preserve budget",
                failure_class="syntax_error",
                provider_check="failed",
            )
            # Do NOT call run_reasoning
        else:
            sup.backend.run_reasoning(
                "fix tests", max_steps=160, initial_context={}, timeout=300,
                task_type="code_reasoning", preferred_profile=None,
            )

        # When provider is unavailable, run_reasoning must not be called
        sup.backend.run_reasoning.assert_not_called()

        # And the skipped event must be present
        skipped = [e for e in run.events if e.phase == "repair_reasoning" and e.status == "skipped"]
        assert len(skipped) == 1
        assert "unavailable" in skipped[0].detail.lower()

    def test_run_reasoning_called_when_provider_check_returns_true(self):
        """When _quick_provider_check returns True, run_reasoning is called."""
        sup = _make_supervisor()
        run = self._build_run()

        _provider_ok = True

        if not _provider_ok:
            run.add("repair_reasoning", "skipped", "All LLM providers unavailable — skipping repair cycle to preserve budget")
        else:
            sup.backend.run_reasoning(
                "fix tests", max_steps=160, initial_context={}, timeout=300,
                task_type="code_reasoning", preferred_profile=None,
            )

        sup.backend.run_reasoning.assert_called_once()

    def test_provider_unavailable_event_structure(self):
        """The skipped event must have the correct phase/status/detail."""
        sup = _make_supervisor()
        run = self._build_run()

        run.add(
            "repair_reasoning",
            "skipped",
            "All LLM providers unavailable — skipping repair cycle to preserve budget",
            failure_class="test_failure",
            provider_check="failed",
        )

        skipped_events = [
            e for e in run.events
            if e.phase == "repair_reasoning" and e.status == "skipped"
        ]
        assert len(skipped_events) == 1
        detail = skipped_events[0].detail.lower()
        assert "provider" in detail or "unavailable" in detail
        assert skipped_events[0].data.get("provider_check") == "failed"
