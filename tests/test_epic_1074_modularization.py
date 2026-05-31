"""Tests for Epic #1074 — SelfRepairSupervisor modularization.

Validates RunPhase enum, extracted constants, and state machine structure.
"""

import pytest
from igris.core.self_repair_supervisor import (
    RunPhase,
    DEFAULT_REPAIR_TIMEOUT_SECONDS,
    DEFAULT_BASELINE_TIMEOUT_SECONDS,
    DEFAULT_SMOKE_TIMEOUT_SECONDS,
    DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER_PING_TIMEOUT_SECONDS,
    DEFAULT_MAX_REPAIR_CYCLES,
    SUPERVISOR_BRANCH_PREFIX,
    NO_DIFF_SIGNAL_THRESHOLD,
    REPAIRABLE_FAILURES,
    FAILURE_ERROR_CODES,
    CAPABILITY_LIMIT_SIGNALS,
    CAPABILITY_LIMIT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# RunPhase enum
# ---------------------------------------------------------------------------

class TestRunPhase:
    """RunPhase provides named constants for all supervisor states."""

    def test_pre_run_phases_exist(self):
        assert RunPhase.CREATED == "created"
        assert RunPhase.PREFLIGHT == "preflight"
        assert RunPhase.BASELINE_TESTS == "baseline_tests"

    def test_reasoning_phases_exist(self):
        assert RunPhase.PLANNING == "planning"
        assert RunPhase.REASONING == "reasoning"

    def test_validation_phases_exist(self):
        assert RunPhase.DIFF_REVIEW == "diff_review"
        assert RunPhase.TARGETED_TESTS == "targeted_tests"
        assert RunPhase.FULL_TESTS == "full_tests"
        assert RunPhase.SMOKE == "smoke"
        assert RunPhase.SEMANTIC_GATE == "semantic_gate"

    def test_repair_phases_exist(self):
        assert RunPhase.REPAIR == "repair"
        assert RunPhase.REPAIR_REASONING == "repair_reasoning"
        assert RunPhase.API_ESCALATION == "api_escalation"

    def test_decomposition_phases_exist(self):
        assert RunPhase.DECOMPOSITION_REQUEST == "decomposition_request"
        assert RunPhase.SUBISSUE_CREATION == "subissue_creation"
        assert RunPhase.SUBMISSION_AUTORUN == "submission_autorun"

    def test_delivery_phases_exist(self):
        assert RunPhase.DELIVERY == "delivery"
        assert RunPhase.PR_CREATION == "pr_creation"
        assert RunPhase.MERGE == "merge"

    def test_terminal_phases_exist(self):
        assert RunPhase.COMPLETED == "completed"
        assert RunPhase.BLOCKED == "blocked"
        assert RunPhase.FAILED == "failed"
        assert RunPhase.CANCELLED == "cancelled"
        assert RunPhase.INTERRUPTED == "interrupted"

    def test_meta_phases_exist(self):
        assert RunPhase.WATCHDOG == "watchdog"
        assert RunPhase.BUDGET == "execution_budget"

    def test_all_phases_are_strings(self):
        """Every RunPhase attribute is a non-empty string."""
        for attr in dir(RunPhase):
            if attr.startswith("_"):
                continue
            val = getattr(RunPhase, attr)
            if callable(val):
                continue
            assert isinstance(val, str), f"RunPhase.{attr} should be a string, got {type(val)}"
            assert val, f"RunPhase.{attr} should be non-empty"

    def test_all_phase_values_are_unique(self):
        """No two RunPhase constants share the same string value."""
        values = []
        for attr in dir(RunPhase):
            if attr.startswith("_"):
                continue
            val = getattr(RunPhase, attr)
            if callable(val):
                continue
            if isinstance(val, str):
                values.append(val)
        assert len(values) == len(set(values)), f"Duplicate RunPhase values: {[v for v in values if values.count(v) > 1]}"

    def test_phases_are_snake_case(self):
        """Phase values use snake_case for consistent event filtering."""
        for attr in dir(RunPhase):
            if attr.startswith("_"):
                continue
            val = getattr(RunPhase, attr)
            if not isinstance(val, str) or callable(val):
                continue
            # All chars should be lowercase letters, digits, or underscores
            assert val == val.lower(), f"RunPhase.{attr}={val!r} should be lowercase"
            assert " " not in val, f"RunPhase.{attr}={val!r} should not contain spaces"


# ---------------------------------------------------------------------------
# Extracted constants
# ---------------------------------------------------------------------------

class TestExtractedConstants:
    """Verify that extracted constants have sensible default values."""

    def test_repair_timeout_positive(self):
        assert DEFAULT_REPAIR_TIMEOUT_SECONDS > 0

    def test_baseline_timeout_positive(self):
        assert DEFAULT_BASELINE_TIMEOUT_SECONDS > 0

    def test_smoke_timeout_positive(self):
        assert DEFAULT_SMOKE_TIMEOUT_SECONDS > 0

    def test_preflight_timeout_positive(self):
        assert DEFAULT_PREFLIGHT_TIMEOUT_SECONDS > 0

    def test_provider_ping_timeout_small(self):
        """Provider ping must be fast — well under a full reasoning timeout."""
        assert DEFAULT_PROVIDER_PING_TIMEOUT_SECONDS <= 30

    def test_max_repair_cycles_at_least_1(self):
        assert DEFAULT_MAX_REPAIR_CYCLES >= 1

    def test_branch_prefix_format(self):
        """Branch prefix must be a non-empty string suitable for git branch names."""
        assert isinstance(SUPERVISOR_BRANCH_PREFIX, str)
        assert SUPERVISOR_BRANCH_PREFIX
        assert " " not in SUPERVISOR_BRANCH_PREFIX

    def test_no_diff_signal_threshold_positive(self):
        assert NO_DIFF_SIGNAL_THRESHOLD >= 1


# ---------------------------------------------------------------------------
# Pre-existing constants still intact
# ---------------------------------------------------------------------------

class TestExistingConstantsIntact:
    """RunPhase extraction must not break pre-existing module constants."""

    def test_repairable_failures_unchanged(self):
        assert "pytest_failure" in REPAIRABLE_FAILURES
        assert "syntax_error" in REPAIRABLE_FAILURES
        assert "reasoning_loop_blocked" in REPAIRABLE_FAILURES

    def test_failure_error_codes_complete(self):
        assert FAILURE_ERROR_CODES.get("pytest_failure") == "E001"
        assert FAILURE_ERROR_CODES.get("syntax_error") == "E003"

    def test_capability_limit_signals(self):
        assert "reasoning_timeout" in CAPABILITY_LIMIT_SIGNALS
        assert "no_diff_repair" in CAPABILITY_LIMIT_SIGNALS

    def test_capability_limit_threshold(self):
        assert CAPABILITY_LIMIT_THRESHOLD >= 2
