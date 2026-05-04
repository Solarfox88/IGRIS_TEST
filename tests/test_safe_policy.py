"""Tests for strict safety policy (Sprint 13)."""

from __future__ import annotations

import pytest

from igris.core.safe_policy import (
    PolicyDecision,
    check_command_policy,
    get_policy_status,
    reset_rate_limits,
)


@pytest.fixture(autouse=True)
def clean_rate_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


class TestAllowlist:
    def test_allowed_command(self):
        d = check_command_policy("git_status")
        assert d.allowed is True
        assert "allowlist" in d.checks_passed

    def test_unknown_command_blocked(self):
        d = check_command_policy("rm_rf_everything")
        assert d.allowed is False
        assert "allowlist" in d.checks_failed

    def test_all_known_commands_pass(self):
        from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
        for cmd_id in ALLOWED_COMMANDS:
            reset_rate_limits()
            d = check_command_policy(cmd_id)
            assert d.allowed is True, f"{cmd_id} should be allowed"


class TestBlockedList:
    def test_blocked_command(self):
        # These are in BLOCKED_COMMAND_IDS but not in ALLOWED_COMMANDS,
        # so they fail at allowlist first
        d = check_command_policy("git_push")
        assert d.allowed is False

    def test_shell_exec_blocked(self):
        d = check_command_policy("shell_exec")
        assert d.allowed is False

    def test_sudo_blocked(self):
        d = check_command_policy("sudo")
        assert d.allowed is False


class TestRateLimit:
    def test_within_limits(self):
        for _ in range(4):
            d = check_command_policy("git_status")
            assert d.allowed is True

    def test_burst_limit(self):
        for _ in range(5):
            check_command_policy("git_status")
        d = check_command_policy("git_status")
        assert d.allowed is False
        assert "rate_limit" in d.checks_failed
        assert "Burst limit" in d.reason


class TestContextValidation:
    def test_path_traversal_blocked(self):
        d = check_command_policy("git_status", context={"project_root": "/tmp/../etc"})
        assert d.allowed is False
        assert "context_validation" in d.checks_failed

    def test_normal_context_ok(self):
        d = check_command_policy("git_status", context={"project_root": "/home/user/project"})
        assert d.allowed is True


class TestPolicyDecision:
    def test_to_dict(self):
        d = PolicyDecision(
            allowed=True, command_id="test",
            reason="ok with sk-abcdefghijklmnopqrstuvwxyz",
            checks_passed=["a"], checks_failed=[],
        )
        dd = d.to_dict()
        assert dd["allowed"] is True
        assert "sk-" not in dd["reason"]

    def test_policy_status(self):
        s = get_policy_status()
        assert "allowed_commands" in s
        assert "blocked_commands" in s
        assert "rate_limit_max" in s
        assert "remaining_capacity" in s
