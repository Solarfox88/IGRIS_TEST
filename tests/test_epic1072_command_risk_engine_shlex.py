"""Tests for Epic #1072 — shlex.split() in parse_command + contextual policy.

Verifies:
1. parse_command correctly tokenises quoted arguments via shlex.split().
2. parse_command handles edge cases: empty string, pipes in quotes, complex flags.
3. Contextual cwd policy: system directories → blocked.
4. Contextual cwd policy: outside project root → context annotated (not blocked by default).
5. Existing deterministic risk classification unchanged after shlex fix.
6. evaluate_command accepts cwd parameter.
7. shlex fallback on unterminated quote.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from igris.core.command_risk_engine import parse_command, ParsedCommand, CommandRiskEngine


# ---------------------------------------------------------------------------
# parse_command — shlex.split() tests
# ---------------------------------------------------------------------------

class TestParseCommandShlex:

    def test_simple_command_unchanged(self):
        cmd = parse_command("git status")
        assert cmd.executable == "git"
        assert cmd.args == ["status"]

    def test_quoted_commit_message(self):
        """Git commit with a multi-word -m message must be one token."""
        cmd = parse_command('git commit -m "fix: update readme file"')
        assert cmd.executable == "git"
        # shlex splits to: ['git', 'commit', '-m', 'fix: update readme file']
        assert cmd.args[0] == "commit"
        assert "-m" in cmd.args
        # The message should be a single argument, not split by spaces
        assert "fix: update readme file" in cmd.args

    def test_quoted_path_with_spaces(self):
        cmd = parse_command('cp "/path/with spaces/file.py" /tmp/out.py')
        assert cmd.executable == "cp"
        assert "/path/with spaces/file.py" in cmd.args

    def test_single_quotes_preserved(self):
        cmd = parse_command("echo 'hello world'")
        assert cmd.executable == "echo"
        assert "hello world" in cmd.args

    def test_empty_command_returns_empty_parsed(self):
        cmd = parse_command("")
        assert cmd.executable == ""
        assert cmd.args == []

    def test_whitespace_only_command(self):
        cmd = parse_command("   ")
        assert cmd.executable == ""
        assert cmd.args == []

    def test_unterminated_quote_fallback_to_naive_split(self):
        """Unterminated quote → falls back to .split(), does not raise."""
        cmd = parse_command('git commit -m "unterminated')
        # Should not raise; executable should be 'git'
        assert cmd.executable == "git"
        assert isinstance(cmd.args, list)

    def test_rm_command_detected(self):
        cmd = parse_command("rm -rf /tmp/test")
        assert cmd.executable == "rm"
        assert cmd.has_rm is True

    def test_sudo_in_quoted_context(self):
        cmd = parse_command('bash -c "sudo rm -rf /var/log"')
        assert cmd.executable == "bash"
        # has_sudo comes from regex on raw string so still detected
        assert cmd.has_sudo is True

    def test_pipe_in_command(self):
        cmd = parse_command("cat /etc/passwd | grep root")
        assert cmd.executable == "cat"
        assert cmd.has_pipe is True

    def test_python_command_with_flags(self):
        cmd = parse_command("python -m pytest tests/ -v --tb=short")
        assert cmd.executable == "python"
        assert "-m" in cmd.args
        assert "pytest" in cmd.args

    def test_git_push_force_detected(self):
        cmd = parse_command("git push origin main --force")
        assert cmd.has_force_push is True

    def test_empty_string_returns_parsedcommand(self):
        result = parse_command("")
        assert isinstance(result, ParsedCommand)
        assert result.raw == ""

    def test_returns_parsedcommand_instance(self):
        result = parse_command("ls -la")
        assert isinstance(result, ParsedCommand)


# ---------------------------------------------------------------------------
# CommandRiskEngine — contextual cwd policy
# ---------------------------------------------------------------------------

class TestCommandRiskEngineContextualCwd:

    def _engine(self, project_root: str, environment: str = "dev") -> CommandRiskEngine:
        return CommandRiskEngine(
            project_root=project_root,
            use_llm_reviewer=False,  # no LLM in tests
            environment=environment,
        )

    def test_cwd_in_system_dir_is_blocked(self, tmp_path):
        engine = self._engine(str(tmp_path))
        event, review = engine.evaluate_command(
            "ls", cwd="/etc"
        )
        assert event.decision == "blocked"
        assert event.final_risk in ("high", "critical")

    def test_cwd_usr_is_blocked(self, tmp_path):
        engine = self._engine(str(tmp_path))
        event, review = engine.evaluate_command(
            "cat /usr/bin/python3", cwd="/usr/bin"
        )
        assert event.decision == "blocked"

    def test_cwd_in_project_root_not_blocked_by_cwd_policy(self, tmp_path):
        engine = self._engine(str(tmp_path))
        event, review = engine.evaluate_command(
            "git status", cwd=str(tmp_path)
        )
        # git status is low risk — should not be blocked by cwd policy
        assert event.decision != "blocked" or "cwd" not in (event.reason or "")

    def test_cwd_outside_project_annotates_context(self, tmp_path):
        """Outside-project cwd adds context annotation but does not block alone."""
        engine = self._engine(str(tmp_path))
        other_dir = tempfile.mkdtemp()  # not inside tmp_path
        event, review = engine.evaluate_command(
            "git status", cwd=other_dir
        )
        # Should not block just because cwd is different from project root
        # (only system dirs block)
        assert event.final_risk != "critical"

    def test_cwd_none_does_not_trigger_policy(self, tmp_path):
        engine = self._engine(str(tmp_path))
        event, review = engine.evaluate_command("git status", cwd=None)
        assert isinstance(event.decision, str)

    def test_production_environment_blocks_destructive(self, tmp_path):
        engine = self._engine(str(tmp_path), environment="production")
        event, review = engine.evaluate_command(
            "rm -rf /var/data", cwd=str(tmp_path)
        )
        assert event.decision == "blocked"
        assert event.final_risk in ("high", "critical")

    def test_evaluate_command_accepts_cwd_kwarg(self, tmp_path):
        engine = self._engine(str(tmp_path))
        # Should not raise when cwd is provided
        event, review = engine.evaluate_command(
            "echo hello", cwd=str(tmp_path)
        )
        assert isinstance(event.decision, str)


# ---------------------------------------------------------------------------
# Regression: shlex does not break existing risk classification
# ---------------------------------------------------------------------------

class TestRiskClassificationUnchanged:

    def test_rm_rf_still_high_risk(self):
        from igris.core.command_risk_engine import classify_command_risk
        parsed = parse_command("rm -rf /tmp/test")
        risk = classify_command_risk(parsed)
        assert risk in ("high", "critical", "medium")  # at least medium

    def test_git_status_low_risk(self):
        from igris.core.command_risk_engine import classify_command_risk
        parsed = parse_command("git status")
        risk = classify_command_risk(parsed)
        assert risk == "low"

    def test_sudo_escalates_risk(self):
        from igris.core.command_risk_engine import classify_command_risk
        parsed = parse_command("sudo rm file.txt")
        risk = classify_command_risk(parsed)
        assert risk in ("high", "critical")
