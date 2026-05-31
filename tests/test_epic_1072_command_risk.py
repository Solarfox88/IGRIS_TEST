"""Tests for Epic #1072 — Command Risk Engine improvements.

Covers: contextual policy (production vs dev), destructive pre-check,
and dry-run mode.
"""

import pytest
from igris.core.command_risk_engine import CommandRiskEngine


# ---------------------------------------------------------------------------
# Destructive pre-check
# ---------------------------------------------------------------------------

class TestDestructivePrecheck:
    """is_destructive() correctly identifies dangerous commands."""

    def test_rm_rf_is_destructive(self):
        engine = CommandRiskEngine(environment="dev")
        assert engine.is_destructive("rm -rf /tmp/test") is True

    def test_git_reset_hard_is_destructive(self):
        engine = CommandRiskEngine(environment="dev")
        assert engine.is_destructive("git reset --hard HEAD") is True

    def test_git_clean_is_destructive(self):
        engine = CommandRiskEngine(environment="dev")
        assert engine.is_destructive("git clean -fd") is True

    def test_safe_command_not_destructive(self):
        engine = CommandRiskEngine(environment="dev")
        assert engine.is_destructive("ls -la") is False
        assert engine.is_destructive("pytest tests/") is False
        assert engine.is_destructive("git status") is False

    def test_truncate_table_is_destructive(self):
        engine = CommandRiskEngine(environment="dev")
        assert engine.is_destructive("psql -c 'TRUNCATE TABLE users'") is True


# ---------------------------------------------------------------------------
# Contextual policy — production environment
# ---------------------------------------------------------------------------

class TestContextualPolicyProduction:
    """In production, destructive commands are critical; high risk is blocked."""

    def test_rm_rf_blocked_in_production(self):
        engine = CommandRiskEngine(environment="production", use_llm_reviewer=False)
        event, review = engine.evaluate_command("rm -rf /data")
        assert event.decision == "blocked"
        assert event.final_risk == "critical"

    def test_rm_rf_needs_approval_in_dev(self):
        engine = CommandRiskEngine(environment="dev", use_llm_reviewer=False)
        event, review = engine.evaluate_command("rm -rf /data")
        # In dev, destructive but not auto-blocked (follows standard policy)
        assert event.decision in ("blocked", "needs_approval")

    def test_high_risk_blocked_in_production(self):
        """Commands that evaluate to 'high' are blocked in production."""
        engine = CommandRiskEngine(environment="production", use_llm_reviewer=False)
        # Force push is a high-risk command
        event, review = engine.evaluate_command("git push --force origin main")
        assert event.decision == "blocked"

    def test_low_risk_allowed_in_production(self):
        """Low-risk commands are still allowed in production."""
        engine = CommandRiskEngine(environment="production", use_llm_reviewer=False)
        event, review = engine.evaluate_command("git status")
        assert event.decision == "allowed"
        assert event.final_risk == "low"

    def test_environment_stored_on_engine(self):
        engine = CommandRiskEngine(environment="staging")
        assert engine.environment == "staging"

    def test_staging_escalates_destructive_to_high(self):
        """Destructive commands in staging are escalated to at least 'high'."""
        engine = CommandRiskEngine(environment="staging", use_llm_reviewer=False)
        # rm -rf is destructive; in staging it should be at least high
        event, review = engine.evaluate_command("rm -rf /tmp/data")
        assert event.final_risk in ("high", "critical")


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRunMode:
    """In dry-run mode, commands are classified but decision is 'dry_run'."""

    def test_dry_run_decision_on_safe_command(self):
        engine = CommandRiskEngine(dry_run=True, use_llm_reviewer=False)
        event, review = engine.evaluate_command("ls -la")
        assert event.decision == "dry_run"

    def test_dry_run_decision_on_dangerous_command(self):
        """Even 'rm -rf' returns dry_run, not blocked."""
        engine = CommandRiskEngine(dry_run=True, use_llm_reviewer=False)
        event, review = engine.evaluate_command("rm -rf /")
        assert event.decision == "dry_run"

    def test_dry_run_still_classifies_risk(self):
        """dry_run mode still reports the risk level."""
        engine = CommandRiskEngine(dry_run=True, use_llm_reviewer=False)
        event, review = engine.evaluate_command("ls -la")
        assert event.final_risk == "low"

    def test_dry_run_no_llm_call(self):
        """In dry-run mode, the LLM reviewer is never called."""
        engine = CommandRiskEngine(dry_run=True, use_llm_reviewer=True)
        # With use_llm_reviewer=True but dry_run=True, the LLM path is skipped.
        # If LLM were called on a medium+ command, it would attempt subprocess
        # calls that would fail. But in dry_run mode, we should return early.
        # We verify by checking that evaluate_command succeeds and returns dry_run.
        event, review = engine.evaluate_command("git push --force origin main")
        # dry_run mode must skip LLM and return decision="dry_run"
        assert event.decision == "dry_run"

    def test_dry_run_flag_accessible(self):
        engine = CommandRiskEngine(dry_run=True)
        assert engine.dry_run is True

        engine2 = CommandRiskEngine(dry_run=False)
        assert engine2.dry_run is False

    def test_dry_run_event_logged(self):
        """Even in dry-run mode, events are recorded in the log."""
        engine = CommandRiskEngine(dry_run=True, use_llm_reviewer=False)
        engine.evaluate_command("ls -la")
        engine.evaluate_command("git status")
        assert len(engine.get_event_log()) == 2

    def test_dry_run_reason_contains_risk(self):
        """dry_run reason string mentions the would-be risk."""
        engine = CommandRiskEngine(dry_run=True, use_llm_reviewer=False)
        event, _ = engine.evaluate_command("git status")
        assert "low" in event.reason or "dry_run" in event.reason
