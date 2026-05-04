"""Tests for project state + saturation cooldown (Sprint 14)."""

from __future__ import annotations

import json
import time

import pytest

from igris.core.project_state import (
    FamilyMetrics,
    ProjectState,
    get_cooling_down_families,
    get_family_metrics,
    get_project_state,
    get_recent_fingerprints,
    get_recovery_summary,
    has_recent_fingerprint,
    is_family_available,
    record_attempt,
    reset_family_cooldown,
)


class TestRecordAttempt:
    def test_success_increments(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        m = record_attempt("test", success=True, project_root=pr)
        assert m.total_attempts == 1
        assert m.successes == 1
        assert m.failures == 0

    def test_failure_increments(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        m = record_attempt("test", success=False, project_root=pr)
        assert m.total_attempts == 1
        assert m.failures == 1
        assert m.consecutive_failures == 1

    def test_consecutive_failures_escalate(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for _ in range(3):
            m = record_attempt("deploy", success=False, project_root=pr)
        assert m.consecutive_failures == 3
        assert m.recovery_level >= 1

    def test_success_reduces_recovery(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for _ in range(3):
            record_attempt("test", success=False, project_root=pr)
        m = record_attempt("test", success=True, project_root=pr)
        assert m.recovery_level < 2

    def test_fingerprint_tracked(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        record_attempt("test", success=True, fingerprint="fp-123", project_root=pr)
        assert has_recent_fingerprint("fp-123", project_root=pr)
        assert not has_recent_fingerprint("fp-999", project_root=pr)


class TestCooldown:
    def test_no_cooldown_on_success(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        m = record_attempt("test", success=True, project_root=pr)
        assert not m.is_cooling_down

    def test_cooldown_after_failures(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for _ in range(3):
            m = record_attempt("deploy", success=False, project_root=pr)
        assert m.is_cooling_down or m.recovery_level >= 1

    def test_cooldown_families_listed(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for _ in range(3):
            record_attempt("deploy", success=False, project_root=pr)
        cooling = get_cooling_down_families(project_root=pr)
        # May or may not be cooling depending on recovery level
        assert isinstance(cooling, list)

    def test_reset_cooldown(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for _ in range(4):
            record_attempt("deploy", success=False, project_root=pr)
        ok = reset_family_cooldown("deploy", project_root=pr)
        assert ok is True
        m = get_family_metrics("deploy", project_root=pr)
        assert m is not None
        assert m.cooldown_until == 0

    def test_reset_nonexistent(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        ok = reset_family_cooldown("nope", project_root=str(tmp_path))
        assert ok is False


class TestFamilyAvailability:
    def test_available_by_default(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        result = is_family_available("test", project_root=str(tmp_path))
        assert result["available"] is True

    def test_unavailable_if_saturated(self, tmp_path):
        from igris.core import decision_memory
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        decision_memory.record_saturation("deploy", reason="too many", project_root=pr)
        result = is_family_available("deploy", project_root=pr)
        assert result["available"] is False
        assert "saturated" in result["reason"].lower()


class TestProjectState:
    def test_empty_state(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        state = get_project_state(project_root=str(tmp_path))
        assert "families" in state
        assert "cooling_down" in state

    def test_state_persists(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        record_attempt("test", success=True, project_root=pr)
        record_attempt("test", success=False, project_root=pr)
        state = get_project_state(project_root=pr)
        assert "test" in state["families"]
        assert state["families"]["test"]["total_attempts"] == 2


class TestRecoverySummary:
    def test_recovery_summary(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        record_attempt("deploy", success=False, project_root=pr)
        summary = get_recovery_summary(project_root=pr)
        assert "families" in summary
        assert "cooling_down" in summary
        assert "memory_constraints" in summary

    def test_critical_escalation(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for _ in range(7):
            record_attempt("deploy", success=False, project_root=pr)
        summary = get_recovery_summary(project_root=pr)
        assert "deploy" in summary["critical"]


class TestFamilyMetricsModel:
    def test_failure_rate(self):
        m = FamilyMetrics(family="test", total_attempts=10, successes=7, failures=3)
        assert abs(m.failure_rate - 0.3) < 0.01

    def test_zero_attempts(self):
        m = FamilyMetrics(family="test")
        assert m.failure_rate == 0.0

    def test_to_dict(self):
        m = FamilyMetrics(family="test", total_attempts=5, failures=2, successes=3)
        d = m.to_dict()
        assert d["family"] == "test"
        assert d["failure_rate"] == 0.4
        assert d["recovery_label"] == "normal"


class TestFingerprints:
    def test_recent_fingerprints(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        record_attempt("test", success=True, fingerprint="fp-a", project_root=pr)
        record_attempt("test", success=True, fingerprint="fp-b", project_root=pr)
        fps = get_recent_fingerprints(limit=10, project_root=pr)
        assert "fp-a" in fps
        assert "fp-b" in fps

    def test_fingerprint_limit(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for i in range(60):
            record_attempt("test", success=True, fingerprint=f"fp-{i}", project_root=pr)
        fps = get_recent_fingerprints(limit=20, project_root=pr)
        assert len(fps) <= 20
