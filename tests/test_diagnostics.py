"""Tests for operational diagnostics (Sprint 12)."""

from __future__ import annotations

import time

import pytest

from igris.core.diagnostics import (
    DiagnosticFinding,
    DiagnosticReport,
    check_blocked_accumulation,
    check_family_failure_health,
    check_observation_loop,
    check_recovery_escalation,
    check_task_starvation,
    get_diagnostic_summary,
    run_diagnostics,
)


def _ts(offset_seconds: int = 0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_seconds))


class TestTaskStarvation:
    def test_no_pending(self):
        tasks = [{"id": 1, "status": "completed", "created_at": _ts(-600)}]
        findings = check_task_starvation(tasks)
        assert findings == []

    def test_fresh_pending_ok(self):
        tasks = [{"id": 1, "status": "pending", "created_at": _ts(-10)}]
        findings = check_task_starvation(tasks)
        assert findings == []

    def test_stale_pending_warning(self):
        tasks = [{"id": 1, "status": "pending", "created_at": _ts(-600)}]
        findings = check_task_starvation(tasks)
        assert len(findings) == 1
        assert findings[0].category == "starvation"
        assert findings[0].severity == "warning"

    def test_many_stale_critical(self):
        tasks = [{"id": i, "status": "pending", "created_at": _ts(-600)} for i in range(6)]
        findings = check_task_starvation(tasks)
        assert any(f.severity == "critical" for f in findings)

    def test_large_backlog_warning(self):
        tasks = [{"id": i, "status": "pending", "created_at": _ts(-10)} for i in range(15)]
        findings = check_task_starvation(tasks)
        assert any("backlog" in f.detail.lower() for f in findings)


class TestObservationLoop:
    def test_no_loop(self):
        events = [{"type": "task", "detail": f"task {i}", "title": f"t{i}"} for i in range(10)]
        findings = check_observation_loop(events)
        assert findings == []

    def test_loop_detected(self):
        events = [{"type": "task", "detail": "run test suite", "title": "test"} for _ in range(12)]
        findings = check_observation_loop(events)
        assert len(findings) >= 1
        assert findings[0].category == "observation_loop"

    def test_few_events_no_check(self):
        events = [{"type": "task", "detail": "test", "title": "test"}]
        findings = check_observation_loop(events)
        assert findings == []


class TestBlockedAccumulation:
    def test_no_blocked(self):
        tasks = [{"id": 1, "status": "pending"}]
        findings = check_blocked_accumulation(tasks)
        assert findings == []

    def test_few_blocked_ok(self):
        tasks = [{"id": i, "status": "blocked", "blocked_reason": "dep"} for i in range(2)]
        findings = check_blocked_accumulation(tasks)
        assert findings == []

    def test_blocked_warning(self):
        tasks = [{"id": i, "status": "blocked", "blocked_reason": "dep"} for i in range(3)]
        findings = check_blocked_accumulation(tasks)
        assert len(findings) == 1
        assert findings[0].category == "blocked_accumulation"

    def test_many_blocked_critical(self):
        tasks = [{"id": i, "status": "blocked", "blocked_reason": "fail"} for i in range(6)]
        findings = check_blocked_accumulation(tasks)
        assert any(f.severity == "critical" for f in findings)


class TestFamilyFailureHealth:
    def test_no_failures(self, tmp_path):
        from igris.core import decision_memory
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        findings = check_family_failure_health(project_root=str(tmp_path))
        assert findings == []

    def test_high_failure_rate(self, tmp_path):
        from igris.core import decision_memory
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for i in range(4):
            decision_memory.record_failure(
                title=f"fail {i}", family="test", task_id=str(i),
                reason=f"err {i}", project_root=pr,
            )
        findings = check_family_failure_health(project_root=pr)
        assert any(f.category == "family_failure" for f in findings)


class TestRecoveryEscalation:
    def test_no_escalation(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        findings = check_recovery_escalation(project_root=str(tmp_path))
        assert findings == []

    def test_many_failures_escalation(self, tmp_path):
        from igris.core import decision_memory
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for i in range(12):
            decision_memory.record_failure(
                title=f"fail {i}", family="deploy", task_id=str(i),
                reason=f"err {i}", project_root=pr,
            )
        findings = check_recovery_escalation(project_root=pr)
        assert any(f.category == "recovery_escalation" for f in findings)


class TestFullDiagnostics:
    def test_healthy_system(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [{"id": 1, "status": "completed", "created_at": _ts()}]
        events = [{"type": "info", "detail": "started"}]
        report = run_diagnostics(tasks, events, project_root=str(tmp_path))
        assert report.summary["healthy"] is True
        assert report.summary["completed"] == 1

    def test_unhealthy_system(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [
            {"id": i, "status": "blocked", "blocked_reason": "fail", "created_at": _ts(-600)}
            for i in range(5)
        ]
        events = []
        report = run_diagnostics(tasks, events, project_root=str(tmp_path))
        assert report.summary["healthy"] is False
        assert len(report.findings) >= 1

    def test_summary(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [{"id": 1, "status": "pending", "created_at": _ts()}]
        summary = get_diagnostic_summary(tasks, [], project_root=str(tmp_path))
        assert "healthy" in summary
        assert "task_stats" in summary
        assert summary["task_stats"]["pending"] == 1


class TestDiagnosticModels:
    def test_finding_to_dict(self):
        f = DiagnosticFinding(
            category="starvation", severity="warning",
            title="test", detail="detail with sk-abcdefghijklmnopqrstuvwxyz",
        )
        d = f.to_dict()
        assert d["category"] == "starvation"
        assert "sk-" not in d["detail"]

    def test_report_to_dict(self):
        r = DiagnosticReport(findings=[
            DiagnosticFinding(category="test", severity="info", title="t", detail="d"),
        ])
        d = r.to_dict()
        assert d["finding_count"] == 1
        assert d["has_critical"] is False
