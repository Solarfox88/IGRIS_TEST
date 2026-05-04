"""Tests for explainable task selection (Sprint 13)."""

from __future__ import annotations

import pytest

from igris.core.task_selection_explain import (
    CandidateExplanation,
    SelectionExplanation,
    explain_task_selection,
)
from igris.models.task import Task, TaskStatus


def _task(id: int, desc: str, family: str = "other", priority: int = 0,
          risk: str = "low", status: str = "pending") -> Task:
    return Task(
        id=id, description=desc, family=family, priority=priority,
        risk=risk, status=TaskStatus(status),
    )


class TestExplainSelection:
    def test_selects_highest_priority(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [
            _task(1, "low priority task", priority=0),
            _task(2, "high priority task", priority=5),
        ]
        exp = explain_task_selection(tasks, project_root=str(tmp_path))
        assert exp.selected is not None
        assert exp.selected["id"] == 2
        assert exp.selection_source in ("advisory", "fallback")

    def test_no_pending_tasks(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [_task(1, "done", status="completed")]
        exp = explain_task_selection(tasks, project_root=str(tmp_path))
        assert exp.selected is None
        assert "No task selected" in exp.summary

    def test_candidates_scored(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [
            _task(1, "safe task", risk="low"),
            _task(2, "risky task", risk="high"),
        ]
        exp = explain_task_selection(tasks, project_root=str(tmp_path))
        assert len(exp.candidates) == 2
        for c in exp.candidates:
            assert "score" in c.to_dict()

    def test_high_risk_penalized(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [
            _task(1, "safe task", risk="low", priority=0),
            _task(2, "risky task", risk="high", priority=0),
        ]
        exp = explain_task_selection(tasks, project_root=str(tmp_path))
        scores = {c.task_id: c.score for c in exp.candidates}
        assert scores[1] > scores[2]

    def test_blocked_family_penalized(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [
            _task(1, "deploy something", family="deploy"),
            _task(2, "safe task", family="doc"),
        ]
        exp = explain_task_selection(
            tasks, blocked_families=["deploy"], project_root=str(tmp_path),
        )
        scores = {c.task_id: c.score for c in exp.candidates}
        assert scores[2] > scores[1]

    def test_summary_includes_counts(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        tasks = [_task(1, "do something")]
        exp = explain_task_selection(tasks, project_root=str(tmp_path))
        assert "candidates evaluated" in exp.summary

    def test_failure_history_penalty(self, tmp_path):
        from igris.core import decision_memory
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        pr = str(tmp_path)
        for i in range(4):
            decision_memory.record_failure(
                title=f"fail {i}", family="test", task_id=str(i),
                reason=f"err {i}", project_root=pr,
            )
        tasks = [
            _task(1, "run tests", family="test"),
            _task(2, "write docs", family="doc"),
        ]
        exp = explain_task_selection(tasks, project_root=pr)
        scores = {c.task_id: c.score for c in exp.candidates}
        assert scores[2] > scores[1]


class TestModels:
    def test_candidate_to_dict(self):
        c = CandidateExplanation(
            task_id=1, title="test with sk-abcdefghijklmnopqrstuvwxyz",
            family="test", priority=0, risk="low", status="pending",
        )
        d = c.to_dict()
        assert "sk-" not in d["title"]

    def test_explanation_to_dict(self):
        e = SelectionExplanation(summary="test summary")
        d = e.to_dict()
        assert d["summary"] == "test summary"
        assert d["candidates"] == []
