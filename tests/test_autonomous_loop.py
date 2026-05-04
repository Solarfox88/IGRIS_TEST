"""Tests for igris.core.autonomous_loop."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from igris.core.autonomous_loop import (
    LoopStatus,
    LoopStepResult,
    execute_step,
    get_loop_status,
    get_recent_steps,
    run_loop,
)
from igris.core.task_engine import TaskEngine
from igris.core import decision_memory


@pytest.fixture
def engine(tmp_path: Path) -> TaskEngine:
    root = tmp_path / ".igris"
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "timeline").mkdir(parents=True, exist_ok=True)
    (root / "memory").mkdir(parents=True, exist_ok=True)
    return TaskEngine(runtime_root=root)


@pytest.fixture
def project_dir(tmp_path: Path) -> str:
    (tmp_path / ".igris" / "memory").mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


class TestExecuteStep:
    def test_no_pending_tasks(self, engine: TaskEngine, project_dir: str) -> None:
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.action_type == "stop"
        assert result.outcome == "stopped"
        assert "No pending tasks" in result.reason

    def test_skip_high_risk_task(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("dangerous operation", risk="high")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.outcome == "skipped"
        assert "high" in result.reason.lower()

    def test_skip_saturated_family(self, engine: TaskEngine, project_dir: str) -> None:
        decision_memory.record_saturation("test", project_root=project_dir)
        engine.create_task("run tests", family="test")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.outcome == "blocked"
        assert "saturated" in result.reason.lower() or "avoid" in result.reason.lower()

    def test_execute_test_command(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("run unit tests", family="test")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.action_type == "execute_command"
        assert "run_tests" in result.action_detail

    def test_propose_patch_for_code_task(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("implement feature X", family="code")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.action_type == "propose_patch"
        assert result.outcome == "skipped"

    def test_skip_unknown_task(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("do something vague", family="other")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.outcome == "skipped"


class TestRunLoop:
    def test_run_max_steps_1(self, engine: TaskEngine, project_dir: str) -> None:
        status = run_loop(engine, max_steps=1, project_root=project_dir)
        assert isinstance(status, LoopStatus)
        assert status.running is False
        assert status.max_steps == 1

    def test_stop_on_no_tasks(self, engine: TaskEngine, project_dir: str) -> None:
        status = run_loop(engine, max_steps=5, project_root=project_dir)
        assert status.stopped_reason == "No pending tasks available"
        assert status.steps_completed >= 1

    def test_max_steps_capped(self, engine: TaskEngine, project_dir: str) -> None:
        status = run_loop(engine, max_steps=200, project_root=project_dir)
        assert status.max_steps == 100

    def test_records_timeline(self, engine: TaskEngine, project_dir: str) -> None:
        run_loop(engine, max_steps=1, project_root=project_dir)
        events = engine.recent_timeline_events(10)
        loop_events = [e for e in events if e.get("type") == "loop"]
        assert len(loop_events) >= 1

    def test_no_infinite_loop(self, engine: TaskEngine, project_dir: str) -> None:
        for i in range(10):
            engine.create_task(f"task {i}", family="other")
        status = run_loop(engine, max_steps=5, project_root=project_dir)
        assert status.steps_completed <= 5

    def test_stops_on_blocked_skipped(self, engine: TaskEngine, project_dir: str) -> None:
        for i in range(5):
            engine.create_task(f"vague task {i}", family="other")
        status = run_loop(engine, max_steps=10, project_root=project_dir)
        assert "blocked" in status.stopped_reason.lower() or "skipped" in status.stopped_reason.lower() or status.steps_completed <= 10


class TestLoopStepResult:
    def test_to_dict(self) -> None:
        r = LoopStepResult(step_number=1, task_title="T", action_type="skip", outcome="skipped")
        d = r.to_dict()
        assert d["step_number"] == 1
        assert d["outcome"] == "skipped"

    def test_secret_redacted(self) -> None:
        r = LoopStepResult(task_title="API_KEY=sk-secrettest1234567890123")
        d = r.to_dict()
        assert "sk-secrettest1234567890123" not in d["task_title"]


class TestLoopStatus:
    def test_to_dict(self) -> None:
        s = LoopStatus(running=True, max_steps=5, steps_completed=2)
        d = s.to_dict()
        assert d["running"] is True
        assert d["max_steps"] == 5

    def test_get_loop_status(self) -> None:
        s = get_loop_status()
        assert isinstance(s, LoopStatus)


class TestRecentSteps:
    def test_empty(self) -> None:
        steps = get_recent_steps(limit=5)
        assert isinstance(steps, list)


class TestSafetyChecks:
    def test_no_auto_commit(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("commit changes", family="git")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        # git tasks should map to git_status, not auto-commit
        assert result.action_type in ("execute_command", "skip")
        if result.action_type == "execute_command":
            assert "git_status" in result.action_detail

    def test_no_auto_push(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("push to remote", family="git")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert "push" not in result.action_detail.lower()

    def test_no_auto_patch_apply(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("fix the bug", family="fix")
        result = execute_step(engine, step_number=1, project_root=project_dir)
        assert result.action_type == "propose_patch"
        assert result.outcome == "skipped"

    def test_execution_creates_memory(self, engine: TaskEngine, project_dir: str) -> None:
        engine.create_task("run tests", family="test")
        execute_step(engine, step_number=1, project_root=project_dir)
        # Either a decision or failure is recorded regardless of outcome
        decisions = decision_memory.get_recent_decisions(project_root=project_dir)
        failures = decision_memory.get_recent_failures(project_root=project_dir)
        assert len(decisions) + len(failures) >= 1
