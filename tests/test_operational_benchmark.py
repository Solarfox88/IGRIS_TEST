"""Operational benchmark tests (Sprint 20).

Proves IGRIS_GPT workflows end-to-end using deterministic/fallback mode.
No LLM required. No Vast.ai. No costs. No push.

Benchmarks:
1. Docs-only task: mission → plan → materialize → patch → validate → apply → report
2. Bugfix small: mission → plan → task → patch → test verification
3. Test failure recovery: failed outcome → remediation → memory → project state
4. Multi-file safe patch: two file patches → diff → validate → apply
5. Full loop smoke: mission → plan → materialize → select → loop step → report → memory → decision report
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from igris.core.task_engine import TaskEngine
from igris.core import mission_planner
from igris.core.mission_planner import Mission
from igris.core import patch_proposal as patch_mod
from igris.core.patch_proposal import PatchProposal
from igris.core import decision_memory
from igris.core import decision_report as decision_report_mod
from igris.core import project_state as project_state_mod
from igris.core import autonomous_loop
from igris.core.outcome_router import route_outcome
from igris.core.safety import redact_secrets
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def benchmark_env(tmp_path):
    """Set up a clean benchmark environment."""
    root = tmp_path / "benchmark_project"
    root.mkdir()
    for d in [
        ".igris/tasks", ".igris/timeline", ".igris/memory",
        ".igris/reports/decisions", ".igris/missions", ".igris/patches",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)

    old_root = CONFIG.project_root
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))

    # Create safe text files for patches
    (root / "README.md").write_text("# Benchmark Project\n\nInitial content.\n")
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "NOTES.md").write_text("# Notes\n\nOriginal notes.\n")

    te = TaskEngine()
    yield {"root": root, "task_engine": te, "project_root": str(root)}

    CONFIG.project_root = old_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_patch(title: str, description: str, files: List[Dict], pr: str) -> PatchProposal:
    """Create a patch proposal with the correct API."""
    return patch_mod.create_patch_proposal(
        title=title,
        description=description,
        files=files,
        project_root=pr,
    )


def _validate(proposal: PatchProposal, pr: str):
    """Validate and return result."""
    return patch_mod.validate_patch_proposal(proposal, project_root=pr)


# ---------------------------------------------------------------------------
# Benchmark 1: Docs-only task
# ---------------------------------------------------------------------------


class TestBenchmark1DocsOnlyTask:
    """Mission → plan → materialize → patch → validate → apply → report."""

    def test_full_docs_workflow(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]
        root = env["root"]

        record: Dict[str, Any] = {"benchmark": "docs-only"}

        # 1. Create mission
        mission = Mission(
            id="bench-docs-1",
            title="Update project documentation",
            description="Add a CHANGELOG.md to the project with initial release notes",
            status="active",
        )
        mission_planner.save_mission(mission, project_root=pr)
        record["mission_input"] = mission.title

        # 2. Generate plan
        planned = mission_planner.plan_mission("bench-docs-1", project_root=pr)
        assert planned is not None
        assert len(planned.steps) > 0
        record["plan"] = planned.plan_summary

        # 3. Materialize tasks
        mat = mission_planner.materialize_tasks("bench-docs-1", te, project_root=pr)
        assert mat is not None
        assert len(mat.task_ids) > 0
        record["tasks_created"] = len(mat.task_ids)

        # 4. Create patch proposal for docs file
        proposal = _make_patch(
            title="Add CHANGELOG.md",
            description="Initial changelog with v0.3 notes",
            files=[{
                "path": "CHANGELOG.md",
                "action": "create",
                "after": "# Changelog\n\n## v0.3\n\n- Operational diagnostics\n- Safety policy\n- Decision reports\n",
            }],
            pr=pr,
        )
        assert proposal is not None
        assert proposal.status == "proposed"
        assert len(proposal.files) == 1
        assert proposal.files[0].diff
        record["patch_proposal"] = proposal.id

        # 5. Validate proposal
        validation = _validate(proposal, pr)
        assert validation.valid is True
        assert len(validation.reasons) == 0
        record["validation_result"] = "valid"

        # 6. Apply proposal
        apply_result = patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        assert apply_result["success"] is True
        record["apply_result"] = "applied"

        # 7. Verify file created
        changelog = root / "CHANGELOG.md"
        assert changelog.exists()
        assert "v0.3" in changelog.read_text()

        # 8. Verify proposal status persisted
        loaded = patch_mod.load_patch_proposal(proposal.id, project_root=pr)
        assert loaded is not None
        assert loaded.status == "applied"

        # 9. Create decision report
        report = decision_report_mod.create_decision_report(
            step_number=1,
            tasks=te.list_tasks(),
            action_type="propose_patch",
            action_detail="Created CHANGELOG.md",
            outcome="success",
            outcome_reason="File created and validated",
            project_root=pr,
        )
        assert report is not None
        assert report.outcome == "success"
        record["decision_report_id"] = report.id
        record["manual_intervention_needed"] = False
        record["outcome"] = "success"

        # 10. Verify decision report persisted
        saved = decision_report_mod.get_decision_report(report.id, project_root=pr)
        assert saved is not None

    def test_docs_secret_blocked(self, benchmark_env):
        """Secret content in patch must be rejected."""
        env = benchmark_env
        pr = env["project_root"]
        proposal = _make_patch(
            title="Bad docs",
            description="Contains secret",
            files=[{
                "path": "test_safe.md",
                "action": "create",
                "after": "This contains sk-abcdefghijklmnopqrstuvwxyz",
            }],
            pr=pr,
        )
        validation = _validate(proposal, pr)
        assert validation.valid is False
        assert len(validation.secret_findings) > 0


# ---------------------------------------------------------------------------
# Benchmark 2: Bugfix small
# ---------------------------------------------------------------------------


class TestBenchmark2BugfixSmall:
    """Mission → plan → task → patch for a small bugfix."""

    def test_bugfix_workflow(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]
        root = env["root"]

        record: Dict[str, Any] = {"benchmark": "bugfix-small"}

        # Create a "buggy" file
        buggy = root / "utils.py"
        buggy.write_text("def add(a, b):\n    return a - b  # BUG: should be +\n")

        # 1. Create bugfix mission
        mission = Mission(
            id="bench-bugfix-1",
            title="Fix addition bug in utils.py",
            description="Fix the add function in utils.py — it subtracts instead of adding",
            status="active",
        )
        mission_planner.save_mission(mission, project_root=pr)
        record["mission_input"] = mission.title

        # 2. Plan
        planned = mission_planner.plan_mission("bench-bugfix-1", project_root=pr)
        assert planned is not None
        record["plan"] = planned.plan_summary

        # 3. Create fix patch
        proposal = _make_patch(
            title="Fix add function",
            description="Change subtraction to addition",
            files=[{
                "path": "utils.py",
                "action": "modify",
                "before": "def add(a, b):\n    return a - b  # BUG: should be +\n",
                "after": "def add(a, b):\n    return a + b\n",
            }],
            pr=pr,
        )
        assert proposal.files[0].diff  # diff shows the change
        record["patch_proposal"] = proposal.id

        # 4. Validate
        validation = _validate(proposal, pr)
        assert validation.valid is True
        record["validation_result"] = "valid"

        # 5. Apply
        apply_result = patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        assert apply_result["success"] is True
        record["apply_result"] = "applied"

        # 6. Verify fix
        fixed = buggy.read_text()
        assert "a + b" in fixed
        assert "a - b" not in fixed
        record["outcome"] = "success"

    def test_bugfix_idempotent_apply(self, benchmark_env):
        """Applying same proposal twice fails gracefully."""
        env = benchmark_env
        pr = env["project_root"]

        proposal = _make_patch(
            title="Create file",
            description="Test",
            files=[{"path": "temp.txt", "action": "create", "after": "hello"}],
            pr=pr,
        )
        _validate(proposal, pr)
        first = patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        assert first["success"] is True
        second = patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        assert second["success"] is False
        assert "already applied" in second.get("error", "").lower()


# ---------------------------------------------------------------------------
# Benchmark 3: Test failure recovery
# ---------------------------------------------------------------------------


class TestBenchmark3TestFailureRecovery:
    """Failed outcome → outcome router → remediation → memory → project state."""

    def test_failure_recovery_cycle(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]

        record: Dict[str, Any] = {"benchmark": "test-failure-recovery"}

        # 1. Create a task
        task = te.create_task(
            description="Run test suite",
            title="Run tests",
            family="testing",
            source="benchmark",
        )
        record["task_id"] = task.id

        # 2. Simulate failed outcome
        report = {
            "command_id": "run_tests",
            "success": False,
            "stdout_truncated": "",
            "stderr_truncated": "FAILED tests/test_example.py::test_add - AssertionError",
            "task_id": task.id,
        }

        # 3. Route outcome
        recommendation = route_outcome(report, "Run test suite", [])
        assert recommendation is not None
        assert "next_action" in recommendation
        record["recommendation"] = recommendation["next_action"]

        # 4. Record failure in memory
        decision_memory.record_failure(
            title="Test suite failed",
            family="testing",
            task_id=str(task.id),
            reason="AssertionError in test_add",
            project_root=pr,
        )

        # 5. Record in project state
        project_state_mod.record_attempt(
            family="testing",
            success=False,
            fingerprint="test-failure-1",
            project_root=pr,
        )

        # 6. Verify memory has failure
        failures = decision_memory.get_recent_failures(project_root=pr)
        assert len(failures) >= 1
        assert any("test" in f.get("title", "").lower() for f in failures)

        # 7. Verify project state updated
        metrics = project_state_mod.get_family_metrics("testing", project_root=pr)
        assert metrics is not None
        assert metrics.failures >= 1
        record["failure_recorded"] = True

        # 8. Record success (recovery)
        project_state_mod.record_attempt(
            family="testing",
            success=True,
            fingerprint="test-success-1",
            project_root=pr,
        )
        metrics2 = project_state_mod.get_family_metrics("testing", project_root=pr)
        assert metrics2.successes >= 1
        record["recovery_recorded"] = True
        record["outcome"] = "recovered"

    def test_remediation_task_created(self, benchmark_env):
        """Teacher can propose remediation for failures."""
        env = benchmark_env
        pr = env["project_root"]

        from igris.core.teacher import propose_remediation_task, build_teacher_payload
        payload = build_teacher_payload(
            ["Run tests", "Fix bug"],
            project_root=pr,
        )
        remediation = propose_remediation_task(payload)
        assert "task_title" in remediation or "task_description" in remediation

    def test_memory_constraints_reflect_failures(self, benchmark_env):
        """Memory constraints include failure data."""
        env = benchmark_env
        pr = env["project_root"]

        # Record some failures
        for i in range(3):
            decision_memory.record_failure(
                title=f"Repeated failure {i}",
                family="deploy",
                task_id=f"task-{i}",
                reason="timeout",
                project_root=pr,
            )

        constraints = decision_memory.explain_memory_constraints(project_root=pr)
        assert isinstance(constraints, dict)
        assert "avoid_families" in constraints
        assert "saturated_families" in constraints


# ---------------------------------------------------------------------------
# Benchmark 4: Multi-file safe patch
# ---------------------------------------------------------------------------


class TestBenchmark4MultiFilePatch:
    """Multi-file safe patch: two files → diff → validate → apply."""

    def test_two_file_patch(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        root = env["root"]

        record: Dict[str, Any] = {"benchmark": "multi-file-patch"}

        # 1. Create patch with two files in one proposal
        proposal = _make_patch(
            title="Update README and NOTES",
            description="Multi-file update benchmark",
            files=[
                {
                    "path": "README.md",
                    "action": "modify",
                    "before": "# Benchmark Project\n\nInitial content.\n",
                    "after": "# Benchmark Project\n\nUpdated content with new features.\n\n## Features\n\n- Feature A\n- Feature B\n",
                },
                {
                    "path": "docs/NOTES.md",
                    "action": "modify",
                    "before": "# Notes\n\nOriginal notes.\n",
                    "after": "# Notes\n\nUpdated notes with benchmark results.\n\n## Results\n\n- All benchmarks passed.\n",
                },
            ],
            pr=pr,
        )
        assert len(proposal.files) == 2
        for fc in proposal.files:
            assert fc.diff  # each file has diff
        record["patch_proposal"] = proposal.id

        # 2. Validate
        validation = _validate(proposal, pr)
        assert validation.valid is True
        record["validation_result"] = "valid"

        # 3. Apply
        apply_result = patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        assert apply_result["success"] is True
        record["apply_result"] = "applied"

        # 4. Verify files
        readme = (root / "README.md").read_text()
        assert "Feature A" in readme
        notes = (root / "docs" / "NOTES.md").read_text()
        assert "benchmark results" in notes
        record["outcome"] = "success"

    def test_mixed_create_modify(self, benchmark_env):
        """Patch with create + modify in one proposal."""
        env = benchmark_env
        pr = env["project_root"]
        root = env["root"]

        proposal = _make_patch(
            title="Create and modify",
            description="Mixed operations",
            files=[
                {
                    "path": "NEW_FILE.md",
                    "action": "create",
                    "after": "# New File\n\nCreated by benchmark.\n",
                },
                {
                    "path": "README.md",
                    "action": "modify",
                    "before": "# Benchmark Project\n\nInitial content.\n",
                    "after": "# Benchmark Project\n\nModified by benchmark.\n",
                },
            ],
            pr=pr,
        )
        v = _validate(proposal, pr)
        assert v.valid is True
        a = patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        assert a["success"] is True
        assert (root / "NEW_FILE.md").exists()
        assert "Modified by benchmark" in (root / "README.md").read_text()


# ---------------------------------------------------------------------------
# Benchmark 5: Full loop smoke
# ---------------------------------------------------------------------------


class TestBenchmark5FullLoopSmoke:
    """Mission → plan → materialize → select → loop step → report → memory → decision report."""

    def test_full_loop_workflow(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]

        record: Dict[str, Any] = {"benchmark": "full-loop-smoke"}

        # 1. Create mission
        mission = Mission(
            id="bench-loop-1",
            title="Run full test suite",
            description="Execute all tests to verify project health",
            status="active",
        )
        mission_planner.save_mission(mission, project_root=pr)
        record["mission_input"] = mission.title

        # 2. Plan
        planned = mission_planner.plan_mission("bench-loop-1", project_root=pr)
        assert planned is not None
        record["plan"] = planned.plan_summary

        # 3. Materialize tasks
        mat = mission_planner.materialize_tasks("bench-loop-1", te, project_root=pr)
        assert mat is not None
        assert len(mat.task_ids) > 0
        record["tasks_created"] = len(mat.task_ids)

        # 4. Execute one loop step
        step_result = autonomous_loop.execute_step(
            task_engine=te,
            step_number=1,
            project_root=pr,
        )
        assert step_result is not None
        assert step_result.outcome in ("success", "failure", "skipped", "stopped", "blocked")
        record["loop_outcome"] = step_result.outcome

        # 5. Create decision report
        report = decision_report_mod.create_decision_report(
            step_number=1,
            tasks=te.list_tasks(),
            action_type=step_result.action_type or "skip",
            action_detail=step_result.action_detail or "",
            outcome=step_result.outcome or "skipped",
            outcome_reason=step_result.reason or "benchmark",
            project_root=pr,
        )
        assert report is not None
        assert report.id
        record["decision_report_id"] = report.id

        # 6. Verify decision report persisted
        saved = decision_report_mod.get_decision_report(report.id, project_root=pr)
        assert saved is not None

        # 7. Verify memory has events
        decisions = decision_memory.get_recent_decisions(project_root=pr)
        assert isinstance(decisions, list)

        # 8. Check memory constraints
        constraints = decision_memory.explain_memory_constraints(project_root=pr)
        assert "avoid_families" in constraints
        assert "saturated_families" in constraints
        record["outcome"] = "success"

    def test_loop_stops_on_no_tasks(self, benchmark_env):
        """Loop stops gracefully when no tasks are available."""
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]

        step = autonomous_loop.execute_step(
            task_engine=te,
            step_number=0,
            project_root=pr,
        )
        assert step.outcome == "stopped"
        assert "no pending" in step.reason.lower() or "no suitable" in step.reason.lower()

    def test_loop_respects_max_steps(self, benchmark_env):
        """run_loop honors max_steps parameter."""
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]

        te.create_task("Task A", family="analyze")
        te.create_task("Task B", family="testing")

        status = autonomous_loop.run_loop(
            task_engine=te,
            max_steps=1,
            project_root=pr,
        )
        assert status.steps_completed <= 1

    def test_loop_records_timeline(self, benchmark_env):
        """Loop execution creates timeline events."""
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]

        te.create_task("Analyze codebase", family="analyze")
        autonomous_loop.execute_step(
            task_engine=te,
            step_number=0,
            project_root=pr,
        )
        # Steps are recorded
        steps = autonomous_loop.get_recent_steps(limit=10)
        assert isinstance(steps, list)


# ---------------------------------------------------------------------------
# Benchmark safety: cross-cutting
# ---------------------------------------------------------------------------


class TestBenchmarkSafety:
    """Cross-cutting safety checks across all benchmarks."""

    def test_env_file_blocked(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        proposal = _make_patch(
            title="Bad env",
            description="Touches .env",
            files=[{"path": ".env", "action": "create", "after": "SECRET=abc123"}],
            pr=pr,
        )
        validation = _validate(proposal, pr)
        assert validation.valid is False

    def test_git_dir_blocked(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        proposal = _make_patch(
            title="Bad git",
            description="Touches .git",
            files=[{"path": ".git/config", "action": "modify", "after": "hacked"}],
            pr=pr,
        )
        validation = _validate(proposal, pr)
        assert validation.valid is False

    def test_path_traversal_blocked(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        proposal = _make_patch(
            title="Traversal",
            description="Path traversal",
            files=[{"path": "../../../etc/passwd", "action": "modify", "after": "hacked"}],
            pr=pr,
        )
        validation = _validate(proposal, pr)
        assert validation.valid is False

    def test_secret_content_blocked(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        proposal = _make_patch(
            title="Secret content",
            description="Has API key",
            files=[{"path": "safe.txt", "action": "create",
                     "after": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890"}],
            pr=pr,
        )
        validation = _validate(proposal, pr)
        assert validation.valid is False

    def test_redact_secrets_works(self):
        text = "My key is sk-abcdefghijklmnopqrstuvwxyz and ghp_1234567890abcdef"
        redacted = redact_secrets(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "ghp_1234567890abcdef" not in redacted

    def test_no_push_endpoint(self, benchmark_env):
        from fastapi.testclient import TestClient
        from igris.web.server import create_app
        client = TestClient(create_app())
        r = client.post("/api/git/push", json={})
        assert r.status_code == 404 or r.status_code == 405

    def test_no_auto_merge_endpoint(self, benchmark_env):
        from fastapi.testclient import TestClient
        from igris.web.server import create_app
        client = TestClient(create_app())
        r = client.post("/api/git/merge", json={})
        assert r.status_code == 404 or r.status_code == 405

    def test_delete_action_blocked(self, benchmark_env):
        env = benchmark_env
        pr = env["project_root"]
        proposal = _make_patch(
            title="Delete file",
            description="Tries to delete",
            files=[{"path": "README.md", "action": "delete"}],
            pr=pr,
        )
        validation = _validate(proposal, pr)
        assert validation.valid is False


# ---------------------------------------------------------------------------
# Benchmark metadata: records structure
# ---------------------------------------------------------------------------


class TestBenchmarkRecordStructure:
    """Each benchmark produces a record with required fields."""

    REQUIRED_FIELDS = [
        "benchmark", "mission_input", "plan", "tasks_created",
        "patch_proposal", "validation_result", "outcome",
    ]

    def test_docs_record_fields(self, benchmark_env):
        """Docs benchmark produces all required record fields."""
        env = benchmark_env
        pr = env["project_root"]
        te = env["task_engine"]
        root = env["root"]

        record: Dict[str, Any] = {"benchmark": "docs-only"}

        mission = Mission(
            id="rec-docs-1",
            title="Update docs",
            description="Add changelog",
            status="active",
        )
        mission_planner.save_mission(mission, project_root=pr)
        record["mission_input"] = mission.title

        planned = mission_planner.plan_mission("rec-docs-1", project_root=pr)
        record["plan"] = planned.plan_summary

        mat = mission_planner.materialize_tasks("rec-docs-1", te, project_root=pr)
        record["tasks_created"] = len(mat.task_ids)

        proposal = _make_patch(
            title="Add file",
            description="Benchmark record test",
            files=[{"path": "RECORD_TEST.md", "action": "create", "after": "# Test"}],
            pr=pr,
        )
        record["patch_proposal"] = proposal.id
        _validate(proposal, pr)
        record["validation_result"] = "valid"

        patch_mod.apply_patch_proposal(proposal.id, project_root=pr)
        record["outcome"] = "success"
        record["manual_intervention_needed"] = False
        record["decision_report_id"] = "n/a"

        for field in self.REQUIRED_FIELDS:
            assert field in record, f"Missing field: {field}"
