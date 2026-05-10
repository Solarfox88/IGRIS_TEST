import json
import subprocess
import threading
import time

from fastapi.testclient import TestClient

from igris.core.self_repair_supervisor import (
    CommandResult,
    LocalSupervisorBackend,
    RankSupervisorConfig,
    RUN_STORE,
    SelfRepairSupervisor,
    SupervisorEvent,
    SupervisorRun,
    classify_failure,
    get_supervised_run,
    start_supervised_rank_async,
)
from igris.web.server import create_app


class FakeBackend:
    def __init__(self):
        self.status = CommandResult(True, "")
        self.baseline = CommandResult(True, "baseline ok")
        self.smoke_result = CommandResult(True, "smoke ok")
        self.full_tests = [CommandResult(True, "full ok")]
        self.targeted = CommandResult(True, "targeted ok")
        self.reasoning_results = [{
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_status.py"],
            "final_summary": "done",
            "loop_id": "loop-1",
            "goal": "rank task with tests",
        }]
        self.diff_stat = CommandResult(True, " igris/web/server.py | 2 ++")
        self.diff = CommandResult(True, "+safe")
        self.commands = []
        self.test_timeouts = []
        self.restore_result = CommandResult(True, "restored")
        self.last_reasoning_context = None
        self.reasoning_contexts = []

    def git_status(self):
        self.commands.append("git_status")
        return self.status

    def git_log_head(self):
        return CommandResult(True, "abc123 head")

    def create_branch(self, branch):
        self.commands.append(f"branch:{branch}")
        return CommandResult(True, branch)

    def run_reasoning(self, goal, max_steps, initial_context, timeout=300):
        self.commands.append(f"reasoning:{initial_context}")
        self.commands.append(f"reasoning_timeout:{timeout}")
        self.last_reasoning_context = initial_context
        self.reasoning_contexts.append(initial_context)
        if self.reasoning_results:
            return self.reasoning_results.pop(0)
        return {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair",
            "goal": goal,
        }

    def git_diff_stat(self):
        return self.diff_stat

    def git_diff(self):
        return self.diff

    def run_tests(self, targets=None, timeout=240):
        self.commands.append(f"tests:{targets or 'full'}")
        self.test_timeouts.append(timeout)
        if targets:
            return self.targeted
        if self.full_tests:
            return self.full_tests.pop(0)
        return CommandResult(True, "full ok")

    def run_test_diagnostics(self, timeout=120):
        self.commands.append(f"diagnostics:{timeout}")
        return CommandResult(False, "FAILED tests/test_example.py::test_x", "", 1)

    def smoke(self, endpoints, restart_command=""):
        self.commands.append(f"smoke:{endpoints}:{restart_command}")
        return self.smoke_result

    def commit(self, message, files=None):
        self.commands.append("commit")
        return CommandResult(True, "commit")

    def push_branch(self, branch):
        self.commands.append("push")
        return CommandResult(True, "push")

    def open_pr(self, branch, title, body):
        self.commands.append("pr")
        return CommandResult(True, "https://example.test/pr/1")

    def wait_ci(self):
        self.commands.append("ci")
        return CommandResult(True, "ci ok")

    def merge_pr(self):
        self.commands.append("merge")
        return CommandResult(True, "merged")

    def pull_main(self):
        self.commands.append("pull")
        return CommandResult(True, "pulled")

    def create_issue(self, title, body):
        self.commands.append("issue")
        return CommandResult(True, "issue")

    def restore_dangerous_diff(self):
        self.commands.append("restore")
        return self.restore_result


def _config(**overrides):
    data = {
        "goal": "Rank A controlled task with tests",
        "rank_id": "A",
        "max_rank_attempts": 2,
        "max_repair_cycles": 1,
        "required_smoke_endpoints": ["http://127.0.0.1:7778/api/health"],
        "targeted_tests": ["tests/test_rank_status.py"],
        "dry_run": True,
    }
    data.update(overrides)
    return RankSupervisorConfig.from_dict(data)


def test_config_defaults_to_real_github_when_github_flags_enabled():
    config = RankSupervisorConfig.from_dict({
        "goal": "rank",
        "allow_github_pr": True,
        "allow_merge_if_green": True,
    })

    assert config.dry_run is False


def test_config_preserves_explicit_dry_run_with_github_flags_enabled():
    config = RankSupervisorConfig.from_dict({
        "goal": "rank",
        "allow_github_pr": True,
        "allow_merge_if_green": True,
        "dry_run": True,
    })

    assert config.dry_run is True


def test_config_infers_targeted_test_from_goal():
    config = RankSupervisorConfig.from_dict({
        "goal": (
            "Add endpoint and dedicated tests in "
            "tests/test_system_version_summary.py. Run pytest."
        ),
    })

    assert config.targeted_tests == ["tests/test_system_version_summary.py"]


def test_failure_classifier_detects_max_steps_as_repairable_infrastructure_failure():
    failure = classify_failure({"status": "stopped", "stop_reason": "max_steps", "files_modified": []})
    assert failure == "max_steps"


def test_failure_classifier_detects_reasoning_timeout_as_blocked_loop():
    failure = classify_failure({"status": "blocked", "stop_reason": "reasoning_timeout", "files_modified": []})
    assert failure == "reasoning_loop_blocked"


def test_failure_classifier_prioritizes_pytest_failure_over_reasoning_timeout():
    failure = classify_failure(
        {"status": "blocked", "stop_reason": "reasoning_timeout", "files_modified": []},
        targeted_tests=CommandResult(False, "FAILED tests/test_rank_ui_card.py::test_rank_ui_card_endpoint_available", "", 1),
    )
    assert failure == "pytest_failure"


def test_failure_classifier_detects_ast_validation_block_as_syntax_error():
    failure = classify_failure(
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": (
                "Python AST validation failed for 'tests/test_rank_ui_card.py': "
                "expected an indented block after function definition"
            ),
        }
    )
    assert failure == "syntax_error"


def test_failure_classifier_detects_pytest_failure():
    failure = classify_failure(full_tests=CommandResult(False, "FAILED tests/test_x.py", "", 1))
    assert failure == "pytest_failure"


def test_failure_classifier_detects_missing_targeted_test_file():
    failure = classify_failure(
        targeted_tests=CommandResult(
            False,
            "ERROR: file or directory not found: tests/test_rank_s_dashboard.py",
            "",
            4,
        )
    )
    assert failure == "missing_tests"


def test_failure_classifier_prioritizes_llm_unavailable_over_missing_tests():
    failure = classify_failure(
        reasoning_result={
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "No suitable LLM provider available; deterministic fallback",
        },
        targeted_tests=CommandResult(
            False,
            "ERROR: file or directory not found: tests/test_rank_s_dashboard.py",
            "",
            4,
        ),
    )
    assert failure == "infrastructure_bug"


def test_failure_classifier_detects_destructive_diff():
    failure = classify_failure(diff="-def create_app():\n+def removed():\n")
    assert failure == "destructive_diff"


def test_failure_classifier_allows_test_file_rewrite_without_marking_destructive():
    diff = """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
index 1111111..2222222 100644
--- a/tests/test_rank_ui_card.py
+++ b/tests/test_rank_ui_card.py
@@ -1,8 +1,8 @@
-import pytest
 from fastapi.testclient import TestClient

 from igris.web.server import create_app


 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
"""

    failure = classify_failure(diff=diff)

    assert failure != "destructive_diff"


def test_failure_classifier_allows_test_file_rewrite_without_marking_invalid_bootstrap():
    diff = """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
index 1111111..2222222 100644
--- a/tests/test_rank_ui_card.py
+++ b/tests/test_rank_ui_card.py
@@ -1,8 +1,8 @@
-from fastapi.testclient import TestClient
 from fastapi.testclient import TestClient

 from igris.web.server import create_app


 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
"""

    failure = classify_failure(diff=diff)

    assert failure != "invalid_bootstrap"


def test_failure_classifier_does_not_mark_html_class_changes_as_destructive():
    diff = """diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
index 1111111..2222222 100644
--- a/igris/web/templates/index.html
+++ b/igris/web/templates/index.html
@@ -1,5 +1,5 @@
-<div class="panel card">
+<div class="panel card rank-a">
 </div>
"""
    failure = classify_failure(diff=diff)

    assert failure != "destructive_diff"


def test_failure_classifier_detects_invalid_bootstrap_smoke_failure():
    smoke = CommandResult(False, '{"app":"IGRIS_GPT","rank":"A++","status":"ok","capability":"ui-visible-supervised"}', "Invalid bootstrap response for /api/health", 1)

    failure = classify_failure(smoke=smoke)

    assert failure == "invalid_bootstrap"


def test_command_result_serializes_bytes_safely():
    result = CommandResult(False, b"stdout bytes", b"stderr bytes", 124)
    data = result.to_dict()

    assert data["output"] == "stdout bytes"
    assert data["error"] == "stderr bytes"


def test_local_backend_runs_commands_in_isolated_child_process(monkeypatch, tmp_path):
    captured = {}

    class Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return Proc()

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_x.py::test_y")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak")
    monkeypatch.setenv("VASTAI_API_KEY", "")
    monkeypatch.setenv("PROJECT_ROOT", "/service/root")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = LocalSupervisorBackend(str(tmp_path)).run_tests(timeout=31)

    assert result.success
    assert captured["timeout"] == 31
    assert captured["start_new_session"] is True
    assert captured["close_fds"] is True
    assert captured["env"]["IGRIS_SUPERVISOR_CHILD"] == "1"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert "PYTEST_CURRENT_TEST" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "VASTAI_API_KEY" not in captured["env"]
    assert "PROJECT_ROOT" not in captured["env"]


def test_local_backend_runs_reasoning_in_bounded_worker(monkeypatch, tmp_path):
    captured = {}

    class Proc:
        returncode = 0
        stdout = json.dumps({
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_status.py"],
        })
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return Proc()

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = LocalSupervisorBackend(str(tmp_path)).run_reasoning(
        "rank goal",
        max_steps=7,
        initial_context={"rank_test": "A"},
        timeout=42,
    )

    payload = json.loads(captured["input"])
    assert result["status"] == "finished"
    assert captured["cmd"][-2:] == ["-m", "igris.core.supervisor_reasoning_worker"]
    assert captured["timeout"] == 42
    assert captured["start_new_session"] is True
    assert payload["goal"] == "rank goal"
    assert payload["max_steps"] == 7
    assert payload["initial_context"]["rank_test"] == "A"


def test_local_backend_rejects_bootstrap_smoke_payloads(monkeypatch, tmp_path):
    import igris.core.self_repair_supervisor as mod

    responses = {
        "http://127.0.0.1:7778/api/health": '{"app":"IGRIS_GPT","rank":"A++","status":"ok","capability":"ui-visible-supervised"}',
        "http://127.0.0.1:7778/api/readiness": '{"project_root_exists":true,"project_root_is_dir":true,"templates":true,"static":true,"agents_registered":true}',
        "http://127.0.0.1:7778/api/ping": '{"pong":true}',
    }

    class Proc:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["curl", "-fsS", "http://127.0.0.1:7778/api/health"]:
            return Proc(stdout=responses["http://127.0.0.1:7778/api/health"])
        if cmd[:3] == ["curl", "-fsS", "http://127.0.0.1:7778/api/readiness"]:
            return Proc(stdout=responses["http://127.0.0.1:7778/api/readiness"])
        if cmd[:3] == ["curl", "-fsS", "http://127.0.0.1:7778/api/ping"]:
            return Proc(stdout=responses["http://127.0.0.1:7778/api/ping"])
        return Proc(stdout="ok")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = LocalSupervisorBackend(str(tmp_path)).smoke([
        "http://127.0.0.1:7778/api/health",
        "http://127.0.0.1:7778/api/readiness",
        "http://127.0.0.1:7778/api/ping",
    ])

    assert not result.success
    assert "Invalid bootstrap response for http://127.0.0.1:7778/api/health" in result.error


def test_local_backend_classifies_reasoning_timeout(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=kwargs["timeout"],
            output=b"partial",
            stderr=b"timed out",
        )

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = LocalSupervisorBackend(str(tmp_path)).run_reasoning(
        "rank goal",
        max_steps=7,
        initial_context={},
        timeout=42,
    )

    assert result["status"] == "blocked"
    assert result["stop_reason"] == "reasoning_timeout"
    assert "timed out" in result["final_summary"]


def test_supervisor_event_serializes_bytes_safely():
    event = SupervisorEvent(
        phase="baseline_tests",
        status="failure",
        detail=b"Command timed out",
        data={"raw": b"bytes"},
    )
    data = event.to_dict()

    assert data["detail"] == "Command timed out"
    assert data["data"]["raw"] == "bytes"


def test_supervisor_does_not_proceed_when_workspace_dirty():
    backend = FakeBackend()
    backend.status = CommandResult(True, " M igris/web/server.py\n")
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_config())

    assert run.status == "blocked"
    assert run.failure_class == "workspace_dirty"
    assert not any(cmd.startswith("branch:") for cmd in backend.commands)


def test_supervisor_blocks_merge_when_full_pytest_fails():
    backend = FakeBackend()
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(False, "1 failed", "", 1),
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_config(dry_run=False, allow_github_pr=True, allow_merge_if_green=True, max_repair_cycles=0))

    assert run.status == "blocked"
    assert run.failure_class == "pytest_failure"
    assert "merge" not in backend.commands


def test_supervisor_completes_by_verification_when_reasoning_blocks_after_changes():
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "blocked",
        "stop_reason": "blocked",
        "files_modified": ["igris/web/server.py", "tests/test_rank_status.py"],
        "final_summary": "changed files but final report blocked",
        "goal": "rank task with tests",
    }]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert "issue" not in backend.commands
    assert not any(event.phase == "failure" for event in run.events)


def test_supervisor_completes_by_verification_when_reasoning_timeout_has_diff():
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "blocked",
        "stop_reason": "reasoning_timeout",
        "files_modified": [],
        "final_summary": "Command timed out",
        "goal": "rank task with tests",
    }]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert "issue" not in backend.commands
    assert not any(event.phase == "failure" for event in run.events)
    assert run.report["degraded_completion"] is True
    assert run.report["completion_mode"] == "verified_diff"


def test_supervisor_records_post_merge_smoke_and_degraded_completion():
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "blocked",
        "stop_reason": "reasoning_timeout",
        "files_modified": [],
        "final_summary": "Command timed out after producing changes",
        "goal": "rank task with tests",
    }]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(dry_run=False, allow_github_pr=True, allow_merge_if_green=True, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert any(event.phase == "completion" and event.status == "degraded" for event in run.events)
    assert any(event.phase == "post_merge_smoke" and event.status == "success" for event in run.events)
    assert run.report["degraded_completion"] is True
    assert run.report["completion_mode"] == "verified_diff"
    assert run.report["post_merge_smoke"] is True
    assert run.report["manual_remaining"] == ""
    assert backend.commands.index("pull") < len(backend.commands) - 1


def test_supervisor_defers_post_merge_smoke_when_runtime_refresh_is_required():
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "blocked",
        "stop_reason": "reasoning_timeout",
        "files_modified": ["igris/web/server.py", "tests/test_rank_summary_card.py"],
        "final_summary": "Command timed out after producing changes",
        "goal": "rank task with tests",
    }]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            dry_run=False,
            allow_github_pr=True,
            allow_merge_if_green=True,
            defer_service_restart=True,
            max_repair_cycles=1,
        )
    )

    assert run.status == "completed"
    assert any(event.phase == "post_merge_smoke" and event.status == "deferred" for event in run.events)
    assert run.report["runtime_refresh_required"] is True
    assert run.report["post_merge_smoke"] is False
    assert run.report["degraded_completion"] is True
    assert sum(1 for command in backend.commands if command.startswith("smoke:")) == 2


def test_supervisor_reports_manual_remaining_when_merge_is_disabled():
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(dry_run=False, allow_github_pr=True, allow_merge_if_green=False, max_repair_cycles=0)
    )

    assert run.status == "completed"
    assert run.report["manual_remaining"] == "merge disabled by config"
    assert "merge" not in backend.commands


def test_supervisor_reports_manual_remaining_for_dry_run_delivery():
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(dry_run=True, max_repair_cycles=0)
    )

    assert run.status == "completed"
    assert run.report["manual_remaining"] == "delivery skipped by dry_run"


def test_supervisor_runs_baseline_diagnostics_before_blocking():
    backend = FakeBackend()
    backend.full_tests = [CommandResult(False, "progress dots", "", 1)]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(test_timeout_seconds=90)
    )

    assert run.status == "blocked"
    assert run.failure_class == "pytest_failure"
    assert "diagnostics:90" in backend.commands
    assert any(event.phase == "baseline_diagnostics" for event in run.events)


def test_supervisor_records_stdout_and_error_for_failed_diagnostics():
    backend = FakeBackend()
    backend.full_tests = [CommandResult(False, "progress dots", "pytest timed out", 124)]
    backend.run_test_diagnostics = lambda timeout=120: CommandResult(
        False,
        "partial verbose output",
        "Command timed out",
        124,
    )
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(test_timeout_seconds=90)
    )

    detail = next(
        event.detail
        for event in run.events
        if event.phase == "baseline_diagnostics" and event.status == "failure"
    )

    assert run.status == "blocked"
    assert "partial verbose output" in detail
    assert "Command timed out" in detail


def test_supervisor_produces_blocked_report_after_repair_budget_exhausted():
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "stopped",
        "stop_reason": "max_steps",
        "files_modified": [],
        "final_summary": "stopped",
        "goal": "rank task with tests",
    }]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_config(max_repair_cycles=0))

    assert run.status == "blocked"
    assert run.failure_class == "max_steps"
    assert run.report["blocked_reason"]


def test_supervisor_handles_destructive_diff_without_unbound_local_exception():
    backend = FakeBackend()
    backend.diff = CommandResult(True, "-def create_app():\n+def removed():\n")
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 2 +-")
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_rank_attempts=1, max_repair_cycles=0)
    )

    assert run.status == "blocked"
    assert run.failure_class == "destructive_diff"
    assert any(event.phase == "safety" and event.status == "blocked" for event in run.events)
    assert all(event.phase != "exception" for event in run.events)


def test_supervisor_passes_after_one_repair_cycle():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "stopped",
            "stop_reason": "max_steps",
            "files_modified": [],
            "final_summary": "stopped",
            "goal": "rank task with tests",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair ok",
            "goal": "repair",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_status.py"],
            "final_summary": "rank ok",
            "goal": "rank task with tests",
        },
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_config(max_rank_attempts=2, max_repair_cycles=1))

    assert run.status == "completed"
    assert run.repair_cycles_used == 1
    assert any(event.phase == "repair_reasoning" for event in run.events)


def test_supervisor_extends_attempts_after_repair_on_final_configured_attempt():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "stopped",
            "stop_reason": "max_steps",
            "files_modified": [],
            "final_summary": "stopped",
            "goal": "rank task with tests",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair ok",
            "goal": "repair",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_status.py"],
            "final_summary": "rank ok after repair",
            "goal": "rank task with tests",
        },
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full failed attempt"),
        CommandResult(True, "repair validation ok"),
        CommandResult(True, "rank full final"),
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_rank_attempts=1, max_repair_cycles=2)
    )

    assert run.status == "completed"
    assert run.repair_cycles_used == 1
    assert sum(1 for command in backend.commands if command.startswith("branch:")) == 2
    assert any(event.phase == "rank_attempt_extension" for event in run.events)


def test_supervisor_deduplicates_repair_issue_for_same_failure_in_single_run():
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.reasoning_results = [
        {
            "status": "stopped",
            "stop_reason": "max_steps",
            "files_modified": [],
            "final_summary": "stopped",
            "goal": "rank task with tests",
        },
        {
            "status": "blocked",
            "stop_reason": "ask_user",
            "files_modified": [],
            "final_summary": "needs human",
            "goal": "repair",
        },
        {
            "status": "stopped",
            "stop_reason": "max_steps",
            "files_modified": [],
            "final_summary": "stopped again",
            "goal": "rank task with tests",
        },
        {
            "status": "blocked",
            "stop_reason": "ask_user",
            "files_modified": [],
            "final_summary": "needs human again",
            "goal": "repair",
        },
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            dry_run=False,
            allow_github_pr=True,
            allow_merge_if_green=False,
            max_rank_attempts=2,
            max_repair_cycles=2,
        )
    )

    assert run.status == "blocked"
    assert sum(1 for command in backend.commands if command == "issue") == 1
    assert sum(1 for event in run.events if event.phase == "repair_issue" and event.status == "success") == 1
    assert sum(1 for event in run.events if event.phase == "repair_issue" and event.status == "skipped") == 1


def test_supervisor_blocks_and_restores_when_repair_reasoning_blocks():
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.reasoning_results = [
        {
            "status": "stopped",
            "stop_reason": "max_steps",
            "files_modified": [],
            "final_summary": "stopped",
            "goal": "rank task with tests",
        },
        {
            "status": "blocked",
            "stop_reason": "ask_user",
            "files_modified": [],
            "final_summary": "needs human",
            "goal": "repair",
        },
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_rank_attempts=2, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert run.repair_cycles_used == 1
    assert "restore" in backend.commands
    assert sum(1 for command in backend.commands if command.startswith("branch:")) == 2
    assert any(event.phase == "repair_restore" for event in run.events)


def test_supervisor_blocks_and_restores_when_repair_tests_fail():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "stopped",
            "stop_reason": "max_steps",
            "files_modified": [],
            "final_summary": "stopped",
            "goal": "rank task with tests",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair",
            "goal": "repair",
        },
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
        CommandResult(False, "repair tests failed", "", 1),
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_rank_attempts=2, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert run.repair_cycles_used == 1
    assert "restore" in backend.commands
    assert sum(1 for command in backend.commands if command.startswith("branch:")) == 2
    assert any(event.phase == "repair_restore" for event in run.events)


def test_supervisor_defers_restart_when_configured():
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(service_restart_command="sudo -n systemctl restart igris", defer_service_restart=True)
    )

    assert run.status == "completed"
    assert any(event.phase == "service_restart" and event.status == "deferred" for event in run.events)
    assert all(not command.endswith(":sudo -n systemctl restart igris") for command in backend.commands)


def test_supervisor_records_running_events_and_test_timeout():
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(test_timeout_seconds=45)
    )

    assert run.status == "completed"
    running_phases = [
        event.phase for event in run.events
        if event.status == "running"
    ]
    assert "baseline_tests" in running_phases
    assert "baseline_smoke" in running_phases
    assert "targeted_tests" in running_phases
    assert "full_pytest" in running_phases
    assert "smoke" in running_phases
    assert backend.test_timeouts == [45, 45, 45]


def test_supervisor_records_rank_reasoning_running_and_timeout_budget():
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(reasoning_timeout_seconds=55)
    )

    assert run.status == "completed"
    assert any(event.phase == "rank_reasoning" and event.status == "running" for event in run.events)
    assert "reasoning_timeout:55" in backend.commands


def test_supervisor_passes_requested_rank_test_file_to_reasoning_context(tmp_path):
    backend = FakeBackend()
    run = SelfRepairSupervisor(str(tmp_path), backend=backend).run(
        _config(targeted_tests=["tests/test_rank_status.py"])
    )

    assert run.status == "completed"
    assert backend.last_reasoning_context["must_create_test_file"] == "tests/test_rank_status.py"
    assert backend.last_reasoning_context["expected_endpoint_file"] == "igris/web/server.py"
    assert backend.last_reasoning_context["must_not_ask_user"] is True
    assert "TestClient(create_app())" in backend.last_reasoning_context["fastapi_test_policy"]
    assert "Do not import app" in backend.last_reasoning_context["fastapi_test_policy"]
    assert "Create tests/test_rank_status.py directly" in backend.last_reasoning_context["anti_loop_instruction"]


def test_supervisor_does_not_set_must_create_when_targeted_test_file_exists(tmp_path):
    test_file = tmp_path / "tests" / "test_rank_ui_card.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    backend = FakeBackend()
    run = SelfRepairSupervisor(str(tmp_path), backend=backend).run(
        _config(targeted_tests=["tests/test_rank_ui_card.py"])
    )

    assert run.status == "completed"
    assert "must_create_test_file" not in backend.last_reasoning_context
    assert backend.last_reasoning_context["targeted_test_file_exists"] == "tests/test_rank_ui_card.py"
    assert "Edit this file in place" in backend.last_reasoning_context["targeted_test_policy"]


def test_supervisor_sets_must_create_when_targeted_test_file_is_missing(tmp_path):
    backend = FakeBackend()
    run = SelfRepairSupervisor(str(tmp_path), backend=backend).run(
        _config(targeted_tests=["tests/test_rank_ui_card.py"])
    )

    assert run.status == "completed"
    assert backend.last_reasoning_context["must_create_test_file"] == "tests/test_rank_ui_card.py"
    assert "Create tests/test_rank_ui_card.py directly" in backend.last_reasoning_context["anti_loop_instruction"]
    assert "targeted_test_file_exists" not in backend.last_reasoning_context


def test_supervisor_requires_ui_visibility_for_ui_goals():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_ui_card.py"],
            "final_summary": "backend only",
            "goal": "Add UI-visible rank card",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/static/js/app.js"],
            "final_summary": "ui repair",
            "goal": "repair",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [
                "igris/web/server.py",
                "tests/test_rank_ui_card.py",
                "igris/web/static/js/app.js",
            ],
            "final_summary": "ui plus backend",
            "goal": "Add UI-visible rank card",
        },
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
        CommandResult(True, "repair ok"),
        CommandResult(True, "rank full ok"),
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=2, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert run.repair_cycles_used == 1
    assert run.failure_class == "missing_ui_visibility" or not run.failure_class
    assert backend.last_reasoning_context["must_add_ui_visibility"] is True
    assert "Backend-only changes are not enough" in backend.last_reasoning_context["ui_visibility_policy"]
    assert any(
        context.get("supervised_repair") is True
        and context.get("must_not_ask_user") is True
        and context.get("must_add_ui_visibility") is True
        for context in backend.reasoning_contexts
    )
    assert any(
        event.phase == "repair_completion"
        and event.status == "degraded"
        for event in run.events
    )
    assert any(
        event.phase == "rank_reasoning"
        and event.data.get("ui_visibility_required") is True
        for event in run.events
    )


def test_rank_ui_card_contract_detector_true(tmp_path):
    server = tmp_path / "igris" / "web" / "server.py"
    server.parent.mkdir(parents=True, exist_ok=True)
    server.write_text(
        """def create_app():
    @app.get('/api/rank/ui-card')
    async def get_rank_ui_card():
        return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}
""",
        encoding="utf-8",
    )

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=FakeBackend())

    assert supervisor._rank_ui_card_contract_satisfied() is True


def test_supervisor_context_enforces_ui_only_when_contract_already_satisfied(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: True)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=1, max_repair_cycles=0)
    )

    assert run.status in {"completed", "blocked"}
    assert backend.last_reasoning_context["ui_contract_already_satisfied"] is True
    assert "Do not modify this route" in backend.last_reasoning_context["ui_contract_policy"]


def test_supervisor_completes_ui_mission_as_verified_noop_when_already_satisfied(monkeypatch):
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "timed out with no edits",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
    ]
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: True)
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_visibility_signal_present", lambda self: True)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=1, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert run.report["completion_mode"] == "already_satisfied"
    assert run.report["degraded_completion"] is True
    assert run.report["manual_remaining"] == ""
    assert run.report["no_op_completion"] is True
    assert "issue" not in backend.commands
    assert not any(event.phase == "failure" for event in run.events)
    assert any(
        event.phase == "completion" and event.data.get("mode") == "already_satisfied"
        for event in run.events
    )


def test_supervisor_does_not_noop_complete_when_ui_contract_is_not_satisfied(monkeypatch):
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "timed out with no edits",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
    ]
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: False)
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_visibility_signal_present", lambda self: False)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=1, max_repair_cycles=0)
    )

    assert run.status == "blocked"
    assert run.failure_class == "reasoning_loop_blocked"


def test_supervisor_blocks_immediately_when_llm_provider_is_unavailable():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "No suitable LLM provider available; deterministic fallback",
            "goal": "Add /api/rank/s-dashboard endpoint and tests",
        }
    ]
    backend.targeted = CommandResult(
        False,
        "ERROR: file or directory not found: tests/test_rank_s_dashboard.py",
        "",
        4,
    )

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_rank_attempts=2,
            max_repair_cycles=3,
        )
    )

    assert run.status == "blocked"
    assert run.failure_class == "infrastructure_bug"
    assert not any(event.phase == "repair_issue" for event in run.events)
    assert any(
        event.phase == "blocked"
        and "No suitable LLM provider available" in event.detail
        for event in run.events
    )


def test_supervisor_context_does_not_lock_ui_card_contract_for_non_ui_card_goal(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: True)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            goal="Add Rank S dashboard visibility for /api/rank/s-dashboard",
            max_rank_attempts=1,
            max_repair_cycles=0,
        )
    )

    assert run.status in {"completed", "blocked"}
    assert backend.last_reasoning_context["ui_contract_already_satisfied"] is False
    assert "ui_contract_policy" not in backend.last_reasoning_context


def test_supervisor_does_not_noop_complete_non_ui_card_goal_when_ui_card_contract_is_satisfied(monkeypatch):
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "blocked while editing rank s endpoint",
            "goal": "Add Rank S dashboard visibility for /api/rank/s-dashboard",
        }
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,4 +1,5 @@
+@app.get('/api/rank/s-dashboard')
""",
    )
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
    ]
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: True)
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_visibility_signal_present", lambda self: True)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            goal="Add Rank S dashboard visibility for /api/rank/s-dashboard",
            rank_id="S",
            max_rank_attempts=1,
            max_repair_cycles=0,
        )
    )

    assert run.status == "blocked"
    assert run.failure_class == "missing_ui_visibility"
    assert run.report.get("completion_mode") != "already_satisfied"
    assert "restore" not in backend.commands


def test_supervisor_restores_protected_ui_contract_test_edits_and_completes_noop(monkeypatch):
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "timed out while editing tests",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.diff_stat = CommandResult(True, " tests/test_rank_ui_card.py | 10 ++++++++++")
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
@@ -1,8 +1,10 @@
 def test_rank_ui_card_endpoint_available():
+    assert response.json() == {"app":"IGRIS_GPT","rank":"A++","status":"ok","capability":"ui-visible-supervised"}
""",
    )
    backend.full_tests = [CommandResult(True, "baseline ok")]
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: True)
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_visibility_signal_present", lambda self: True)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=1, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert run.report["completion_mode"] == "already_satisfied"
    assert run.report["no_op_completion"] is True
    assert "restore" in backend.commands
    assert "issue" not in backend.commands


def test_supervisor_does_not_restore_protected_contract_edits_when_ui_surface_changes(monkeypatch):
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "timed out with ui edits",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.diff_stat = CommandResult(
        True,
        " igris/web/templates/index.html | 1 +\n tests/test_rank_ui_card.py | 1 +",
    )
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
@@ -20,6 +20,7 @@
+<div id="rank-ui-card-visibility" hidden></div>
diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
@@ -1,8 +1,9 @@
+assert response.status_code == 200
""",
    )
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
    ]
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_card_contract_satisfied", lambda self: True)
    monkeypatch.setattr(SelfRepairSupervisor, "_rank_ui_visibility_signal_present", lambda self: True)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=1, max_repair_cycles=0)
    )

    assert run.status == "completed"
    assert "restore" not in backend.commands


def test_supervisor_infers_ui_visibility_from_diff_when_reasoning_metadata_is_empty():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "timed out after partial edits",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.diff_stat = CommandResult(True, " igris/web/static/js/app.js | 1 +")
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/static/js/app.js b/igris/web/static/js/app.js
@@ -10,6 +10,7 @@
+const rankUiVisible = true;
""",
    )
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=1, max_repair_cycles=0)
    )

    assert run.status == "completed"
    assert run.failure_class == ""
    assert any(
        event.phase == "ui_visibility"
        and event.status == "success"
        and event.data.get("inferred_from_diff") is True
        for event in run.events
    )


def test_supervisor_retries_repair_validation_failures_for_rank_reasons():
    backend = FakeBackend()
    backend.diff = CommandResult(True, "+ui")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "needs repair",
            "goal": "rank task with tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/static/js/app.js"],
            "final_summary": "repair produced a diff",
            "goal": "repair",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [
                "igris/web/server.py",
                "tests/test_rank_ui_card.py",
                "igris/web/static/js/app.js",
            ],
            "final_summary": "rank repaired",
            "goal": "rank task with tests",
        },
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "rank full ok"),
        CommandResult(False, "repair validation failed"),
        CommandResult(True, "rank full ok"),
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(goal="Add UI-visible rank card", max_rank_attempts=2, max_repair_cycles=1)
    )

    assert run.status == "completed"
    assert run.repair_cycles_used == 1
    assert any(event.phase == "repair_retry" for event in run.events)


def test_supervisor_retries_repair_validation_failures_for_pytest_failure_class():
    backend = FakeBackend()
    backend.diff = CommandResult(True, "+safe")
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair produced a diff",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(False, "repair validation failed", "", 1)]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-pytest-retry", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Rank task with tests", max_repair_cycles=1),
        "pytest_failure",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "pytest_failure"
        for event in run.events
    )


def test_supervisor_retries_no_diff_repairs_for_syntax_error():
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "repair attempt produced no valid diff",
            "goal": "repair",
        }
    ]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-syntax-retry", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "syntax_error",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "syntax_error"
        for event in run.events
    )


def test_supervisor_rejects_invalid_ui_test_diff_before_validation_pytest():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
@@ -1,8 +1,8 @@
 from fastapi.testclient import TestClient

 from igris.web.server import create_app


 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
-    response = client.get("/api/rank/ui-card")
+    response = client.post("/api/rank/ui-card", json={"key": "value"})

     assert response.status_code == 200
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_ui_card.py"],
            "final_summary": "ui repair",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
    ]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-1", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "missing_ui_visibility",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_rejects_product_only_ui_repair_diff_before_validation_pytest():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
index 1111111..2222222 100644
--- a/tests/test_rank_ui_card.py
+++ b/tests/test_rank_ui_card.py
@@ -1,9 +1,18 @@
 from fastapi.testclient import TestClient
 
 from igris.web.server import create_app
 
 
 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
     response = client.get("/api/rank/ui-card")
 
     assert response.status_code == 200
+    assert response.json() == {
+        "app": "IGRIS_GPT",
+        "rank": "A++",
+        "status": "ok",
+        "capability": "ui-visible-supervised",
+    }
""",
    )
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["tests/test_rank_ui_card.py"],
            "final_summary": "edited product test during repair",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "baseline ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-product-only", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "reasoning_loop_blocked",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_accepts_product_only_ui_repair_diff_for_pytest_failure():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
index 1111111..2222222 100644
--- a/igris/web/templates/index.html
+++ b/igris/web/templates/index.html
@@ -48,6 +48,7 @@
   <section id="rank-dashboard">
+    <p id="rank-ui-card-status">ui-visible-supervised</p>
   </section>
diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
index 1111111..2222222 100644
--- a/tests/test_rank_ui_card.py
+++ b/tests/test_rank_ui_card.py
@@ -1,9 +1,16 @@
 from fastapi.testclient import TestClient

 from igris.web.server import create_app


 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
     response = client.get("/api/rank/ui-card")

     assert response.status_code == 200
+    assert response.json() == {
+        "app": "IGRIS_GPT",
+        "rank": "A++",
+        "status": "ok",
+        "capability": "ui-visible-supervised",
+    }
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/templates/index.html", "tests/test_rank_ui_card.py"],
            "final_summary": "repair pytest failure for ui mission",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-ui-pytest-repair", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "pytest_failure",
        1,
    )

    assert result is True
    assert "restore" not in backend.commands
    assert any(command.startswith("tests:") for command in backend.commands)
    assert not any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_rejects_invalid_ui_test_diff_for_pytest_failure_repairs():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
@@ -1,8 +1,9 @@
 from fastapi.testclient import TestClient

 from igris.web.server import create_app


 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
     response = client.get("/api/rank/ui-card")
     assert response.status_code == 200
-    assert response.json() == {"app": "IGRIS_GPT", "rank": "A++", "status": "ok", "capability": "ui-visible-supervised"}
+    assert "data" in response.json()
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_ui_card.py"],
            "final_summary": "invalid test assertion",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-ui-pytest-invalid", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "pytest_failure",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_accepts_safe_ui_surface_repair_for_missing_ui_visibility():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/static/js/app.js b/igris/web/static/js/app.js
index 1111111..2222222 100644
--- a/igris/web/static/js/app.js
+++ b/igris/web/static/js/app.js
@@ -10,6 +10,7 @@
 function renderDashboard() {
   const root = document.getElementById("dashboard-root");
+  root.setAttribute("data-rank-ui-card", "visible");
 }
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/static/js/app.js"],
            "final_summary": "ui visibility repair",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-ui-accept", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "missing_ui_visibility",
        1,
    )

    assert result is True
    assert any(command.startswith("tests:") for command in backend.commands)
    assert not any(event.phase == "repair_retry" for event in run.events)


def test_supervisor_accepts_safe_ui_surface_repair_for_reasoning_loop_blocked():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
index 1111111..2222222 100644
--- a/igris/web/templates/index.html
+++ b/igris/web/templates/index.html
@@ -20,6 +20,7 @@
 <main id="dashboard-root">
+  <div id="rank-ui-card-visibility" hidden></div>
 </main>
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/templates/index.html"],
            "final_summary": "ui visibility repair",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-ui-accept-reasoning", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "reasoning_loop_blocked",
        1,
    )

    assert result is True
    assert any(command.startswith("tests:") for command in backend.commands)
    assert not any(event.phase == "repair_retry" for event in run.events)


def test_supervisor_rejects_missing_ui_visibility_repair_when_only_tests_change():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
index 1111111..2222222 100644
--- a/tests/test_rank_ui_card.py
+++ b/tests/test_rank_ui_card.py
@@ -1,7 +1,12 @@
 from fastapi.testclient import TestClient

 from igris.web.server import create_app

 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
     response = client.get("/api/rank/ui-card")
+    assert response.json() == {
+        "app": "IGRIS_GPT",
+        "rank": "A++",
+        "status": "ok",
+        "capability": "ui-visible-supervised",
+    }
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_ui_card.py"],
            "final_summary": "tests only",
            "goal": "repair",
        }
    ]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-ui-reject", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "missing_ui_visibility",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_accepts_missing_tests_repair_for_mission_endpoint(tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
index 1111111..2222222 100644
--- a/tests/test_rank_s_dashboard.py
+++ b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,12 @@
+from fastapi.testclient import TestClient
+
+from igris.web.server import create_app
+
+def test_rank_s_dashboard_contract():
+    client = TestClient(create_app())
+    response = client.get("/api/rank/s-dashboard")
+    assert response.status_code == 200
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "create targeted test",
            "goal": "repair missing tests",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="run-missing-tests-accept", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "missing_tests",
        1,
    )

    assert result is True
    assert any(command.startswith("tests:") for command in backend.commands)
    assert not any(event.phase == "repair_retry" for event in run.events)


def test_supervisor_rejects_missing_tests_repair_with_unrelated_endpoints(tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
index 1111111..2222222 100644
--- a/tests/test_rank_s_dashboard.py
+++ b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,16 @@
+from fastapi.testclient import TestClient
+
+from igris.web.server import create_app
+
+def test_rank_s_dashboard_endpoint():
+    client = TestClient(create_app())
+    response = client.get("/api/rank/status")
+    assert response.status_code == 200
+
+def test_dashboard_endpoint(client):
+    response = client.get("/dashboard")
+    assert response.status_code == 200
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "wrong endpoint and fixture usage",
            "goal": "repair missing tests",
        }
    ]

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="run-missing-tests-reject", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "missing_tests",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_scaffolds_missing_tests_target_file(tmp_path):
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=FakeBackend())
    config = _config(
        goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
        targeted_tests=["tests/test_rank_s_dashboard.py"],
    )

    result = supervisor._scaffold_missing_tests_target(config)

    assert result.success
    scaffold_path = tmp_path / "tests/test_rank_s_dashboard.py"
    assert scaffold_path.exists()
    content = scaffold_path.read_text(encoding="utf-8")
    assert 'response = client.get("/api/rank/s-dashboard")' in content
    assert "TestClient(create_app())" in content


def test_supervisor_missing_tests_repair_uses_scaffold_fallback(monkeypatch, tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
""",
    )
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 4 ++++")
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "wrong repair scope",
            "goal": "repair missing tests",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)

    def _fake_scaffold(config):
        backend.diff = CommandResult(
            True,
            """diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,8 @@
+from fastapi.testclient import TestClient
+from igris.web.server import create_app
+def test_rank_s_dashboard_contract():
+    client = TestClient(create_app())
+    response = client.get('/api/rank/s-dashboard')
+    assert response.status_code == 200
""",
        )
        backend.diff_stat = CommandResult(True, " tests/test_rank_s_dashboard.py | 8 ++++++++")
        return CommandResult(True, "scaffolded")

    monkeypatch.setattr(supervisor, "_scaffold_missing_tests_target", _fake_scaffold)
    run = SupervisorRun(run_id="run-missing-tests-scaffold", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            max_repair_cycles=1,
        ),
        "missing_tests",
        1,
    )

    assert result is True
    assert any(event.phase == "repair_scaffold" and event.status == "success" for event in run.events)
    assert any(command.startswith("tests:") for command in backend.commands)
    assert not any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_supervisor_missing_tests_accepts_untracked_scaffold(monkeypatch, tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
""",
    )
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 4 ++++")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "No suitable LLM provider available; deterministic fallback",
            "goal": "repair missing tests",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)

    def _fake_scaffold(config):
        target = tmp_path / "tests/test_rank_s_dashboard.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (
                "from fastapi.testclient import TestClient\n\n"
                "from igris.web.server import create_app\n\n\n"
                "def test_rank_s_dashboard_contract():\n"
                "    client = TestClient(create_app())\n"
                "    response = client.get('/api/rank/s-dashboard')\n"
                "    assert response.status_code == 200\n"
            ),
            encoding="utf-8",
        )
        backend.diff = CommandResult(True, "")
        backend.diff_stat = CommandResult(True, "")
        return CommandResult(True, "scaffolded")

    monkeypatch.setattr(supervisor, "_scaffold_missing_tests_target", _fake_scaffold)
    run = SupervisorRun(run_id="run-missing-tests-untracked-scaffold", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "missing_tests",
        1,
    )

    assert result is True
    assert any(event.phase == "repair_scaffold" and event.status == "success" for event in run.events)
    assert any(
        event.phase == "repair_scaffold_diff" and event.data.get("synthesized_untracked") is True
        for event in run.events
    )
    assert not any(
        event.phase == "repair_restore"
        and "Scaffolded missing-tests diff was invalid; restored." in event.detail
        for event in run.events
    )
    assert any(command.startswith("tests:") for command in backend.commands)


def test_supervisor_preserves_valid_missing_tests_scaffold_when_repair_pytest_fails(monkeypatch, tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
""",
    )
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 4 ++++")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "No suitable LLM provider available; deterministic fallback",
            "goal": "repair missing tests",
        }
    ]
    backend.full_tests = [CommandResult(False, "FAILED tests/test_rank_s_dashboard.py::test_rank_s_dashboard_contract", "", 1)]

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)

    def _fake_scaffold(config):
        backend.diff = CommandResult(
            True,
            """diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,8 @@
+from fastapi.testclient import TestClient
+from igris.web.server import create_app
+def test_rank_s_dashboard_contract():
+    client = TestClient(create_app())
+    response = client.get('/api/rank/s-dashboard')
+    assert response.status_code == 200
""",
        )
        backend.diff_stat = CommandResult(True, " tests/test_rank_s_dashboard.py | 8 ++++++++")
        return CommandResult(True, "scaffolded")

    monkeypatch.setattr(supervisor, "_scaffold_missing_tests_target", _fake_scaffold)
    run = SupervisorRun(run_id="run-missing-tests-preserve-scaffold", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "missing_tests",
        1,
    )

    assert result is True
    assert backend.commands.count("restore") == 1
    assert any(
        event.phase == "repair_completion" and event.status == "degraded"
        for event in run.events
    )
    assert not any(
        event.phase == "repair_retry"
        and "Repair validation failed; retrying" in event.detail
        for event in run.events
    )


def test_supervisor_re_scaffolds_targeted_test_after_pytest_failure_restore(tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,5 @@
+@app.get('/api/rank/s-dashboard')
+async def get_rank_s_dashboard():
+    return {'status': 'ok'}
""",
    )
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 3 +++")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "repair timed out",
            "goal": "repair pytest failure",
        }
    ]
    backend.full_tests = [CommandResult(False, "FAILED tests/test_rank_s_dashboard.py::test_api_rank_s_dashboard", "", 1)]
    backend.restore_result = CommandResult(True, "Removing tests/test_rank_s_dashboard.py")

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="run-pytest-re-scaffold", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "pytest_failure",
        1,
    )

    assert result is True
    assert backend.commands.count("restore") == 1
    assert (tmp_path / "tests/test_rank_s_dashboard.py").exists()
    assert any(event.phase == "repair_scaffold" and event.status == "success" for event in run.events)
    assert any(
        event.phase == "repair_completion"
        and "re-scaffolded targeted tests" in event.detail
        for event in run.events
    )
    assert not any(
        event.phase == "repair_retry"
        and "Repair validation failed; retrying" in event.detail
        for event in run.events
    )


def test_supervisor_re_scaffolds_targeted_test_for_no_diff_pytest_failure(tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.diff_stat = CommandResult(True, "")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "repair timed out with no diff",
            "goal": "repair pytest failure",
        }
    ]
    backend.restore_result = CommandResult(True, "Removing tests/test_rank_s_dashboard.py")

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="run-pytest-no-diff-re-scaffold", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "pytest_failure",
        1,
    )

    assert result is True
    assert backend.commands.count("restore") == 1
    assert (tmp_path / "tests/test_rank_s_dashboard.py").exists()
    assert any(event.phase == "repair_scaffold" and event.status == "success" for event in run.events)
    assert any(
        event.phase == "repair_completion"
        and "No-diff pytest repair restored and re-scaffolded targeted tests" in event.detail
        for event in run.events
    )
    assert not any(
        event.phase == "repair_retry"
        and "Repair reasoning produced no validated diff" in event.detail
        for event in run.events
    )


def test_supervisor_retries_destructive_repair_diff_for_retryable_failure():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,6 +1,5 @@
-def create_app() -> FastAPI:
     @app.get('/api/rank/ui-card')
     async def get_rank_ui_card():
         return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_ui_card.py"],
            "final_summary": "ui repair",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.full_tests = [CommandResult(True, "baseline ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-4", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "reasoning_loop_blocked",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "destructive_diff"
        for event in run.events
    )


def test_supervisor_rejects_invalid_fastapi_bootstrap_diff_before_validation_pytest():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -56,12 +56,13 @@ def create_app() -> FastAPI:
     @app.get('/api/rank/ui-card')
     async def get_rank_ui_card():
     return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}

-    @app.get('/api/status')
-    async def api_status() -> Dict[str, object]:
-        provider, model = provider_router.choose_provider()
-        return {"provider": provider, "model": model, "safe": True}
+    return JSONResponse(content={'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'})
@@ -2798,6 +2799,10 @@ def run_app(application: FastAPI, host: str = "0.0.0.0", port: int = 7778) -> No
     @app.get('/api/rank/ui-card')
     async def get_rank_ui_card():
         return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}
+
+    @app.get('/api/rank/ui-card')
+    async def get_rank_ui_card():
+        return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_ui_card.py"],
            "final_summary": "ui repair",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.full_tests = [CommandResult(True, "baseline ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-3", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "missing_ui_visibility",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "invalid_bootstrap"
        for event in run.events
    )


def test_supervisor_re_scaffolds_targeted_tests_after_invalid_bootstrap_restore(tmp_path):
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -56,12 +56,13 @@ def create_app() -> FastAPI:
-    @app.get('/api/status')
+    return JSONResponse(content={'app': 'IGRIS_GPT', 'rank': 'S', 'status': 'ok'})
""",
    )
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "repair timed out",
            "goal": "repair pytest failure",
        }
    ]

    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="run-invalid-bootstrap-re-scaffold", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal="Add /api/rank/s-dashboard endpoint and tests/test_rank_s_dashboard.py coverage",
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "pytest_failure",
        1,
    )

    assert result is True
    assert backend.commands.count("restore") == 1
    assert (tmp_path / "tests/test_rank_s_dashboard.py").exists()
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "invalid_bootstrap"
        for event in run.events
    )
    assert any(
        event.phase == "repair_completion"
        and "restore-based retry path" in event.detail
        for event in run.events
    )


def test_supervisor_rejects_ui_test_diff_that_asserts_non_contract_keys():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_ui_card.py b/tests/test_rank_ui_card.py
@@ -1,8 +1,9 @@
 from fastapi.testclient import TestClient

 from igris.web.server import create_app


 def test_rank_ui_card_endpoint_available():
     client = TestClient(create_app())
     response = client.get("/api/rank/ui-card")
     assert response.status_code == 200
-    assert response.json() == {"app": "IGRIS_GPT", "rank": "A++", "status": "ok", "capability": "ui-visible-supervised"}
+    assert "data" in response.json()
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_ui_card.py"],
            "final_summary": "ui repair",
            "goal": "Add UI-visible rank card",
        }
    ]
    backend.full_tests = [CommandResult(True, "baseline ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-2", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "missing_ui_visibility",
        1,
    )

    assert result is True
    assert "restore" in backend.commands
    assert not any(command.startswith("tests:") for command in backend.commands)
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "wrong_file_edit"
        for event in run.events
    )


def test_async_supervisor_start_is_observable_before_work_finishes(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class SlowSupervisor:
        def __init__(self, project_root):
            self.project_root = project_root

        def run(self, config, run=None):
            assert config.defer_service_restart is True
            started.set()
            release.wait(timeout=2)
            run.status = "completed"
            run.outcome = "Completed"
            run.add("done", "success", "finished")
            return run

    import igris.core.self_repair_supervisor as mod

    RUN_STORE.clear()
    monkeypatch.setattr(mod, "SelfRepairSupervisor", SlowSupervisor)
    run = start_supervised_rank_async({"goal": "rank", "rank_id": "A"}, "/tmp/project")

    assert run.status == "running"
    assert get_supervised_run(run.run_id) is run
    assert started.wait(timeout=1)
    assert get_supervised_run(run.run_id).status == "running"
    release.set()
    deadline = time.time() + 2
    while time.time() < deadline and get_supervised_run(run.run_id).status == "running":
        time.sleep(0.01)
    assert get_supervised_run(run.run_id).status == "completed"


def test_rank_supervisor_api_dry_run_blocks_dirty_repo():
    client = TestClient(create_app())
    resp = client.post("/api/rank/run-supervised", json={"goal": "rank", "dry_run": True})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["run_id"]
    detail = client.get(f"/api/rank/runs/{data['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] in {"running", "completed", "blocked"}
    listed = client.get("/api/rank/runs")
    assert listed.status_code == 200
