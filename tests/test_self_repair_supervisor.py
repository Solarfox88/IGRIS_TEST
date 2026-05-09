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


def test_failure_classifier_detects_pytest_failure():
    failure = classify_failure(full_tests=CommandResult(False, "FAILED tests/test_x.py", "", 1))
    assert failure == "pytest_failure"


def test_failure_classifier_detects_destructive_diff():
    failure = classify_failure(diff="-def create_app():\n+def removed():\n")
    assert failure == "destructive_diff"


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

    assert run.status == "blocked"
    assert run.failure_class == "max_steps"
    assert "restore" in backend.commands
    assert sum(1 for command in backend.commands if command.startswith("branch:")) == 1
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

    assert run.status == "blocked"
    assert "restore" in backend.commands
    assert sum(1 for command in backend.commands if command.startswith("branch:")) == 1
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


def test_supervisor_passes_requested_rank_test_file_to_reasoning_context():
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(targeted_tests=["tests/test_rank_status.py"])
    )

    assert run.status == "completed"
    assert backend.last_reasoning_context["must_create_test_file"] == "tests/test_rank_status.py"
    assert backend.last_reasoning_context["expected_endpoint_file"] == "igris/web/server.py"
    assert backend.last_reasoning_context["must_not_ask_user"] is True
    assert "TestClient(create_app())" in backend.last_reasoning_context["fastapi_test_policy"]
    assert "Do not import app" in backend.last_reasoning_context["fastapi_test_policy"]
    assert "Create tests/test_rank_status.py directly" in backend.last_reasoning_context["anti_loop_instruction"]


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


def test_rank_supervisor_api_dry_run_blocks_dirty_repo(monkeypatch):
    class DirtySupervisor:
        def run(self, config):
            backend = FakeBackend()
            backend.status = CommandResult(True, " M file.py\n")
            return SelfRepairSupervisor("/tmp/project", backend=backend).run(config)

    import igris.core.self_repair_supervisor as mod

    def fake_start(data, project_root):
        run = DirtySupervisor().run(RankSupervisorConfig.from_dict(data))
        mod.RUN_STORE[run.run_id] = run
        return run

    monkeypatch.setattr(mod, "start_supervised_rank_async", fake_start)
    client = TestClient(create_app())
    resp = client.post("/api/rank/run-supervised", json={"goal": "rank", "dry_run": True})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "blocked"
    detail = client.get(f"/api/rank/runs/{data['run_id']}")
    assert detail.status_code == 200
    listed = client.get("/api/rank/runs")
    assert listed.status_code == 200
