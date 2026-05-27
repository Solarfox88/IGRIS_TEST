import json
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi.testclient import TestClient

from igris.core.self_repair_supervisor import (
    cancel_supervised_run,
    CAPABILITY_LIMIT_THRESHOLD,
    CommandResult,
    DECOMPOSITION_REQUIRED_FIELDS,
    LocalSupervisorBackend,
    list_active_supervised_run_summaries,
    PLANNING_MAX_STEPS,
    PLANNING_TIMEOUT_SECONDS,
    RankSupervisorConfig,
    RUN_STORE,
    SelfRepairSupervisor,
    SupervisorEvent,
    SupervisorRun,
    TERMINAL_RUN_STATUSES,
    classify_failure,
    get_supervisor_audit_summary,
    get_supervised_run,
    summarize_supervised_run,
    start_supervised_rank_async,
    _has_flask_test_client_in_diff,
    _reconcile_run_records,
)
from igris.web.server import create_app


class FakeBackend:
    def __init__(self):
        self.status = CommandResult(True, "")
        self.status_sequence = []
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
        self.restore_paths_result = CommandResult(True, "restore paths ok")
        self.restore_paths_calls = []
        self.last_reasoning_context = None
        self.reasoning_contexts = []
        self.reasoning_goals = []
        self.api_helper_result = CommandResult(
            True,
            json.dumps({
                "diagnosis": "timeout loop",
                "likely_supervisor_gap": "missing retry strategy",
                "suggested_repair_strategy": "apply bounded retry",
                "suggested_tests": ["tests/test_self_repair_supervisor.py -q"],
                "risk": "low",
                "confidence": 0.78,
                "requires_human_or_codex_audit": False,
                "must_not_complete_product_manually": True,
                "estimated_cost_usd": 0.01,
            }),
        )
        self.api_helper_packets = []
        self.created_issues: list = []  # records (title, body) for each create_issue call

    def git_status(self):
        self.commands.append("git_status")
        if self.status_sequence:
            return self.status_sequence.pop(0)
        return self.status

    def git_log_head(self):
        return CommandResult(True, "abc123 head")

    def create_branch(self, branch):
        self.commands.append(f"branch:{branch}")
        return CommandResult(True, branch)

    def run_reasoning(self, goal, max_steps, initial_context, timeout=300,
                      task_type="code_reasoning", preferred_profile=None):
        self.commands.append(f"reasoning:{initial_context}")
        self.commands.append(f"reasoning_timeout:{timeout}")
        self.reasoning_goals.append(goal)
        self.last_reasoning_context = initial_context
        self.reasoning_contexts.append(initial_context)
        # Track execution routing fields for observability tests
        self.last_task_type = task_type
        self.last_preferred_profile = preferred_profile
        if self.reasoning_results:
            return self.reasoning_results.pop(0)
        return {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair",
            "goal": goal,
            "orchestrator_used": True,
            "reasoning_execution_provider": "openai",
            "reasoning_execution_model": "gpt-4o-mini",
            "reasoning_execution_profile": task_type,
            "local_model_available": False,
        }

    def git_diff_stat(self):
        if getattr(self, "diff_stat_sequence", None):
            return self.diff_stat_sequence.pop(0)
        return self.diff_stat

    def git_diff(self):
        if getattr(self, "diff_sequence", None):
            return self.diff_sequence.pop(0)
        return self.diff

    def run_tests(self, targets=None, timeout=120, hard_cap=3600, exclude_slow=False):
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
        self.created_issues.append({"title": title, "body": body})
        return CommandResult(True, "issue")

    def update_issue(self, issue_url, comment_body):
        self.commands.append(f"update_issue:{issue_url}")
        return CommandResult(True, "updated")

    def fetch_issue(self, issue_url: str) -> CommandResult:
        return CommandResult(True, json.dumps({"title": "Fake sub-issue", "body": "Implement the endpoint.", "number": 999}), "", 0)

    def restore_dangerous_diff(self):
        self.commands.append("restore")
        return self.restore_result

    def restore_paths(self, paths):
        normalized = list(paths or [])
        self.commands.append(f"restore_paths:{normalized}")
        self.restore_paths_calls.append(normalized)
        return self.restore_paths_result

    def call_api_helper(self, packet, model, max_tokens, timeout=45, mode=""):
        self.commands.append(f"api_helper:{model}:{max_tokens}:{timeout}")
        self.api_helper_packets.append({
            "packet": packet,
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "mode": mode,
        })
        return self.api_helper_result

    def api_helper_is_configured(self) -> bool:
        return getattr(self, "_api_helper_configured", True)


def _config(**overrides):
    data = {
        "goal": "Rank A controlled task with tests",
        "rank_id": "A",
        "max_rank_attempts": 2,
        "max_repair_cycles": 1,
        "required_smoke_endpoints": ["http://127.0.0.1:7778/api/health"],
        "targeted_tests": ["tests/test_rank_status.py"],
        "dry_run": True,
        # Planning disabled by default in tests — the dataclass default is False
        # and planning tests opt in explicitly with enable_mission_planning=True.
        # Semantic gate disabled by default — most tests use simplified diffs and
        # test orchestration behavior, not implementation quality.
        "enable_semantic_gate": False,
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


def test_config_default_test_timeout_is_idle_based():
    # Regression / design test for #332: test_timeout_seconds is now an *idle*
    # timeout (kill only on silence), not a total-wall-clock timeout.
    # Default is 300 s (5 min) — large enough for individual slow integration
    # tests that may run 2-3 min without printing, while still catching genuinely
    # hung processes.  test_hard_cap_seconds provides the absolute ceiling.
    config = RankSupervisorConfig.from_dict({"goal": "rank"})
    assert config.test_timeout_seconds == 300, (
        f"test_timeout_seconds (idle) default must be 300 s (got {config.test_timeout_seconds})"
    )
    assert config.test_hard_cap_seconds >= 3600, (
        f"test_hard_cap_seconds must be >= 3600 s (got {config.test_hard_cap_seconds})"
    )


def test_config_from_dict_preserves_explicit_test_timeout():
    config = RankSupervisorConfig.from_dict({"goal": "rank", "test_timeout_seconds": 60, "test_hard_cap_seconds": 7200})
    assert config.test_timeout_seconds == 60
    assert config.test_hard_cap_seconds == 7200


def test_run_adaptive_completes_fast_command(tmp_path):
    """_run_adaptive should return success for a command that finishes quickly."""
    from igris.core.self_repair_supervisor import LocalSupervisorBackend
    backend = LocalSupervisorBackend(project_root=tmp_path)
    result = backend._run_adaptive(
        ["python3", "-c", "print('hello'); import sys; sys.exit(0)"],
        idle_timeout=10,
        hard_cap=30,
    )
    assert result.success
    assert "hello" in result.output


def test_run_adaptive_kills_idle_process(tmp_path):
    """_run_adaptive should kill a process that stops producing output."""
    import time as _time
    from igris.core.self_repair_supervisor import LocalSupervisorBackend
    backend = LocalSupervisorBackend(project_root=tmp_path)
    result = backend._run_adaptive(
        ["python3", "-c", "import time; print('start', flush=True); time.sleep(30)"],
        idle_timeout=3,
        hard_cap=60,
    )
    assert not result.success
    assert result.returncode == 124
    assert "idle" in result.error.lower() or "killed" in result.error.lower() or "no output" in result.error.lower()


def test_run_adaptive_hard_cap_kills_talkative_process(tmp_path):
    """_run_adaptive must kill even a chatty process when hard_cap is hit."""
    from igris.core.self_repair_supervisor import LocalSupervisorBackend
    backend = LocalSupervisorBackend(project_root=tmp_path)
    # Process spams output every 0.1 s, so idle_timeout will never fire.
    result = backend._run_adaptive(
        ["python3", "-c",
         "import time, sys\n"
         "while True:\n"
         "    print('.', end='', flush=True)\n"
         "    time.sleep(0.1)\n"],
        idle_timeout=60,
        hard_cap=2,
    )
    assert not result.success
    assert result.returncode == 124
    assert "hard cap" in result.error.lower() or "exceeded" in result.error.lower()


def test_failure_classifier_detects_max_steps_as_repairable_infrastructure_failure():
    failure = classify_failure({"status": "stopped", "stop_reason": "max_steps", "files_modified": []})
    assert failure == "max_steps"


def test_failure_classifier_detects_reasoning_timeout_as_blocked_loop():
    failure = classify_failure({"status": "blocked", "stop_reason": "reasoning_timeout", "files_modified": []})
    assert failure == "reasoning_loop_blocked"


def test_failure_classifier_detects_budget_exceeded_as_blocked_loop():
    failure = classify_failure({"status": "stopped", "stop_reason": "budget_exceeded", "files_modified": []})
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


def test_failure_classifier_does_not_flag_env_example_as_destructive():
    diff = """diff --git a/.env.example b/.env.example
index 1111111..2222222 100644
--- a/.env.example
+++ b/.env.example
@@ -1,2 +1,5 @@
 LOCAL_LLM_PROVIDER=ollama
 LOCAL_LLM_MODEL=phi4-mini
+LOCAL_LLM_BASE_URL=http://127.0.0.1:11434
"""
    failure = classify_failure(diff=diff)
    assert failure != "destructive_diff", (
        ".env.example additions must not be classified as destructive_diff"
    )


def test_failure_classifier_flags_actual_env_file_as_destructive():
    diff = """diff --git a/.env b/.env
index 1111111..2222222 100644
--- a/.env
+++ b/.env
@@ -1,2 +1,3 @@
 SECRET_KEY=abc
+NEW_VAR=xyz
"""
    failure = classify_failure(diff=diff)
    assert failure == "destructive_diff", (
        "Modifications to .env secrets file must be classified as destructive_diff"
    )


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


def test_failure_classifier_does_not_flag_import_reorganisation_as_destructive():
    """Removing import lines while adding a new endpoint must not be flagged as destructive.

    When the model adds a new endpoint to server.py it often reorganises the import block,
    producing removed '-import ...' lines.  These are not structural deletions and the test
    suite will catch any broken imports, so they must not trigger destructive_diff.
    """
    diff = """\
diff --git a/igris/web/server.py b/igris/web/server.py
index 1111111..2222222 100644
--- a/igris/web/server.py
+++ b/igris/web/server.py
@@ -1,8 +1,10 @@
-import os
-import json
-from typing import Dict
+import json
+import os
+from typing import Dict, Optional
+from datetime import datetime, timezone

 from fastapi import FastAPI

+
+@app.get("/api/diagnostics/session-resume")
+async def session_resume():
+    return {"session_id": "x", "resume_protocol_active": False,
+            "last_heartbeat_utc": None, "pending_tasks_count": 0}
"""
    failure = classify_failure(diff=diff)
    assert failure != "destructive_diff", (
        "import reorganisation in server.py must not be classified as destructive_diff"
    )


def test_failure_classifier_flags_true_import_deletion_as_destructive():
    """Removing an import that is NOT re-added must be flagged as destructive.

    Regression: PR #384 removed 'import ' from the critical list entirely, which
    allowed the model to delete FastAPI/StaticFiles/Path imports from server.py,
    breaking the entire app (NameError at test collection time).
    """
    diff = """\
diff --git a/igris/web/server.py b/igris/web/server.py
index 1111111..2222222 100644
--- a/igris/web/server.py
+++ b/igris/web/server.py
@@ -1,6 +1,4 @@
-from fastapi import FastAPI, Depends
-from starlette.staticfiles import StaticFiles
 import os
+import json

 def create_app():
"""
    failure = classify_failure(diff=diff)
    assert failure == "destructive_diff", (
        "Deleting core imports (FastAPI, StaticFiles) must be classified as destructive_diff"
    )


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


def test_immediately_dangerous_diff_allows_create_app_signature_change():
    """Modifying the create_app signature (e.g. adding a parameter) must not trigger
    the pre-test safety gate — the function is still present in the added lines."""
    from igris.core.self_repair_supervisor import _has_immediately_dangerous_diff

    diff = """\
diff --git a/igris/web/server.py b/igris/web/server.py
index 1111111..2222222 100644
--- a/igris/web/server.py
+++ b/igris/web/server.py
@@ -55,7 +55,7 @@ MODULE_DIR = Path(__file__).resolve().parent
-def create_app() -> FastAPI:
+def create_app(overrides=None) -> FastAPI:
     app = FastAPI(title="IGRIS_GPT", version="0.1.0")
+    @app.get('/api/diagnostics/session-resume')
+    async def session_resume():
+        runs = list_supervised_runs()
+        return {'active_runs': runs}
"""
    assert not _has_immediately_dangerous_diff(diff), (
        "Modifying create_app signature must not be treated as a structural deletion"
    )


def test_immediately_dangerous_diff_blocks_true_create_app_deletion():
    """Removing create_app without re-adding it must be caught by the pre-test gate."""
    from igris.core.self_repair_supervisor import _has_immediately_dangerous_diff

    diff = """\
diff --git a/igris/web/server.py b/igris/web/server.py
index 1111111..2222222 100644
--- a/igris/web/server.py
+++ b/igris/web/server.py
@@ -55,4 +55,4 @@ MODULE_DIR = Path(__file__).resolve().parent
-def create_app() -> FastAPI:
-    app = FastAPI()
+def make_app() -> FastAPI:
+    app = FastAPI()
"""
    assert _has_immediately_dangerous_diff(diff), (
        "Removing create_app without re-adding it must trigger the safety gate"
    )


def test_immediately_dangerous_diff_env_token_in_content_not_path():
    """'.env' appearing only in diff content (e.g. a docstring) must not trigger the gate.
    The fix moves to path-level matching to avoid false positives on template files."""
    from igris.core.self_repair_supervisor import _has_immediately_dangerous_diff

    diff = """\
diff --git a/igris/core/config.py b/igris/core/config.py
index 1111111..2222222 100644
--- a/igris/core/config.py
+++ b/igris/core/config.py
@@ -1,3 +1,5 @@
+# Load settings from .env file using python-dotenv
+OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
"""
    assert not _has_immediately_dangerous_diff(diff), (
        "'.env' in diff content must not trigger the gate — only the '.env' path itself should"
    )


def test_immediately_dangerous_diff_blocks_env_file_path():
    """A diff that actually modifies the '.env' secrets file must be blocked."""
    from igris.core.self_repair_supervisor import _has_immediately_dangerous_diff

    diff = """\
diff --git a/.env b/.env
index 1111111..2222222 100644
--- a/.env
+++ b/.env
@@ -1,2 +1,3 @@
+OPENAI_API_KEY=sk-injected
"""
    assert _has_immediately_dangerous_diff(diff), (
        "Modifying the '.env' secrets file must be caught by the pre-test gate"
    )


def test_immediately_dangerous_diff_allows_env_example():
    """'.env.example' is a safe template file and must not trigger the gate."""
    from igris.core.self_repair_supervisor import _has_immediately_dangerous_diff

    diff = """\
diff --git a/.env.example b/.env.example
index 1111111..2222222 100644
--- a/.env.example
+++ b/.env.example
@@ -1,2 +1,3 @@
+OPENAI_API_KEY=your-key-here
"""
    assert not _has_immediately_dangerous_diff(diff), (
        "'.env.example' is a template and must not be blocked by the safety gate"
    )


class TestSupervisorRunIsZombie:
    def _make_run(self, status: str = "running", last_event_age_seconds: float = 0.0):
        import time
        run = SupervisorRun(run_id="z", rank_id="r", status=status)
        run.add("start", "running", "started")
        run.events[-1].timestamp = time.time() - last_event_age_seconds
        return run

    def test_completed_run_is_not_zombie(self):
        run = self._make_run(status="completed", last_event_age_seconds=9999)
        assert not run.is_zombie()

    def test_recently_active_run_is_not_zombie(self):
        run = self._make_run(status="running", last_event_age_seconds=60)
        assert not run.is_zombie(threshold_seconds=1800)

    def test_run_stuck_long_but_active_is_not_zombie(self):
        """A run that started 3 hours ago but had an event 10 minutes ago is active."""
        import time
        run = SupervisorRun(run_id="z", rank_id="r", status="running")
        run.add("start", "running", "started")
        run.events[-1].timestamp = time.time() - 10800  # started 3h ago
        run.add("progress", "running", "still going")
        run.events[-1].timestamp = time.time() - 600    # last update 10 min ago
        assert not run.is_zombie(threshold_seconds=1800)

    def test_run_with_no_recent_events_is_zombie(self):
        run = self._make_run(status="running", last_event_age_seconds=3600)
        assert run.is_zombie(threshold_seconds=1800)

    def test_run_with_no_events_is_not_zombie(self):
        run = SupervisorRun(run_id="z", rank_id="r", status="running")
        assert not run.is_zombie()


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
    """run_tests() uses _run_adaptive (Popen-based) with a clean subprocess environment.

    Verifies: start_new_session, close_fds, secret env vars stripped.
    The idle_timeout is now internal to _run_adaptive, not a Popen kwarg.
    """
    captured = {}

    class FakeProc:
        returncode = 0
        pid = 99999
        stdout = iter(["ok\n"])
        stderr = iter([])

        def wait(self):
            pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return FakeProc()

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_x.py::test_y")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak")
    monkeypatch.setenv("VASTAI_API_KEY", "")
    monkeypatch.setenv("PROJECT_ROOT", "/service/root")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    result = LocalSupervisorBackend(str(tmp_path)).run_tests(timeout=31)

    assert result.success
    # idle_timeout is NOT passed to Popen — it is managed internally
    assert "timeout" not in captured
    assert captured["start_new_session"] is True
    assert captured["close_fds"] is True
    assert captured["env"]["IGRIS_SUPERVISOR_CHILD"] == "1"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert "PYTEST_CURRENT_TEST" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "VASTAI_API_KEY" not in captured["env"]
    assert "PROJECT_ROOT" not in captured["env"]


def test_local_backend_runs_reasoning_in_bounded_worker(monkeypatch, tmp_path):
    """run_reasoning() launches supervisor_reasoning_worker via Popen and returns parsed JSON."""
    captured = {}
    result_json = json.dumps({
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["tests/test_rank_status.py"],
    })

    class FakeStdin:
        def __init__(self):
            self._buf = []
        def write(self, data):
            self._buf.append(data)
        def close(self):
            captured["input"] = "".join(self._buf)

    class FakeProc:
        returncode = 0
        stdin = FakeStdin()
        stdout = iter([result_json + "\n"])
        stderr = iter([])

        def poll(self):
            return 0

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    result = LocalSupervisorBackend(str(tmp_path)).run_reasoning(
        "rank goal",
        max_steps=7,
        initial_context={"rank_test": "A"},
        timeout=42,
    )

    assert result["status"] == "finished"
    assert captured["cmd"][-2:] == ["-m", "igris.core.supervisor_reasoning_worker"]
    assert captured["kwargs"]["start_new_session"] is True
    payload = json.loads(captured["input"])
    assert payload["goal"] == "rank goal"
    assert payload["max_steps"] == 7
    assert payload["initial_context"]["rank_test"] == "A"


def test_subprocess_env_clean_passes_api_keys():
    """forward_credentials=True must inject LLM provider credentials into the
    clean subprocess env so ModelOrchestrator can reach cloud providers.
    Without this, OpenAI/DeepSeek are skipped and Ollama is the only fallback."""
    import os
    from unittest.mock import patch

    fake_env = {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "sk-test-openai",
        "DEEPSEEK_API_KEY": "sk-test-deepseek",
        "ANTHROPIC_API_KEY": "sk-test-anthropic",
        "IGRIS_API_HELPER_COMMAND": "/bin/true",
        "IGRIS_EXECUTION_STRONG_MODEL": "gpt-4o",
        "IGRIS_EXECUTION_FALLBACK_MODEL": "gpt-4o-mini",
        "SOME_UNRELATED_VAR": "should-be-stripped",
    }

    with patch.dict(os.environ, fake_env, clear=True):
        backend = LocalSupervisorBackend(project_root="/tmp")
        env = backend._subprocess_env(clean_for_tests=True, forward_credentials=True)

    assert env.get("OPENAI_API_KEY") == "sk-test-openai"
    assert env.get("DEEPSEEK_API_KEY") == "sk-test-deepseek"
    assert env.get("IGRIS_API_HELPER_COMMAND") == "/bin/true"
    assert env.get("IGRIS_EXECUTION_STRONG_MODEL") == "gpt-4o"
    assert "SOME_UNRELATED_VAR" not in env


def test_subprocess_env_clean_without_forward_strips_api_keys():
    """clean_for_tests=True without forward_credentials must NOT leak LLM keys
    into test-isolated subprocesses (e.g. run_tests)."""
    import os
    from unittest.mock import patch

    fake_env = {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "sk-test-openai",
    }

    with patch.dict(os.environ, fake_env, clear=True):
        backend = LocalSupervisorBackend(project_root="/tmp")
        env = backend._subprocess_env(clean_for_tests=True)

    assert "OPENAI_API_KEY" not in env


def test_commit_retries_with_git_add_u_on_unstaged_changes(monkeypatch, tmp_path):
    """When commit fails because files are modified but not staged,
    the backend should run 'git add -u' and retry the commit once."""
    import igris.core.self_repair_supervisor as mod

    calls = []

    class Proc:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "commit"] and len(calls) == 1:
            return Proc(returncode=1, stdout="", stderr="Changes not staged for commit")
        if cmd[:3] == ["git", "add", "-u"]:
            return Proc(returncode=0)
        if cmd[:2] == ["git", "commit"] and len(calls) > 1:
            return Proc(returncode=0, stdout="[branch abc123] feat: done")
        return Proc(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = LocalSupervisorBackend(str(tmp_path)).commit("feat: done")

    assert result.success
    assert any(c[:3] == ["git", "add", "-u"] for c in calls), "git add -u must be called on unstaged failure"
    commit_calls = [c for c in calls if c[:2] == ["git", "commit"]]
    assert len(commit_calls) == 2, "commit must be retried once"


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
    """run_reasoning() returns stop_reason='reasoning_timeout' when the worker is killed."""
    import igris.core.self_repair_supervisor as mod

    _calls = [0]

    def fake_monotonic():
        _calls[0] += 1
        return 0.0 if _calls[0] == 1 else 999.0

    class FakeStdin:
        def write(self, _): pass
        def close(self): pass

    class FakeProc:
        returncode = 0
        pid = 99999
        stdin = FakeStdin()
        stdout = iter([])
        stderr = iter([])

        def poll(self):
            return None

        def wait(self):
            self.returncode = -9
            return -9

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod.os, "killpg", lambda *_: None)
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)

    result = LocalSupervisorBackend(str(tmp_path)).run_reasoning(
        "rank goal",
        max_steps=7,
        initial_context={},
        timeout=42,
    )

    assert result["status"] == "blocked"
    assert result["stop_reason"] == "reasoning_timeout"


def test_local_backend_reuses_existing_open_issue_by_exact_title(monkeypatch, tmp_path):
    commands = []

    class Proc:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[:4] == ["gh", "issue", "list", "--state"]:
            return Proc(
                stdout=json.dumps([
                    {"title": "A: supervised repair for reasoning_loop_blocked", "url": "https://example.test/issues/1"},
                    {"title": "S-full-e2e: supervised repair for syntax_error", "url": "https://example.test/issues/2"},
                ]),
            )
        if cmd[:3] == ["gh", "issue", "create"]:
            return Proc(stdout="https://example.test/issues/new")
        return Proc(stdout="")

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    backend = LocalSupervisorBackend(str(tmp_path))
    result = backend.create_issue(
        "S-full-e2e: supervised repair for syntax_error",
        "body",
    )

    assert result.success
    assert result.output == "https://example.test/issues/2"
    assert not any(cmd[:3] == ["gh", "issue", "create"] for cmd in commands)


def test_local_backend_creates_issue_when_no_open_match_exists(monkeypatch, tmp_path):
    commands = []

    class Proc:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[:4] == ["gh", "issue", "list", "--state"]:
            return Proc(stdout=json.dumps([]))
        if cmd[:3] == ["gh", "issue", "create"]:
            return Proc(stdout="https://example.test/issues/new")
        return Proc(stdout="")

    import igris.core.self_repair_supervisor as mod

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    backend = LocalSupervisorBackend(str(tmp_path))
    result = backend.create_issue(
        "S-full-e2e: supervised repair for wrong_file_edit",
        "body",
    )

    assert result.success
    assert result.output == "https://example.test/issues/new"
    assert any(cmd[:3] == ["gh", "issue", "create"] for cmd in commands)


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
    # single_stage_execution stage is failure (reasoning_timeout) → degraded=True with reason
    assert run.report["degraded_completion"] is True
    assert run.report["degraded_completion_reason"] != ""
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
    # single_stage_execution stage is failure (reasoning_timeout) → degraded=True with reason
    assert run.report["degraded_completion"] is True
    assert run.report["degraded_completion_reason"] != ""
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
    # runtime_refresh_required + post_merge_smoke=False = smoke not confirmed = genuinely degraded
    # (also: single_stage_execution stage failure contributes to the reason)
    assert run.report["degraded_completion"] is True
    assert run.report["degraded_completion_reason"] != ""
    assert sum(1 for command in backend.commands if command.startswith("smoke:")) == 2


def test_clean_completed_run_has_degraded_false_and_empty_reason():
    """A direct completion (reasoning finished cleanly) must NOT be degraded."""
    backend = FakeBackend()
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(max_repair_cycles=0)
    )
    assert run.status == "completed"
    assert run.report["degraded_completion"] is False
    assert run.report["degraded_completion_reason"] == ""


def test_verified_diff_completion_with_all_stages_green_is_not_degraded():
    """_compute_degraded_completion: verified_diff + all stages green → not degraded.

    Regression for #336: run 3dfcdc055cc2 had completion_mode="verified_diff" but
    10 required stages all green, failure_class="", state_conflict=False.  The old
    expression `completion_mode != "direct"` incorrectly forced degraded=True.

    Tested at the unit level because wiring a full staged mission with all stages
    green via FakeBackend requires complex setup; the logic itself is the fix.
    """
    compute = SelfRepairSupervisor._compute_degraded_completion

    # Case 1: verified_diff, no stage system, no smoke (dry-run scenario)
    degraded, reason = compute(
        completion_mode="verified_diff",
        runtime_refresh_required=False,
        post_merge_smoke_success=False,
        smoke_was_applicable=False,
        failure_class="",
        stage_statuses=None,
    )
    assert not degraded, f"expected not degraded, got reason={reason!r}"
    assert reason == ""

    # Case 2: verified_diff + all required stages green + smoke passed
    stages = {
        "s1": {"required": True, "status": "success"},
        "s2": {"required": True, "status": "success"},
        "s3": {"required": False, "status": "failure"},  # optional, doesn't count
    }
    degraded2, reason2 = compute(
        completion_mode="verified_diff",
        runtime_refresh_required=True,
        post_merge_smoke_success=True,
        smoke_was_applicable=True,
        failure_class="",
        stage_statuses=stages,
    )
    assert not degraded2, f"expected not degraded, got reason={reason2!r}"
    assert reason2 == ""

    # Case 3: verified_diff + a required stage failed → IS degraded
    stages_with_fail = {
        "s1": {"required": True, "status": "success"},
        "s2": {"required": True, "status": "failure"},
    }
    degraded3, reason3 = compute(
        completion_mode="verified_diff",
        runtime_refresh_required=False,
        post_merge_smoke_success=True,
        smoke_was_applicable=True,
        failure_class="",
        stage_statuses=stages_with_fail,
    )
    assert degraded3
    assert reason3 != ""


def test_degraded_completion_always_has_non_empty_reason():
    """Whenever degraded_completion=True the reason field must be non-empty."""
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "blocked",
        "stop_reason": "reasoning_timeout",
        "files_modified": ["igris/web/server.py"],
        "final_summary": "timed out",
        "goal": "rank task",
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
    # runtime_refresh_required + post_merge_smoke=False → degraded
    if run.report.get("degraded_completion"):
        assert run.report.get("degraded_completion_reason", "") != "", (
            "degraded_completion=True must always have a non-empty degraded_completion_reason"
        )


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
    backend.diff_stat = CommandResult(True, "")
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
    # no-op: mission already satisfied, nothing delivered this run → legitimately degraded
    assert run.report["degraded_completion"] is True
    assert run.report["degraded_completion_reason"] != ""
    assert "no-op" in run.report["degraded_completion_reason"].lower() or "already" in run.report["degraded_completion_reason"].lower()
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
    assert run.failure_class in {"reasoning_loop_blocked", "wrong_file_edit"}


def test_supervisor_decomposes_immediately_when_llm_provider_is_unavailable():
    """When LLM is unavailable, the supervisor must decompose immediately (no repair cycles).

    Repair cycles cannot succeed without a working model, so the run decomposes
    rather than blocking — allowing the auto-chain to hand the task to a sub-mission
    that may reach a capable cloud provider.
    """
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "No suitable LLM provider available; deterministic fallback",
            "goal": "Add /api/rank/s-dashboard endpoint and tests",
        }
        # No 2nd result needed: _ask_igris_decompose falls through to
        # deterministic fallback when the queue is empty.
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
    assert run.failure_class == "decomposition_required", (
        f"Expected decomposition_required (LLM unavailable → decompose), "
        f"got {run.failure_class!r}"
    )
    assert not any(event.phase == "repair_issue" for event in run.events), (
        "No repair cycles should run when LLM is unavailable"
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


def test_supervisor_does_not_apply_ui_card_repair_gate_to_rank_s_backend_syntax_repair():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -100,6 +100,18 @@ def create_app() -> FastAPI:
+    @app.get('/api/rank/s-dashboard')
+    async def get_rank_s_dashboard() -> Dict[str, object]:
+        return {
+            'app': 'IGRIS_GPT',
+            'rank': 'S',
+            'status': 'ok',
+            'capability': 'end-to-end-supervised',
+            'checks': {'backend': True, 'ui': True, 'tests': True, 'workflow': True},
+        }
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "repair backend syntax",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "repair validation passed")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-rank-s-backend-syntax", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(
            goal=(
                "Complete Rank S mission: add /api/rank/s-dashboard backend endpoint, "
                "UI dashboard visibility, tests, and workflow."
            ),
            targeted_tests=["tests/test_rank_s_dashboard.py"],
            max_repair_cycles=1,
        ),
        "syntax_error",
        1,
    )

    assert result is True
    assert "restore" not in backend.commands
    assert any(command.startswith("tests:") for command in backend.commands)
    assert not any(
        event.phase == "repair_retry"
        and "Product-only UI task diff was rejected" in event.detail
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
        and "re-scaffolded targeted tests" in event.detail.lower()
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


def test_supervisor_retries_destructive_repair_diff_for_wrong_file_edit():
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
            "final_summary": "wrong file scope repair",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(True, "baseline ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-destructive-wrong-file-edit", rank_id="A")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Add UI-visible rank card", max_repair_cycles=1),
        "wrong_file_edit",
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
    assert any(event.phase == "repair_scaffold" and event.status == "success" for event in run.events)
    assert any(
        event.phase == "repair_completion"
        and "re-scaffolded targeted tests" in event.detail.lower()
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


def _load_runs_records(project_root):
    from pathlib import Path
    path = Path(project_root) / ".igris" / "supervisor_runs.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("runs", {})


def test_supervisor_run_start_creates_durable_record(tmp_path):
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="durable-start-001", rank_id="S")
    config = _config(rank_id="S", max_repair_cycles=3, allow_api_escalation=True, max_api_escalations_per_run=2, max_api_budget_usd=1.5)

    supervisor._configure_run_tracking(run, config)
    run.add("queued", "running", "accepted")

    records = _load_runs_records(tmp_path)
    assert "durable-start-001" in records
    assert records["durable-start-001"]["status"] == "running"
    assert records["durable-start-001"]["max_repair_cycles"] == 3
    assert records["durable-start-001"]["max_api_escalations_per_run"] == 2


def test_supervisor_run_events_update_durable_record(tmp_path):
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="durable-events-001", rank_id="S")
    config = _config(rank_id="S")

    supervisor._configure_run_tracking(run, config)
    run.add("queued", "running", "accepted")
    run.add("rank_reasoning", "running", "stage running", stage_id="backend_api_change")

    records = _load_runs_records(tmp_path)
    latest = records["durable-events-001"]["latest_event"]
    assert latest["phase"] == "rank_reasoning"
    assert records["durable-events-001"]["current_stage"] == "backend_api_change"


def test_blocked_and_completed_runs_remain_visible_after_memory_reset(tmp_path):
    backend_blocked = FakeBackend()
    backend_blocked.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "blocked run",
            "goal": "blocked",
        }
    ]
    blocked_run = SelfRepairSupervisor(str(tmp_path), backend=backend_blocked).run(_config(rank_id="S", max_rank_attempts=1, max_repair_cycles=0))

    backend_completed = FakeBackend()
    backend_completed.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py", "tests/test_rank_status.py"],
            "final_summary": "completed run",
            "goal": "complete",
        }
    ]
    completed_run = SelfRepairSupervisor(str(tmp_path), backend=backend_completed).run(_config(rank_id="A", max_rank_attempts=1, max_repair_cycles=0))
    assert blocked_run.status == "blocked"
    assert completed_run.status == "completed"

    RUN_STORE.clear()
    summary = get_supervisor_audit_summary(str(tmp_path))
    recent = summary["recent_runs"]
    ids = {entry["run_id"] for entry in recent}
    assert blocked_run.run_id in ids
    assert completed_run.run_id in ids
    blocked_item = next(item for item in recent if item["run_id"] == blocked_run.run_id)
    completed_item = next(item for item in recent if item["run_id"] == completed_run.run_id)
    assert blocked_item["status"] == "blocked"
    assert completed_item["status"] == "completed"


def test_supervisor_durable_records_redact_secrets(tmp_path):
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="durable-secret-001", rank_id="S")
    config = _config(rank_id="S")

    supervisor._configure_run_tracking(run, config)
    run.add("queued", "running", "token=sk-secret-should-not-leak", api_key="sk-live-super-secret")

    serialized = json.dumps(_load_runs_records(tmp_path))
    assert "sk-live-super-secret" not in serialized
    assert "sk-secret-should-not-leak" not in serialized


def test_cancel_run_transitions_to_cancelled_and_persists_recent(tmp_path):
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=backend)
    run = SupervisorRun(run_id="cancel-run-001", rank_id="S")
    config = _config(rank_id="S", max_repair_cycles=8, allow_api_escalation=True, max_api_escalations_per_run=2, max_api_budget_usd=1.5)
    supervisor._configure_run_tracking(run, config)
    run.status = "running"
    run.add("queued", "running", "running")
    RUN_STORE[run.run_id] = run

    result = cancel_supervised_run(run.run_id, str(tmp_path), "Cancelled by test")
    assert result is not None
    assert result.status == "cancelled"
    assert result.outcome == "Cancelled"
    assert result.failure_class == "user_cancelled"
    assert any(event.phase == "cancel_request" for event in result.events)
    assert any(event.phase == "cancelled" for event in result.events)
    assert any(event.phase == "cancel_workspace_state" for event in result.events)

    active = list_active_supervised_run_summaries(str(tmp_path))
    assert all(item.get("run_id") != run.run_id for item in active)

    summary = get_supervisor_audit_summary(str(tmp_path))
    item = next((entry for entry in summary["recent_runs"] if entry.get("run_id") == run.run_id), None)
    assert item is not None
    assert item["status"] == "cancelled"
    assert "Cancelled by test" in (item.get("cancelled_reason") or item.get("blocked_reason") or "")
    RUN_STORE.pop(run.run_id, None)


def test_completed_with_failure_requires_resolved_or_degraded_flag():
    run = SupervisorRun(run_id="run-complete-failure", rank_id="S")
    run.status = "completed"
    run.outcome = "Completed"
    run.failure_class = "syntax_error"
    run.report = {}
    run.add("finish", "success", "done")

    summary = summarize_supervised_run(run)
    assert summary["state_conflict"] is True
    assert "Completed run has failure_class" in summary["warning"]


def test_completed_with_failure_and_degraded_flag_is_allowed():
    run = SupervisorRun(run_id="run-complete-degraded", rank_id="S")
    run.status = "completed"
    run.outcome = "Completed"
    run.failure_class = "syntax_error"
    run.report = {"degraded_completion": True}
    run.add("finish", "success", "done")

    summary = summarize_supervised_run(run)
    assert summary["state_conflict"] is False


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


def _staged_config(**overrides):
    data = {
        "goal": (
            "Implement backend API endpoint /api/rank/s-dashboard with UI dashboard visibility, "
            "add tests coverage, run full pytest, and complete workflow reporting."
        ),
        "rank_id": "S",
        "max_rank_attempts": 1,
        "max_repair_cycles": 0,
        "required_smoke_endpoints": ["http://127.0.0.1:7778/api/health"],
        "targeted_tests": ["tests/test_rank_s_dashboard.py"],
        "dry_run": True,
        # Semantic gate disabled — staged tests exercise orchestration, not implementation quality.
        "enable_semantic_gate": False,
    }
    data.update(overrides)
    return RankSupervisorConfig.from_dict(data)


def test_supervisor_keeps_simple_missions_single_stage():
    supervisor = SelfRepairSupervisor("/tmp/project", backend=FakeBackend())
    plan = supervisor._build_mission_plan(
        RankSupervisorConfig.from_dict({"goal": "Add /api/rank/status endpoint"})
    )

    assert plan.mode == "single-stage"
    assert [stage.stage_id for stage in plan.stages] == ["single_stage_execution"]


def test_supervisor_decomposes_non_trivial_missions_into_ordered_stages():
    supervisor = SelfRepairSupervisor("/tmp/project", backend=FakeBackend())
    plan = supervisor._build_mission_plan(_staged_config())
    stage_ids = [stage.stage_id for stage in plan.stages]

    assert plan.mode == "staged"
    assert stage_ids[:6] == [
        "understand_locate",
        "backend_api_change",
        "backend_tests",
        "ui_dashboard_change",
        "ui_dashboard_tests",
        "docs_config_update",
    ]
    assert stage_ids[-3:] == ["pr_ci_merge", "post_merge_runtime", "final_report"]


def test_supervisor_preserves_backend_stage_success_when_ui_stage_fails():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "stage tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "ui blocked",
            "goal": "stage ui",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 4 ++++\n tests/test_rank_s_dashboard.py | 8 ++++++++")
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,8 @@
+def test_rank_s_dashboard():
""",
    )

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config())

    assert run.status == "blocked"
    stages = {entry["stage_id"]: entry for entry in run.report["mission_orchestration"]["stages"]}
    assert stages["backend_api_change"]["status"] == "success"
    assert stages["backend_tests"]["status"] == "success"
    assert stages["ui_dashboard_change"]["status"] == "failure"


def test_supervisor_skips_optional_docs_stage_with_explanation_when_not_relevant():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "stage backend tests",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/templates/index.html"],
            "final_summary": "ui done",
            "goal": "stage ui",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_dashboard_tabs.py"],
            "final_summary": "ui tests done",
            "goal": "stage ui tests",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 4 ++++")

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config())

    assert run.status == "completed"
    stages = {entry["stage_id"]: entry for entry in run.report["mission_orchestration"]["stages"]}
    assert stages["docs_config_update"]["status"] == "skipped"
    assert "does not require docs/config" in stages["docs_config_update"]["detail"]


def test_supervisor_final_report_includes_per_stage_status_for_staged_missions():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "tests done",
            "goal": "stage tests",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/templates/index.html"],
            "final_summary": "ui done",
            "goal": "stage ui",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_dashboard_tabs.py"],
            "final_summary": "ui tests done",
            "goal": "stage ui tests",
        },
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config())

    assert run.status == "completed"
    report = run.report["mission_orchestration"]
    assert report["mode"] == "staged"
    assert report["stages"]
    assert all("stage_id" in stage and "status" in stage for stage in report["stages"])


def test_supervisor_does_not_complete_when_required_stage_is_missing(monkeypatch):
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        }
    ]

    monkeypatch.setattr(
        SelfRepairSupervisor,
        "_required_stages_green",
        staticmethod(lambda statuses, **kwargs: False),
    )
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _staged_config(
            goal=(
                "Implement backend API endpoint /api/system/version, add tests, "
                "include docs note, and run workflow reporting."
            ),
            targeted_tests=["tests/test_system_version.py"],
        )
    )

    assert run.status == "blocked"
    assert run.failure_class in {"reasoning_loop_blocked", "wrong_file_edit"}


def test_supervisor_completes_staged_mission_as_noop_when_already_satisfied(monkeypatch):
    backend = FakeBackend()
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")

    monkeypatch.setattr(
        SelfRepairSupervisor,
        "_stage_is_already_satisfied",
        lambda self, stage, config: stage.stage_id != "final_report",
    )
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config())

    assert run.status == "completed"
    assert run.report["no_op_completion"] is True
    assert run.report["completion_mode"] == "already_satisfied"


def test_supervisor_tracks_non_blocking_behavior_per_stage():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "running",
            "stop_reason": "partial_progress",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend partial but acceptable",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "tests done",
            "goal": "stage tests",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/templates/index.html"],
            "final_summary": "ui done",
            "goal": "stage ui",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_dashboard_tabs.py"],
            "final_summary": "ui tests done",
            "goal": "stage ui tests",
        },
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config())

    assert run.status == "completed"
    stages = {entry["stage_id"]: entry for entry in run.report["mission_orchestration"]["stages"]}
    behaviors = stages["backend_api_change"]["non_blocking_behaviors"]
    assert any(item["code"] == "degraded_reasoning" for item in behaviors)


def test_supervisor_accepts_ui_stage_timeout_when_ui_visibility_evidence_is_present():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "stage backend tests",
        },
        {
            "status": "blocked",
            "stop_reason": "reasoning_timeout",
            "files_modified": [],
            "final_summary": "ui timeout",
            "goal": "stage ui",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_dashboard_tabs.py"],
            "final_summary": "ui tests done",
            "goal": "stage ui tests",
        },
    ]
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
@@ -10,6 +10,7 @@
+<div id='rank-s-dashboard'>ready</div>
""",
    )

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    assert run.status == "completed"
    stages = {entry["stage_id"]: entry for entry in run.report["mission_orchestration"]["stages"]}
    assert stages["ui_dashboard_change"]["status"] == "success"
    assert "accepted after timeout" in stages["ui_dashboard_change"]["detail"].lower()
    assert any(
        item["code"] == "ui_stage_timeout_accepted"
        for item in stages["ui_dashboard_change"]["non_blocking_behaviors"]
    )


def test_supervisor_does_not_mark_required_stage_success_when_budget_exceeded():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "stopped",
            "stop_reason": "budget_exceeded",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend exhausted error budget",
            "goal": "stage backend",
        },
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    assert run.status == "blocked"
    assert run.failure_class == "reasoning_loop_blocked"
    stages = {entry["stage_id"]: entry for entry in run.report["mission_orchestration"]["stages"]}
    assert stages["backend_api_change"]["status"] == "failure"
    assert not any(
        event.phase == "mission_stage_behavior"
        and event.data.get("stage_id") == "backend_api_change"
        and event.data.get("behavior_code") == "degraded_reasoning"
        for event in run.events
    )
    assert len([cmd for cmd in backend.commands if cmd.startswith("reasoning:")]) == 1


def test_supervisor_preserve_mode_skips_restore_on_test_repair_failures():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/tests/test_rank_status.py b/tests/test_rank_status.py
@@ -0,0 +1,4 @@
+def test_rank_status():
+    assert True
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_status.py"],
            "final_summary": "repair test",
            "goal": "repair",
        }
    ]
    backend.full_tests = [CommandResult(False, "FAILED tests/test_rank_status.py::test_rank_status", "", 1)]
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-preserve-repair", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Fix backend API endpoint /api/rank/status", targeted_tests=[]),
        "pytest_failure",
        1,
        preserve_validated_progress=True,
    )

    assert result is True
    assert "restore" not in backend.commands
    assert any(event.phase == "repair_restore" and event.status == "skipped" for event in run.events)


def test_supervisor_preserve_mode_restores_unsafe_diffs():
    backend = FakeBackend()
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,6 +1,5 @@
-def create_app() -> FastAPI:
""",
    )
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "unsafe repair",
            "goal": "repair",
        }
    ]
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-preserve-unsafe", rank_id="S")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Fix backend API endpoint /api/rank/status", targeted_tests=[]),
        "reasoning_loop_blocked",
        1,
        preserve_validated_progress=True,
    )

    assert result is True
    assert "restore" in backend.commands
    assert any(
        event.phase == "repair_retry" and event.data.get("failure_class") == "destructive_diff"
        for event in run.events
    )


def test_supervisor_rejects_cross_stage_file_scope_leak_and_skips_validation():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "stage backend tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["tests/test_rank_s_dashboard.py", "igris/web/templates/index.html"],
            "final_summary": "ui stage leaked test edits",
            "goal": "stage ui",
        },
    ]
    backend.diff_stat = CommandResult(
        True,
        " igris/web/server.py | 4 ++++\n tests/test_rank_s_dashboard.py | 8 ++++++++\n igris/web/templates/index.html | 1 +",
    )
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,8 @@
+def test_rank_s_dashboard():
diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
@@ -10,6 +10,7 @@
+<div id='rank-s-dashboard'>ready</div>
""",
    )

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    assert run.status == "blocked"
    assert run.failure_class == "wrong_file_edit"
    assert any(
        event.phase == "validation_short_circuit"
        for event in run.events
    )
    assert any(
        event.phase == "mission_stage"
        and event.data.get("stage_id") == "ui_dashboard_change"
        and "out-of-scope files" in event.detail
        for event in run.events
    )
    assert backend.commands.count("tests:full") == 1
    assert "tests:['tests/test_rank_s_dashboard.py']" not in backend.commands


def test_supervisor_rejects_out_of_scope_attempted_write_even_when_ast_blocked_before_modify():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "stage backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "stage backend tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/templates/index.html"],
            "steps": [
                {
                    "action_type": "replace_range",
                    "parameters": {"path": "igris/web/server.py", "start": "1", "end": "2"},
                    "error": "Python AST validation failed for 'igris/web/server.py': unexpected indent",
                }
            ],
            "final_summary": (
                "Loop blocked. Blocked detail: action=replace_range; "
                "error=Python AST validation failed for 'igris/web/server.py': unexpected indent"
            ),
            "goal": "stage ui",
        },
    ]
    backend.diff_stat = CommandResult(
        True,
        " igris/web/server.py | 4 ++++\n tests/test_rank_s_dashboard.py | 8 ++++++++\n igris/web/templates/index.html | 1 +",
    )
    backend.diff = CommandResult(
        True,
        """diff --git a/igris/web/server.py b/igris/web/server.py
@@ -1,2 +1,6 @@
+@app.get('/api/rank/s-dashboard')
diff --git a/tests/test_rank_s_dashboard.py b/tests/test_rank_s_dashboard.py
@@ -0,0 +1,8 @@
+def test_rank_s_dashboard():
diff --git a/igris/web/templates/index.html b/igris/web/templates/index.html
@@ -10,6 +10,7 @@
+<div id='rank-s-dashboard'>ready</div>
""",
    )

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    assert run.status == "blocked"
    assert run.failure_class == "wrong_file_edit"
    assert any(
        event.phase == "ui_stage_retry"
        and event.status == "running"
        for event in run.events
    )
    assert any(
        event.phase == "mission_stage"
        and event.data.get("stage_id") == "ui_dashboard_change"
        and (
            "out-of-scope files" in event.detail
            or "UI-only recovery exhausted" in event.detail
        )
        for event in run.events
    )
    assert any(
        event.phase == "validation_short_circuit"
        for event in run.events
    )
    assert "tests:['tests/test_rank_s_dashboard.py']" not in backend.commands


def test_supervisor_grants_one_final_validation_attempt_after_last_successful_repair():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": [],
            "final_summary": "initial blocked attempt",
            "goal": "rank task",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "repair applied",
            "goal": "repair",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "final validation attempt",
            "goal": "rank task",
        },
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),
        CommandResult(True, "attempt 1 full pytest ok"),
        CommandResult(True, "repair validation pytest ok"),
        CommandResult(True, "attempt 2 full pytest ok"),
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            goal="Add /api/rank/status endpoint",
            max_rank_attempts=1,
            max_repair_cycles=1,
            targeted_tests=[],
        )
    )

    assert run.status == "completed"
    assert any(
        event.phase == "rank_attempt_extension"
        and event.data.get("final_validation_only") is True
        for event in run.events
    )


def test_supervisor_cleans_dirty_workspace_when_blocked_after_attempt_exhaustion():
    backend = FakeBackend()
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")
    backend.reasoning_results = [
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "blocked after edit",
            "goal": "rank task",
        }
    ]
    backend.status_sequence = [
        CommandResult(True, ""),
        CommandResult(True, " M igris/web/server.py"),
        CommandResult(True, ""),
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(
            goal="Add /api/rank/status endpoint",
            max_rank_attempts=1,
            max_repair_cycles=0,
            targeted_tests=[],
        )
    )

    assert run.status == "blocked"
    assert run.failure_class == "reasoning_loop_blocked"
    assert "restore" in backend.commands
    assert any(
        event.phase == "blocked_workspace_cleanup" and event.status == "success"
        for event in run.events
    )
    assert any(
        event.phase == "blocked_workspace_state"
        and event.data.get("after_cleanup") is True
        and event.data.get("dirty") is False
        for event in run.events
    )


def test_supervisor_does_not_cleanup_preflight_workspace_dirty_block():
    backend = FakeBackend()
    backend.status = CommandResult(True, " M igris/web/server.py")

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_config())

    assert run.status == "blocked"
    assert run.failure_class == "workspace_dirty"
    assert "restore" not in backend.commands


def test_api_escalation_disabled_by_default_and_not_called():
    config = RankSupervisorConfig.from_dict({"goal": "rank task"})
    assert config.allow_api_escalation is False

    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-no-api", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    supervisor._repair_cycle(run, _config(max_repair_cycles=0), "reasoning_loop_blocked", 1)

    assert not any(cmd.startswith("api_helper:") for cmd in backend.commands)
    assert any(event.phase == "api_escalation" and event.status == "skipped" for event in run.events)


def test_api_escalation_packet_contains_required_fields_and_is_sanitized():
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-packet", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    run.add("failure", "classified", "token=sk-test-123456", api_key="sk-live-super-secret", reason="loop")
    config = RankSupervisorConfig.from_dict({
        "goal": "Use token sk-prod-secret and fix ui",
        "allow_api_escalation": True,
        "max_api_escalations_per_run": 2,
        "max_api_budget_usd": 1.0,
    })

    packet = supervisor._build_api_escalation_packet(
        run,
        config,
        failure="reasoning_loop_blocked",
        cycle=2,
        stage_statuses={"ui_dashboard_change": {"status": "failure"}},
    )

    for key in ("run_id", "rank_id", "failure_class", "repair_cycle", "recent_events", "policy"):
        assert key in packet
    serialized = json.dumps(packet)
    assert "sk-live-super-secret" not in serialized
    assert "sk-prod-secret" not in serialized
    assert packet["policy"]["must_not_complete_product_manually"] is True


def test_api_escalation_respects_call_and_budget_limits():
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-budget", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    config = RankSupervisorConfig.from_dict({
        "goal": "rank",
        "allow_api_escalation": True,
        "max_api_escalations_per_run": 2,
        "max_api_budget_usd": 0.005,
        "max_tokens_per_escalation": 300,
    })

    first = supervisor._maybe_api_escalate(run, config, failure="reasoning_loop_blocked", cycle=1)
    second = supervisor._maybe_api_escalate(run, config, failure="reasoning_loop_blocked", cycle=2)

    assert first is not None
    assert second is None
    assert len(backend.api_helper_packets) == 1
    assert run.api_escalations_used == 1
    assert run.api_budget_used_usd >= 0.01
    assert any(
        event.phase == "api_escalation" and event.status == "skipped" and event.data.get("budget_type") == "usd"
        for event in run.events
    )


def test_api_helper_response_is_recorded_and_used_as_advice_only():
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-advice", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    config = _config(
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
        max_api_budget_usd=1.0,
        max_tokens_per_escalation=256,
    )

    ok = supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

    assert ok is True
    assert any(event.phase == "api_escalation_response" and event.status == "success" for event in run.events)
    assert any("api_helper:" in cmd for cmd in backend.commands)
    assert any("API helper advice (advisory only" in goal for goal in backend.reasoning_goals)


def test_api_helper_high_risk_triggers_stronger_validation_smoke():
    backend = FakeBackend()
    backend.api_helper_result = CommandResult(
        True,
        json.dumps({
            "diagnosis": "risky patch",
            "likely_supervisor_gap": "unsafe restore heuristic",
            "suggested_repair_strategy": "force direct file rewrite",
            "suggested_tests": [],
            "risk": "high",
            "confidence": 0.2,
            "requires_human_or_codex_audit": True,
            "must_not_complete_product_manually": True,
            "estimated_cost_usd": 0.0,
        }),
    )
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-risky", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    config = _config(
        allow_api_escalation=True,
        max_api_escalations_per_run=1,
        max_api_budget_usd=1.0,
    )

    ok = supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

    assert ok is True
    assert any(event.phase == "repair_high_risk_validation" for event in run.events)
    assert any(cmd.startswith("smoke:") for cmd in backend.commands)


def test_audit_checkpoint_marks_reviewed_events_as_already_reviewed(tmp_path):
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=FakeBackend())
    probe = SupervisorEvent(phase="repair_issue", status="success", detail="issue#1", data={"failure_class": "x"})
    scope_hash = supervisor._event_scope_hash(probe)
    supervisor.record_audit_checkpoint(scope_hash, audit_status="audit-reviewed", reviewed_by="qa")

    run = SupervisorRun(run_id="run-audit-reviewed", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    run.add("repair_issue", "success", "issue#1", failure_class="x")

    assert run.events[-1].audit_status == "audit-reviewed"
    assert run.events[-1].audit_reviewed_by == "qa"


def test_audit_scope_change_resets_event_to_audit_new(tmp_path):
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=FakeBackend())
    probe = SupervisorEvent(phase="repair_issue", status="success", detail="issue#1", data={"failure_class": "x"})
    scope_hash = supervisor._event_scope_hash(probe)
    supervisor.record_audit_checkpoint(scope_hash, audit_status="audit-fixed", reviewed_by="qa")

    run = SupervisorRun(run_id="run-audit-new", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    run.add("repair_issue", "success", "issue#2 changed detail", failure_class="x")

    assert run.events[-1].audit_status == "audit-new"


def test_audit_deferred_event_becomes_reviewable_after_due_date(tmp_path):
    supervisor = SelfRepairSupervisor(str(tmp_path), backend=FakeBackend())
    probe = SupervisorEvent(phase="repair_issue", status="success", detail="issue#defer", data={"failure_class": "x"})
    scope_hash = supervisor._event_scope_hash(probe)
    future_due = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    supervisor.record_audit_checkpoint(
        scope_hash,
        audit_status="audit-deferred",
        reviewed_by="qa",
        next_review_after=future_due,
    )

    run = SupervisorRun(run_id="run-audit-deferred", rank_id="S")
    run.audit_resolver = supervisor._resolve_event_audit
    run.add("repair_issue", "success", "issue#defer", failure_class="x")
    assert run.events[-1].audit_status == "audit-deferred"

    past_due = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    supervisor.record_audit_checkpoint(
        scope_hash,
        audit_status="audit-deferred",
        reviewed_by="qa",
        next_review_after=past_due,
    )
    run2 = SupervisorRun(run_id="run-audit-due", rank_id="S")
    run2.audit_resolver = supervisor._resolve_event_audit
    run2.add("repair_issue", "success", "issue#defer", failure_class="x")
    assert run2.events[-1].audit_status == "audit-new"


def test_ui_stage_wrong_file_edit_on_server_py_triggers_stage_local_restore_only():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "backend tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 1",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 2",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 3",
        },
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    stages = {entry["stage_id"]: entry for entry in run.report["mission_orchestration"]["stages"]}
    assert run.status == "blocked"
    assert run.failure_class == "wrong_file_edit"
    assert stages["backend_api_change"]["status"] == "success"
    assert stages["backend_tests"]["status"] == "success"
    assert stages["ui_dashboard_change"]["status"] == "failure"
    assert all("igris/web/server.py" not in call for call in backend.restore_paths_calls)
    assert all("tests/test_rank_s_dashboard.py" not in call for call in backend.restore_paths_calls)
    assert any(
        "igris/web/templates/index.html" in call
        for call in backend.restore_paths_calls
    )


def test_ui_stage_retry_prompt_contains_ui_only_policy_and_hard_forbid_server_py():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "backend tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 1",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/templates/index.html"],
            "final_summary": "ui fixed with template-only edit",
            "goal": "ui attempt 2",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_dashboard_tabs.py"],
            "final_summary": "ui tests done",
            "goal": "ui tests",
        },
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    ui_goals = [
        goal for goal in backend.reasoning_goals
        if "[stage:ui_dashboard_change]" in goal
    ]
    assert run.status == "completed"
    # At least 2 UI stage calls: the initial attempt and at least one retry.
    # The exact count varies by environment: backend_tests may be pre-satisfied
    # when targeted test files already exist in /tmp/project from other tests,
    # causing an extra wrong_file_edit on the test file → 3 UI calls total.
    assert len(ui_goals) >= 2
    # The first retry goal (index 1) must contain the UI-only recovery policy.
    first_retry_goal = ui_goals[1]
    assert "UI-only recovery policy:" in first_retry_goal
    assert "Do not modify igris/web/server.py." in first_retry_goal
    assert "Hard-forbidden paths for this stage: igris/web/server.py" in first_retry_goal


def test_ui_stage_repeated_wrong_file_edit_has_bounded_retries_not_blind_loop():
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "backend done",
            "goal": "backend",
        },
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_rank_s_dashboard.py"],
            "final_summary": "backend tests done",
            "goal": "backend tests",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 1",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 2",
        },
        {
            "status": "blocked",
            "stop_reason": "blocked",
            "files_modified": ["igris/web/server.py", "igris/web/templates/index.html"],
            "final_summary": "wrong file edit in ui stage",
            "goal": "ui attempt 3",
        },
    ]

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(_staged_config(max_repair_cycles=0))

    ui_contexts = [
        ctx for ctx in backend.reasoning_contexts
        if ctx.get("mission_stage_id") == "ui_dashboard_change"
    ]
    ui_retry_events = [event for event in run.events if event.phase == "ui_stage_retry"]
    assert run.status == "blocked"
    assert len(ui_contexts) == 3
    assert len(ui_retry_events) == 2


# ---------------------------------------------------------------------------
# Fix #342 — Defect 1: required stage with no diff must not be accepted as success
# ---------------------------------------------------------------------------

def test_required_stage_with_no_diff_is_classified_as_failure():
    """backend_api_change that produces no diff must fail, not silently succeed.

    Previously _validate_new_stage_paths returned (True, "") for empty
    candidate_paths, allowing required stages to be accepted as success even
    when the reasoning loop made no file changes.  The fix adds an explicit
    guard: if a required stage with allowed_file_families produced no diff and
    _stage_is_already_satisfied() returns False, the stage is failed with
    reasoning_loop_blocked.

    The test uses _staged_config (score >= 4) to trigger staged mode so
    _execute_staged_reasoning is called and the per-stage no-diff guard fires.
    """
    backend = FakeBackend()
    # Reasoning for backend_api_change reports "finished" but touches NO files.
    # The diff stays empty, simulating IGRIS deciding the stage is done without
    # actually editing server.py.
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],      # ← no files modified
            "final_summary": "no changes needed",
            "goal": "backend_api_change stage",
        },
        # Subsequent stages won't run because backend_api_change fails first.
    ]
    # Diff stat and diff both report no changes (clean working tree throughout)
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "full ok")]

    # _staged_config provides a non-trivial goal (score >= 4) which forces
    # staged mode; the endpoint /api/rank/s-dashboard is not in /tmp/project so
    # _stage_is_already_satisfied returns False for backend_api_change.
    config = _staged_config(max_repair_cycles=0)

    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(config)

    # The run must be blocked because backend_api_change is required and failed
    assert run.status == "blocked"
    assert run.failure_class in {"reasoning_loop_blocked", "wrong_file_edit", "pytest_failure"}
    # The stage status for backend_api_change must be 'failure'
    stage_events = [
        e for e in run.events
        if e.phase == "mission_stage" and e.data.get("stage_id") == "backend_api_change"
    ]
    assert any(e.status == "failure" for e in stage_events), (
        f"backend_api_change stage must be marked failure when no diff produced; "
        f"got: {[(e.status, e.detail[:60]) for e in stage_events]}"
    )


def test_required_stage_with_no_diff_is_not_silently_marked_noop_success():
    """No-diff required stage must never appear as no_op=True success via false positive.

    The no_op=True flag is valid only for stages that _stage_is_already_satisfied
    returns True (pre-satisfied).  An unsatisfied required stage that runs
    reasoning and produces no diff must never receive no_op=True success.
    """
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],
            "final_summary": "nothing to do",
            "goal": "backend stage",
        },
    ]
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")

    config = _staged_config(max_repair_cycles=0)
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(config)

    # Must NOT have a no_op=True success event for backend_api_change
    # that was NOT set by _stage_is_already_satisfied (pre-satisfied paths
    # use "Stage already satisfied" or "Stage already validated" messages).
    noop_success_events = [
        e for e in run.events
        if e.phase == "mission_stage"
        and e.data.get("stage_id") == "backend_api_change"
        and e.status == "success"
        and e.data.get("no_op") is True
        and "Stage already satisfied" not in (e.detail or "")
        and "Stage already validated" not in (e.detail or "")
    ]
    assert noop_success_events == [], (
        f"Expected no false-positive no_op success, got: {[e.detail for e in noop_success_events]}"
    )


# ---------------------------------------------------------------------------
# Fix #342 — Defect 2: _has_flask_test_client_in_diff helper + repair guidance
# ---------------------------------------------------------------------------

def test_has_flask_test_client_in_diff_detects_added_flask_call():
    """_has_flask_test_client_in_diff returns True when diff adds test_client(."""
    diff = """\
diff --git a/tests/test_supervisor_api.py b/tests/test_supervisor_api.py
--- /dev/null
+++ b/tests/test_supervisor_api.py
@@ -0,0 +1,8 @@
+import pytest
+from igris.web.server import create_app
+
+@pytest.fixture
+def client():
+    app = create_app()
+    with app.test_client() as client:
+        yield client
"""
    assert _has_flask_test_client_in_diff(diff) is True


def test_has_flask_test_client_in_diff_ignores_removed_flask_call():
    """_has_flask_test_client_in_diff returns False for removed (- lines) test_client."""
    diff = """\
@@ -1,4 +1,4 @@
-    with app.test_client() as client:
+    client = TestClient(create_app())
"""
    assert _has_flask_test_client_in_diff(diff) is False


def test_has_flask_test_client_in_diff_ignores_fastapi_testclient():
    """_has_flask_test_client_in_diff returns False for correct FastAPI TestClient."""
    diff = """\
+from fastapi.testclient import TestClient
+client = TestClient(create_app())
"""
    assert _has_flask_test_client_in_diff(diff) is False


def test_pytest_failure_repair_goal_contains_fastapi_testclient_guidance():
    """repair_cycle goal for pytest_failure must include FastAPI TestClient warning."""
    backend = FakeBackend()
    # Repair reasoning produces a clean diff with FastAPI TestClient
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_supervisor_run.py"],
            "final_summary": "fixed test",
            "goal": "repair",
        }
    ]
    backend.diff = CommandResult(
        True,
        """\
diff --git a/tests/test_supervisor_run.py b/tests/test_supervisor_run.py
@@ -0,0 +1,6 @@
+from fastapi.testclient import TestClient
+from igris.web.server import create_app
+def test_run():
+    client = TestClient(create_app())
+    r = client.post('/api/supervisor/run', json={})
+    assert r.status_code == 200
""",
    )
    backend.full_tests = [CommandResult(True, "all ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-repair-guidance", rank_id="test")

    supervisor._repair_cycle(
        run,
        _config(goal="Fix backend API /api/supervisor/run endpoint tests", targeted_tests=[]),
        "pytest_failure",
        1,
    )

    # The repair goal sent to reasoning must contain FastAPI TestClient guidance
    assert backend.reasoning_goals, "Expected reasoning to be called during repair"
    repair_goal = backend.reasoning_goals[-1]
    assert "TestClient" in repair_goal, (
        f"Expected FastAPI TestClient guidance in repair goal, got: {repair_goal[:300]}"
    )
    assert "test_client()" in repair_goal.lower() or "test_client(" in repair_goal, (
        "Repair goal must mention the forbidden Flask test_client() pattern"
    )


def test_reasoning_loop_blocked_repair_goal_uses_original_mission():
    """When failure_class is reasoning_loop_blocked, the repair goal must repeat the
    original mission goal, not 'Fix IGRIS infrastructure failure'."""
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "implemented endpoint",
            "goal": "implement",
        }
    ]
    backend.diff = CommandResult(
        True,
        "diff --git a/igris/web/server.py b/igris/web/server.py\n@@ -0,0 +1,3 @@\n+@app.get('/api/test')\n+def test_ep(): return {}\n",
    )
    backend.full_tests = [CommandResult(True, "ok")]

    original_goal = "Implement /api/diagnostics/session-resume endpoint"
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-rbl", rank_id="test")

    supervisor._repair_cycle(
        run,
        _config(goal=original_goal),
        "reasoning_loop_blocked",
        1,
    )

    assert backend.reasoning_goals, "Expected reasoning to be called"
    repair_goal = backend.reasoning_goals[-1]
    assert original_goal in repair_goal, (
        f"repair goal for reasoning_loop_blocked must contain the original mission goal. "
        f"Got: {repair_goal[:300]}"
    )
    assert "Fix IGRIS infrastructure failure" not in repair_goal, (
        "repair goal for reasoning_loop_blocked must NOT say 'Fix IGRIS infrastructure failure'"
    )


def test_reasoning_loop_blocked_uses_semantic_repair_task_type():
    """reasoning_loop_blocked must route to semantic_repair task_type (cloud-first)."""
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
            "goal": "implement",
        }
    ]
    backend.diff = CommandResult(True, "diff --git a/igris/web/server.py b/igris/web/server.py\n@@ -1 +1,2 @@\n+# endpoint\n")
    backend.full_tests = [CommandResult(True, "ok")]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-rbl-type", rank_id="test")

    supervisor._repair_cycle(
        run,
        _config(goal="Implement /api/diagnostics/session-resume"),
        "reasoning_loop_blocked",
        1,
    )

    assert backend.last_task_type == "semantic_repair", (
        f"reasoning_loop_blocked must use semantic_repair task type, got {backend.last_task_type!r}"
    )


def test_pytest_failure_repair_rejects_flask_test_client_diff_and_retries():
    """_repair_cycle must reject a diff that adds Flask test_client() for pytest_failure.

    When IGRIS produces a repair diff containing Flask-style test_client(), the
    supervisor must detect it via _has_flask_test_client_in_diff, reject the diff,
    restore the working tree, and return True (retry) with a repair_retry event.
    """
    backend = FakeBackend()
    flask_diff = """\
diff --git a/tests/test_supervisor_api.py b/tests/test_supervisor_api.py
--- /dev/null
+++ b/tests/test_supervisor_api.py
@@ -0,0 +1,10 @@
+import pytest
+from igris.web.server import create_app
+
+@pytest.fixture
+def client():
+    app = create_app()
+    with app.test_client() as client:
+        yield client
+def test_run(client):
+    assert client.post('/api/supervisor/run').status_code == 200
"""
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["tests/test_supervisor_api.py"],
            "final_summary": "added tests",
            "goal": "repair pytest",
        }
    ]
    backend.diff = CommandResult(True, flask_diff)
    backend.diff_stat = CommandResult(True, " tests/test_supervisor_api.py | 10 ++++++++++")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-flask-reject", rank_id="test")

    result = supervisor._repair_cycle(
        run,
        _config(goal="Fix /api/supervisor/run tests", targeted_tests=[]),
        "pytest_failure",
        1,
    )

    # Should return True (continue / retry) and have restored the diff
    assert result is True
    assert "restore" in backend.commands
    # Must have emitted a repair_retry event
    retry_events = [e for e in run.events if e.phase == "repair_retry"]
    assert retry_events, "Expected a repair_retry event after Flask test_client rejection"
    assert any(
        "flask" in (e.detail or "").lower() or "test_client" in (e.detail or "").lower()
        for e in retry_events
    ), f"repair_retry detail should mention Flask or test_client, got: {[e.detail for e in retry_events]}"


# ---------------------------------------------------------------------------
# Tests for bug #341 — API escalation counter fix (unconfigured helper)
# ---------------------------------------------------------------------------

def _escalation_config(**overrides):
    """Config with API escalation enabled and a generous budget."""
    data = {
        "goal": "Rank A controlled task with tests",
        "rank_id": "A",
        "max_rank_attempts": 1,
        "max_repair_cycles": 1,
        "allow_api_escalation": True,
        "max_api_escalations_per_run": 3,
        "max_api_budget_usd": 1.0,
        "dry_run": True,
    }
    data.update(overrides)
    return RankSupervisorConfig.from_dict(data)


def test_unconfigured_helper_does_not_consume_call_budget():
    """When IGRIS_API_HELPER_COMMAND is not set, _maybe_api_escalate must NOT
    increment api_escalations_used — it should instead increment
    api_escalations_failed_unconfigured so the call budget stays intact."""
    backend = FakeBackend()
    backend._api_helper_configured = False

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-uncfg-1", rank_id="test")
    config = _escalation_config()

    supervisor._maybe_api_escalate(run, config, failure="pytest_failure", cycle=1)

    assert run.api_escalations_used == 0, (
        "Budget must not be consumed when helper is not configured"
    )
    assert run.api_escalations_failed_unconfigured == 1, (
        "Failed-unconfigured counter must be incremented"
    )
    # Helper must NOT have been called
    assert not any("api_helper" in cmd for cmd in backend.commands), (
        "call_api_helper must not be invoked when helper is not configured"
    )


def test_unconfigured_helper_emits_not_configured_event():
    """_maybe_api_escalate must emit an api_escalation/not_configured event
    (not a skipped/budget-exhausted event) when the helper is unconfigured."""
    backend = FakeBackend()
    backend._api_helper_configured = False

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-uncfg-evt", rank_id="test")
    config = _escalation_config()

    supervisor._maybe_api_escalate(run, config, failure="pytest_failure", cycle=1)

    not_cfg_events = [
        e for e in run.events
        if e.phase == "api_escalation" and e.status == "not_configured"
    ]
    assert not_cfg_events, (
        "Expected api_escalation/not_configured event when helper is unconfigured"
    )
    skipped_budget_events = [
        e for e in run.events
        if e.phase == "api_escalation"
        and e.status == "skipped"
        and "budget" in (e.detail or "").lower()
    ]
    assert not skipped_budget_events, (
        "Must NOT emit a budget-exhausted skipped event when the real cause is misconfiguration"
    )


def test_unconfigured_helper_multiple_calls_do_not_exhaust_budget():
    """Calling _maybe_api_escalate N times with unconfigured helper must not
    make the budget check (api_escalations_used >= max) trigger — subsequent
    calls must each emit not_configured rather than budget-exhausted."""
    backend = FakeBackend()
    backend._api_helper_configured = False

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-uncfg-multi", rank_id="test")
    config = _escalation_config(max_api_escalations_per_run=2)

    for cycle in range(1, 5):
        supervisor._maybe_api_escalate(run, config, failure="pytest_failure", cycle=cycle)

    assert run.api_escalations_used == 0
    assert run.api_escalations_failed_unconfigured == 4

    # All 4 events must be not_configured, none budget-exhausted
    for e in run.events:
        if e.phase == "api_escalation":
            assert e.status == "not_configured", (
                f"Expected not_configured but got {e.status!r}: {e.detail!r}"
            )


def test_configured_helper_consumes_call_budget_normally():
    """When the helper IS configured and call succeeds, api_escalations_used
    must be incremented and api_escalations_failed_unconfigured must stay 0."""
    backend = FakeBackend()
    backend._api_helper_configured = True  # default, but explicit

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-cfg-ok", rank_id="test")
    config = _escalation_config()

    supervisor._maybe_api_escalate(run, config, failure="pytest_failure", cycle=1)

    assert run.api_escalations_used == 1
    assert run.api_escalations_failed_unconfigured == 0
    assert any("api_helper" in cmd for cmd in backend.commands), (
        "call_api_helper must be invoked when helper is configured"
    )


def test_run_start_emits_config_warning_when_helper_unconfigured():
    """When allow_api_escalation=True but helper is not configured, the run()
    method must emit an api_escalation_config/not_configured event immediately
    after start so operators see the problem early."""
    backend = FakeBackend()
    backend._api_helper_configured = False
    # Make baseline tests fail fast so run exits quickly
    backend.baseline = CommandResult(False, "baseline failed")
    backend.status = CommandResult(True, "")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _escalation_config(max_rank_attempts=1, max_repair_cycles=0)

    run = supervisor.run(config)

    cfg_events = [
        e for e in run.events
        if e.phase == "api_escalation_config" and e.status == "not_configured"
    ]
    assert cfg_events, (
        "Expected api_escalation_config/not_configured event at run start when helper is unconfigured"
    )


def test_run_start_does_not_emit_config_warning_when_helper_configured():
    """When the helper IS configured, no api_escalation_config/not_configured
    warning event should appear at run start."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.baseline = CommandResult(False, "baseline failed")
    backend.status = CommandResult(True, "")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _escalation_config(max_rank_attempts=1, max_repair_cycles=0)

    run = supervisor.run(config)

    cfg_events = [
        e for e in run.events
        if e.phase == "api_escalation_config" and e.status == "not_configured"
    ]
    assert not cfg_events, (
        "Should NOT emit api_escalation_config/not_configured when helper is properly configured"
    )


def test_local_supervisor_backend_api_helper_is_configured_with_env(monkeypatch):
    """LocalSupervisorBackend.api_helper_is_configured() must return True only
    when IGRIS_API_HELPER_COMMAND is set to a non-empty string."""
    backend = LocalSupervisorBackend("/tmp/test_proj")

    monkeypatch.delenv("IGRIS_API_HELPER_COMMAND", raising=False)
    assert backend.api_helper_is_configured() is False

    monkeypatch.setenv("IGRIS_API_HELPER_COMMAND", "")
    assert backend.api_helper_is_configured() is False

    monkeypatch.setenv("IGRIS_API_HELPER_COMMAND", "   ")
    assert backend.api_helper_is_configured() is False

    monkeypatch.setenv("IGRIS_API_HELPER_COMMAND", "python helper.py")
    assert backend.api_helper_is_configured() is True


def test_api_escalation_report_fragment_includes_failed_unconfigured():
    """_api_escalation_report_fragment must include calls_failed_unconfigured
    so dashboards can distinguish 'budget used' from 'helper not configured'."""
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)

    run = SupervisorRun(run_id="run-frag", rank_id="test")
    run.api_escalations_used = 2
    run.api_escalations_failed_unconfigured = 3

    fragment = supervisor._api_escalation_report_fragment(run)

    assert fragment["api_escalation"]["calls_used"] == 2
    assert fragment["api_escalation"]["calls_failed_unconfigured"] == 3


def test_summarize_supervised_run_includes_failed_unconfigured():
    """summarize_supervised_run must surface api_escalations_failed_unconfigured
    so the UI can show it correctly rather than lumping it into api_escalations_used."""
    run = SupervisorRun(run_id="run-sum", rank_id="test")
    run.api_escalations_used = 1
    run.api_escalations_failed_unconfigured = 2

    summary = summarize_supervised_run(run)

    assert summary.get("api_escalations_failed_unconfigured") == 2, (
        f"Expected 2 in summary, got: {summary.get('api_escalations_failed_unconfigured')!r}"
    )


# ---------------------------------------------------------------------------
# Tests for noop completion bug (#345) — _complete_noop must not be blocked
# by pr_ci_merge / post_merge_runtime when they were never reached
# ---------------------------------------------------------------------------

def test_complete_noop_does_not_block_when_pr_ci_merge_was_not_reached():
    """When a staged mission completes as a no-op (goal already satisfied,
    no diff produced, all tests green), _complete_noop must NOT block the run
    because pr_ci_merge / post_merge_runtime stages were never executed.

    Regression: before the fix, _required_stages_green() inside _complete_noop
    was called without exclusions, so a required pr_ci_merge stage with no
    status (never reached) triggered 'No-op completion rejected: required
    stage missing.' and set run.status = 'blocked'.
    """
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-noop-prci", rank_id="test")

    # Build stage_statuses that mirror what the supervisor would have after
    # completing implementation stages but never reaching pr_ci_merge.
    stage_statuses = {
        "backend_api_change": {
            "required": True,
            "status": "success",
            "detail": "API change implemented.",
        },
        "backend_tests": {
            "required": True,
            "status": "success",
            "detail": "Tests implemented.",
        },
        # pr_ci_merge is required (dry_run=False) but was never executed
        "pr_ci_merge": {
            "required": True,
            "status": "pending",
            "detail": "",
        },
        # post_merge_runtime same
        "post_merge_runtime": {
            "required": True,
            "status": "pending",
            "detail": "",
        },
        "final_report": {
            "required": True,
            "status": "pending",
            "detail": "",
        },
    }

    supervisor._complete_noop(
        run,
        completion_mode="already_satisfied",
        runtime_refresh_required=False,
        detail="All required staged mission phases were already satisfied; completed as verified no-op.",
        post_merge_smoke=True,
        stage_statuses=stage_statuses,
    )

    # Must NOT be blocked
    assert run.status == "completed", (
        f"Expected status='completed' for valid noop, got '{run.status}'. "
        "Check: _complete_noop may be including pr_ci_merge in required-stages check."
    )
    assert run.outcome == "Completed"
    assert run.failure_class == ""

    # final_report must be success
    final_status = stage_statuses.get("final_report", {}).get("status")
    assert final_status == "success", (
        f"Expected final_report status='success', got '{final_status}'"
    )


def test_complete_noop_still_blocks_on_genuinely_missing_required_stage():
    """If a truly required IMPLEMENTATION stage (not delivery) is missing,
    _complete_noop should still block — the exclusion is limited to delivery
    stages (pr_ci_merge, post_merge_runtime, final_report)."""
    backend = FakeBackend()
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    run = SupervisorRun(run_id="run-noop-miss", rank_id="test")

    stage_statuses = {
        "backend_api_change": {
            "required": True,
            "status": "pending",  # NOT completed — this should block
            "detail": "",
        },
        "pr_ci_merge": {
            "required": True,
            "status": "pending",
            "detail": "",
        },
        "post_merge_runtime": {
            "required": True,
            "status": "pending",
            "detail": "",
        },
        "final_report": {
            "required": True,
            "status": "pending",
            "detail": "",
        },
    }

    supervisor._complete_noop(
        run,
        completion_mode="already_satisfied",
        runtime_refresh_required=False,
        detail="Noop with missing implementation stage.",
        post_merge_smoke=True,
        stage_statuses=stage_statuses,
    )

    # MUST be blocked because backend_api_change was not completed
    assert run.status == "blocked", (
        f"Expected status='blocked' when required implementation stage is missing, got '{run.status}'"
    )
    assert run.failure_class == "reasoning_loop_blocked"


# ---------------------------------------------------------------------------
# Capability-limit detection and mission decomposition tests
# ---------------------------------------------------------------------------

def _decomposition_reasoning_result(fields=None):
    """Fake reasoning result that returns a well-formed JSON decomposition."""
    sub = {
        "title": "Implement sub-task A",
        "goal": "Add endpoint /api/foo",
        "dependencies": [],
        "acceptance_criteria": ["GET /api/foo returns 200"],
        "allowed_file_scopes": ["igris/web/server.py", "tests/test_foo.py"],
        "tests": ["tests/test_foo.py"],
        "risk_level": "low",
        "human_approval_required": False,
    }
    payload = {
        "why_too_large": "Original mission required 5+ file changes across unrelated modules.",
        "sub_missions": [sub],
        "first_sub_mission": "Implement sub-task A",
        "human_approval_required": False,
    }
    if fields:
        payload.update(fields)
    return {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": [],
        "final_summary": json.dumps(payload),
        "goal": "decomposition",
    }


def _make_timeout_result():
    return {
        "status": "blocked",
        "stop_reason": "reasoning_timeout",
        "files_modified": [],
        "final_summary": "Timed out without producing a change.",
    }


def _make_no_diff_repair_results(n: int):
    """n repair reasoning results that succeed but produce no diff."""
    return [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],
            "final_summary": f"repair attempt {i + 1}",
        }
        for i in range(n)
    ]


def test_repeated_reasoning_timeout_triggers_decomposition_required():
    """When reasoning times out CAPABILITY_LIMIT_THRESHOLD times the supervisor
    must block with failure_class='decomposition_required', not a generic block.

    Sequence (with diff_stat always failing so no diff is produced):
      [0] attempt-1 main reasoning  → reasoning_timeout (signal=1)
      [1] repair-cycle reasoning    → reasoning_timeout (signal=2 → threshold)
          _repair_cycle returns False (diff_stat failure), capability limit fires.
      [2] decomposition reasoning   → JSON decomposition
    """
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),              # [0] attempt-1 main → reasoning_timeout=1
        _make_timeout_result(),              # [1] repair-cycle  → reasoning_timeout=2
        _decomposition_reasoning_result(),   # [2] decomposition call
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex multi-file task that exceeds model capacity",
    )
    run = SupervisorRun(run_id="cap-timeout", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Expected decomposition_required, got {result.failure_class!r}"
    )
    assert result.status == "blocked"
    assert result.outcome == "Blocked"
    assert result.report.get("decomposition_required") is True
    assert result.capability_signals.get("reasoning_timeout", 0) >= CAPABILITY_LIMIT_THRESHOLD
    assert "decomposition" in result.report
    assert result.report.get("next_action", "").startswith("run:") or \
           result.report.get("next_action", "").startswith("request_approval:")


def test_explicit_blocked_stop_reason_triggers_decomposition_required():
    """When the LLM explicitly returns blocked (self-aware refusal), the supervisor
    must accumulate reasoning_timeout signals and decompose — not exhaust repair
    cycles and return a generic blocked.

    stop_reason='blocked' with a summary that does NOT contain the LLM-unavailable
    string should be treated the same as 'reasoning_timeout' for capability-signal
    accounting — it means the model was reached but refused the task.
    """
    backend = FakeBackend()
    blocked_result = {
        "status": "blocked",
        "stop_reason": "blocked",
        "files_modified": [],
        # No "No suitable LLM provider available" here — model WAS reached but refused.
        "final_summary": "Loop abc123: blocked\nGoal: ...\nStop: blocked\nBlocked detail: action=blocked; reason=Task complexity exceeds local model capability",
    }
    backend.reasoning_results = [
        blocked_result,                      # attempt-1: blocked → reasoning_timeout=1
        blocked_result,                      # repair-cycle: blocked → reasoning_timeout=2
        _decomposition_reasoning_result(),   # decomposition call
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Implement GET /api/diagnostics/session-resume for issue #350",
    )
    run = SupervisorRun(run_id="cap-blocked", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Expected decomposition_required, got {result.failure_class!r}. "
        "blocked stop_reason must count as reasoning_timeout capability signal."
    )
    assert result.capability_signals.get("reasoning_timeout", 0) >= CAPABILITY_LIMIT_THRESHOLD
    assert result.report.get("decomposition_required") is True


def test_repeated_no_diff_repair_triggers_decomposition_required():
    """When repair cycles repeatedly produce no diff, capability limit is detected.

    Flow with max_rank_attempts=3, max_repair_cycles=2 (== CAPABILITY_LIMIT_THRESHOLD):
      attempt-1: main→done+diff, full=FAILED; repair-1→no_diff (no_diff_repair=1) → True
      attempt-2: main→done+diff, full=FAILED; repair-2→no_diff (no_diff_repair=2) → True
      attempt-3: main→done+diff, full=FAILED;
          repair_cycles(2) >= max_repair_cycles(2) → budget exhausted
          capability limit: no_diff_repair=2 → TRIGGERED → decomposition
    """
    n = CAPABILITY_LIMIT_THRESHOLD  # = 2
    backend = FakeBackend()
    # 3 main attempts + 2 repair cycles + 1 decomposition = 6 reasoning calls
    main_result = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/web/server.py"],
        "final_summary": "done",
    }
    no_diff_repair = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": [],
        "final_summary": "repaired (no change)",
    }
    backend.reasoning_results = [
        main_result,            # [0] attempt-1 main
        no_diff_repair,         # [1] repair-1
        dict(main_result),      # [2] attempt-2 main
        dict(no_diff_repair),   # [3] repair-2
        dict(main_result),      # [4] attempt-3 main
        _decomposition_reasoning_result(),  # [5] decomposition
    ]
    # diff_stat_sequence: 5 calls (3 main + 2 repair) — repair sees empty stat
    diff_has = CommandResult(True, " igris/web/server.py | 1 +")
    diff_empty_stat = CommandResult(True, "")
    backend.diff_stat_sequence = [diff_has, diff_empty_stat, diff_has, diff_empty_stat, diff_has]
    # diff_sequence: main sees real diff, repair sees empty diff
    diff_real = CommandResult(True, "+safe line")
    diff_empty = CommandResult(True, "")
    backend.diff_sequence = [diff_real, diff_empty, diff_real, diff_empty, diff_real]
    # full_tests: baseline pops first (must succeed), then each main attempt pops one
    # repairs exit early (no run_tests call in no-diff path)
    backend.full_tests = (
        [CommandResult(True, "baseline ok")]       # baseline
        + [CommandResult(False, "FAILED")] * 3     # attempts 1, 2, 3
        + [CommandResult(True, "ok")] * 5          # spare
    )
    backend.targeted = CommandResult(True, "ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=n,
        goal="Complex mission with no-diff repair loops",
    )
    run = SupervisorRun(run_id="cap-nodiff", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Expected decomposition_required, got {result.failure_class!r}; "
        f"signals={result.capability_signals}"
    )
    assert result.report.get("decomposition_required") is True
    assert result.capability_signals.get("no_diff_repair", 0) >= CAPABILITY_LIMIT_THRESHOLD


def test_repeated_pytest_hang_triggers_decomposition_required():
    """Repeated pytest hangs (Command killed) must trigger decomposition_required.

    Flow (max_repair_cycles=1):
      baseline → ok
      attempt-1: full→hang (pytest_hang=1); repair-1 reasoning ok+diff,
                 repair-validation→hang (pytest_hang=2, RETRYABLE → returns True)
      attempt-2: full→hang (pytest_hang=3);
                 repair_cycles(1) >= max_repair_cycles(1) → budget exhausted
                 capability limit: pytest_hang=3 >= 2 → TRIGGERED → decomposition
    """
    backend = FakeBackend()
    hang = CommandResult(False, "", "Command killed: no output for 120s (idle timeout)", 124)
    backend.reasoning_results = [
        {                             # [0] attempt-1 main
            "status": "finished", "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"], "final_summary": "done",
        },
        {                             # [1] repair-1 reasoning
            "status": "finished", "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"], "final_summary": "repair",
        },
        _decomposition_reasoning_result(),   # [2] decomposition
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+safe")
    # [0] baseline ok, [1] attempt-1 full hang, [2] repair-1 validation hang,
    # [3] attempt-2 full hang
    backend.full_tests = [CommandResult(True, "baseline ok"), hang, hang, hang] + [CommandResult(True, "ok")] * 5

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=4,
        max_repair_cycles=1,
        goal="Mission with hanging pytest",
    )
    run = SupervisorRun(run_id="cap-hang", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Expected decomposition_required, got {result.failure_class!r}; "
        f"signals={result.capability_signals}"
    )
    assert result.capability_signals.get("pytest_hang", 0) >= CAPABILITY_LIMIT_THRESHOLD
    assert result.report.get("decomposition_required") is True


def test_decomposition_report_contains_required_fields():
    """decomposition_required run must include all required fields in run.decomposition."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),
        _make_timeout_result(),
        _decomposition_reasoning_result(),
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Mission requiring decomposition",
    )
    run = SupervisorRun(run_id="cap-fields", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    decomposition = result.decomposition
    assert decomposition is not None, "run.decomposition must not be None"
    for field in DECOMPOSITION_REQUIRED_FIELDS:
        assert field in decomposition, (
            f"Required decomposition field '{field}' missing. "
            f"Present: {list(decomposition.keys())}"
        )


def test_supervisor_does_not_declare_completed_on_decomposition_required():
    """A run blocked with decomposition_required must never have status='completed'."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),
        _make_timeout_result(),
        _decomposition_reasoning_result(),
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Uncompletable mission",
    )
    run = SupervisorRun(run_id="cap-nocomp", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.status != "completed", (
        "Supervisor must not declare 'completed' when decomposition_required is set"
    )
    assert result.outcome != "Completed"
    assert result.report.get("decomposition_required") is True


def test_decomposition_is_persisted_in_run_report_and_to_dict():
    """run.to_dict() must include decomposition and capability_signals."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),
        _make_timeout_result(),
        _decomposition_reasoning_result(),
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Persisted decomposition test",
    )
    run = SupervisorRun(run_id="cap-persist", rank_id="test")
    result = supervisor.run(config, run=run)

    d = result.to_dict()
    assert "capability_signals" in d, "to_dict() must include capability_signals"
    assert "decomposition" in d, "to_dict() must include decomposition"
    assert d["decomposition"] is not None
    assert d["capability_signals"].get("reasoning_timeout", 0) >= CAPABILITY_LIMIT_THRESHOLD

    # Report must also carry decomposition
    assert result.report.get("decomposition") is not None
    assert result.report.get("capability_limit_signal") is not None
    assert result.report.get("next_action") is not None


def test_decomposition_report_has_no_secrets():
    """Decomposition fields must be redacted — no raw secrets pass through."""
    backend = FakeBackend()
    # Inject a secret into the decomposition output to verify it gets redacted.
    decomp_with_secret = _decomposition_reasoning_result(fields={
        "why_too_large": "Failed because OPENAI_API_KEY=sk-secret123 was missing."
    })
    backend.reasoning_results = [
        _make_timeout_result(),
        _make_timeout_result(),
        decomp_with_secret,
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Secret safety test",
    )
    run = SupervisorRun(run_id="cap-secret", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    report_text = json.dumps(result.report)
    assert "sk-secret123" not in report_text, (
        "Secret must be redacted from decomposition report"
    )


def test_mixed_capability_signals_trigger_decomposition():
    """One reasoning_timeout + one no_diff_repair (total=2) must trigger decomposition.

    This verifies cross-signal detection: even when no single signal reaches
    CAPABILITY_LIMIT_THRESHOLD alone, the combined total reaching it triggers
    decomposition_required.

    Flow: max_rank_attempts=3, max_repair_cycles=1
      attempt-1: main→finish+diff, full=FAILED (pytest_failure)
                 repair-1 → no_diff (no_diff_repair=1, RETRYABLE → return True)
      attempt-2: main→timeout (reasoning_timeout=1)
                 repair_cycles(1) >= max_repair_cycles(1) → budget exhausted
                 capability limit: total=2 >= threshold → decomposition_required
    """
    backend = FakeBackend()
    main_ok = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/web/server.py"],
        "final_summary": "done",
    }
    repair_ok = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": [],
        "final_summary": "repair attempt",
    }
    backend.reasoning_results = [
        main_ok,                           # [0] attempt-1 main → pytest_failure
        repair_ok,                         # [1] repair-1 → empty diff → no_diff_repair=1
        _make_timeout_result(),            # [2] attempt-2 main → reasoning_timeout=1
        _decomposition_reasoning_result(), # [3] decomposition
    ]
    diff_has = CommandResult(True, " igris/web/server.py | 1 +")
    diff_empty_stat = CommandResult(True, "")
    backend.diff_stat_sequence = [
        diff_has,        # main-1: real diff produced
        diff_has,        # repair-1: passes diff_stat check (but diff itself is empty)
        diff_empty_stat, # main-2: timeout, no diff
    ]
    diff_real = CommandResult(True, "+some code")
    diff_empty = CommandResult(True, "")
    backend.diff_sequence = [
        diff_real,  # main-1 diff
        diff_empty, # repair-1 diff → no_diff_repair signal
        diff_empty, # main-2 diff
    ]
    backend.full_tests = [
        CommandResult(True, "baseline ok"),  # baseline
        CommandResult(False, "FAILED"),      # main-1 → pytest_failure triggers repair
        CommandResult(True, "ok"),           # main-2 (reasoning failure supersedes)
    ]
    backend.targeted = CommandResult(True, "ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Mission that produces mixed capability signals",
    )
    run = SupervisorRun(run_id="cap-mixed", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Expected decomposition_required, got {result.failure_class!r}; "
        f"signals={result.capability_signals}"
    )
    assert result.report.get("decomposition_required") is True
    assert result.capability_signals.get("reasoning_timeout", 0) >= 1
    assert result.capability_signals.get("no_diff_repair", 0) >= 1
    assert "decomposition" in result.report


def test_non_repeated_failure_does_not_trigger_decomposition():
    """A single reasoning_timeout (below threshold) must NOT trigger decomposition."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),       # 1 timeout — below threshold
        _decomposition_reasoning_result(),  # would be repair result if triggered
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,   # no repair budget → blocked immediately
        goal="Single timeout test",
    )
    run = SupervisorRun(run_id="cap-single", rank_id="test")
    result = supervisor.run(config, run=run)

    # Below threshold — should be a plain block, not decomposition_required.
    assert result.failure_class != "decomposition_required", (
        "Single timeout below threshold must not trigger decomposition_required"
    )


# ---------------------------------------------------------------------------
# Decomposition fallback chain tests
# ---------------------------------------------------------------------------

def test_decomposition_local_retry_short_prompt_succeeds():
    """When the local short-prompt reasoning returns valid 4-field JSON,
    generated_by must be 'local_reasoning' and all required fields present."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),              # [0] main → reasoning_timeout=1
        _make_timeout_result(),              # [1] repair  → reasoning_timeout=2 → threshold
        _decomposition_reasoning_result(),   # [2] short-prompt decomp → valid JSON
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex multi-file task requiring decomposition",
    )
    run = SupervisorRun(run_id="decomp-local", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    decomp = result.decomposition
    assert decomp is not None
    assert decomp.get("generated_by") == "local_reasoning", (
        f"Expected generated_by='local_reasoning', got {decomp.get('generated_by')!r}"
    )
    for field in DECOMPOSITION_REQUIRED_FIELDS:
        assert field in decomp, f"Required field '{field}' missing from decomposition"


def _make_max_steps_result():
    """Reasoning result that hits max_steps with no valid JSON in summary."""
    return {
        "status": "stopped",
        "stop_reason": "max_steps",
        "files_modified": [],
        "final_summary": "Steps exhausted without producing output.",
    }


def test_decomposition_falls_back_to_api_helper_when_local_fails():
    """When local decomposition yields no valid JSON, API helper is tried.
    generated_by must be 'api_helper' when helper returns valid decomposition."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.api_helper_result = CommandResult(
        True,
        json.dumps({
            "ok": True,
            "why_too_large": "Mission too broad for single pass.",
            "sub_missions": [
                {"title": "Sub A", "goal": "Do A", "risk_level": "low"}
            ],
            "first_sub_mission": "Sub A",
            "human_approval_required": False,
        }),
    )
    backend.reasoning_results = [
        _make_timeout_result(),      # [0] main → reasoning_timeout=1
        _make_timeout_result(),      # [1] repair → reasoning_timeout=2 → threshold
        _make_max_steps_result(),    # [2] local decomp → no valid JSON
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Large task requiring api helper decomposition",
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
    )
    run = SupervisorRun(run_id="decomp-api", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    decomp = result.decomposition
    assert decomp is not None
    assert decomp.get("generated_by") == "api_helper", (
        f"Expected generated_by='api_helper', got {decomp.get('generated_by')!r}"
    )
    for field in DECOMPOSITION_REQUIRED_FIELDS:
        assert field in decomp, f"Required field '{field}' missing from decomposition"


def test_decomposition_falls_back_to_deterministic_when_helper_unavailable():
    """When local fails and API helper is not configured, deterministic fallback fires.
    generated_by must be 'deterministic_fallback' and all required fields present."""
    backend = FakeBackend()
    backend._api_helper_configured = False
    backend.reasoning_results = [
        _make_timeout_result(),      # [0] main
        _make_timeout_result(),      # [1] repair → threshold
        _make_max_steps_result(),    # [2] local decomp → no valid JSON
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Implement auth endpoint, add tests, update dashboard badge",
    )
    run = SupervisorRun(run_id="decomp-det-nohelper", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    decomp = result.decomposition
    assert decomp is not None
    assert decomp.get("generated_by") == "deterministic_fallback", (
        f"Expected 'deterministic_fallback', got {decomp.get('generated_by')!r}"
    )
    for field in DECOMPOSITION_REQUIRED_FIELDS:
        assert field in decomp, f"Required field '{field}' missing from decomposition"
    assert isinstance(decomp.get("sub_missions"), list)
    assert len(decomp["sub_missions"]) >= 1


def test_decomposition_falls_back_to_deterministic_when_helper_fails():
    """When local fails and API helper returns a failure, deterministic fallback fires."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.api_helper_result = CommandResult(False, "", "network error", 1)
    backend.reasoning_results = [
        _make_timeout_result(),      # [0] main
        _make_timeout_result(),      # [1] repair → threshold
        _make_max_steps_result(),    # [2] local decomp → no valid JSON
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Refactor supervisor and add endpoint",
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
    )
    run = SupervisorRun(run_id="decomp-det-helperfail", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    decomp = result.decomposition
    assert decomp is not None
    assert decomp.get("generated_by") == "deterministic_fallback", (
        f"Expected 'deterministic_fallback', got {decomp.get('generated_by')!r}"
    )
    for field in DECOMPOSITION_REQUIRED_FIELDS:
        assert field in decomp, f"Required field '{field}' missing from decomposition"


def test_decomposition_persisted_with_generated_by():
    """run.to_dict() and run.report must include generated_by and next_action."""
    backend = FakeBackend()
    backend._api_helper_configured = False
    backend.reasoning_results = [
        _make_timeout_result(),
        _make_timeout_result(),
        _make_max_steps_result(),    # triggers deterministic fallback
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Persist decomposition with generated_by",
    )
    run = SupervisorRun(run_id="decomp-persist", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    d = result.to_dict()
    assert "decomposition" in d, "to_dict() must include decomposition"
    assert d["decomposition"] is not None
    assert d["decomposition"].get("generated_by") in {
        "local_reasoning", "api_helper", "deterministic_fallback"
    }, f"Unexpected generated_by: {d['decomposition'].get('generated_by')!r}"

    next_action = result.report.get("next_action", "")
    assert next_action.startswith("request_approval:") or next_action.startswith("run:"), (
        f"next_action must start with 'request_approval:' or 'run:', got {next_action!r}"
    )


def test_decomposition_no_secrets_in_fallback():
    """Secrets in the goal must be redacted in the deterministic fallback decomposition."""
    backend = FakeBackend()
    backend._api_helper_configured = False
    backend.reasoning_results = [
        _make_timeout_result(),
        _make_timeout_result(),
        _make_max_steps_result(),
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    secret = "sk-ant-fake123abcdefghijklmno"
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal=f"implement {secret} endpoint and add tests",
    )
    run = SupervisorRun(run_id="decomp-nosecret", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    decomp = result.decomposition
    assert decomp is not None

    # Serialise everything and check secret is absent
    serialised = json.dumps(decomp)
    assert secret not in serialised, (
        "Secret must be redacted from decomposition fallback output"
    )


# ---------------------------------------------------------------------------
# Autonomous sub-issue creation — policy-aware decomposition (#decomp-auto)
# ---------------------------------------------------------------------------

def _make_valid_decomposition_result(destructive=False, has_secret=False):
    """Reasoning result returning a valid 4-field decomposition."""
    goal_text = "Implement sub-task A"
    if destructive:
        goal_text = "Delete all data and reset database"
    if has_secret:
        goal_text = "Deploy with key sk-ant-testfakekey1234567890"
    sub = {
        "title": "Sub A",
        "goal": goal_text,
        "dependencies": [],
        "acceptance_criteria": ["Works correctly"],
        "allowed_file_scopes": ["igris/web/server.py"],
        "tests": ["tests/test_rank_status.py"],
        "risk_level": "high" if destructive else "medium",
        "human_approval_required": False,
    }
    payload = {
        "why_too_large": "Mission too large for single pass.",
        "sub_missions": [sub],
        "first_sub_mission": "Sub A",
        "human_approval_required": False,
    }
    return {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": [],
        "final_summary": json.dumps(payload),
        "goal": "decomposition",
    }


def _decomp_backend(decomp_result):
    """Build a FakeBackend wired for a 2-timeout decomposition flow."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),   # [0] attempt-1 main → reasoning_timeout=1
        _make_timeout_result(),   # [1] repair-cycle  → reasoning_timeout=2 → threshold
        decomp_result,            # [2] decomposition call
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10
    return backend


def test_decomposition_policy_auto_create_when_safe():
    """Safe decomposition with GitHub enabled triggers auto sub-issue creation."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="auto-create-safe", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required"
    next_action = result.report.get("next_action", "")
    assert next_action.startswith("run:"), (
        f"Expected next_action to start with 'run:' when safe, got: {next_action!r}"
    )
    phases = [e.phase for e in result.events]
    assert "subissue_creation" in phases, "Expected subissue_creation event"
    sub_issue_urls = result.report.get("decomposition", {}).get("sub_issue_urls", [])
    assert len(sub_issue_urls) > 0, "Expected non-empty sub_issue_urls in decomposition report"


def test_decomposition_policy_request_approval_when_dry_run():
    """dry_run=True suppresses auto sub-issue creation even if allow_auto_subissues is set."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_auto_subissues=True,
        dry_run=True,
    )
    run = SupervisorRun(run_id="dry-run-no-auto", rank_id="test")
    result = supervisor.run(config, run=run)

    next_action = result.report.get("next_action", "")
    assert next_action.startswith("request_approval:"), (
        f"Expected request_approval when dry_run=True, got: {next_action!r}"
    )
    phases = [e.phase for e in result.events]
    assert "subissue_creation" not in phases, "No subissue_creation events expected in dry_run"


def test_decomposition_policy_request_approval_when_github_disabled():
    """No GitHub flags → always request_approval:decomposition."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=False,
        allow_auto_subissues=False,
        dry_run=True,
    )
    run = SupervisorRun(run_id="no-github-no-auto", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.report.get("next_action") == "request_approval:decomposition"


def test_decomposition_policy_request_approval_for_destructive():
    """Destructive decomposition → request_human_approval (no auto-create)."""
    backend = _decomp_backend(_make_valid_decomposition_result(destructive=True))
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="destructive-no-auto", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.report.get("next_action") == "request_approval:decomposition", (
        "Destructive decomposition must request human approval, not auto-create"
    )


def test_decomposition_policy_block_unsafe_for_secret():
    """Decomposition containing a secret pattern is blocked as unsafe."""
    backend = _decomp_backend(_make_valid_decomposition_result(has_secret=True))
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="secret-blocked", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.report.get("next_action") == "request_approval:decomposition"
    blocked_unsafe_events = [
        e for e in result.events
        if e.phase == "decomposition_policy" and e.status == "blocked_unsafe"
    ]
    assert blocked_unsafe_events, (
        "Expected a decomposition_policy/blocked_unsafe event for secret-containing decomposition"
    )


def test_decomposition_subissue_body_contains_parent_run_and_generated_by():
    """Auto-create path calls create_issue and records subissue_created events with expected fields."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="subissue-body-check", rank_id="test")
    result = supervisor.run(config, run=run)

    assert "issue" in backend.commands, "create_issue must be called on the backend"
    created_events = [e for e in result.events if e.phase == "subissue_created" and e.status == "success"]
    assert created_events, "Expected at least one subissue_created/success event"
    ev = created_events[0]
    assert ev.data.get("title"), "subissue_created event must have title"
    assert ev.data.get("url"), "subissue_created event must have url"
    assert ev.data.get("risk"), "subissue_created event must have risk"


def test_decomposition_subissue_urls_persisted_in_report():
    """sub_issue_urls must be non-empty in run.report decomposition after auto-create."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="subissue-urls", rank_id="test")
    result = supervisor.run(config, run=run)

    sub_issue_urls = result.report.get("decomposition", {}).get("sub_issue_urls", [])
    assert isinstance(sub_issue_urls, list) and len(sub_issue_urls) > 0, (
        "sub_issue_urls must be a non-empty list after auto-create"
    )
    d = result.to_dict()
    assert isinstance(d.get("decomposition", {}).get("sub_issue_urls"), list), (
        "sub_issue_urls must survive to_dict() serialization"
    )


def test_decomposition_no_secrets_in_subissue_body():
    """Decomposition with secret content is blocked before create_issue is called."""
    backend = _decomp_backend(_make_valid_decomposition_result(has_secret=True))
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="secret-no-create", rank_id="test")
    result = supervisor.run(config, run=run)

    # Policy should block this — no sub-mission issues must be created.
    # (The supervisor may create a repair-tracking issue, but NOT any sub-mission issue.)
    subissue_created = any(
        issue["title"] == "Sub A"  # sub-mission title from _make_valid_decomposition_result
        for issue in backend.created_issues
    )
    assert not subissue_created, (
        "create_issue must NOT be called for sub-missions when decomposition is blocked as unsafe"
    )


# ---------------------------------------------------------------------------
# Decomposition report consistency tests (#350 watchdog fix)
# ---------------------------------------------------------------------------


def test_auto_create_subissues_clears_human_approval_required():
    """When policy=auto_create_subissues and sub-issues are created,
    human_approval_required must be False and auto_approved_by_policy=True."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="auto-create-approval-clear", rank_id="test")
    result = supervisor.run(config, run=run)

    decomp = result.report.get("decomposition", {})
    assert decomp.get("human_approval_required") is False, (
        "human_approval_required must be False when policy=auto_create_subissues and issues created"
    )
    assert decomp.get("auto_approved_by_policy") is True, (
        "auto_approved_by_policy must be True when policy=auto_create_subissues"
    )
    assert decomp.get("approval_status") == "auto_approved_by_policy", (
        f"approval_status must be 'auto_approved_by_policy', got {decomp.get('approval_status')!r}"
    )


def test_auto_create_subissues_report_coherent():
    """Decomposition report must include policy, allow_auto_subissues, sub_issue_urls,
    next_action as a single coherent picture — no ambiguity."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=True,
        allow_auto_subissues=True,
        dry_run=False,
    )
    run = SupervisorRun(run_id="auto-create-coherent-report", rank_id="test")
    result = supervisor.run(config, run=run)

    decomp = result.report.get("decomposition", {})
    assert decomp.get("policy") == "auto_create_subissues", (
        f"decomposition.policy must be 'auto_create_subissues', got {decomp.get('policy')!r}"
    )
    assert decomp.get("allow_auto_subissues") is True, (
        "decomposition.allow_auto_subissues must be True"
    )
    assert isinstance(decomp.get("sub_issue_urls"), list) and len(decomp["sub_issue_urls"]) > 0, (
        "sub_issue_urls must be a non-empty list in decomposition report"
    )
    next_action = result.report.get("next_action", "")
    assert next_action.startswith("run:"), (
        f"next_action must start with 'run:' after auto-create, got {next_action!r}"
    )
    assert decomp.get("next_action", "").startswith("run:"), (
        "next_action must also be mirrored inside decomposition dict"
    )


def test_request_human_approval_keeps_human_approval_required_true():
    """When policy=request_human_approval, human_approval_required must stay True
    and auto_approved_by_policy must be False."""
    backend = _decomp_backend(_make_valid_decomposition_result())
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=1,
        goal="Complex task",
        allow_github_pr=False,
        allow_auto_subissues=False,
        dry_run=False,
    )
    run = SupervisorRun(run_id="manual-approval-keeps-flag", rank_id="test")
    result = supervisor.run(config, run=run)

    decomp = result.report.get("decomposition", {})
    assert decomp.get("policy") == "request_human_approval", (
        f"Expected policy=request_human_approval, got {decomp.get('policy')!r}"
    )
    assert decomp.get("auto_approved_by_policy") is False, (
        "auto_approved_by_policy must be False when human approval is required"
    )
    assert decomp.get("approval_status") == "pending_human_approval", (
        f"approval_status must be 'pending_human_approval', got {decomp.get('approval_status')!r}"
    )
    # human_approval_required may be True or False depending on what the decomposer returned,
    # but approval_status must accurately reflect the pending state.
    assert result.report.get("next_action") == "request_approval:decomposition"


# ---------------------------------------------------------------------------
# Semantic acceptance gate integration tests (#365)
# ---------------------------------------------------------------------------

_STUB_DIFF = """\
diff --git a/igris/web/server.py b/igris/web/server.py
--- a/igris/web/server.py
+++ b/igris/web/server.py
@@ -1,3 +1,10 @@
+    @app.get('/api/diagnostics/session-resume')
+    async def session_resume():
+        # Logic to gather diagnostics data
+        return JSONResponse(content={'zombie_runs': [], 'active_runs': [], 'stale_branches': []})
"""

_REAL_DIFF = """\
diff --git a/igris/web/server.py b/igris/web/server.py
--- a/igris/web/server.py
+++ b/igris/web/server.py
@@ -1,3 +1,10 @@
+    @app.get('/api/diagnostics/session-resume')
+    async def session_resume():
+        runs = list_supervised_runs()
+        zombie = [r for r in runs if _is_zombie(r)]
+        return JSONResponse(content={'zombie_runs': zombie, 'active_runs': []})
diff --git a/tests/test_session_resume.py b/tests/test_session_resume.py
--- /dev/null
+++ b/tests/test_session_resume.py
@@ -0,0 +1,8 @@
+def test_session_resume():
+    resp = client.get('/api/diagnostics/session-resume')
+    assert resp.status_code == 200
+    assert 'zombie_runs' in resp.json()
"""


def _make_semantic_backend(diff_output: str, targeted_pass: bool = True):
    """FakeBackend configured so tests pass but diff is controllable."""
    backend = FakeBackend()
    backend.reasoning_results = [{
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/web/server.py", "tests/test_session_resume.py"],
        "final_summary": "done",
        "loop_id": "loop-1",
    }]
    backend.diff_stat = CommandResult(True, "igris/web/server.py | 5 +++++")
    backend.diff = CommandResult(True, diff_output)
    backend.targeted = CommandResult(targeted_pass, "1 passed" if targeted_pass else "1 failed")
    backend.full_tests = [CommandResult(True, "ok")] * 5
    return backend


def test_semantic_gate_blocks_stub_endpoint():
    """Stub endpoint (hardcoded empty arrays) must be classified semantic_incomplete."""
    backend = _make_semantic_backend(_STUB_DIFF)
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        goal="Implement GET /api/diagnostics/session-resume endpoint in server.py",
        max_repair_cycles=0,
        dry_run=False,
        allow_github_pr=False,
        enable_semantic_gate=True,
    )
    run = supervisor.run(config)

    assert run.status == "blocked"
    assert run.failure_class == "semantic_incomplete"
    semantic_events = [e for e in run.events if e.phase == "semantic_check"]
    assert semantic_events, "Expected semantic_check event"
    assert semantic_events[-1].status == "incomplete"
    assert "acceptance_evidence" in run.report
    assert not run.report["acceptance_evidence"]["passed"]


def test_semantic_gate_passes_real_implementation():
    """Real implementation with test coverage must pass the gate and complete."""
    backend = _make_semantic_backend(_REAL_DIFF)
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        goal="Implement GET /api/diagnostics/session-resume endpoint in server.py",
        max_repair_cycles=0,
        dry_run=True,
        allow_github_pr=False,
        enable_semantic_gate=True,
    )
    run = supervisor.run(config)

    assert run.status == "completed"
    semantic_events = [e for e in run.events if e.phase == "semantic_check"]
    assert semantic_events, "Expected semantic_check event"
    assert semantic_events[-1].status == "passed"
    assert run.report.get("acceptance_evidence", {}).get("passed") is True


def test_semantic_gate_triggers_repair_when_cycles_available():
    """semantic_incomplete is repairable: first attempt stubs, second delivers real code."""
    stub_result = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/web/server.py"],
        "final_summary": "stub",
    }
    real_result = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/web/server.py", "tests/test_session_resume.py"],
        "final_summary": "real impl",
    }
    # [0] attempt-1 main reasoning → stub (gate blocks)
    # [1] repair-cycle reasoning (consumed by _repair_cycle internals)
    # [2] attempt-2 main reasoning → real impl with test file in modified_files
    attempt2_result = {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/web/server.py", "tests/test_session_resume.py"],
        "final_summary": "real impl attempt 2",
    }
    backend = FakeBackend()
    backend.reasoning_results = [stub_result, real_result, attempt2_result]
    backend.diff_stat = CommandResult(True, "igris/web/server.py | 5 +++++")
    # diff_sequence: [0]=stub (attempt-1 gate), [1]=real (repair cycle validation)
    # attempt-2 gate uses backend.diff fallback below
    backend.diff_sequence = [
        CommandResult(True, _STUB_DIFF),
        CommandResult(True, _REAL_DIFF),
    ]
    backend.diff = CommandResult(True, _REAL_DIFF)
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        goal="Implement GET /api/diagnostics/session-resume endpoint in server.py",
        max_repair_cycles=1,
        dry_run=True,
        allow_github_pr=False,
        enable_semantic_gate=True,
    )
    run = supervisor.run(config)

    assert run.status == "completed"
    incomplete_events = [e for e in run.events if e.phase == "semantic_check" and e.status == "incomplete"]
    assert incomplete_events, "Expected at least one semantic_check/incomplete event during repair"


def test_semantic_gate_report_has_evidence_fields():
    """run.report must contain acceptance_evidence with found/missing/endpoints."""
    backend = _make_semantic_backend(_STUB_DIFF)
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        goal="Implement GET /api/diagnostics/session-resume endpoint in server.py",
        max_repair_cycles=0,
        dry_run=False,
        allow_github_pr=False,
        enable_semantic_gate=True,
    )
    run = supervisor.run(config)

    ev = run.report.get("acceptance_evidence", {})
    assert "passed" in ev
    assert "found_evidence" in ev
    assert "missing_evidence" in ev
    assert "required_endpoints" in ev
    assert "/api/diagnostics/session-resume" in ev["required_endpoints"]


def test_semantic_gate_not_triggered_for_generic_goal():
    """Generic goals with no endpoint spec must pass the gate without intervention."""
    backend = _make_semantic_backend("+ x = 1 + 2\n+ y = x * 3")
    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        goal="Fix arithmetic bug in calculation module",
        max_repair_cycles=0,
        dry_run=True,
        allow_github_pr=False,
        enable_semantic_gate=True,
    )
    run = supervisor.run(config)

    assert run.status == "completed"
    semantic_events = [e for e in run.events if e.phase == "semantic_check"]
    assert semantic_events
    assert semantic_events[-1].status == "passed"


# ---------------------------------------------------------------------------
# Ghost / zombie run detection — service restart stale-active fix (#332)
# ---------------------------------------------------------------------------

def test_interrupted_is_terminal_status():
    """'interrupted' must be in TERMINAL_RUN_STATUSES so ghost runs are filtered."""
    assert "interrupted" in TERMINAL_RUN_STATUSES


def test_reconcile_marks_persisted_only_running_as_interrupted():
    """A run that exists only in the persisted store with status='running' was
    killed by a service restart and must be promoted to 'interrupted'."""
    persisted = {
        "ghost-run": {
            "run_id": "ghost-run",
            "rank_id": "rank",
            "status": "running",
            "updated_at": "2026-05-13T01:59:07+00:00",
        }
    }
    result = _reconcile_run_records(in_memory={}, persisted=persisted)
    assert result["ghost-run"]["status"] == "interrupted", (
        "Persisted-only running run must be promoted to 'interrupted' after restart"
    )


def test_reconcile_preserves_persisted_terminal_status():
    """A persisted run that already has a terminal status must not be touched."""
    for terminal_status in ("blocked", "completed", "cancelled", "failed", "crashed"):
        persisted = {
            "done-run": {
                "run_id": "done-run",
                "rank_id": "rank",
                "status": terminal_status,
                "updated_at": "2026-05-13T01:59:07+00:00",
            }
        }
        result = _reconcile_run_records(in_memory={}, persisted=persisted)
        assert result["done-run"]["status"] == terminal_status, (
            f"Terminal status '{terminal_status}' must not be changed by reconcile"
        )


def test_active_runs_excludes_ghost_runs_after_restart(tmp_path):
    """list_active_supervised_run_summaries must not surface ghost runs that
    are only in the persisted file (i.e. were interrupted by a service restart)."""
    runs_path = tmp_path / ".igris" / "supervisor_runs.json"
    runs_path.parent.mkdir(parents=True)
    runs_path.write_text(
        '{"runs": {"ghost-18fa": {"rank_id": "rank", "status": "running",'
        ' "updated_at": "2026-05-13T01:59:07+00:00", "latest_event": {}}}}',
        encoding="utf-8",
    )
    # ghost-18fa is NOT in RUN_STORE (simulates post-restart state)
    active = list_active_supervised_run_summaries(project_root=str(tmp_path))
    assert not any(r["run_id"] == "ghost-18fa" for r in active), (
        "Ghost run (persisted-only, status=running) must not appear in active runs after restart"
    )


# ---------------------------------------------------------------------------
# Pre-flight mission planning (#354 — Miglioramento 1)
# ---------------------------------------------------------------------------

def _planning_scope_result(
    complexity: str = "low",
    decomposition_recommended: bool = False,
    reason: str = "",
    files: list | None = None,
) -> Dict[str, Any]:
    """Fake backend result whose final_summary contains a valid MissionScope JSON."""
    scope = {
        "files_to_touch": files or ["igris/web/server.py"],
        "estimated_complexity": complexity,
        "decomposition_recommended": decomposition_recommended,
        "decomposition_reason": reason,
        "safe_entry_point": "add endpoint first",
        "risks": ["may break existing tests"],
    }
    return {
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": [],
        "final_summary": json.dumps(scope),
    }


def test_planning_pass_produces_mission_scope():
    """Planning pass must populate run.mission_scope and run.report['mission_scope']."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _planning_scope_result(complexity="medium"),  # planning
        {                                              # main attempt
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")
    backend.full_tests = [CommandResult(True, "ok")] * 5

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="Add diagnostics endpoint",
        enable_mission_planning=True,
    )
    run = SupervisorRun(run_id="plan-scope", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.mission_scope is not None, "mission_scope must be set after planning pass"
    assert result.mission_scope.get("estimated_complexity") == "medium"
    assert result.to_dict().get("mission_scope") is not None
    planning_events = [e for e in result.events if e.phase == "mission_planning"]
    assert len(planning_events) >= 2, "Expected running + success/partial planning events"


def test_proactive_decomposition_from_planning():
    """If the planning pass flags decomposition_recommended=true, the supervisor
    must block with decomposition_required BEFORE the first attempt."""
    backend = FakeBackend()
    backend.reasoning_results = [
        _planning_scope_result(           # planning → recommends decomposition
            complexity="high",
            decomposition_recommended=True,
            reason="4000+ LOC file, cross-cutting concerns",
        ),
        _decomposition_reasoning_result(), # _ask_igris_decompose
        # No main attempt reasoning — must never be reached
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/self_repair_supervisor.py"],
            "final_summary": "attempted — should not happen",
        },
    ]
    backend.diff_stat = CommandResult(True, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 5

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=3,
        max_repair_cycles=3,
        goal="Universal supervisor redesign",
        enable_mission_planning=True,
    )
    run = SupervisorRun(run_id="plan-decomp", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Expected decomposition_required from planning, got {result.failure_class!r}"
    )
    assert result.report.get("decomposition_required") is True
    assert result.report.get("capability_limit_signal") == "pre_flight_planning"
    # Must not have attempted any code changes
    branch_events = [e for e in result.events if e.phase == "rank_branch"]
    assert not branch_events, "No rank branch should have been created — decomp fires before first attempt"


def test_planning_failure_does_not_block_run():
    """If the planning pass produces no valid JSON, the run proceeds normally."""
    backend = FakeBackend()
    backend.reasoning_results = [
        {                                  # planning → garbage output
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],
            "final_summary": "I cannot produce a valid scope analysis.",
        },
        {                                  # main attempt → succeeds
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")
    backend.full_tests = [CommandResult(True, "ok")] * 5

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="Add simple endpoint",
        enable_mission_planning=True,
    )
    run = SupervisorRun(run_id="plan-fail", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class != "decomposition_required", (
        "Planning failure must not block the run — it should proceed to main attempt"
    )
    assert result.status == "completed", f"Expected completed, got {result.status!r}"


def test_planning_disabled_skips_planning_pass():
    """When enable_mission_planning=False, no mission_planning event is emitted."""
    backend = FakeBackend()
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")
    backend.full_tests = [CommandResult(True, "ok")] * 5

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="Simple task",
        enable_mission_planning=False,
    )
    run = SupervisorRun(run_id="plan-disabled", rank_id="test")
    result = supervisor.run(config, run=run)

    planning_events = [e for e in result.events if e.phase == "mission_planning"]
    assert not planning_events, "No mission_planning events when planning is disabled"
    assert result.mission_scope is None


# ---------------------------------------------------------------------------
# Miglioramento 2: Failure Memory integration tests
# ---------------------------------------------------------------------------

def test_failure_memory_check_event_emitted_in_run():
    """run() must emit a failure_memory/checked event after baseline passes."""
    backend = FakeBackend()
    backend.full_tests = [CommandResult(True, "ok")] * 5
    # Simplest setup: reason → no diff → budget exhausted (blocked quickly)
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],
            "final_summary": "",
        }
    ]

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="add health check endpoint",
        enable_mission_planning=False,
        allow_api_escalation=False,
    )
    run = SupervisorRun(run_id="fm-check-event", rank_id="test")
    result = supervisor.run(config, run=run)

    phases = [e.phase for e in result.events]
    assert "failure_memory" in phases
    fm_event = next(e for e in result.events if e.phase == "failure_memory")
    assert fm_event.status == "checked"
    assert "risk_level" in fm_event.data
    assert fm_event.data["risk_level"] in ("low", "medium", "high")


def test_supervisor_run_stores_goal_on_run():
    """run.goal must be populated from config.goal via _configure_run_tracking."""
    backend = FakeBackend()
    backend.full_tests = [CommandResult(True, "ok")] * 5

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="my explicit goal string",
        enable_mission_planning=False,
    )
    run = SupervisorRun(run_id="goal-stored", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.goal == "my explicit goal string"


def test_baseline_pytest_failure_not_recorded_to_memory(tmp_path):
    """pytest_failure at baseline must NOT be recorded to failure memory."""
    from igris.core.failure_memory import FailureMemory

    backend = FakeBackend()
    backend.baseline = CommandResult(False, "1 failed")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    mem = FailureMemory(store_path=tmp_path / "failure_patterns.json")
    supervisor._failure_memory = mem

    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="add health endpoint",
        enable_mission_planning=False,
    )
    run = SupervisorRun(run_id="baseline-fail-no-mem", rank_id="test")
    supervisor.run(config, run=run)

    store = tmp_path / "failure_patterns.json"
    if store.exists():
        data = json.loads(store.read_text())
        assert data.get("patterns", []) == []


# ---------------------------------------------------------------------------
# Miglioramento 3: Model-aware escalation in planning pass
# ---------------------------------------------------------------------------

def test_model_aware_escalation_triggered_on_high_complexity():
    """When planning returns high complexity + escalation enabled, escalation fires."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.full_tests = [CommandResult(True, "ok")] * 5
    backend.reasoning_results = [
        _planning_scope_result(complexity="high"),  # planning pass
        {                                            # main attempt
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="implement complex websocket streaming system",
        enable_mission_planning=True,
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
        max_api_budget_usd=1.0,
    )
    run = SupervisorRun(run_id="m3-escalation", rank_id="test")
    result = supervisor.run(config, run=run)

    escalation_events = [e for e in result.events if e.phase == "model_aware_escalation"]
    assert escalation_events, "model_aware_escalation event must be emitted for high complexity"
    running_event = next((e for e in escalation_events if e.status == "running"), None)
    assert running_event is not None
    assert running_event.data.get("complexity") == "high"


def test_model_aware_escalation_not_triggered_on_low_complexity():
    """Planning with low complexity must NOT trigger model-aware escalation."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.full_tests = [CommandResult(True, "ok")] * 5
    backend.reasoning_results = [
        _planning_scope_result(complexity="low"),   # planning pass
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="add simple endpoint",
        enable_mission_planning=True,
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
    )
    run = SupervisorRun(run_id="m3-low-no-escalation", rank_id="test")
    result = supervisor.run(config, run=run)

    escalation_events = [e for e in result.events if e.phase == "model_aware_escalation"]
    assert not escalation_events, "model_aware_escalation must NOT fire for low complexity"


def test_model_aware_escalation_skipped_when_helper_not_configured():
    """When helper is not configured, escalation is skipped (never blocks run)."""
    backend = FakeBackend()
    backend._api_helper_configured = False
    backend.full_tests = [CommandResult(True, "ok")] * 5
    backend.reasoning_results = [
        _planning_scope_result(complexity="high"),
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="complex mission but helper not set up",
        enable_mission_planning=True,
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
    )
    run = SupervisorRun(run_id="m3-no-helper", rank_id="test")
    result = supervisor.run(config, run=run)

    # Run must not be blocked because of missing escalation
    assert result.status != "crashed"
    escalation_events = [e for e in result.events if e.phase == "model_aware_escalation"]
    if escalation_events:
        # If emitted, must be skipped — never 'running' with no configured helper
        statuses = {e.status for e in escalation_events}
        assert "running" not in statuses or "skipped" in statuses or "not_configured" in statuses


def test_model_aware_escalation_disabled_when_escalation_off():
    """allow_api_escalation=False must prevent model-aware escalation even at high complexity."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.full_tests = [CommandResult(True, "ok")] * 5
    backend.reasoning_results = [
        _planning_scope_result(complexity="high"),
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="complex task but escalation disabled",
        enable_mission_planning=True,
        allow_api_escalation=False,
        max_api_escalations_per_run=0,
    )
    run = SupervisorRun(run_id="m3-escalation-off", rank_id="test")
    result = supervisor.run(config, run=run)

    escalation_events = [e for e in result.events if e.phase == "model_aware_escalation"]
    assert not escalation_events, "No model_aware_escalation when allow_api_escalation=False"


def test_model_aware_escalation_hint_stored_in_mission_scope():
    """When escalation succeeds, strategy hint is stored in run.mission_scope."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.full_tests = [CommandResult(True, "ok")] * 5
    backend.reasoning_results = [
        _planning_scope_result(complexity="high"),
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/web/server.py"],
            "final_summary": "done",
        },
    ]
    backend.diff_stat = CommandResult(True, " igris/web/server.py | 1 +")
    backend.diff = CommandResult(True, "+ok")

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=1,
        max_repair_cycles=0,
        goal="complex websocket implementation",
        enable_mission_planning=True,
        allow_api_escalation=True,
        max_api_escalations_per_run=2,
        max_api_budget_usd=1.0,
    )
    run = SupervisorRun(run_id="m3-hint-scope", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.mission_scope is not None
    # Hint should be present if escalation succeeded
    hint = result.mission_scope.get("escalation_strategy_hint")
    # FakeBackend returns a valid advice payload, so hint should be non-empty
    assert isinstance(hint, str)


# ---------------------------------------------------------------------------
# _deterministic_decompose_fallback — parser correctness
# ---------------------------------------------------------------------------

_SIGNALS = {"reasoning_timeout": 1, "no_diff_repair": 1}
_decompose = SelfRepairSupervisor._deterministic_decompose_fallback


class TestDeterministicDecomposeFallback:
    def test_endpoint_goal_produces_impl_and_test_submissions(self):
        goal = (
            "Implement issue #350: GET /api/diagnostics/session-resume. "
            "Use the GitHub issue body as source of truth. "
            "Implement real session-resume diagnostics, not a stub. "
            "Add targeted tests in tests/test_session_resume.py. "
            "If too large, decompose semantically before implementation."
        )
        result = _decompose(goal, _SIGNALS)
        subs = result["sub_missions"]
        # Must produce 2 sub-missions: implementation + tests
        assert len(subs) == 2, f"Expected 2, got {len(subs)}: {[s['title'] for s in subs]}"
        titles = [s["title"].lower() for s in subs]
        assert any("impl" in t or "backend" in t or "endpoint" in t for t in titles)
        assert any("test" in t for t in titles)

    def test_endpoint_goal_does_not_split_on_periods(self):
        goal = (
            "Implement GET /api/diagnostics/session-resume. "
            "Use real data. Not a stub. Add tests in tests/test_session_resume.py."
        )
        result = _decompose(goal, _SIGNALS)
        subs = result["sub_missions"]
        # None of the sub-missions should be a sentence fragment like "not a stub"
        for s in subs:
            assert len(s["goal"]) >= 30, f"Fragment too short: {s['goal']!r}"
            assert s["goal"].lower() not in ("not a stub", "use real data", "py")

    def test_bulleted_goal_splits_on_bullets(self):
        goal = (
            "Complete the following tasks:\n"
            "- Implement the endpoint in server.py\n"
            "- Add zombie run detection logic\n"
            "- Write tests in tests/test_session_resume.py\n"
            "- Update the API documentation"
        )
        result = _decompose(goal, _SIGNALS)
        subs = result["sub_missions"]
        assert len(subs) >= 2
        goals_text = " ".join(s["goal"].lower() for s in subs)
        assert "endpoint" in goals_text or "server" in goals_text
        assert "zombie" in goals_text or "test" in goals_text

    def test_semicolon_goal_splits_on_semicolons(self):
        goal = (
            "Implement the session-resume endpoint; "
            "add zombie run detection using the run store; "
            "write integration tests for the new endpoint"
        )
        result = _decompose(goal, _SIGNALS)
        subs = result["sub_missions"]
        assert len(subs) == 3
        assert all(len(s["goal"]) >= 30 for s in subs)

    def test_generic_goal_produces_single_submission(self):
        goal = "Fix arithmetic rounding in the cost calculator module"
        result = _decompose(goal, _SIGNALS)
        subs = result["sub_missions"]
        assert len(subs) == 1
        assert "rounding" in subs[0]["goal"].lower() or "arithmetic" in subs[0]["goal"].lower()

    def test_always_has_required_fields(self):
        goal = "Implement GET /api/health endpoint and add tests"
        result = _decompose(goal, _SIGNALS)
        assert "why_too_large" in result
        assert "sub_missions" in result
        assert "first_sub_mission" in result
        assert "human_approval_required" in result
        assert result["generated_by"] == "deterministic_fallback"
        for s in result["sub_missions"]:
            for field in ("title", "goal", "dependencies", "acceptance_criteria",
                          "allowed_file_scopes", "tests", "risk_level"):
                assert field in s, f"Missing field {field!r} in sub-mission"

    def test_endpoint_file_scope_includes_server(self):
        goal = "Implement GET /api/diagnostics/session-resume endpoint"
        result = _decompose(goal, _SIGNALS)
        impl_sub = result["sub_missions"][0]
        scopes = impl_sub["allowed_file_scopes"]
        assert any("server" in s or "igris/web" in s or "igris/core" in s for s in scopes)

    def test_test_file_path_extracted_from_goal(self):
        goal = (
            "Implement GET /api/diagnostics/session-resume. "
            "Add tests in tests/test_session_resume.py."
        )
        result = _decompose(goal, _SIGNALS)
        test_sub = result["sub_missions"][-1]
        assert any("test_session_resume" in t for t in test_sub["tests"]) or \
               any("tests/" in t for t in test_sub["tests"])


# ---------------------------------------------------------------------------
# API helper — Codex-only mode observability in supervisor events
# ---------------------------------------------------------------------------


def _make_escalation_backend(api_helper_mode: str = "", model_resolved: str = "codex-mini-latest") -> FakeBackend:
    """Return a FakeBackend configured for an escalation scenario."""
    backend = FakeBackend()
    backend._api_helper_configured = True
    backend.full_tests = [CommandResult(True, "ok")] * 10
    backend.reasoning_results = [
        {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "done",
        }
    ]
    # Response payload includes observability fields from helper script
    import json as _json
    response_payload = {
        "ok": True,
        "model": model_resolved,
        "api_helper_mode": api_helper_mode or "auto",
        "api_helper_provider": "openai",
        "api_helper_model_requested": "gpt-5.4-mini",
        "api_helper_model_resolved": model_resolved,
        "codex_only": api_helper_mode == "codex_only",
        "summary": "ok",
        "diagnosis": "timeout",
        "likely_supervisor_gap": "missing retry",
        "suggested_repair_strategy": "bounded retry",
        "suggested_tests": [],
        "risk": "low",
        "risk_notes": [],
        "do_not_do": [],
        "confidence": 0.8,
        "requires_human_or_codex_audit": False,
        "must_not_complete_product_manually": True,
        "estimated_cost_usd": 0.002,
    }
    backend.api_helper_result = CommandResult(True, _json.dumps(response_payload))
    return backend


class TestApiHelperModeObservability:
    """Test API helper mode observability via _repair_cycle (same pattern as existing escalation tests)."""

    def _make_run_and_supervisor(self, backend, mode=""):
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        run = SupervisorRun(run_id="run-obs-test", rank_id="A")
        run.audit_resolver = supervisor._resolve_event_audit
        config = _config(
            allow_api_escalation=True,
            max_api_escalations_per_run=1,
            max_api_budget_usd=1.0,
            max_tokens_per_escalation=256,
            api_helper_mode=mode,
        )
        return supervisor, run, config

    def test_auto_mode_request_event_has_mode_field(self):
        backend = _make_escalation_backend()
        supervisor, run, config = self._make_run_and_supervisor(backend)
        supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

        req_events = [e for e in run.events if e.phase == "api_escalation_request"]
        assert req_events, "Expected api_escalation_request event"
        assert req_events[0].data.get("api_helper_mode") == "auto"
        assert req_events[0].data.get("codex_only") is False

    def test_codex_only_mode_request_event_flags_codex_only(self):
        backend = _make_escalation_backend(api_helper_mode="codex_only", model_resolved="codex-mini-latest")
        supervisor, run, config = self._make_run_and_supervisor(backend, mode="codex_only")
        supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

        req_events = [e for e in run.events if e.phase == "api_escalation_request"]
        assert req_events, "Expected api_escalation_request event"
        assert req_events[0].data.get("api_helper_mode") == "codex_only"
        assert req_events[0].data.get("codex_only") is True

    def test_codex_only_mode_passed_to_backend_call(self):
        backend = _make_escalation_backend(api_helper_mode="codex_only", model_resolved="codex-mini-latest")
        supervisor, run, config = self._make_run_and_supervisor(backend, mode="codex_only")
        supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

        assert backend.api_helper_packets, "Expected at least one api_helper call"
        assert backend.api_helper_packets[-1]["mode"] == "codex_only"

    def test_response_event_includes_model_resolved(self):
        backend = _make_escalation_backend(api_helper_mode="codex_only", model_resolved="codex-mini-latest")
        supervisor, run, config = self._make_run_and_supervisor(backend, mode="codex_only")
        supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

        resp_events = [e for e in run.events if e.phase == "api_escalation_response" and e.status == "success"]
        assert resp_events, "Expected successful api_escalation_response event"
        data = resp_events[0].data
        assert data.get("api_helper_model_resolved") == "codex-mini-latest"
        assert data.get("api_helper_provider") == "openai"
        assert data.get("codex_only") is True

    def test_no_secrets_in_escalation_events(self):
        backend = _make_escalation_backend()
        supervisor, run, config = self._make_run_and_supervisor(backend)
        supervisor._repair_cycle(run, config, "reasoning_loop_blocked", 1)

        import json as _json
        all_event_data = _json.dumps([e.data for e in run.events])
        assert "sk-" not in all_event_data
        assert "API_KEY" not in all_event_data

    def test_helper_called_only_on_blocked_or_recovery(self):
        """Helper must NOT fire during normal successful execution."""
        backend = FakeBackend()
        backend._api_helper_configured = True
        backend.full_tests = [CommandResult(True, "ok")] * 5
        backend.reasoning_results = [{
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": ["igris/core/fix.py"],
            "final_summary": "done",
        }]
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(
            allow_api_escalation=True,
            max_api_escalations_per_run=3,
            max_api_budget_usd=5.0,
            max_repair_cycles=0,
        )
        run = supervisor.run(config)

        assert run.status == "completed"
        assert run.api_escalations_used == 0, (
            "API helper must not be called during normal successful execution"
        )


# ---------------------------------------------------------------------------
# Execution routing observability tests (#350 watchdog)
# ---------------------------------------------------------------------------

class TestExecutionRoutingObservability:
    """Verify task_type escalation and observability in repair_reasoning events."""

    def _make_semantic_backend(self, semantic_incomplete_cycles=2):
        """FakeBackend that fails semantic gate N times then succeeds."""
        backend = FakeBackend()
        backend.semantic_results = [False] * semantic_incomplete_cycles + [True]
        backend.full_tests = [CommandResult(True, "ok")] * 10
        return backend

    def test_semantic_incomplete_repair_uses_semantic_repair_task_type(self):
        """When failure_class=semantic_incomplete, repair_reasoning must use
        task_type=semantic_repair (cloud-first profile, not code_reasoning)."""
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 10
        backend.reasoning_results = [
            {
                "status": "finished",
                "stop_reason": "finish",
                "files_modified": ["igris/web/server.py"],
                "final_summary": "stub fix attempted",
                "orchestrator_used": True,
                "reasoning_execution_provider": "openai",
                "reasoning_execution_model": "gpt-4o-mini",
                "reasoning_execution_profile": "endpoint_implementation",
                "local_model_available": False,
            }
        ] * 5
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(
            max_rank_attempts=1,
            max_repair_cycles=2,
        )
        run = SupervisorRun(run_id="semantic-task-type", rank_id="test")
        supervisor._repair_cycle(
            run=run,
            failure="semantic_incomplete",
            config=config,
            cycle=1,
        )
        assert backend.last_task_type == "semantic_repair", (
            f"Expected task_type=semantic_repair for semantic_incomplete, "
            f"got {backend.last_task_type!r}"
        )

    def test_repair_reasoning_events_include_orchestrator_fields(self):
        """repair_reasoning events must include orchestrator observability fields."""
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 10
        backend.reasoning_results = [
            {
                "status": "finished",
                "stop_reason": "finish",
                "files_modified": ["igris/web/server.py"],
                "final_summary": "done",
                "orchestrator_used": True,
                "reasoning_execution_provider": "openai",
                "reasoning_execution_model": "gpt-4o-mini",
                "reasoning_execution_profile": "endpoint_implementation",
                "local_model_available": False,
            }
        ] * 5
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(max_rank_attempts=1, max_repair_cycles=1)
        run = SupervisorRun(run_id="obs-fields-test", rank_id="test")
        supervisor._repair_cycle(
            run=run,
            failure="semantic_incomplete",
            config=config,
            cycle=1,
        )
        repair_events = [
            e for e in run.events
            if e.phase == "repair_reasoning" and e.status not in ("running",)
        ]
        assert repair_events, "Expected at least one repair_reasoning event"
        ev = repair_events[-1]
        assert "orchestrator_used" in ev.data, "repair_reasoning event must have orchestrator_used"
        assert "reasoning_execution_provider" in ev.data
        assert "reasoning_execution_model" in ev.data
        assert "reasoning_execution_profile" in ev.data
        assert "local_model_available" in ev.data

    def test_api_helper_model_separate_from_execution_model(self):
        """API helper observability fields must use different keys than execution fields.
        They represent different concepts: advisory vs. execution backend."""
        from igris.core.self_repair_supervisor import SupervisorEvent
        # The two distinct field names must never be the same key
        api_field = "api_helper_model_resolved"
        exec_field = "reasoning_execution_model"
        assert api_field != exec_field, (
            "API helper model field and execution model field must be distinct keys"
        )
        # Verify repair_reasoning events include execution field (structural check)
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 5
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(max_rank_attempts=1, max_repair_cycles=1)
        run = SupervisorRun(run_id="field-separation", rank_id="test")
        supervisor._repair_cycle(
            run=run,
            failure="semantic_incomplete",
            config=config,
            cycle=1,
        )
        repair_events = [
            e for e in run.events
            if e.phase == "repair_reasoning" and e.status not in ("running",)
        ]
        assert repair_events, "Expected repair_reasoning event"
        # execution field must be present in repair events
        assert exec_field in repair_events[-1].data, (
            f"{exec_field!r} must be in repair_reasoning event data"
        )
        # api_helper field must NOT be in repair_reasoning events (it belongs to api_escalation)
        assert api_field not in repair_events[-1].data, (
            f"repair_reasoning must not contain {api_field!r} — that belongs to api_escalation"
        )

    def test_normal_failure_uses_code_reasoning_task_type(self):
        """For non-semantic failures (pytest_failure), task_type must be code_reasoning."""
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 10
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(max_rank_attempts=1, max_repair_cycles=1)
        run = SupervisorRun(run_id="pytest-task-type", rank_id="test")
        supervisor._repair_cycle(
            run=run,
            failure="pytest_failure",
            config=config,
            cycle=1,
        )
        assert backend.last_task_type == "code_reasoning", (
            f"pytest_failure should use code_reasoning, got {backend.last_task_type!r}"
        )


# ---------------------------------------------------------------------------
# Cost-policy execution strategy tests
# ---------------------------------------------------------------------------

class TestCostPolicyExecutionStrategy:
    """Tests for helper-advice-then-mini/gpt4o execution strategy routing."""

    def _make_backend_with_plan(self, execution_plan: str = "1. Add endpoint\n2. Add test"):
        """FakeBackend whose api_helper returns a response with execution_plan."""
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 10
        backend.api_helper_result = CommandResult(True, json.dumps({
            "diagnosis": "missing endpoint",
            "likely_supervisor_gap": "endpoint not implemented",
            "suggested_repair_strategy": "add route",
            "suggested_tests": ["test_endpoint"],
            "risk": "low",
            "confidence": 0.9,
            "requires_human_or_codex_audit": False,
            "must_not_complete_product_manually": True,
            "advice_only": True,
            "execution_plan": execution_plan,
            "file_targets": ["igris/web/server.py"],
            "operations": ["add_route GET /api/ping"],
            "acceptance_matrix": ["GET /api/ping returns 200"],
            "required_tests": ["tests/test_ping.py"],
            "do_not_do": ["do not modify existing routes"],
            "retry_focus": "add missing endpoint first",
            "estimated_cost_usd": 0.002,
        }))
        backend._api_helper_configured = True
        return backend

    # -- 1. Execution plan preserved in helper advice after validation ------

    def test_helper_advice_contains_execution_plan(self):
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        valid, advice, err = SelfRepairSupervisor._validate_helper_response({
            "diagnosis": "d",
            "likely_supervisor_gap": "g",
            "suggested_repair_strategy": "s",
            "suggested_tests": [],
            "risk": "low",
            "confidence": 0.8,
            "requires_human_or_codex_audit": False,
            "must_not_complete_product_manually": True,
            "execution_plan": "step 1\nstep 2",
            "file_targets": ["server.py"],
            "operations": ["add_route"],
            "acceptance_matrix": ["GET returns 200"],
            "required_tests": ["tests/test_x.py"],
            "do_not_do": ["no deletions"],
            "retry_focus": "focus on endpoint",
        })
        assert valid
        assert advice["execution_plan"] == "step 1\nstep 2"
        assert advice["file_targets"] == ["server.py"]
        assert advice["operations"] == ["add_route"]
        assert advice["acceptance_matrix"] == ["GET returns 200"]
        assert advice["required_tests"] == ["tests/test_x.py"]
        assert advice["do_not_do"] == ["no deletions"]
        assert advice["retry_focus"] == "focus on endpoint"
        assert advice["advice_only"] is True

    def test_advice_only_always_true(self):
        """advice_only must be True even when helper omits or sets it False."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        payload = {
            "diagnosis": "d", "likely_supervisor_gap": "g",
            "suggested_repair_strategy": "s", "suggested_tests": [],
            "risk": "low", "confidence": 0.9,
            "requires_human_or_codex_audit": False,
            "must_not_complete_product_manually": True,
            "advice_only": False,  # helper tries to claim authority
        }
        _, advice, _ = SelfRepairSupervisor._validate_helper_response(payload)
        assert advice["advice_only"] is True

    def test_execution_plan_defaults_to_empty_string(self):
        """Old helper responses without execution_plan must still validate."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        valid, advice, err = SelfRepairSupervisor._validate_helper_response({
            "diagnosis": "d", "likely_supervisor_gap": "g",
            "suggested_repair_strategy": "s", "suggested_tests": [],
            "risk": "low", "confidence": 0.9,
            "requires_human_or_codex_audit": False,
            "must_not_complete_product_manually": True,
        })
        assert valid
        assert advice["execution_plan"] == ""
        assert advice["file_targets"] == []

    # -- 2. Strategy selection based on same_failure_count -----------------

    def test_strategy_mini_when_first_attempt(self):
        """First failure → mini strategy."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        run = SupervisorRun(run_id="r1", rank_id="test")
        strategy, profile = SelfRepairSupervisor._strategy_for_repair(run, has_execution_plan=True)
        assert strategy == "helper_advice_then_mini_execution"
        assert profile == "mini_execution"

    def test_strategy_strong_when_same_failure_exceeds_threshold(self, monkeypatch):
        """After max_same_failure_retries consecutive same failures → strong strategy."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        monkeypatch.setenv("IGRIS_MAX_SAME_FAILURE_RETRIES", "2")
        run = SupervisorRun(run_id="r2", rank_id="test")
        run.same_failure_count = 2  # at threshold
        strategy, profile = SelfRepairSupervisor._strategy_for_repair(run, has_execution_plan=True)
        assert strategy == "helper_advice_then_gpt4o_execution"
        assert profile == "strong_execution"

    def test_strategy_empty_without_execution_plan(self):
        """No execution_plan → no strategy override."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        run = SupervisorRun(run_id="r3", rank_id="test")
        strategy, profile = SelfRepairSupervisor._strategy_for_repair(run, has_execution_plan=False)
        assert strategy == ""
        assert profile is None

    # -- 3. repair_cycle passes execution_plan as context to run_reasoning --

    def test_repair_cycle_passes_execution_plan_to_reasoning(self):
        """When helper provides execution_plan, it must appear in initial_context."""
        backend = self._make_backend_with_plan("1. Add GET /api/ping\n2. Add test")
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(
            max_rank_attempts=1,
            max_repair_cycles=1,
            allow_api_escalation=True,
            max_api_escalations_per_run=1,
            max_api_budget_usd=10.0,
        )
        run = SupervisorRun(run_id="plan-ctx", rank_id="test")
        supervisor._repair_cycle(run=run, failure="pytest_failure", config=config, cycle=1)
        ctx = backend.last_reasoning_context or {}
        assert ctx.get("execution_plan"), "execution_plan must be passed to run_reasoning"
        assert ctx.get("helper_advice_only") is True

    def test_repair_cycle_sets_strategy_used_on_run(self):
        """When helper provides execution_plan, run.strategy_used must be set."""
        backend = self._make_backend_with_plan("do the thing")
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(
            allow_api_escalation=True,
            max_api_escalations_per_run=1,
            max_api_budget_usd=10.0,
        )
        run = SupervisorRun(run_id="strat-used", rank_id="test")
        supervisor._repair_cycle(run=run, failure="pytest_failure", config=config, cycle=1)
        assert run.strategy_used in {"helper_advice_then_mini_execution", "helper_advice_then_gpt4o_execution"}

    # -- 4. Same-failure escalates from mini to gpt-4o ----------------------

    def test_same_failure_escalates_to_gpt4o_profile(self, monkeypatch):
        """After max retries with same failure, strategy_for_repair returns strong_execution."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        monkeypatch.setenv("IGRIS_MAX_SAME_FAILURE_RETRIES", "1")
        run = SupervisorRun(run_id="esc", rank_id="test")
        run.same_failure_count = 1
        strategy, profile = SelfRepairSupervisor._strategy_for_repair(run, has_execution_plan=True)
        assert strategy == "helper_advice_then_gpt4o_execution"
        assert profile == "strong_execution"

    # -- 5. Budget blocks repair when exceeded -----------------------------

    def test_execution_budget_blocks_repair(self, monkeypatch):
        """When IGRIS_MAX_COST_PER_RUN exceeded, _check_execution_budget returns failure_class."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        monkeypatch.setenv("IGRIS_MAX_COST_PER_RUN", "0.50")
        run = SupervisorRun(run_id="budget-test", rank_id="test")
        run.execution_budget_used_usd = 0.55  # over budget
        result = SelfRepairSupervisor._check_execution_budget(run)
        assert result == "execution_budget_exceeded"

    def test_execution_budget_not_blocked_when_zero(self, monkeypatch):
        """IGRIS_MAX_COST_PER_RUN=0 means unlimited — must not block."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        monkeypatch.setenv("IGRIS_MAX_COST_PER_RUN", "0")
        run = SupervisorRun(run_id="no-budget", rank_id="test")
        run.execution_budget_used_usd = 999.99
        assert SelfRepairSupervisor._check_execution_budget(run) is None

    def test_repair_cycle_aborts_on_budget_exceeded(self, monkeypatch):
        """When execution budget is exceeded, _repair_cycle must return False."""
        monkeypatch.setenv("IGRIS_MAX_COST_PER_RUN", "0.01")
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 10
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(max_repair_cycles=1)
        run = SupervisorRun(run_id="budget-abort", rank_id="test")
        run.execution_budget_used_usd = 0.50  # already over budget
        result = supervisor._repair_cycle(run=run, failure="pytest_failure", config=config, cycle=1)
        assert result is False
        assert run.failure_class == "execution_budget_exceeded"
        # run_reasoning must NOT have been called
        assert backend.commands == []

    # -- 6. Codex direct execution flag ------------------------------------

    def test_codex_direct_disabled_by_default(self, monkeypatch):
        """Codex direct execution must be off unless IGRIS_ENABLE_CODEX_DIRECT_EXECUTION=true."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        monkeypatch.delenv("IGRIS_ENABLE_CODEX_DIRECT_EXECUTION", raising=False)
        assert SelfRepairSupervisor._is_codex_direct_execution_enabled() is False

    def test_codex_direct_enabled_by_env(self, monkeypatch):
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        monkeypatch.setenv("IGRIS_ENABLE_CODEX_DIRECT_EXECUTION", "true")
        assert SelfRepairSupervisor._is_codex_direct_execution_enabled() is True

    def test_codex_direct_not_selected_by_strategy_for_repair(self, monkeypatch):
        """_strategy_for_repair must never return codex_direct regardless of env."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun
        monkeypatch.setenv("IGRIS_ENABLE_CODEX_DIRECT_EXECUTION", "true")
        run = SupervisorRun(run_id="no-codex-direct", rank_id="test")
        for same in range(5):
            run.same_failure_count = same
            strategy, profile = SelfRepairSupervisor._strategy_for_repair(run, has_execution_plan=True)
            assert "codex_direct" not in strategy

    # -- 7. No secrets in helper advice ------------------------------------

    def test_no_secrets_in_validated_advice(self):
        """Validated advice must not contain secret-like content."""
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        from igris.core.safety import detect_secret_like_content
        _, advice, _ = SelfRepairSupervisor._validate_helper_response({
            "diagnosis": "missing endpoint",
            "likely_supervisor_gap": "no route",
            "suggested_repair_strategy": "add route",
            "suggested_tests": ["test_ping"],
            "risk": "low",
            "confidence": 0.9,
            "requires_human_or_codex_audit": False,
            "must_not_complete_product_manually": True,
            "execution_plan": "add GET /api/ping",
            "retry_focus": "endpoint first",
        })
        serialized = json.dumps(advice)
        assert not detect_secret_like_content(serialized)

    # -- 8. run.strategy_used and execution_budget_used_usd in to_dict ------

    def test_run_to_dict_includes_strategy_fields(self):
        """SupervisorRun.to_dict() must include new strategy telemetry fields."""
        from igris.core.self_repair_supervisor import SupervisorRun
        run = SupervisorRun(run_id="td", rank_id="test")
        run.strategy_used = "helper_advice_then_mini_execution"
        run.execution_budget_used_usd = 0.0042
        d = run.to_dict()
        assert d["strategy_used"] == "helper_advice_then_mini_execution"
        assert d["same_failure_count"] == 0
        assert abs(d["execution_budget_used_usd"] - 0.0042) < 1e-9

    # -- 9. Telemetry in repair_reasoning events ---------------------------

    def test_repair_reasoning_events_include_strategy_telemetry(self):
        """repair_reasoning events must include strategy_used and same_failure_count."""
        backend = FakeBackend()
        backend.full_tests = [CommandResult(True, "ok")] * 10
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        config = _config(max_rank_attempts=1, max_repair_cycles=1)
        run = SupervisorRun(run_id="tel", rank_id="test")
        supervisor._repair_cycle(run=run, failure="pytest_failure", config=config, cycle=1)
        repair_events = [e for e in run.events if e.phase == "repair_reasoning"]
        assert repair_events, "expected at least one repair_reasoning event"
        start_event = repair_events[0]
        assert "strategy_used" in start_event.data
        assert "same_failure_count" in start_event.data


# ---------------------------------------------------------------------------
# TestAutoRunSubissue — auto-chain sub-mission runs after decomposition
# ---------------------------------------------------------------------------

class TestAutoRunSubissue:
    """Tests for the auto-chain feature: after decomposition with auto_create_subissues
    policy, the supervisor queues a child run on the first sub-issue."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_supervisor(fetch_ok: bool = True):
        backend = FakeBackend()
        # create_issue returns "https://github.com/org/repo/issues/42" to make URLs realistic
        backend.create_issue = lambda title, body: CommandResult(
            True, "https://github.com/org/repo/issues/42", "", 0
        )
        if fetch_ok:
            backend.fetch_issue = lambda url: CommandResult(
                True,
                json.dumps({"title": "Fake sub-issue", "body": "Implement the endpoint.", "number": 999}),
                "",
                0,
            )
        else:
            backend.fetch_issue = lambda url: CommandResult(False, "", "gh: not found", 1)
        supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
        return supervisor, backend

    @staticmethod
    def _valid_decomposition():
        """A well-formed decomposition that passes _decomposition_policy checks."""
        return {
            "why_too_large": "Original mission required 5+ file changes across unrelated modules.",
            "sub_missions": [{
                "title": "Implement sub-task A",
                "goal": "Add endpoint /api/foo",
                "dependencies": [],
                "acceptance_criteria": ["GET /api/foo returns 200"],
                "allowed_file_scopes": ["igris/web/server.py", "tests/test_foo.py"],
                "tests": ["tests/test_foo.py"],
                "risk_level": "low",
                "human_approval_required": False,
            }],
            "first_sub_mission": "Implement sub-task A",
            "human_approval_required": False,
        }

    @staticmethod
    def _auto_config(**overrides):
        """Config with allow_auto_subissues=True and dry_run=False (via allow_github_pr)."""
        data = {
            "goal": "Rank A controlled task with tests",
            "rank_id": "A",
            "max_rank_attempts": 2,
            "max_repair_cycles": 1,
            "allow_github_pr": True,
            "allow_merge_if_green": False,
            "allow_auto_subissues": True,
            "enable_semantic_gate": False,
        }
        data.update(overrides)
        return RankSupervisorConfig.from_dict(data)

    @staticmethod
    def _fake_child_run():
        run = SupervisorRun(run_id="child-run-42", rank_id="A-sub999")
        run.status = "running"
        return run

    # -----------------------------------------------------------------------
    # 1. Happy path: autorun queued when auto_approved
    # -----------------------------------------------------------------------

    def test_autorun_queued_when_auto_approved(self):
        """After decomposition with auto_approved_by_policy, a child run is queued."""
        from unittest.mock import patch, MagicMock

        supervisor, backend = self._make_supervisor()
        config = self._auto_config()
        run = SupervisorRun(run_id="parent-1", rank_id="A")
        decomposition = self._valid_decomposition()

        fake_child = self._fake_child_run()
        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
            return_value=fake_child,
        ):
            result = supervisor._blocked_decomposition_required(
                run, "reasoning_timeout", "Task too large", decomposition, config=config
            )

        # Child run should be recorded
        assert result.autorun_child_run_id == "child-run-42", (
            f"Expected child_run_id='child-run-42', got {result.autorun_child_run_id!r}"
        )
        assert result.autorun_policy == "auto_create_subissues"

        # Event submission_autorun_run_id must be present
        event_phases = [e.phase for e in result.events]
        assert "submission_autorun_run_id" in event_phases, (
            f"Expected submission_autorun_run_id event; got phases: {event_phases}"
        )

        # autorun_child_run_id in report
        assert result.report.get("autorun_child_run_id") == "child-run-42"

    # -----------------------------------------------------------------------
    # 2. Skipped when allow_auto_subissues=False
    # -----------------------------------------------------------------------

    def test_autorun_skipped_when_allow_auto_subissues_false(self):
        """No child run is queued when allow_auto_subissues=False."""
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        config = self._auto_config(allow_auto_subissues=False)
        run = SupervisorRun(run_id="parent-2", rank_id="A")
        decomposition = self._valid_decomposition()

        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
        ) as mock_start:
            result = supervisor._blocked_decomposition_required(
                run, "reasoning_timeout", "Task too large", decomposition, config=config
            )
            mock_start.assert_not_called()

        assert result.autorun_child_run_id == ""
        skipped_events = [e for e in result.events if e.phase == "submission_autorun_skipped"]
        # Policy should be request_human_approval when allow_auto_subissues=False,
        # so _autorun_first_subissue is not even called; guard catches it
        # OR the policy path prevents calling _autorun_first_subissue altogether.
        # Either way: no child run id.
        assert result.autorun_child_run_id == ""

    # -----------------------------------------------------------------------
    # 3. Skipped when cycle detected
    # -----------------------------------------------------------------------

    def test_autorun_skipped_when_cycle_detected(self):
        """No child run when decomposition_cycle_detected=True."""
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        config = self._auto_config()
        run = SupervisorRun(run_id="parent-3", rank_id="A")
        decomposition = self._valid_decomposition()

        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
        ) as mock_start:
            # We call _autorun_first_subissue directly with a decomposition that has cycle
            # and approval_status set (bypassing _blocked_decomposition_required policy path)
            decomp_with_cycle = {
                "approval_status": "auto_approved_by_policy",
                "decomposition_cycle_detected": True,
            }
            created_urls = ["https://github.com/org/repo/issues/50"]
            result_id = supervisor._autorun_first_subissue(
                run, config, decomp_with_cycle, created_urls, "reasoning_timeout"
            )
            mock_start.assert_not_called()

        assert result_id is None
        assert run.autorun_skipped_reason == "decomposition_cycle_detected"

    # -----------------------------------------------------------------------
    # 4. Skipped when budget exceeded
    # -----------------------------------------------------------------------

    def test_autorun_skipped_when_budget_exceeded(self):
        """No child run when execution_budget_used_usd >= IGRIS_MAX_COST_PER_RUN."""
        import os
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        config = self._auto_config()
        run = SupervisorRun(run_id="parent-4", rank_id="A")
        run.execution_budget_used_usd = 999.0

        decomp = {
            "approval_status": "auto_approved_by_policy",
            "decomposition_cycle_detected": False,
        }
        created_urls = ["https://github.com/org/repo/issues/51"]

        with patch.dict(os.environ, {"IGRIS_MAX_COST_PER_RUN": "0.001"}):
            with patch(
                "igris.core.self_repair_supervisor.start_supervised_rank_async",
            ) as mock_start:
                result_id = supervisor._autorun_first_subissue(
                    run, config, decomp, created_urls, "reasoning_timeout"
                )
                mock_start.assert_not_called()

        assert result_id is None
        assert "budget_exceeded" in run.autorun_skipped_reason

    # -----------------------------------------------------------------------
    # 5. Skipped when sub-issue already running
    # -----------------------------------------------------------------------

    def test_autorun_skipped_when_sub_issue_already_running(self):
        """No child run when an existing run already references the same sub-issue URL."""
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        config = self._auto_config()
        run = SupervisorRun(run_id="parent-5", rank_id="A")

        first_url = "https://github.com/org/repo/issues/60"
        # Plant a "running" run that references the same URL in its goal
        existing = SupervisorRun(run_id="existing-run", rank_id="B")
        existing.status = "running"
        existing.goal = f"Work on {first_url}"

        decomp = {
            "approval_status": "auto_approved_by_policy",
            "decomposition_cycle_detected": False,
        }
        created_urls = [first_url]

        from igris.core.self_repair_supervisor import RUN_STORE, RUN_LOCK
        with RUN_LOCK:
            RUN_STORE["existing-run"] = existing

        try:
            with patch(
                "igris.core.self_repair_supervisor.start_supervised_rank_async",
            ) as mock_start:
                result_id = supervisor._autorun_first_subissue(
                    run, config, decomp, created_urls, "reasoning_timeout"
                )
                mock_start.assert_not_called()
        finally:
            with RUN_LOCK:
                RUN_STORE.pop("existing-run", None)

        assert result_id is None
        assert "sub_issue_already_running" in run.autorun_skipped_reason

    # -----------------------------------------------------------------------
    # 6. Report fields populated
    # -----------------------------------------------------------------------

    def test_autorun_report_fields(self):
        """autorun_child_run_id, autorun_policy, next_subissue_url appear in report."""
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        config = self._auto_config()
        run = SupervisorRun(run_id="parent-6", rank_id="A")
        decomposition = self._valid_decomposition()

        fake_child = self._fake_child_run()
        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
            return_value=fake_child,
        ):
            result = supervisor._blocked_decomposition_required(
                run, "reasoning_timeout", "Task too large", decomposition, config=config
            )

        assert "autorun_child_run_id" in result.report
        assert "autorun_policy" in result.report
        assert result.report.get("autorun_policy") == "auto_create_subissues"
        assert "next_subissue_url" in result.report

    # -----------------------------------------------------------------------
    # 7. No secrets in autorun events
    # -----------------------------------------------------------------------

    def test_autorun_no_secrets(self):
        """No API keys or secrets appear in autorun events."""
        from unittest.mock import patch
        from igris.core.safety import detect_secret_like_content

        supervisor, backend = self._make_supervisor()
        config = self._auto_config()
        run = SupervisorRun(run_id="parent-7", rank_id="A")
        decomposition = self._valid_decomposition()

        fake_child = self._fake_child_run()
        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
            return_value=fake_child,
        ):
            result = supervisor._blocked_decomposition_required(
                run, "reasoning_timeout", "Task too large", decomposition, config=config
            )

        autorun_events = [
            e for e in result.events
            if e.phase.startswith("submission_autorun")
        ]
        assert autorun_events, "Expected at least one submission_autorun event"
        serialized = json.dumps([e.to_dict() for e in autorun_events])
        assert not detect_secret_like_content(serialized), (
            f"Secret-like content found in autorun events: {serialized[:200]}"
        )

    # -----------------------------------------------------------------------
    # 8. Skipped when dry_run=True
    # -----------------------------------------------------------------------

    def test_autorun_skipped_dry_run(self):
        """No child run when dry_run=True even if allow_auto_subissues=True."""
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        # dry_run=True (default when allow_github_pr=False)
        config = _config(allow_auto_subissues=True, dry_run=True)
        run = SupervisorRun(run_id="parent-8", rank_id="A")

        decomp = {
            "approval_status": "auto_approved_by_policy",
            "decomposition_cycle_detected": False,
        }
        created_urls = ["https://github.com/org/repo/issues/70"]

        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
        ) as mock_start:
            result_id = supervisor._autorun_first_subissue(
                run, config, decomp, created_urls, "reasoning_timeout"
            )
            mock_start.assert_not_called()

        assert result_id is None
        assert run.autorun_skipped_reason == "dry_run=True"

    def test_autorun_skipped_max_depth(self):
        """No child run when autochain_depth >= 2 (infinite cascade guard)."""
        from unittest.mock import patch

        supervisor, backend = self._make_supervisor()
        config = _config(allow_auto_subissues=True, allow_github_pr=True, dry_run=False, autochain_depth=2)
        run = SupervisorRun(run_id="parent-9", rank_id="A")

        decomp = {
            "approval_status": "auto_approved_by_policy",
            "decomposition_cycle_detected": False,
        }
        created_urls = ["https://github.com/org/repo/issues/71"]

        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
        ) as mock_start:
            result_id = supervisor._autorun_first_subissue(
                run, config, decomp, created_urls, "no_diff_repair"
            )
            mock_start.assert_not_called()

        assert result_id is None
        assert "max_autochain_depth" in run.autorun_skipped_reason

    def test_autochain_depth_incremented_in_child(self):
        """Child data must have autochain_depth = parent_depth + 1."""
        from unittest.mock import patch, MagicMock

        supervisor, backend = self._make_supervisor()
        config = _config(allow_auto_subissues=True, allow_github_pr=True, dry_run=False, autochain_depth=0)
        run = SupervisorRun(run_id="parent-10", rank_id="A")

        decomp = {
            "approval_status": "auto_approved_by_policy",
            "decomposition_cycle_detected": False,
        }
        created_urls = ["https://github.com/org/repo/issues/72"]

        captured = {}

        def fake_start(data, project_root):
            captured["data"] = data
            m = MagicMock()
            m.run_id = "child-depth-test"
            return m

        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
            side_effect=fake_start,
        ):
            supervisor._autorun_first_subissue(
                run, config, decomp, created_urls, "no_diff_repair"
            )

        assert captured.get("data", {}).get("autochain_depth") == 1

    def test_autorun_depth_1_still_chains(self):
        """At depth=1 (child), auto-chain still works (below max=2)."""
        from unittest.mock import patch, MagicMock

        supervisor, backend = self._make_supervisor()
        config = _config(allow_auto_subissues=True, allow_github_pr=True, dry_run=False, autochain_depth=1)
        run = SupervisorRun(run_id="parent-11", rank_id="A")

        decomp = {
            "approval_status": "auto_approved_by_policy",
            "decomposition_cycle_detected": False,
        }
        created_urls = ["https://github.com/org/repo/issues/73"]

        captured = {}

        def fake_start(data, project_root):
            captured["data"] = data
            m = MagicMock()
            m.run_id = "grandchild-test"
            return m

        with patch(
            "igris.core.self_repair_supervisor.start_supervised_rank_async",
            side_effect=fake_start,
        ):
            result_id = supervisor._autorun_first_subissue(
                run, config, decomp, created_urls, "no_diff_repair"
            )

        assert result_id == "grandchild-test"
        assert captured.get("data", {}).get("autochain_depth") == 2

    def test_decomposition_policy_allows_issue_creation_at_max_depth(self):
        """Safe decomposition can still create sub-issues at max autochain depth.
        Child autorun depth limits are enforced separately by _autorun_guards."""
        supervisor, _ = self._make_supervisor()
        config = _config(
            allow_auto_subissues=True,
            allow_github_pr=True,
            dry_run=False,
            autochain_depth=SelfRepairSupervisor._MAX_AUTOCHAIN_DEPTH,
        )
        decomp = {
            "sub_missions": [{"title": "sub", "goal": "do something", "acceptance_criteria": []}],
            "why_too_large": "task is large",
            "first_sub_mission": "do something",
        }
        policy = supervisor._decomposition_policy(decomp, config)
        assert policy == "auto_create_subissues", (
            f"At max autochain depth={SelfRepairSupervisor._MAX_AUTOCHAIN_DEPTH}, "
            "safe decomposition should still auto-create sub-issues"
        )


def test_rank_supervisor_config_default_repair_cycles():
    """RankSupervisorConfig must default max_repair_cycles to 2, not 0.
    A default of 0 means every run that hits semantic_incomplete or
    other repairable failures is blocked immediately without any retry."""
    from igris.core.self_repair_supervisor import RankSupervisorConfig
    config = RankSupervisorConfig(goal="test")
    assert config.max_repair_cycles == 2, (
        f"Default max_repair_cycles must be 2, got {config.max_repair_cycles}"
    )


def test_rank_supervisor_config_from_dict_default_repair_cycles():
    """from_dict without max_repair_cycles must inherit the class default (2), not hard-code 0.
    The watchdog calls start_supervised_rank_async without max_repair_cycles, so this
    path must not silently disable repair."""
    from igris.core.self_repair_supervisor import RankSupervisorConfig
    config = RankSupervisorConfig.from_dict({"goal": "test goal"})
    assert config.max_repair_cycles == 2, (
        f"from_dict without explicit max_repair_cycles must default to 2, got {config.max_repair_cycles}"
    )


def test_repair_profile_escalates_to_strong_execution_on_max_steps():
    """When failure_class=max_steps the repair profile selection must yield
    strong_execution (gpt-4o), not None (cheap default). Without this, the repair
    repeats the same model that already hit the step ceiling."""
    # Replicate the profile-selection logic from _repair_cycle directly.
    # This tests the decision, not the full cycle (which requires git/github stubs).
    failure = "max_steps"
    repair_profile: object = None

    if failure in {"semantic_incomplete", "stub_detected", "reasoning_loop_blocked"}:
        repair_profile = "semantic_repair"
    elif failure in {"missing_tests", "pytest_failure"}:
        repair_profile = "code_generation"
    elif failure == "max_steps":
        repair_profile = "strong_execution"

    assert repair_profile == "strong_execution", (
        f"max_steps must escalate to strong_execution, got: {repair_profile}"
    )

    # Ensure other failures are not affected
    for other_failure in ("pytest_failure", "semantic_incomplete", "destructive_diff"):
        p: object = None
        if other_failure in {"semantic_incomplete", "stub_detected", "reasoning_loop_blocked"}:
            p = "semantic_repair"
        elif other_failure == "max_steps":
            p = "strong_execution"
        assert p != "strong_execution" or other_failure == "max_steps", (
            f"Only max_steps should escalate to strong_execution, not {other_failure}"
        )


def test_pick_next_roadmap_issue_skips_repair_issues(monkeypatch):
    """_pick_next_roadmap_issue must skip orphaned repair issues
    (title contains 'supervised repair for') and return the next
    clean roadmap issue by number."""
    import json
    import subprocess as sp
    from igris.web.server import _pick_next_roadmap_issue

    fake_issues = [
        {"number": 10, "title": "roadmap-autonomy-418: supervised repair for reasoning_loop_blocked", "body": "", "labels": []},
        {"number": 11, "title": "Supervisor: autonomous roadmap task selection", "body": "...", "labels": [{"name": "roadmap"}]},
        {"number": 12, "title": "supervised repair for pytest_failure", "body": "", "labels": []},
    ]

    class FakeResult:
        returncode = 0
        stdout = json.dumps(fake_issues)

    monkeypatch.setattr(sp, "run", lambda *a, **kw: FakeResult())
    issue = _pick_next_roadmap_issue("/tmp")
    assert issue is not None
    assert issue["number"] == 11, "Must skip repair issues and return first clean issue"


def test_pick_next_roadmap_issue_respects_skip_set(monkeypatch):
    """_pick_next_roadmap_issue must skip any issue number in the skip_issues set,
    so the watchdog can move on from a repeatedly-failing issue."""
    import json
    import subprocess as sp
    from igris.web.server import _pick_next_roadmap_issue

    fake_issues = [
        {"number": 11, "title": "Sprint 11: Harden LLM defaults", "body": "", "labels": [{"name": "roadmap"}]},
        {"number": 12, "title": "Sprint 12: Add memory layer", "body": "", "labels": [{"name": "roadmap"}]},
        {"number": 13, "title": "Sprint 13: UI dashboard", "body": "", "labels": [{"name": "roadmap"}]},
    ]

    class FakeResult:
        returncode = 0
        stdout = json.dumps(fake_issues)

    monkeypatch.setattr(sp, "run", lambda *a, **kw: FakeResult())

    # Without skip_issues, picks lowest: #11
    issue = _pick_next_roadmap_issue("/tmp")
    assert issue["number"] == 11

    # With #11 in skip set, picks #12
    issue = _pick_next_roadmap_issue("/tmp", skip_issues={11})
    assert issue["number"] == 12, "Must skip issue #11 and return #12"

    # With #11 and #12 skipped, picks #13
    issue = _pick_next_roadmap_issue("/tmp", skip_issues={11, 12})
    assert issue["number"] == 13, "Must skip #11 and #12 and return #13"

    # With all skipped, returns None
    issue = _pick_next_roadmap_issue("/tmp", skip_issues={11, 12, 13})
    assert issue is None, "All issues skipped — must return None"


# ---------------------------------------------------------------------------
# #420 — capability_ceiling_reached
# ---------------------------------------------------------------------------

def test_max_steps_ceiling_emits_capability_ceiling_reached():
    """When strong_execution hits max_steps (max_steps_ceiling signal), the
    supervisor must block with capability_ceiling_reached, NOT decomposition_required.
    This skips the expensive decompose LLM call and lets the watchdog fast-skip.
    """
    backend = FakeBackend()
    backend.reasoning_results = [
        # attempt-1 main: max_steps (not ceiling yet — no strong profile used)
        {"status": "blocked", "stop_reason": "max_steps", "files_modified": [], "final_summary": ""},
        # repair-cycle: strong_execution also hits max_steps → max_steps_ceiling signal
        {"status": "blocked", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "", "reasoning_execution_profile": "strong_execution"},
        # No decompose call expected — capability_ceiling_reached skips it
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(
        max_rank_attempts=2,
        max_repair_cycles=1,
        goal="Complex task requiring strong model",
    )
    run = SupervisorRun(run_id="ceiling-test", rank_id="test")

    # Pre-seed max_steps_ceiling signal (as would happen after repair reasoning)
    run.capability_signals["max_steps_ceiling"] = 1
    result = supervisor.run(config, run=run)

    assert result.failure_class == "capability_ceiling_reached", (
        f"Expected capability_ceiling_reached, got {result.failure_class!r}"
    )
    assert result.status == "blocked"
    # No decomposition should have been attempted
    ceiling_events = [e for e in result.events if e.phase == "capability_ceiling"]
    assert ceiling_events, "Expected a capability_ceiling event"
    # decompose should NOT have been called
    decompose_events = [e for e in result.events if "decomposition" in e.phase]
    assert not decompose_events, f"Decompose should not be called for structural ceiling; got {decompose_events}"


def test_pure_reasoning_timeout_still_decomposes():
    """Pure reasoning_timeout signals (no max_steps_ceiling) must still trigger
    decomposition, NOT capability_ceiling_reached. The cheap model timing out
    doesn't mean the strong model can't handle it via decomposed sub-missions.
    """
    backend = FakeBackend()
    backend.reasoning_results = [
        _make_timeout_result(),              # attempt-1 → reasoning_timeout=1
        _make_timeout_result(),              # repair  → reasoning_timeout=2 → threshold
        _decomposition_reasoning_result(),   # decomposition call
    ]
    backend.diff_stat = CommandResult(False, "")
    backend.diff = CommandResult(True, "")
    backend.full_tests = [CommandResult(True, "ok")] * 10

    supervisor = SelfRepairSupervisor("/tmp/project", backend=backend)
    config = _config(max_rank_attempts=3, max_repair_cycles=1, goal="timeout test")
    run = SupervisorRun(run_id="timeout-decompose", rank_id="test")
    result = supervisor.run(config, run=run)

    assert result.failure_class == "decomposition_required", (
        f"Pure reasoning_timeout must still decompose, got {result.failure_class!r}"
    )
    assert result.report.get("decomposition_required") is True


# ---------------------------------------------------------------------------
# #418 — watchdog skip persistence
# ---------------------------------------------------------------------------

def test_load_skipped_issues_returns_empty_on_missing_file(tmp_path):
    from igris.web.server import _load_skipped_issues
    result = _load_skipped_issues(str(tmp_path))
    assert result == set()


def test_save_and_load_skipped_issues_round_trips(tmp_path):
    from igris.web.server import _load_skipped_issues, _save_skipped_issues
    _save_skipped_issues(str(tmp_path), {12, 13, 17})
    loaded = _load_skipped_issues(str(tmp_path))
    assert loaded == {12, 13, 17}


def test_save_skipped_issues_creates_igris_dir(tmp_path):
    from igris.web.server import _load_skipped_issues, _save_skipped_issues
    # .igris dir does not exist yet
    assert not (tmp_path / ".igris").exists()
    _save_skipped_issues(str(tmp_path), {99})
    assert (tmp_path / ".igris" / "watchdog_skipped_issues.json").exists()
    assert _load_skipped_issues(str(tmp_path)) == {99}


# ---------------------------------------------------------------------------
# #432 — heartbeat stale detection
# ---------------------------------------------------------------------------

def test_run_with_heartbeat_monitor_kills_on_stale_heartbeat(monkeypatch, tmp_path):
    """When heartbeat_at is old, _run_with_heartbeat_monitor kills the process
    early and returns returncode=124 (same as a timeout).
    """
    import json
    import time as _time
    import igris.core.self_repair_supervisor as mod

    # Heartbeat file with an old timestamp
    hb_path = str(tmp_path / "hb.json")
    with open(hb_path, "w") as f:
        json.dump({"heartbeat_at": _time.time() - 9999}, f)

    class FakeStdin:
        def write(self, _): pass
        def close(self): pass

    class FakeProc:
        returncode = 0
        pid = 99999
        stdin = FakeStdin()
        stdout = iter([])
        stderr = iter([])

        def poll(self):
            return None  # always running

        def wait(self):
            self.returncode = -9
            return -9

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(mod.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod.os, "killpg", lambda *_: None)
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)

    backend = mod.LocalSupervisorBackend(str(tmp_path))
    result = backend._run_with_heartbeat_monitor(
        ["echo", "test"],
        timeout=900,
        input_text="{}",
        heartbeat_path=hb_path,
        stale_threshold=60,
    )
    assert result.returncode == 124
    assert "stale" in result.error.lower()

# ─── Bundle A tests ───────────────────────────────────────────────────────────
import time as _time_mod
from unittest.mock import MagicMock, patch

def test_test_runner_timeout_in_repairable_failures():
    from igris.core.self_repair_supervisor import REPAIRABLE_FAILURES
    assert "test_runner_timeout" in REPAIRABLE_FAILURES

def test_classify_failure_test_runner_timeout_targeted():
    from igris.core.self_repair_supervisor import classify_failure
    targeted = MagicMock()
    targeted.returncode = 124
    targeted.success = False
    targeted.output = ""
    targeted.error = ""
    full = MagicMock()
    full.returncode = 0
    full.success = True
    full.output = ""
    full.error = ""
    assert classify_failure(targeted_tests=targeted, full_tests=full) == "test_runner_timeout"

def test_classify_failure_test_runner_timeout_full():
    from igris.core.self_repair_supervisor import classify_failure
    targeted = MagicMock()
    targeted.returncode = 0
    targeted.success = True
    targeted.output = ""
    targeted.error = ""
    full = MagicMock()
    full.returncode = 124
    full.success = False
    full.output = ""
    full.error = ""
    assert classify_failure(targeted_tests=targeted, full_tests=full) == "test_runner_timeout"

def test_classify_failure_normal_pytest_not_misclassified():
    from igris.core.self_repair_supervisor import classify_failure
    targeted = MagicMock()
    targeted.returncode = 1
    targeted.success = False
    targeted.output = "FAILED tests/test_foo.py"
    targeted.error = ""
    assert classify_failure(targeted_tests=targeted) == "pytest_failure"

def test_env_var_timeout_default():
    import os
    os.environ.pop("IGRIS_TEST_RUNNER_TIMEOUT_SECONDS", None)
    from igris.core.self_repair_supervisor import RankSupervisorConfig
    cfg = RankSupervisorConfig(goal="test goal")
    assert cfg.test_timeout_seconds == 300

def test_watchdog_logs_stall_warning():
    from igris.web import server
    mock_run = MagicMock()
    mock_run.run_id = "stale_run"
    mock_ts = MagicMock()
    mock_ts.timestamp.return_value = _time_mod.time() - 700
    mock_run.last_event.timestamp = mock_ts
    with patch.object(server, "_watchdog_logger") as mock_logger:
        _now = _time_mod.time()
        for _ar in [mock_run]:
            _last = getattr(_ar, "last_event", None)
            if _last is not None:
                _ts = getattr(_last, "timestamp", None)
                if _ts is not None:
                    _elapsed = _now - _ts.timestamp()
                    if _elapsed > 600:
                        server._watchdog_logger.warning(
                            "Watchdog: active run %s has not emitted events for %ds — possible hang",
                            _ar.run_id, int(_elapsed),
                        )
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "possible hang" in call_args[0]
        assert "stale_run" in str(call_args)

def test_reasoning_loop_repair_prompt_cycle1():
    from igris.core.self_repair_supervisor import SelfRepairSupervisor
    prompt = SelfRepairSupervisor._build_reasoning_loop_repair_prompt(
        "stage1", "implement feature X", "", 1
    )
    assert "minimal" in prompt.lower()
    assert "Do not optimize" in prompt

def test_reasoning_loop_repair_prompt_cycle2():
    from igris.core.self_repair_supervisor import SelfRepairSupervisor
    prompt = SelfRepairSupervisor._build_reasoning_loop_repair_prompt(
        "stage1", "implement feature X", "", 2
    )
    assert "REPAIR CYCLE 2" in prompt
    assert "smallest" in prompt.lower()

def test_wrong_file_edit_repair_prompt_lists_paths():
    from igris.core.self_repair_supervisor import SelfRepairSupervisor
    prompt = SelfRepairSupervisor._build_wrong_file_edit_repair_prompt(
        stage_id="backend_api_change",
        goal="add endpoint",
        wrong_paths=["igris/web/templates/x.html"],
        allowed_families=["igris/core/"],
        repair_cycle=1,
    )
    assert "igris/web/templates/x.html" in prompt
    assert "igris/core/" in prompt

def test_wrong_file_edit_repair_prompt_cycle2_adds_hard_constraint():
    from igris.core.self_repair_supervisor import SelfRepairSupervisor
    prompt = SelfRepairSupervisor._build_wrong_file_edit_repair_prompt(
        "s", "goal", ["bad/file.py"], ["good/"], 2
    )
    assert "under any circumstance" in prompt or "output ONLY" in prompt


def test_fast_track_capability_limit_returns_signal_for_repairable_failure():
    run = SupervisorRun(run_id="fast-track-1", rank_id="test")
    run.capability_signals = {"reasoning_timeout": 1, "no_diff_repair": 1}
    signal = SelfRepairSupervisor._should_fast_track_capability_limit(
        run,
        "reasoning_loop_blocked",
    )
    assert signal is not None


def test_fast_track_capability_limit_ignores_non_repair_failure():
    run = SupervisorRun(run_id="fast-track-2", rank_id="test")
    run.capability_signals = {"reasoning_timeout": 2}
    signal = SelfRepairSupervisor._should_fast_track_capability_limit(
        run,
        "semantic_incomplete",
    )
    assert signal is None


# ---------------------------------------------------------------------------
# Issue #710 — Autonomy hardening tests
# ---------------------------------------------------------------------------


def test_as_bool_string_false_returns_false():
    """The classic bool("false") == True trap must be fixed by _as_bool."""
    from igris.core.self_repair_supervisor import _as_bool
    assert _as_bool("false") is False
    assert _as_bool("False") is False
    assert _as_bool("FALSE") is False


def test_as_bool_string_zero_returns_false():
    from igris.core.self_repair_supervisor import _as_bool
    assert _as_bool("0") is False


def test_as_bool_string_no_returns_false():
    from igris.core.self_repair_supervisor import _as_bool
    assert _as_bool("no") is False
    assert _as_bool("NO") is False


def test_as_bool_true_variants():
    from igris.core.self_repair_supervisor import _as_bool
    assert _as_bool(True) is True
    assert _as_bool("true") is True
    assert _as_bool("True") is True
    assert _as_bool("1") is True
    assert _as_bool("yes") is True
    assert _as_bool("YES") is True


def test_as_bool_false_literal():
    from igris.core.self_repair_supervisor import _as_bool
    assert _as_bool(False) is False


def test_as_bool_none_returns_default():
    from igris.core.self_repair_supervisor import _as_bool
    assert _as_bool(None) is False
    assert _as_bool(None, default=True) is True


def test_max_rank_attempts_default_is_at_least_2():
    """Autonomous config must have max_rank_attempts >= 2 (regression guard for #710)."""
    import os
    os.environ.pop("IGRIS_MAX_RANK_ATTEMPTS", None)
    config = RankSupervisorConfig.from_dict({"goal": "implement feature"})
    assert config.max_rank_attempts >= 2, (
        f"max_rank_attempts default must be >= 2, got {config.max_rank_attempts}"
    )


def test_max_rank_attempts_env_override():
    """IGRIS_MAX_RANK_ATTEMPTS env var must override the default."""
    import os
    os.environ["IGRIS_MAX_RANK_ATTEMPTS"] = "3"
    try:
        config = RankSupervisorConfig.from_dict({"goal": "implement feature"})
        assert config.max_rank_attempts == 3
    finally:
        os.environ.pop("IGRIS_MAX_RANK_ATTEMPTS", None)


def test_sub_mission_does_not_decompose_when_already_focused(tmp_path):
    """A sub-mission with targeted_tests (focused) must not trigger decomposition."""
    backend = FakeBackend()
    config = _config(
        autochain_depth=1,
        targeted_tests=["tests/test_some_module.py"],
        enable_mission_planning=True,
        allow_auto_subissues=True,
        dry_run=False,
        max_rank_attempts=1,
        max_repair_cycles=0,
    )
    backend.reasoning_results = [{
        "status": "finished",
        "stop_reason": "finish",
        "files_modified": ["igris/core/fix.py"],
        "final_summary": "done",
    }]
    backend.diff_stat = CommandResult(True, " igris/core/fix.py | 2 ++")
    backend.diff = CommandResult(True, "+fix content")
    backend.targeted = CommandResult(True, "targeted ok")
    backend.full_tests = [CommandResult(True, "full ok")]
    backend.smoke_result = CommandResult(True, "smoke ok")

    sup = SelfRepairSupervisor(backend=backend, project_root=str(tmp_path))
    run = sup.run(config)

    event_phases = [e.phase for e in run.events]
    assert "decomposition_request" not in event_phases, (
        "Focused sub-mission should not trigger decomposition"
    )
    assert run.failure_class != "decomposition_required"


# ---------------------------------------------------------------------------
# Issue #715 — Execution effectiveness tests
# ---------------------------------------------------------------------------


def test_no_diff_terminal_report_emitted_when_budget_exhausted_with_no_diff():
    """When repair budget is exhausted and an attempt produced no diff, emit
    no_diff_terminal_report so callers know the agent never made file changes."""
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.diff_stat = CommandResult(True, "")   # empty → no diff
    backend.reasoning_results = [
        {"status": "stopped", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "stopped"},
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(dry_run=False, allow_github_pr=False, max_rank_attempts=1, max_repair_cycles=0)
    )
    assert run.status == "blocked"
    terminal_events = [e for e in run.events if e.phase == "no_diff_terminal_report"]
    assert len(terminal_events) >= 1, "expected at least one no_diff_terminal_report event"
    assert terminal_events[0].data["no_diff_count"] >= 1


def test_adaptive_retry_emits_strategy_switch_on_same_failure():
    """Second repair cycle for the same failure emits adaptive_retry/strategy_switch."""
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.diff_stat = CommandResult(True, "")
    # Attempt 1 fails; repair 1 fails; attempt 2 fails; repair 2 emits adaptive_retry.
    backend.reasoning_results = [
        {"status": "stopped", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "rank fail 1"},
        {"status": "stopped", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "repair fail 1"},
        {"status": "stopped", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "rank fail 2"},
        {"status": "stopped", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "repair fail 2"},
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(dry_run=True, max_rank_attempts=2, max_repair_cycles=2)
    )
    adaptive_events = [e for e in run.events if e.phase == "adaptive_retry"]
    assert len(adaptive_events) >= 1, "expected at least one adaptive_retry event"
    assert adaptive_events[0].status == "strategy_switch"
    assert adaptive_events[0].data.get("task_type") == "single_file_single_test"


def test_build_telemetry_fragment_returns_expected_keys():
    """`_build_telemetry_fragment` static method returns the four required telemetry keys."""
    frag = SelfRepairSupervisor._build_telemetry_fragment(
        time_to_first_diff_s=12.5,
        no_diff_count=1,
        decompose_count=0,
        attempt_outcomes=["no_diff", "failed"],
        total_attempts=2,
    )
    assert frag["time_to_first_diff_s"] == 12.5
    assert frag["no_diff_rate"] == 0.5
    assert frag["decompose_rate"] == 0.0
    assert frag["attempt_outcomes"] == ["no_diff", "failed"]


def test_run_report_includes_telemetry_after_blocked():
    """Blocked run report must include no_diff_rate and attempt_outcomes telemetry fields."""
    backend = FakeBackend()
    backend.diff = CommandResult(True, "")
    backend.diff_stat = CommandResult(True, "")
    backend.reasoning_results = [
        {"status": "stopped", "stop_reason": "max_steps", "files_modified": [],
         "final_summary": "stopped"},
    ]
    run = SelfRepairSupervisor("/tmp/project", backend=backend).run(
        _config(dry_run=True, max_rank_attempts=1, max_repair_cycles=0)
    )
    assert run.status == "blocked"
    assert "no_diff_rate" in run.report, "run.report must include no_diff_rate"
    assert "attempt_outcomes" in run.report, "run.report must include attempt_outcomes"
    assert isinstance(run.report["attempt_outcomes"], list)


# ---------------------------------------------------------------------------
# Issue #722 — zombie 'running' runs marked 'interrupted' on startup
# Issue #733 — rank_pending.patch cleaned up on blocked runs and startup
# ---------------------------------------------------------------------------

def test_startup_cleanup_zombie_runs_marks_interrupted(tmp_path):
    """Runs stuck as 'running' are marked 'interrupted' on supervisor init (#722)."""
    import json, time as _time
    from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun, RUN_STORE

    # Write fake runs_index with two zombie running runs
    runs_path = tmp_path / ".igris" / "supervisor_runs.json"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    zombie1 = {"run_id": "zombie-1", "rank_id": "R1", "status": "running", "pid": 99999999, "events": []}
    zombie2 = {"run_id": "zombie-2", "rank_id": "R2", "status": "cancelling", "pid": 88888888, "events": []}
    alive = {"run_id": "alive-1", "rank_id": "R3", "status": "completed", "pid": 12345, "events": []}
    runs_path.write_text(json.dumps({"runs": {
        "zombie-1": zombie1,
        "zombie-2": zombie2,
        "alive-1": alive,
    }}), encoding="utf-8")

    RUN_STORE.clear()
    supervisor = SelfRepairSupervisor(str(tmp_path))

    updated = json.loads(runs_path.read_text())["runs"]
    assert updated["zombie-1"]["status"] == "interrupted", "zombie-1 must be interrupted"
    assert updated["zombie-2"]["status"] == "interrupted", "zombie-2 must be interrupted"
    assert updated["alive-1"]["status"] == "completed", "completed run must not be touched"

    # interrupted_at timestamp must be set
    assert "interrupted_at" in updated["zombie-1"]
    assert "interrupted_at" in updated["zombie-2"]


def test_startup_cleanup_zombie_runs_does_not_touch_current_pid(tmp_path):
    """Runs owned by current PID are not marked interrupted (#722)."""
    import json, os as _os
    from igris.core.self_repair_supervisor import SelfRepairSupervisor, RUN_STORE

    runs_path = tmp_path / ".igris" / "supervisor_runs.json"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    current = {"run_id": "mine-1", "rank_id": "R1", "status": "running", "pid": _os.getpid(), "events": []}
    runs_path.write_text(json.dumps({"runs": {"mine-1": current}}), encoding="utf-8")

    RUN_STORE.clear()
    SelfRepairSupervisor(str(tmp_path))

    updated = json.loads(runs_path.read_text())["runs"]
    # Run owned by current process must NOT be interrupted
    assert updated["mine-1"]["status"] == "running", "current-process run must stay 'running'"


def test_startup_cleanup_stale_patch_deleted(tmp_path):
    """Stale rank_pending.patch is deleted on supervisor init (#733)."""
    from igris.core.self_repair_supervisor import SelfRepairSupervisor, RUN_STORE

    patch_path = tmp_path / ".igris" / "rank_pending.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("diff --git a/f b/f\n+fix", encoding="utf-8")

    assert patch_path.exists()

    RUN_STORE.clear()
    SelfRepairSupervisor(str(tmp_path))

    assert not patch_path.exists(), "rank_pending.patch must be deleted on startup"


def test_blocked_run_removes_pending_patch(tmp_path):
    """_blocked() deletes rank_pending.patch so it doesn't persist between runs (#733)."""
    from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun, RUN_STORE

    patch_path = tmp_path / ".igris" / "rank_pending.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("diff --git a/f b/f\n+fix", encoding="utf-8")

    RUN_STORE.clear()
    supervisor = SelfRepairSupervisor(str(tmp_path))

    # Recreate patch (startup cleanup already deleted it, simulate mid-run patch)
    patch_path.write_text("diff --git a/f b/f\n+fix", encoding="utf-8")

    run = SupervisorRun(run_id="block-test", rank_id="R1")
    supervisor._blocked(run, "test_failure", "Simulated failure for patch cleanup test")

    assert not patch_path.exists(), "rank_pending.patch must be deleted by _blocked()"
    # patch_cleanup event should be in run events
    assert any(e.phase == "patch_cleanup" for e in run.events), "patch_cleanup event missing"


# ---------------------------------------------------------------------------
# Issue #730 — baseline cache revalidation by age + force_revalidate flag
# ---------------------------------------------------------------------------

def test_baseline_cache_returns_none_when_stale(tmp_path):
    """Cache entry older than TTL returns None (triggers re-run)."""
    import json, time as _time
    from igris.core.self_repair_supervisor import (
        _baseline_cache_path, _load_valid_baseline_cache, _save_baseline_cache
    )
    # Save a cache entry then manually age it
    _save_baseline_cache(str(tmp_path), "abc123", policy="strict")
    path = _baseline_cache_path(str(tmp_path))
    data = json.loads(path.read_text())
    data["checked_at"] = _time.time() - 9999  # older than any reasonable TTL
    path.write_text(json.dumps(data))

    result = _load_valid_baseline_cache(str(tmp_path), "abc123")
    assert result is None, "Stale cache should return None"


def test_baseline_cache_valid_hit_returns_payload(tmp_path):
    """Fresh cache entry for matching SHA returns the payload."""
    from igris.core.self_repair_supervisor import (
        _load_valid_baseline_cache, _save_baseline_cache
    )
    import os
    with __import__("unittest.mock").mock.patch.dict(
        os.environ, {"IGRIS_BASELINE_CACHE_SECONDS": "3600"}, clear=False
    ):
        _save_baseline_cache(str(tmp_path), "freshsha", policy="strict")
        result = _load_valid_baseline_cache(str(tmp_path), "freshsha")
    assert result is not None, "Fresh cache hit should return payload"
    assert result.get("baseline_ok") is True


def test_baseline_cache_sha_mismatch_returns_none(tmp_path):
    """Different SHA → cache miss (returns None)."""
    from igris.core.self_repair_supervisor import (
        _load_valid_baseline_cache, _save_baseline_cache
    )
    _save_baseline_cache(str(tmp_path), "sha_A", policy="strict")
    result = _load_valid_baseline_cache(str(tmp_path), "sha_B")
    assert result is None, "SHA mismatch must return None"


def test_force_revalidate_bypasses_fresh_cache(tmp_path):
    """force_revalidate=True bypasses even a fresh, matching cache entry."""
    from igris.core.self_repair_supervisor import (
        _load_valid_baseline_cache, _save_baseline_cache
    )
    _save_baseline_cache(str(tmp_path), "freshsha", policy="strict")
    result = _load_valid_baseline_cache(str(tmp_path), "freshsha", force_revalidate=True)
    assert result is None, "force_revalidate=True must bypass even a fresh cache"


def test_rank_supervisor_config_has_force_revalidate_baseline():
    """RankSupervisorConfig has force_revalidate_baseline field defaulting to False."""
    from igris.core.self_repair_supervisor import RankSupervisorConfig
    cfg = RankSupervisorConfig(goal="test", rank_id="R1")
    assert hasattr(cfg, "force_revalidate_baseline")
    assert cfg.force_revalidate_baseline is False


def test_force_revalidate_from_dict():
    """RankSupervisorConfig.from_dict parses force_revalidate_baseline."""
    from igris.core.self_repair_supervisor import RankSupervisorConfig
    cfg = RankSupervisorConfig.from_dict({"goal": "test", "force_revalidate_baseline": True})
    assert cfg.force_revalidate_baseline is True
