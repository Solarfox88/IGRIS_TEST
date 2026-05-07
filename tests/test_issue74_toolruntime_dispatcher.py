"""Tests for Issue #74 — ToolRuntime dispatcher, write_file verification,
test discovery, and files_modified tracking.

Covers:
  1. git_status via reasoning loop uses rt.git_status() (no missing .execute)
  2. write_file false success is impossible (hash verification)
  3. fs_write real produces diff / no-op write returns failure
  4. files_modified only tracks real changes
  5. Test discovery (test_*.py, *_test.py, tests/**/*.py, TestClient)
  6. propose_patch / apply_patch routing
"""

import os
import hashlib
import tempfile
import textwrap

import pytest
from unittest.mock import patch, MagicMock

from igris.core.agent_reasoning_loop import (
    AgentReasoningLoop,
    LoopStep,
)
from igris.core.agent_action_schema import AgentAction
from igris.core.code_navigation import CodeNavigator
from igris.core.tool_runtime import ToolRuntime, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_project(files: dict) -> str:
    """Create a temporary project directory with given files."""
    root = tempfile.mkdtemp(prefix="igris_test_74_")
    for rel_path, content in files.items():
        full = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    return root


# ---------------------------------------------------------------------------
# 1. git_status via reasoning loop — no missing .execute
# ---------------------------------------------------------------------------

class TestToolRuntimeDispatcher:
    """ToolRuntime dispatcher must call specific methods, not .execute()."""

    def test_git_status_uses_rt_git_status(self):
        """git_status action should call rt.git_status(), not rt.execute()."""
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="git_status",
            parameters={},
        )

        mock_result = ToolResult(
            tool="git", action="status",
            success=True, output="M file.py",
        )
        with patch.object(ToolRuntime, "git_status", return_value=mock_result) as mock_gs:
            result = loop._execute_tool_runtime(action)

        mock_gs.assert_called_once()
        assert result["success"] is True
        assert "file.py" in result["summary"]

    def test_git_diff_uses_rt_git_diff(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="git_diff",
            parameters={"staged": True},
        )

        mock_result = ToolResult(
            tool="git", action="diff",
            success=True, output="1 file changed",
        )
        with patch.object(ToolRuntime, "git_diff", return_value=mock_result) as mock_gd:
            result = loop._execute_tool_runtime(action)

        mock_gd.assert_called_once_with(staged=True)
        assert result["success"] is True

    def test_run_tests_uses_rt_run_tests(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="run_tests",
            parameters={"args": ["-k", "test_ping"]},
        )

        mock_result = ToolResult(
            tool="test", action="pytest",
            success=True, output="1 passed",
        )
        with patch.object(ToolRuntime, "run_tests", return_value=mock_result) as mock_rt:
            result = loop._execute_tool_runtime(action)

        mock_rt.assert_called_once_with(args=["-k", "test_ping"])
        assert result["success"] is True

    def test_http_check_uses_rt_http_check(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="http_check",
            parameters={"url": "http://localhost:8000/api/ping"},
        )

        mock_result = ToolResult(
            tool="http", action="check",
            success=True, output='{"status": 200}',
        )
        with patch.object(ToolRuntime, "http_check", return_value=mock_result) as mock_hc:
            result = loop._execute_tool_runtime(action)

        mock_hc.assert_called_once_with(url="http://localhost:8000/api/ping")
        assert result["success"] is True

    def test_raw_shell_proposal_blocked(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="raw_shell_proposal",
            parameters={"command": "rm -rf /"},
        )
        result = loop._execute_tool_runtime(action)
        assert result["success"] is False
        assert "Risk Engine" in result["error"]

    def test_unknown_action_returns_error(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="magic_action",
            parameters={},
        )
        result = loop._execute_tool_runtime(action)
        assert result["success"] is False
        assert "not yet integrated" in result["error"]

    def test_no_execute_method_called(self):
        """Ensure .execute() is never called — it doesn't exist."""
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(action_type="git_status", parameters={})

        mock_result = ToolResult(
            tool="git", action="status", success=True, output="clean",
        )
        with patch.object(ToolRuntime, "git_status", return_value=mock_result):
            # If .execute() were called, it would raise AttributeError
            result = loop._execute_tool_runtime(action)

        assert result["success"] is True
        assert not hasattr(ToolRuntime, "execute")


# ---------------------------------------------------------------------------
# 2. write_file false success impossible
# ---------------------------------------------------------------------------

class TestWriteFileVerification:
    """write_file must verify real change via hash."""

    def test_write_file_identical_content_idempotent(self):
        """Writing identical content is idempotent: success=True, no disk I/O. (#76)

        Previous behaviour (success=False) was changed in #76: identical writes
        are now treated as already-done rather than as errors, because callers
        may legitimately retry a write after a crash.
        """
        root = _make_temp_project({
            "existing.py": "print('hello')\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"path": "existing.py", "content": "print('hello')\n"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_write_file(rt, action)
        # Idempotent: already on disk, so success=True
        assert result["success"] is True
        assert "idempotent" in result["summary"] or "existing.py" in loop._files_modified

    def test_write_file_real_change_succeeds(self):
        """Writing different content should return success=True."""
        root = _make_temp_project({
            "existing.py": "print('hello')\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"path": "existing.py", "content": "print('world')\n"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True
        assert "hash" in result["summary"]
        # Verify the file was actually written
        with open(os.path.join(root, "existing.py")) as f:
            assert f.read() == "print('world')\n"

    def test_write_file_new_file_succeeds(self):
        """Writing a new file should succeed."""
        root = _make_temp_project({})
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"path": "new_file.py", "content": "# new\n"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True
        assert os.path.isfile(os.path.join(root, "new_file.py"))

    def test_write_file_missing_path_fails(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"content": "stuff"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_write_file(rt, action)
        assert result["success"] is False
        assert "missing" in result["error"]

    def test_write_file_missing_content_fails(self):
        """content parameter entirely absent should fail.

        Note: content="" (empty string) is now allowed (creates empty file).
        Only content=None or truly absent param triggers the guard.
        """
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"path": "test.py", "content": None},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_write_file(rt, action)
        assert result["success"] is False
        assert "missing" in result["error"]
        assert "missing" in result["error"]


# ---------------------------------------------------------------------------
# 3. files_modified only on real change
# ---------------------------------------------------------------------------

class TestFilesModifiedTracking:
    """files_modified should only contain files with verified changes."""

    def test_write_file_real_change_tracked(self):
        root = _make_temp_project({
            "code.py": "old\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"path": "code.py", "content": "new\n"},
        )
        rt = loop._get_tool_runtime()
        loop._execute_write_file(rt, action)
        assert "code.py" in loop._files_modified

    def test_write_file_idempotent_is_tracked(self):
        """Idempotent write (same content) is tracked for auditability. (#76)

        Previous behaviour: not tracked. New: tracked as already-done write.
        """
        root = _make_temp_project({
            "code.py": "same\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="write_file",
            parameters={"path": "code.py", "content": "same\n"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_write_file(rt, action)
        # Idempotent: file already has this content -> success=True, tracked
        assert result["success"] is True
        assert "code.py" in loop._files_modified

    def test_apply_patch_real_change_tracked(self):
        root = _make_temp_project({
            "target.py": "BEFORE = 1\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="apply_patch",
            parameters={"path": "target.py", "content": "AFTER = 2\n"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_apply_patch(rt, action)
        assert result["success"] is True
        assert "target.py" in loop._files_modified

    def test_propose_patch_does_not_modify_files(self):
        root = _make_temp_project({
            "src.py": "ORIGINAL = 1\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="propose_patch",
            parameters={"path": "src.py", "content": "MODIFIED = 2\n"},
        )
        rt = loop._get_tool_runtime()
        result = loop._execute_propose_patch(rt, action)
        assert result["success"] is True
        # propose_patch should NOT modify the file
        with open(os.path.join(root, "src.py")) as f:
            assert f.read() == "ORIGINAL = 1\n"
        assert "src.py" not in loop._files_modified


# ---------------------------------------------------------------------------
# 4. Test discovery
# ---------------------------------------------------------------------------

class TestDiscoverTests:
    """CodeNavigator.discover_tests should find test files and patterns."""

    def test_finds_test_prefix_files(self):
        root = _make_temp_project({
            "test_main.py": "import pytest\ndef test_hello(): pass\n",
            "utils.py": "def helper(): pass\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_tests()
        assert result.success
        files = [d["file"] for d in result.data]
        assert "test_main.py" in files
        assert "utils.py" not in files

    def test_finds_test_suffix_files(self):
        root = _make_temp_project({
            "main_test.py": "def test_one(): pass\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_tests()
        files = [d["file"] for d in result.data]
        assert "main_test.py" in files

    def test_finds_files_in_tests_directory(self):
        root = _make_temp_project({
            "tests/test_api.py": "def test_ping(): pass\n",
            "tests/conftest.py": "import pytest\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_tests()
        files = [d["file"] for d in result.data]
        assert any("test_api.py" in f for f in files)
        assert any("conftest.py" in f for f in files)

    def test_finds_testclient_usage(self):
        root = _make_temp_project({
            "integration.py": textwrap.dedent("""\
                from fastapi.testclient import TestClient
                from app import app
                client = TestClient(app)
                def test_endpoint():
                    resp = client.get("/api/ping")
                    assert resp.status_code == 200
            """),
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_tests()
        assert result.total_count >= 1
        entry = result.data[0]
        indicators_str = " ".join(entry["indicators"])
        assert "TestClient" in indicators_str

    def test_empty_project_no_tests(self):
        root = _make_temp_project({
            "main.py": "print('no tests')\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_tests()
        assert result.success
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# 5. Full step execution: tool_runtime routing
# ---------------------------------------------------------------------------

class TestToolRuntimeInLoop:
    """Tool runtime actions in the full loop step should route correctly."""

    def test_git_status_step_succeeds(self):
        """A git_status action through _execute_step should not crash."""
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=5)

        mock_action = AgentAction(
            action_type="git_status",
            parameters={},
            reason="Check workspace",
        )
        mock_tr = ToolResult(
            tool="git", action="status", success=True, output="nothing to commit",
        )
        with patch.object(loop, "_decide_action", return_value=(mock_action, [])):
            with patch.object(loop, "_build_context", return_value=MagicMock()):
                with patch.object(ToolRuntime, "git_status", return_value=mock_tr):
                    step = loop._execute_step(1, "test", "")

        assert step.outcome == "success"
        assert step.action_type == "git_status"

    def test_write_file_step_with_real_change(self):
        """write_file through full step should produce real change."""
        root = _make_temp_project({"target.py": "OLD_VAL = 1\n"})
        loop = AgentReasoningLoop(project_root=root, max_steps=5, role="coder")

        mock_action = AgentAction(
            mode="coder",
            action_type="write_file",
            parameters={"path": "target.py", "content": "NEW_VAL = 2\n"},
            reason="Update file",
        )
        with patch.object(loop, "_decide_action", return_value=(mock_action, [])):
            with patch.object(loop, "_build_context", return_value=MagicMock()):
                step = loop._execute_step(1, "test", "")

        assert step.outcome == "success", step.error
        assert "target.py" in loop._files_modified
        # Verify file was actually written
        with open(os.path.join(root, "target.py")) as f:
            assert f.read() == "NEW_VAL = 2\n"

    def test_write_file_step_identical_content_idempotent(self):
        """write_file with identical content is idempotent in step. (#76)

        Previous: outcome=failure. New: outcome=success (idempotent write).
        """
        root = _make_temp_project({"same.py": "content\n"})
        loop = AgentReasoningLoop(project_root=root, max_steps=5, role="coder")

        mock_action = AgentAction(
            mode="coder",
            action_type="write_file",
            parameters={"path": "same.py", "content": "content\n"},
            reason="Overwrite",
        )
        with patch.object(loop, "_decide_action", return_value=(mock_action, [])):
            with patch.object(loop, "_build_context", return_value=MagicMock()):
                step = loop._execute_step(1, "test", "")

        # Idempotent write is now a success, not a failure
        assert step.outcome == "success"
        assert "same.py" in loop._files_modified


# ---------------------------------------------------------------------------
# 6. Shell template routing
# ---------------------------------------------------------------------------

class TestShellTemplateRouting:
    """shell_template should route to rt.shell_execute()."""

    def test_shell_template_calls_shell_execute(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        action = AgentAction(
            action_type="shell_template",
            parameters={"command_id": "ls", "args": ["-la"]},
        )
        mock_result = ToolResult(
            tool="shell", action="ls", success=True, output="total 42",
        )
        with patch.object(ToolRuntime, "shell_execute", return_value=mock_result) as mock_se:
            result = loop._execute_tool_runtime(action)

        mock_se.assert_called_once_with(command_id="ls", args=["-la"])
        assert result["success"] is True
