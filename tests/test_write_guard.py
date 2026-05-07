"""Tests for issue #76 — Destructive Write Guard & Safe Edit Actions.

Validates:
- write_file snippet on large .py file is blocked
- write_file on new small file succeeds
- insert_after adds content without deleting the rest
- replace_range modifies only the explicit range
- files_modified only populated on real diff
- Python AST validation catches corrupt edits
- Symbol guard protects create_app / run_app in server.py
- append_file, insert_before work correctly
"""

import os
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from igris.core.agent_action_schema import AgentAction
from igris.core.agent_reasoning_loop import AgentReasoningLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(tmp_path: str) -> AgentReasoningLoop:
    return AgentReasoningLoop(project_root=tmp_path, max_steps=5)


def _write_tmp(tmp_path: str, rel_path: str, content: str) -> str:
    full = os.path.join(tmp_path, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return full


def _action(action_type: str, **params) -> AgentAction:
    return AgentAction(
        mode="coder",
        action_type=action_type,
        reason="test",
        parameters=params,
    )


def _mock_rt():
    """Return a MagicMock ToolRuntime that writes to disk via fs_write."""
    rt = MagicMock()

    def fs_write_side_effect(path, content, **kwargs):
        result = MagicMock()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            result.success = True
            result.error = ""
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        return result

    rt.fs_write.side_effect = fs_write_side_effect
    return rt


# ---------------------------------------------------------------------------
# 1. Destructive write guard
# ---------------------------------------------------------------------------

LARGE_PY = textwrap.dedent("""\
    \"\"\"A large server module.\"\"\"
    from fastapi import FastAPI

    app = FastAPI()


    def create_app():
        return app


    def run_app():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)


    @app.get("/api/health")
    async def health():
        return {"status": "ok"}


    @app.get("/api/ping")
    async def ping():
        return {"pong": True}


    @app.get("/api/version")
    async def version():
        return {"version": "1.0.0"}


    @app.get("/api/info")
    async def info():
        return {"name": "IGRIS"}


    # Many more lines follow to make this file "large" enough for the guard.
    # Padding to exceed 200 chars threshold.
    _PADDING = "x" * 300
""")

SNIPPET = "@app.get('/api/version-info')\nasync def get_version_info():\n    return {'app': 'IGRIS_GPT', 'status': 'ok'}\n"


class TestDestructiveWriteGuard:
    """write_file snippet replacement on a large existing .py file must be blocked."""

    def test_snippet_on_large_py_is_blocked(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "server.py", LARGE_PY)
        rt = _mock_rt()
        action = _action("write_file", path="server.py", content=SNIPPET)
        result = loop._execute_write_file(rt, action)
        assert result["success"] is False, "expected blocked"
        assert "Destructive write guard" in result["error"]
        rt.fs_write.assert_not_called()

    def test_snippet_on_large_py_does_not_touch_disk(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "igris/web/server.py", LARGE_PY)
        rt = _mock_rt()
        action = _action("write_file", path="igris/web/server.py", content=SNIPPET)
        result = loop._execute_write_file(rt, action)
        assert result["success"] is False
        # File must be unchanged
        with open(os.path.join(str(tmp_path), "igris/web/server.py")) as f:
            assert f.read() == LARGE_PY

    def test_full_replacement_large_py_allowed(self, tmp_path):
        """A full-content replacement (≥30% size) must pass the guard."""
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "server.py", LARGE_PY)
        rt = _mock_rt()
        # New content is the same file plus one more endpoint — ≥ 30% of original
        new_full = LARGE_PY + "\n@app.get('/api/extra')\nasync def extra():\n    return {}\n"
        action = _action("write_file", path="server.py", content=new_full)
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True, result.get("error", "")

    def test_new_small_file_allowed(self, tmp_path):
        """write_file on a brand-new file (no existing file) must succeed."""
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("write_file", path="new_module.py", content="def hello(): pass\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True, result.get("error", "")
        assert "new_module.py" in loop._files_modified

    def test_small_file_replacement_allowed(self, tmp_path):
        """Replacing a small file (< 200 chars) with snippet is always allowed."""
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "tiny.py", "x = 1\n")
        rt = _mock_rt()
        action = _action("write_file", path="tiny.py", content="x = 2\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True

    def test_non_source_extension_not_guarded(self, tmp_path):
        """Binary / unknown extensions bypass the destructive write guard."""
        loop = _make_loop(str(tmp_path))
        big_txt = "a" * 500
        _write_tmp(str(tmp_path), "data.bin", big_txt)
        rt = _mock_rt()
        action = _action("write_file", path="data.bin", content="small")
        result = loop._execute_write_file(rt, action)
        # Guard should not fire for .bin
        assert "Destructive write guard" not in result.get("error", "")


# ---------------------------------------------------------------------------
# 2. files_modified only on real diff
# ---------------------------------------------------------------------------

class TestFilesModifiedTracking:
    """files_modified should reflect idempotent writes correctly."""

    def test_idempotent_write_adds_to_files_modified(self, tmp_path):
        """Writing the same content again is idempotent → still tracked."""
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "same.py", "x = 1\n")
        rt = _mock_rt()
        action = _action("write_file", path="same.py", content="x = 1\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True
        assert "same.py" in loop._files_modified
        # fs_write must NOT have been called (idempotent short-circuit)
        rt.fs_write.assert_not_called()

    def test_real_change_adds_to_files_modified(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "change.py", "x = 1\n")
        rt = _mock_rt()
        action = _action("write_file", path="change.py", content="x = 2\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True
        assert "change.py" in loop._files_modified

    def test_new_file_adds_to_files_modified(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("write_file", path="brand_new.py", content="pass\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True
        assert "brand_new.py" in loop._files_modified


# ---------------------------------------------------------------------------
# 3. Python AST validation
# ---------------------------------------------------------------------------

class TestPythonASTValidation:
    """AST validation must catch corrupted Python edits."""

    def test_valid_python_passes(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("write_file", path="valid.py", content="def foo():\n    pass\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True

    def test_invalid_python_blocked(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("write_file", path="broken.py", content="def foo(:\n    pass\n")
        result = loop._execute_write_file(rt, action)
        assert result["success"] is False
        assert "AST validation" in result["error"] or "SyntaxError" in result["error"]
        rt.fs_write.assert_not_called()

    def test_invalid_python_does_not_write(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("write_file", path="oops.py", content="def :\n    x\n")
        loop._execute_write_file(rt, action)
        assert not os.path.exists(os.path.join(str(tmp_path), "oops.py"))

    def test_symbol_guard_blocks_removal_of_create_app(self, tmp_path):
        """Replacing server.py must not silently remove create_app."""
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "server.py", LARGE_PY)
        rt = _mock_rt()
        # New content is big enough to pass size guard but missing create_app
        new_content = LARGE_PY.replace("def create_app():", "def _create_app_disabled():")
        action = _action("write_file", path="server.py", content=new_content)
        result = loop._execute_write_file(rt, action)
        assert result["success"] is False
        assert "create_app" in result["error"]

    def test_symbol_guard_allows_when_symbols_present(self, tmp_path):
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "server.py", LARGE_PY)
        rt = _mock_rt()
        # Add an endpoint — symbols still present
        new_content = LARGE_PY + "\n@app.get('/new')\nasync def new_ep():\n    return {}\n"
        action = _action("write_file", path="server.py", content=new_content)
        result = loop._execute_write_file(rt, action)
        assert result["success"] is True

    def test_server_route_guard_blocks_top_level_route_without_global_app(self, tmp_path):
        server_py = textwrap.dedent("""\
            from fastapi import FastAPI

            def create_app() -> FastAPI:
                app = FastAPI(title="IGRIS_GPT", version="0.1.0")
                return app

            def run_app():
                pass
        """)
        loop = _make_loop(str(tmp_path))
        _write_tmp(str(tmp_path), "server.py", server_py)
        rt = _mock_rt()
        action = _action(
            "insert_after",
            path="server.py",
            anchor='app = FastAPI(title="IGRIS_GPT", version="0.1.0")',
            content='\n@app.get("/api/version-info")\nasync def get_version_info():\n    return {"app": "IGRIS_GPT", "status": "ok"}\n',
        )

        result = loop._execute_insert_after(rt, action)

        assert result["success"] is False
        assert "Server route guard" in result["error"]


# ---------------------------------------------------------------------------
# 4. insert_after
# ---------------------------------------------------------------------------

class TestInsertAfter:
    """insert_after must add content without removing the rest."""

    def _setup(self, tmp_path, filename="module.py"):
        content = "line1\nline2\nline3\n"
        _write_tmp(str(tmp_path), filename, content)
        return content

    def test_insert_after_anchor(self, tmp_path):
        self._setup(tmp_path)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("insert_after", path="module.py", anchor="line1", content="inserted\n")
        result = loop._execute_insert_after(rt, action)
        assert result["success"] is True
        with open(os.path.join(str(tmp_path), "module.py")) as f:
            text = f.read()
        assert "line1\n" in text
        assert "inserted\n" in text
        assert "line2\n" in text
        assert "line3\n" in text
        # Order check
        assert text.index("line1") < text.index("inserted") < text.index("line2")

    def test_insert_after_missing_anchor_fails(self, tmp_path):
        self._setup(tmp_path)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("insert_after", path="module.py", anchor="NOT_THERE", content="x\n")
        result = loop._execute_insert_after(rt, action)
        assert result["success"] is False
        assert "anchor not found" in result["error"]

    def test_insert_after_tracks_files_modified(self, tmp_path):
        self._setup(tmp_path)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("insert_after", path="module.py", anchor="line2", content="new\n")
        loop._execute_insert_after(rt, action)
        assert "module.py" in loop._files_modified

    def test_insert_after_preserves_existing_content(self, tmp_path):
        """Existing lines must survive the insert."""
        original = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
        _write_tmp(str(tmp_path), "funcs.py", original)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("insert_after", path="funcs.py", anchor="def foo():", content="    # inserted\n")
        result = loop._execute_insert_after(rt, action)
        assert result["success"] is True
        with open(os.path.join(str(tmp_path), "funcs.py")) as f:
            text = f.read()
        assert "def foo():" in text
        assert "def bar():" in text
        assert "# inserted" in text

    def test_insert_after_repeated_route_is_no_change(self, tmp_path):
        original = textwrap.dedent("""\
            from fastapi import FastAPI


            def create_app() -> FastAPI:
                app = FastAPI(title="IGRIS_GPT", version="0.1.0")

                return app
        """)
        route = (
            "\n"
            "    @app.get('/api/version-info')\n"
            "    async def version_info():\n"
            "        return {\"app\": \"IGRIS_GPT\", \"status\": \"ok\"}\n"
        )
        _write_tmp(str(tmp_path), "server.py", original)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action(
            "insert_after",
            path="server.py",
            anchor='app = FastAPI(title="IGRIS_GPT", version="0.1.0")',
            content=route,
        )

        first = loop._execute_insert_after(rt, action)
        second = loop._execute_insert_after(rt, action)

        assert first["success"] is True
        assert second["success"] is True
        assert "no change" in second["summary"]
        with open(os.path.join(str(tmp_path), "server.py")) as f:
            text = f.read()
        assert text.count("@app.get('/api/version-info')") == 1

    def test_insert_after_blocks_app_route_before_app_init(self, tmp_path):
        original = textwrap.dedent("""\
            from fastapi import FastAPI


            def create_app() -> FastAPI:
                \"\"\"Create and configure the FastAPI application.\"\"\"
                app = FastAPI(title="IGRIS_GPT", version="0.1.0")
                return app
        """)
        route = (
            "\n"
            "    @app.get('/api/version-info')\n"
            "    async def version_info():\n"
            "        return {'app': 'IGRIS_GPT', 'status': 'ok'}\n"
        )
        _write_tmp(str(tmp_path), "server.py", original)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action(
            "insert_after",
            path="server.py",
            anchor="def create_app() -> FastAPI:",
            content=route,
        )

        result = loop._execute_insert_after(rt, action)

        assert result["success"] is False
        assert "before app = FastAPI" in result["error"]
        with open(os.path.join(str(tmp_path), "server.py")) as f:
            text = f.read()
        assert "@app.get('/api/version-info')" not in text

    def test_insert_after_allows_app_route_after_app_init(self, tmp_path):
        original = textwrap.dedent("""\
            from fastapi import FastAPI


            def create_app() -> FastAPI:
                app = FastAPI(title="IGRIS_GPT", version="0.1.0")
                return app
        """)
        route = (
            "\n"
            "    @app.get('/api/version-info')\n"
            "    async def version_info():\n"
            "        return {'app': 'IGRIS_GPT', 'status': 'ok'}\n"
        )
        _write_tmp(str(tmp_path), "server.py", original)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action(
            "insert_after",
            path="server.py",
            anchor='app = FastAPI(title="IGRIS_GPT", version="0.1.0")',
            content=route,
        )

        result = loop._execute_insert_after(rt, action)

        assert result["success"] is True, result.get("error", "")
        with open(os.path.join(str(tmp_path), "server.py")) as f:
            text = f.read()
        assert "@app.get('/api/version-info')" in text


# ---------------------------------------------------------------------------
# 5. insert_before
# ---------------------------------------------------------------------------

class TestInsertBefore:
    def test_insert_before_anchor(self, tmp_path):
        _write_tmp(str(tmp_path), "f.py", "line1\nline2\nline3\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("insert_before", path="f.py", anchor="line2", content="between\n")
        result = loop._execute_insert_before(rt, action)
        assert result["success"] is True
        with open(os.path.join(str(tmp_path), "f.py")) as f:
            text = f.read()
        assert text.index("between") < text.index("line2")
        assert "line1" in text and "line3" in text

    def test_insert_before_repeated_content_is_no_change(self, tmp_path):
        _write_tmp(str(tmp_path), "f.py", "line1\nline2\nline3\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("insert_before", path="f.py", anchor="line2", content="between\n")

        first = loop._execute_insert_before(rt, action)
        second = loop._execute_insert_before(rt, action)

        assert first["success"] is True
        assert second["success"] is True
        assert "no change" in second["summary"]
        with open(os.path.join(str(tmp_path), "f.py")) as f:
            text = f.read()
        assert text.count("between") == 1


# ---------------------------------------------------------------------------
# 6. replace_range
# ---------------------------------------------------------------------------

class TestReplaceRange:
    """replace_range must only modify the explicit range."""

    def test_replace_middle_lines(self, tmp_path):
        _write_tmp(str(tmp_path), "r.py", "line1\nline2\nline3\nline4\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("replace_range", path="r.py", start=2, end=3, content="replaced\n")
        result = loop._execute_replace_range(rt, action)
        assert result["success"] is True
        with open(os.path.join(str(tmp_path), "r.py")) as f:
            text = f.read()
        assert "line1" in text
        assert "replaced" in text
        assert "line4" in text
        assert "line2" not in text
        assert "line3" not in text

    def test_replace_range_tracks_files_modified(self, tmp_path):
        _write_tmp(str(tmp_path), "rr.py", "a\nb\nc\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("replace_range", path="rr.py", start=1, end=1, content="z\n")
        loop._execute_replace_range(rt, action)
        assert "rr.py" in loop._files_modified

    def test_replace_range_invalid_range_fails(self, tmp_path):
        _write_tmp(str(tmp_path), "rng.py", "a\nb\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("replace_range", path="rng.py", start=5, end=10, content="x\n")
        result = loop._execute_replace_range(rt, action)
        assert result["success"] is False

    def test_replace_range_invalid_start_end_order(self, tmp_path):
        _write_tmp(str(tmp_path), "ord.py", "a\nb\nc\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("replace_range", path="ord.py", start=3, end=1, content="x\n")
        result = loop._execute_replace_range(rt, action)
        assert result["success"] is False

    def test_replace_range_ast_validation(self, tmp_path):
        _write_tmp(str(tmp_path), "ast_rng.py", "def foo():\n    pass\n\ndef bar():\n    return 1\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        # Replace line 1 with broken Python
        action = _action("replace_range", path="ast_rng.py", start=1, end=1, content="def foo(:\n")
        result = loop._execute_replace_range(rt, action)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# 7. append_file
# ---------------------------------------------------------------------------

class TestAppendFile:
    VERSION_INFO_TEST = textwrap.dedent("""\
        import pytest
        from fastapi.testclient import TestClient

        from igris.web.server import create_app


        app = create_app()


        @pytest.fixture
        def client():
            return TestClient(app)


        def test_version_info(client):
            response = client.get("/api/version-info")
            assert response.status_code == 200
            assert response.json() == {"app": "IGRIS_GPT", "status": "ok"}
    """)

    def test_append_adds_to_end(self, tmp_path):
        _write_tmp(str(tmp_path), "app.py", "x = 1  # existing\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("append_file", path="app.py", content="y = 2  # appended\n")
        result = loop._execute_append_file(rt, action)
        assert result["success"] is True, result.get("error", "")
        with open(os.path.join(str(tmp_path), "app.py")) as f:
            text = f.read()
        assert "x = 1" in text
        assert "y = 2" in text
        assert text.index("x = 1") < text.index("y = 2")

    def test_append_tracks_files_modified(self, tmp_path):
        _write_tmp(str(tmp_path), "ap2.py", "x = 1\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action("append_file", path="ap2.py", content="y = 2\n")
        loop._execute_append_file(rt, action)
        assert "ap2.py" in loop._files_modified

    def test_append_ast_validation(self, tmp_path):
        _write_tmp(str(tmp_path), "aast.py", "def foo():\n    pass\n")
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        # Append broken Python — should be blocked by AST validator
        action = _action("append_file", path="aast.py", content="def bar(:\n    x\n")
        result = loop._execute_append_file(rt, action)
        assert result["success"] is False

    def test_append_file_allows_new_python_module(self, tmp_path):
        os.makedirs(os.path.join(str(tmp_path), "tests"), exist_ok=True)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action(
            "append_file",
            path="tests/test_version_info.py",
            content=self.VERSION_INFO_TEST,
        )
        result = loop._execute_append_file(rt, action)
        assert result["success"] is True, result.get("error", "")
        with open(os.path.join(str(tmp_path), "tests/test_version_info.py")) as f:
            text = f.read()
        assert "def test_version_info" in text

    def test_append_file_blocks_module_content_on_existing_python(self, tmp_path):
        _write_tmp(str(tmp_path), "tests/test_version_info.py", self.VERSION_INFO_TEST)
        loop = _make_loop(str(tmp_path))
        rt = _mock_rt()
        action = _action(
            "append_file",
            path="tests/test_version_info.py",
            content=self.VERSION_INFO_TEST,
        )
        result = loop._execute_append_file(rt, action)
        assert result["success"] is False
        assert "complete Python module" in result["error"]


class TestWriteGuardInLoop:
    """The guard must work end-to-end through AgentReasoningLoop.run()."""

    def test_snippet_write_blocked_in_run(self, tmp_path):
        """An LLM proposing snippet replacement on a large file must fail."""
        _write_tmp(str(tmp_path), "server.py", LARGE_PY)
        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="coder", action_type="write_file",
                    reason="add version endpoint",
                    parameters={"path": "server.py", "content": SNIPPET},
                    risk_hint="low", confidence=0.9,
                ), []
            return AgentAction(
                mode="coder", action_type="blocked",
                reason="write was blocked",
                parameters={"reason": "cannot write snippet"},
                risk_hint="low", confidence=0.9,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Add version-info endpoint")

        assert "server.py" not in result.files_modified
        assert result.steps[0].outcome == "failure"
        assert "Destructive write guard" in result.steps[0].error

    def test_new_file_write_succeeds_in_run(self, tmp_path):
        """An LLM writing a new file must succeed."""
        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="coder", action_type="write_file",
                    reason="create new file",
                    parameters={"path": "new_endpoint.py", "content": "def ping(): pass\n"},
                    risk_hint="low", confidence=0.9,
                ), []
            return AgentAction(
                mode="coder", action_type="finish",
                reason="done",
                parameters={"summary": "Created file"},
                risk_hint="low", confidence=0.95,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Create endpoint file")

        assert result.status == "finished"
        assert "new_endpoint.py" in result.files_modified

    def test_ast_validation_failure_blocks_loop(self, tmp_path):
        """A Python AST validation failure must stop before more actions run."""
        _write_tmp(str(tmp_path), "server.py", "def create_app():\n    app = object()\n    return app\n")
        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="coder",
                    action_type="insert_after",
                    reason="insert invalid handler",
                    parameters={
                        "path": "server.py",
                        "anchor": "def create_app():",
                        "content": "def broken(:\n",
                    },
                    risk_hint="low",
                    confidence=0.9,
                ), []
            return AgentAction(
                mode="coder",
                action_type="read_file_range",
                reason="would continue without AST block",
                parameters={"path": "server.py"},
                risk_hint="low",
                confidence=0.9,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Add version-info endpoint")

        assert result.status == "blocked"
        assert result.stop_reason == "blocked"
        assert result.total_steps == 1
        assert result.steps[0].outcome == "blocked"
        assert "Python AST validation failed" in result.steps[0].error
        assert call_count[0] == 1
        assert "server.py" not in result.files_modified


# ---------------------------------------------------------------------------
# Issue #78 — False-positive secret guard on safe edit methods
# ---------------------------------------------------------------------------

TOKEN_PY = textwrap.dedent("""\
    \"\"\"Module with token variable (legitimate code, not a secret).\"\"\"
    from fastapi import FastAPI

    app = FastAPI(title="IGRIS_GPT", version="0.1.0")


    def create_app():
        return app


    def run_app():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)


    @app.get("/api/approval")
    async def approval(content: dict):
        token = content.get("approval_token", "")
        return {"approved": bool(token)}
""")

ENDPOINT_INSERTION = (
    "\n\n@app.get('/api/version-info')\n"
    "async def version_info():\n"
    "    return {'app': 'IGRIS_GPT', 'status': 'ok'}\n"
)


class TestSafeEditSecretFalsePositive:
    """insert_after / insert_before / replace_range / append_file must not be
    blocked by pre-existing code that happens to match secret-like patterns
    (e.g. ``token=content.get(...)`` already in the file)."""

    def test_insert_after_not_blocked_by_existing_token(self, tmp_path):
        """insert_after on a file with 'token=' variable must succeed."""
        src = tmp_path / "server.py"
        src.write_text(TOKEN_PY, encoding="utf-8")

        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        action = _action(
            "insert_after",
            path="server.py",
            anchor="app = FastAPI",
            content=ENDPOINT_INSERTION,
            success_criteria="endpoint added",
        )
        result = loop._execute_insert_after(loop._get_tool_runtime(), action)

        assert result.get("success"), f"Expected success, got error: {result.get('error')}"
        text = src.read_text(encoding="utf-8")
        assert "version-info" in text
        assert "approval_token" in text  # pre-existing code intact

    def test_insert_before_not_blocked_by_existing_token(self, tmp_path):
        """insert_before on a file with 'token=' variable must succeed."""
        src = tmp_path / "server.py"
        src.write_text(TOKEN_PY, encoding="utf-8")

        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        action = _action(
            "insert_before",
            path="server.py",
            anchor="@app.get(\"/api/approval\")",
            content=ENDPOINT_INSERTION,
            success_criteria="endpoint added",
        )
        result = loop._execute_insert_before(loop._get_tool_runtime(), action)

        assert result.get("success"), f"Expected success, got error: {result.get('error')}"
        text = src.read_text(encoding="utf-8")
        assert "version-info" in text

    def test_append_file_not_blocked_by_existing_token(self, tmp_path):
        """append_file on a file with 'token=' variable must succeed."""
        src = tmp_path / "server.py"
        src.write_text(TOKEN_PY, encoding="utf-8")

        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        action = _action(
            "append_file",
            path="server.py",
            content=ENDPOINT_INSERTION,
            success_criteria="endpoint appended",
        )
        result = loop._execute_append_file(loop._get_tool_runtime(), action)

        assert result.get("success"), f"Expected success, got error: {result.get('error')}"
        text = src.read_text(encoding="utf-8")
        assert "version-info" in text

    def test_replace_range_not_blocked_by_existing_token(self, tmp_path):
        """replace_range on a file with 'token=' variable must succeed."""
        src = tmp_path / "server.py"
        src.write_text(TOKEN_PY, encoding="utf-8")
        lines = TOKEN_PY.splitlines()
        # Replace last line (a comment) with the new endpoint
        last = len(lines)

        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        action = _action(
            "replace_range",
            path="server.py",
            start=last,
            end=last,
            content=ENDPOINT_INSERTION,
            success_criteria="range replaced",
        )
        result = loop._execute_replace_range(loop._get_tool_runtime(), action)

        assert result.get("success"), f"Expected success, got error: {result.get('error')}"
        text = src.read_text(encoding="utf-8")
        assert "version-info" in text

    def test_actual_secret_in_insertion_still_blocked(self, tmp_path):
        """insert_after must still block a real secret in the new content."""
        src = tmp_path / "server.py"
        src.write_text(TOKEN_PY, encoding="utf-8")

        loop = AgentReasoningLoop(project_root=str(tmp_path), max_steps=3)
        action = _action(
            "insert_after",
            path="server.py",
            anchor="app = FastAPI",
            content="\n# sk-ABCDEF1234567890ABCDEF12345678901234567890\n",
            success_criteria="should be blocked",
        )
        result = loop._execute_insert_after(loop._get_tool_runtime(), action)

        assert not result.get("success"), "Should have been blocked by secret guard"
        assert "secret" in result.get("error", "").lower()
