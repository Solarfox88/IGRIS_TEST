"""
FastAPI application for IGRIS_GPT.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from igris.core import anti_loop
from igris.core.task_engine import TaskEngine
from igris.core.teacher import build_teacher_payload
from igris.core import execution_report
from igris.layers.advisory import router as provider_router
from igris.layers.execution import runner as execution_runner
from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
from igris.core import safety
from igris.layers.git_layer.git_status import get_git_info
from igris.models.config import CONFIG
from igris.models.report import GitStatusResponse, TestRunResponse
from igris.agents import build_default_registry
from igris.a2a.agent_card import build_agent_card
from igris.core.project_context import build_project_snapshot
from igris.core.memory import recent_memory_events, append_memory_event

MODULE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = MODULE_DIR / "templates"
STATIC_DIR = MODULE_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="IGRIS_GPT", version="0.1.0")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    sessions: Dict[str, List[Dict[str, str]]] = {}
    task_engine = TaskEngine()

    build_default_registry()

    nonlocal_test_running = {"running": False}
    nonlocal_cmd_running = {"running": False}

    def _redact(text: str) -> str:
        return safety.redact_secrets(text)

    # ---- Root ----

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        template = jinja_env.get_template("index.html")
        return template.render()

    # ---- Status / Config ----

    @app.get("/api/status")
    async def api_status() -> Dict[str, object]:
        provider, model = provider_router.choose_provider()
        return {"provider": provider, "model": model, "safe": True}

    @app.get("/api/config/safe")
    async def api_config_safe() -> Dict[str, object]:
        return CONFIG.safe_dict()

    # ---- Sessions ----

    @app.post("/api/sessions")
    async def create_session() -> Dict[str, str]:
        session_id = str(len(sessions) + 1)
        sessions[session_id] = []
        return {"id": session_id}

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, content: Dict[str, str] = Body(...)) -> Dict[str, str]:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        message = content.get("message", "")
        sessions[session_id].append({"role": "user", "content": message})
        provider_router.choose_provider(for_task="chat")
        response_text = "This is a placeholder response."
        sessions[session_id].append({"role": "assistant", "content": response_text})
        return {"response": response_text}

    # ---- Git ----

    @app.get("/api/git/status", response_model=GitStatusResponse)
    async def api_git_status() -> GitStatusResponse:
        info = get_git_info()
        return GitStatusResponse(
            branch=info.branch, remote=info.remote,
            dirty=info.dirty, changed=info.changed, head=info.head,
        )

    # ---- Routing / Cost ----

    @app.get("/api/routing/history")
    async def api_routing_history() -> Dict[str, object]:
        return {"history": provider_router.get_history()}

    @app.get("/api/cost/summary")
    async def api_cost_summary() -> Dict[str, object]:
        return provider_router.cost_summary()

    @app.get("/api/routing/explain")
    async def api_routing_explain() -> Dict[str, str]:
        return {"explanation": provider_router.explain_routing()}

    # ---- Files ----

    @app.get("/api/files/tree")
    async def api_files_tree() -> Dict[str, object]:
        root = CONFIG.project_root
        tree = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            filtered_dirs: List[str] = []
            for d in list(dirnames):
                sub_path = Path(dirpath) / d
                if safety.is_runtime_artifact(sub_path):
                    dirnames.remove(d)
                    continue
                if d.startswith('.'):
                    dirnames.remove(d)
                    continue
                filtered_dirs.append(d)
            entries = []
            for d in sorted(filtered_dirs):
                entries.append({"type": "dir", "name": d})
            for f in sorted(filenames):
                if f.startswith('.'):
                    continue
                if safety.is_sensitive_filename(f):
                    continue
                sub = Path(dirpath) / f
                if safety.is_runtime_artifact(sub):
                    continue
                entries.append({"type": "file", "name": f})
            tree.append({"path": rel_dir, "entries": entries})
        return {"tree": tree}

    @app.get("/api/files/preview")
    async def api_files_preview(path: str) -> Dict[str, object]:
        root = CONFIG.project_root
        requested = (root / path).resolve()
        # Use enhanced safety check
        decision = safety.check_file_preview(requested, root)
        if not decision.allowed:
            raise HTTPException(status_code=403, detail=decision.reason)
        if not safety.check_path_access(requested, root):
            raise HTTPException(status_code=403, detail="Invalid path")
        if requested.is_dir():
            raise HTTPException(status_code=400, detail="Cannot preview a directory")
        if not requested.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if requested.name.lower() == ".env":
            raise HTTPException(status_code=403, detail="Preview of .env is blocked")
        mime, _ = mimetypes.guess_type(str(requested))
        if mime and not mime.startswith("text"):
            raise HTTPException(status_code=400, detail="Only text files can be previewed")
        try:
            with requested.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read(20_000)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if safety.detect_secret_like_content(content):
            content = safety.redact_secrets(content)
        return {"path": path, "preview": content}

    # ---- Tests ----

    @app.post("/api/tests/run", response_model=TestRunResponse)
    async def api_tests_run() -> TestRunResponse:
        if nonlocal_test_running["running"]:
            raise HTTPException(status_code=409, detail="Test run already in progress")
        nonlocal_test_running["running"] = True
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()
        try:
            result = execution_runner.run_tests()
            finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            duration_ms = int((time.monotonic() - t0) * 1000)
            success = result["returncode"] == 0
            stdout = _redact(result.get("stdout", ""))
            stderr = _redact(result.get("stderr", ""))
            execution_report.create_report(
                command_id="run_tests", capability_id="validation.run_tests",
                returncode=result["returncode"], stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                started_at=started, finished_at=finished, duration_ms=duration_ms,
            )
            task_engine.append_timeline_event({
                "event": "test_run", "success": success,
                "duration_ms": duration_ms,
            })
            return TestRunResponse(success=success, stdout=stdout, stderr=stderr)
        finally:
            nonlocal_test_running["running"] = False

    # ---- Logs ----

    @app.get("/api/logs")
    async def api_logs(lines: int = 200) -> Dict[str, str]:
        log_path = Path("logs/igris.log")
        if not log_path.exists():
            return {"logs": "Log file not found."}
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()[-lines:]
        return {"logs": "".join(data)}

    # ---- Timeline ----

    @app.get("/api/agent/timeline")
    async def api_agent_timeline() -> Dict[str, object]:
        events = task_engine.recent_timeline_events(limit=50)
        return {"timeline": events}

    # ---- Safety ----

    @app.get("/api/safety/status")
    async def api_safety_status() -> Dict[str, object]:
        tasks = [t.description for t in task_engine.tasks]
        counts = anti_loop.compute_family_counts(tasks)
        saturated = anti_loop.saturated_families(counts)
        return {"saturated_families": saturated, "counts": counts}

    # ---- Health / Readiness ----

    @app.get("/api/health")
    async def api_health() -> Dict[str, object]:
        return {"status": "ok", "version": app.version, "time": time.time()}

    @app.get("/api/readiness")
    async def api_readiness() -> Dict[str, object]:
        checks: Dict[str, object] = {}
        root = CONFIG.project_root
        checks["project_root_exists"] = root.exists()
        checks["project_root_is_dir"] = root.is_dir()
        checks["templates"] = TEMPLATES_DIR.exists()
        checks["static"] = STATIC_DIR.exists()
        from igris.agents import list_agents
        checks["agents_registered"] = len(list_agents()) > 0
        return checks

    # ---- Project Context ----

    @app.get("/api/project/context")
    async def api_project_context() -> Dict[str, object]:
        snapshot = build_project_snapshot(task_engine=task_engine)
        return snapshot

    # ---- Memory ----

    @app.get("/api/memory/recent")
    async def api_memory_recent(namespace: str, limit: int = 20) -> Dict[str, object]:
        events = recent_memory_events(namespace, limit)
        return {"events": events}

    # ---- Task management ----

    @app.get("/api/tasks")
    async def api_list_tasks() -> Dict[str, object]:
        tasks = []
        for t in task_engine.tasks:
            tasks.append(t.to_dict())
        return {"tasks": tasks}

    @app.post("/api/tasks")
    async def api_create_task(content: Dict[str, str] = Body(...)) -> Dict[str, object]:
        description = content.get("description")
        if not description:
            raise HTTPException(status_code=400, detail="description is required")
        title = content.get("title")
        source = content.get("source", "user")
        task = task_engine.create_task(description, title=title, source=source)
        return task.to_dict()

    @app.get("/api/tasks/{task_id}")
    async def api_get_task(task_id: int) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_dict()

    @app.post("/api/tasks/{task_id}/complete")
    async def api_complete_task(task_id: int, body: Dict[str, str] = Body(default={})):  # type: ignore[assignment]
        result_text = body.get("result") if isinstance(body, dict) else None
        task = task_engine.complete_task(task_id, result_text)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_dict()

    @app.post("/api/tasks/{task_id}/block")
    async def api_block_task(task_id: int, body: Dict[str, str] = Body(default={})):  # type: ignore[assignment]
        reason = body.get("reason") if isinstance(body, dict) else None
        task = task_engine.block_task(task_id, reason)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_dict()

    # ---- Terminal ----

    @app.get("/api/terminal/commands")
    async def api_terminal_commands() -> Dict[str, object]:
        return {"commands": list(ALLOWED_COMMANDS.keys())}

    @app.post("/api/terminal/run")
    async def api_terminal_run(command: Dict[str, str] = Body(...)) -> Dict[str, object]:
        # Reject if raw 'command' string is passed instead of command_id
        if "command" in command and "command_id" not in command:
            raise HTTPException(status_code=400, detail="Use command_id, not command")
        cmd_id = command.get("command_id")
        if not cmd_id:
            raise HTTPException(status_code=400, detail="command_id is required")
        if not safety.check_command_allowed(cmd_id):
            raise HTTPException(status_code=403, detail="Command not allowed")
        if nonlocal_cmd_running["running"]:
            raise HTTPException(status_code=409, detail="A command is already running")
        nonlocal_cmd_running["running"] = True
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()
        try:
            result = execution_runner.run_safe_command(cmd_id)
            finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            duration_ms = int((time.monotonic() - t0) * 1000)
            stdout = _redact(result.get("stdout", ""))
            stderr = _redact(result.get("stderr", ""))
            execution_report.create_report(
                command_id=cmd_id, capability_id="execution.run_safe_command",
                returncode=result.get("returncode", 1),
                stdout=result.get("stdout", ""), stderr=result.get("stderr", ""),
                started_at=started, finished_at=finished, duration_ms=duration_ms,
            )
            task_engine.append_timeline_event({
                "event": "command_run", "command_id": cmd_id,
                "returncode": result.get("returncode"),
            })
            return {"command_id": cmd_id, "stdout": stdout, "stderr": stderr, "returncode": result.get("returncode")}
        finally:
            nonlocal_cmd_running["running"] = False

    # ---- Reports ----

    @app.get("/api/reports/recent")
    async def api_reports_recent() -> Dict[str, object]:
        return {"reports": execution_report.recent_reports()}

    @app.get("/api/reports/{report_id}")
    async def api_get_report(report_id: str) -> Dict[str, object]:
        report = execution_report.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return report

    # ---- A2A ----

    @app.post("/api/a2a/tasks")
    async def a2a_create_task(task: Dict[str, object] = Body(...)) -> Dict[str, object]:
        description = None
        if isinstance(task, dict):
            description = task.get("description") or task.get("title")
        if not description:
            raise HTTPException(status_code=400, detail="description or title is required")
        created = task_engine.create_task(str(description), source="a2a")
        return created.to_dict()

    @app.get("/api/a2a/tasks/{task_id}")
    async def a2a_get_task(task_id: int) -> Dict[str, object]:
        t = task_engine.get_task(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="Task not found")
        return t.to_dict()

    @app.post("/api/a2a/tasks/{task_id}/messages")
    async def a2a_append_message(task_id: int, message: Dict[str, object] = Body(...)) -> Dict[str, object]:
        t = task_engine.get_task(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="Task not found")
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Invalid message format")
        entry = {
            "task_id": task_id,
            "sender": message.get("sender", "unknown"),
            "content": message.get("content", ""),
        }
        append_memory_event(f"a2a_messages_{task_id}", entry)
        task_engine.append_timeline_event({"event": "a2a_message", "task_id": task_id})
        return {"status": "ok"}

    @app.get("/.well-known/agent-card.json")
    @app.get("/.well-known/agent.json")
    async def well_known_agent(request: Request) -> JSONResponse:
        base_url = str(request.base_url).rstrip("/")
        card = build_agent_card(base_url)
        from dataclasses import asdict
        return JSONResponse(content=asdict(card))

    @app.get("/api/a2a/agent-card")
    async def api_a2a_agent_card(request: Request) -> JSONResponse:
        base_url = str(request.base_url).rstrip("/")
        card = build_agent_card(base_url)
        from dataclasses import asdict
        return JSONResponse(content=asdict(card))

    @app.get("/api/a2a/capabilities")
    async def api_a2a_capabilities() -> Dict[str, object]:
        from igris.agents import list_capabilities
        caps = list_capabilities()
        return {"capabilities": [{"id": c.id, "name": c.name, "description": c.description, "safe": c.safe, "risk": c.risk} for c in caps]}

    return app


def app() -> FastAPI:
    """Factory function for uvicorn ``--factory`` mode."""
    return create_app()


def run_app(application: FastAPI, host: str = "0.0.0.0", port: int = 7778) -> None:
    """Run the FastAPI application using Uvicorn."""
    import uvicorn
    uvicorn.run(application, host=host, port=port, log_level="info")
