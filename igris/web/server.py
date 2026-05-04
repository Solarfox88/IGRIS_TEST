"""
FastAPI application for IGRIS_GPT.

This module exposes a factory to create the FastAPI application and a helper
to run it with Uvicorn.  The application serves both the HTTP API used by
the web UI as well as the root HTML page rendered via Jinja2 templates.
"""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from igris.core import anti_loop
from igris.core.task_engine import TaskEngine
from igris.core.teacher import build_teacher_payload
from igris.layers.advisory import router as provider_router
from igris.layers.execution import runner as execution_runner
from igris.layers.git_layer.git_status import get_git_info
from igris.models.config import CONFIG
from igris.models.report import GitStatusResponse, TestRunResponse

# Determine paths relative to this file
MODULE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = MODULE_DIR / "templates"
STATIC_DIR = MODULE_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="IGRIS_GPT", version="0.1.0")

    # Mount static files for CSS/JS
    if STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="static",
        )

    # Set up Jinja environment manually; FastAPI includes a Templates helper but
    # this manual setup avoids the dependency on starlette.templating during tests.
    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # In‑memory session storage
    sessions: Dict[str, List[Dict[str, str]]] = {}
    task_engine = TaskEngine()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the main UI page."""
        template = jinja_env.get_template("index.html")
        return template.render()

    @app.get("/api/status")
    async def api_status() -> Dict[str, object]:
        provider, model = provider_router.choose_provider()
        return {
            "provider": provider,
            "model": model,
            "safe": True,
        }

    @app.get("/api/config/safe")
    async def api_config_safe() -> Dict[str, object]:
        return CONFIG.safe_dict()

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
        # Choose provider (for now always local) and produce a placeholder response
        provider_router.choose_provider(for_task="chat")
        response_text = "This is a placeholder response."
        sessions[session_id].append({"role": "assistant", "content": response_text})
        return {"response": response_text}

    @app.get("/api/git/status", response_model=GitStatusResponse)
    async def api_git_status() -> GitStatusResponse:
        info = get_git_info()
        return GitStatusResponse(
            branch=info.branch,
            remote=info.remote,
            dirty=info.dirty,
            changed=info.changed,
            head=info.head,
        )

    def _is_safe_path(path: Path, root: Path) -> bool:
        try:
            return root in path.resolve().parents or path.resolve() == root.resolve()
        except Exception:
            return False

    @app.get("/api/files/tree")
    async def api_files_tree() -> Dict[str, object]:
        root = CONFIG.project_root
        tree = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            # Skip hidden directories and .git
            if ".git" in dirnames:
                dirnames.remove(".git")
            if "__pycache__" in dirnames:
                dirnames.remove("__pycache__")
            entries = []
            for d in sorted(dirnames):
                entries.append({"type": "dir", "name": d})
            for f in sorted(filenames):
                entries.append({"type": "file", "name": f})
            tree.append({"path": rel_dir, "entries": entries})
        return {"tree": tree}

    @app.get("/api/files/preview")
    async def api_files_preview(path: str) -> Dict[str, object]:
        root = CONFIG.project_root
        # Normalize and validate path
        requested = (root / path).resolve()
        if not _is_safe_path(requested, root):
            raise HTTPException(status_code=403, detail="Invalid path")
        if requested.is_dir():
            raise HTTPException(status_code=400, detail="Cannot preview a directory")
        if not requested.exists():
            raise HTTPException(status_code=404, detail="File not found")
        # Reject binary files based on mimetype
        mime, _ = mimetypes.guess_type(str(requested))
        if mime and not mime.startswith("text"):
            raise HTTPException(status_code=400, detail="Only text files can be previewed")
        try:
            with requested.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read(10_000)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"path": path, "preview": content}

    @app.post("/api/tests/run", response_model=TestRunResponse)
    async def api_tests_run() -> TestRunResponse:
        result = execution_runner.run_tests()
        success = result["returncode"] == 0
        return TestRunResponse(
            success=success,
            stdout=result["stdout"],
            stderr=result["stderr"],
        )

    @app.get("/api/logs")
    async def api_logs(lines: int = 200) -> Dict[str, str]:
        log_path = Path("logs/igris.log")
        if not log_path.exists():
            return {"logs": "Log file not found."}
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()[-lines:]
        return {"logs": "".join(data)}

    @app.get("/api/agent/timeline")
    async def api_agent_timeline() -> Dict[str, object]:
        # Placeholder timeline; this would normally include plan/action/observation events
        return {"timeline": []}

    @app.get("/api/safety/status")
    async def api_safety_status() -> Dict[str, object]:
        # Compute saturated families from the recent tasks
        tasks = [t.description for t in task_engine.tasks]
        counts = anti_loop.compute_family_counts(tasks)
        saturated = anti_loop.saturated_families(counts)
        return {
            "saturated_families": saturated,
            "counts": counts,
        }

    @app.get("/api/routing/explain")
    async def api_routing_explain() -> Dict[str, str]:
        explanation = provider_router.explain_routing()
        return {"explanation": explanation}

    return app


def run_app(app: FastAPI, host: str = "0.0.0.0", port: int = 7778) -> None:
    """Run the FastAPI application using Uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")