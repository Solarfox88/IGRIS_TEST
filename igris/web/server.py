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
from igris.core.teacher import build_teacher_payload, validate_teacher_assignment, propose_remediation_task
from igris.core import execution_report
from igris.core.chat_engine import chat as chat_llm, check_ollama_available
from igris.core.outcome_router import route_outcome
from igris.core import patch_proposal as patch_mod
from igris.layers.advisory import router as provider_router
from igris.layers.execution import runner as execution_runner
from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
from igris.core import safety
from igris.layers.git_layer.git_status import get_git_info
from igris.layers.git_layer import git_ops
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

    # ---- Sessions / Chat ----

    @app.post("/api/sessions")
    async def create_session() -> Dict[str, str]:
        session_id = str(len(sessions) + 1)
        sessions[session_id] = []
        return {"id": session_id}

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, content: Dict[str, str] = Body(...)) -> Dict[str, object]:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        message = content.get("message", "")
        sessions[session_id].append({"role": "user", "content": message})

        # Use real chat engine
        result = chat_llm(message, history=sessions[session_id][:-1])
        response_text = _redact(result["text"])

        sessions[session_id].append({"role": "assistant", "content": response_text})

        # Record routing decision
        provider_router.record_chat_routing(
            provider=result["provider"], model=result["model"],
            reason=result["routing_reason"], latency_ms=result["latency_ms"],
            fallback_used=result["fallback_used"],
        )

        task_engine.append_timeline_event({
            "type": "chat", "title": "Chat message",
            "detail": f"User: {message[:80]}",
        })

        return {
            "response": response_text,
            "provider": result["provider"],
            "model": result["model"],
            "fallback_used": result["fallback_used"],
            "latency_ms": result["latency_ms"],
        }

    # ---- Git ----

    @app.get("/api/git/status", response_model=GitStatusResponse)
    async def api_git_status() -> GitStatusResponse:
        info = get_git_info()
        return GitStatusResponse(
            branch=info.branch, remote=info.remote,
            dirty=info.dirty, changed=info.changed, head=info.head,
        )

    @app.get("/api/git/diff")
    async def api_git_diff(staged: bool = False) -> Dict[str, object]:
        return git_ops.get_diff(staged=staged)

    @app.get("/api/git/diff/stat")
    async def api_git_diff_stat() -> Dict[str, object]:
        return git_ops.get_diff_stat()

    @app.get("/api/git/branches")
    async def api_git_branches() -> Dict[str, object]:
        return git_ops.list_branches()

    @app.post("/api/git/branch")
    async def api_git_create_branch(request: Request) -> Dict[str, object]:
        content = await request.json()
        name = content.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="Branch name required")
        result = git_ops.create_branch(name)
        if result.get("success"):
            task_engine.append_timeline_event({
                "type": "git", "title": f"Branch created: {result.get('branch')}",
                "detail": "", "severity": "info",
            })
        return result

    @app.post("/api/git/commit-proposal")
    async def api_git_commit_proposal(request: Request) -> Dict[str, object]:
        content = await request.json()
        message = content.get("message", "")
        files = content.get("files")
        if not message:
            raise HTTPException(status_code=400, detail="Commit message required")
        proposal = git_ops.create_commit_proposal(message, files)
        return {
            "message": proposal.message,
            "files": proposal.files,
            "safe": proposal.safe,
            "warnings": proposal.warnings,
            "blocked_files": proposal.blocked_files,
            "secret_files": proposal.secret_files,
            "runtime_artifacts": proposal.runtime_artifacts,
        }

    @app.get("/api/git/safety-check")
    async def api_git_safety_check() -> Dict[str, object]:
        return git_ops.pre_commit_safety_check()

    @app.get("/api/git/pr-summary")
    async def api_git_pr_summary(base: str = "main") -> Dict[str, object]:
        return git_ops.generate_pr_summary(base_branch=base)

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
            report = execution_report.create_report(
                command_id="run_tests", capability_id="validation.run_tests",
                returncode=result["returncode"], stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                started_at=started, finished_at=finished, duration_ms=duration_ms,
            )
            # Route outcome
            recommendation = route_outcome(report, "run tests")
            task_engine.append_timeline_event({
                "type": "test", "title": "Test run",
                "detail": f"{'Passed' if success else 'Failed'} in {duration_ms}ms",
                "related_report_id": report.get("report_id"),
                "severity": "info" if success else "warning",
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
        checks["ollama_available"] = check_ollama_available()
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
        task_engine.append_timeline_event({
            "type": "task", "title": f"Task created: {title or description[:40]}",
            "detail": description[:100], "related_task_id": task.id,
            "severity": "info",
        })
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
        task_engine.append_timeline_event({
            "type": "task", "title": f"Task completed: #{task_id}",
            "detail": result_text or "", "related_task_id": task_id,
            "severity": "info",
        })
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
            report = execution_report.create_report(
                command_id=cmd_id, capability_id="execution.run_safe_command",
                returncode=result.get("returncode", 1),
                stdout=result.get("stdout", ""), stderr=result.get("stderr", ""),
                started_at=started, finished_at=finished, duration_ms=duration_ms,
            )
            # Route outcome
            recommendation = route_outcome(report, f"terminal {cmd_id}")
            task_engine.append_timeline_event({
                "type": "action", "title": f"Command: {cmd_id}",
                "detail": f"exit={result.get('returncode', 1)}, {duration_ms}ms",
                "related_report_id": report.get("report_id"),
                "severity": "info" if result.get("returncode") == 0 else "warning",
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

    # ---- Teacher Remediation ----

    @app.post("/api/teacher/remediate")
    async def api_teacher_remediate(body: Dict[str, object] = Body(default={})) -> Dict[str, object]:
        task_id = body.get("task_id") if isinstance(body, dict) else None
        create = body.get("create", False) if isinstance(body, dict) else False

        # Build teacher payload
        recent_tasks = task_engine.list_tasks()
        history = [t.description for t in recent_tasks]
        payload = build_teacher_payload(tasks=history)

        # Propose remediation
        proposed = propose_remediation_task(payload)

        # Validate if we have enough info
        validation = None
        if proposed.get("task_description"):
            assignment = {
                "diagnosis": proposed.get("reason", ""),
                "selected_family": proposed.get("family", "other"),
                "why_this_family": proposed.get("reason", ""),
                "differentiator": proposed.get("differentiator", ""),
                "task_title": proposed.get("task_title", ""),
                "task_description": proposed.get("task_description", ""),
                "success_criteria": proposed.get("success_criteria", []),
                "safe_command_ids": proposed.get("safe_command_ids", []),
                "expected_next_state": proposed.get("expected_next_state", ""),
                "fallback_if_blocked": proposed.get("fallback_if_blocked", ""),
            }
            validation = validate_teacher_assignment(assignment, history)

        result: Dict[str, object] = {
            "payload": payload,
            "proposed_task": proposed,
            "validation": validation,
            "created_task_id": None,
        }

        # Create the task if requested and valid
        if create and proposed.get("task_description"):
            if validation and validation.get("valid", False):
                created = task_engine.create_task(
                    description=proposed["task_description"],
                    title=proposed.get("task_title"),
                    family=proposed.get("family"),
                    source="teacher",
                )
                result["created_task_id"] = created.id
                task_engine.append_timeline_event({
                    "type": "teacher", "title": "Teacher remediation",
                    "detail": f"Created task: {proposed.get('task_title', '')}",
                    "related_task_id": created.id,
                })

        return result

    # ---- Outcome Router ----

    @app.get("/api/outcome/recent")
    async def api_outcome_recent() -> Dict[str, object]:
        reports = execution_report.recent_reports(limit=10)
        outcomes = []
        for r in reports:
            rec = route_outcome(r)
            outcomes.append(rec)
        return {"outcomes": outcomes}

    # ---- Patch Proposals ----

    @app.get("/api/patches")
    async def api_list_patches() -> Dict[str, object]:
        patches = patch_mod.list_patch_proposals(project_root=str(CONFIG.project_root))
        return {"patches": patches}

    @app.post("/api/patches/propose")
    async def api_propose_patch(request: Request) -> Dict[str, object]:
        content = await request.json()
        title = content.get("title", "Untitled patch")
        description = content.get("description", "")
        task_id = content.get("task_id")
        files = content.get("files", [])
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        proposal = patch_mod.create_patch_proposal(
            title=title,
            description=description,
            files=files,
            task_id=task_id,
            project_root=str(CONFIG.project_root),
        )
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch proposed: {title}",
            "detail": f"{len(files)} file(s)", "severity": "info",
            "related_task_id": task_id,
            "related_patch_id": proposal.id,
        })
        return patch_mod._proposal_to_dict(proposal)

    @app.get("/api/patches/{proposal_id}")
    async def api_get_patch(proposal_id: str) -> Dict[str, object]:
        proposal = patch_mod.load_patch_proposal(proposal_id, project_root=str(CONFIG.project_root))
        if proposal is None:
            raise HTTPException(status_code=404, detail="Proposal not found")
        return patch_mod._proposal_to_dict(proposal)

    @app.post("/api/patches/{proposal_id}/validate")
    async def api_validate_patch(proposal_id: str) -> Dict[str, object]:
        proposal = patch_mod.load_patch_proposal(proposal_id, project_root=str(CONFIG.project_root))
        if proposal is None:
            raise HTTPException(status_code=404, detail="Proposal not found")
        result = patch_mod.validate_patch_proposal(proposal, project_root=str(CONFIG.project_root))
        severity = "info" if result.valid else "warning"
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch validated: {proposal.title}",
            "detail": f"valid={result.valid}, risk={result.risk}",
            "severity": severity,
            "related_patch_id": proposal.id,
        })
        return {
            "proposal_id": proposal_id,
            "status": proposal.status,
            "validation": {
                "valid": result.valid,
                "reasons": result.reasons,
                "blocked_paths": result.blocked_paths,
                "secret_findings": result.secret_findings,
                "risk": result.risk,
            },
        }

    @app.post("/api/patches/{proposal_id}/apply")
    async def api_apply_patch(proposal_id: str) -> Dict[str, object]:
        result = patch_mod.apply_patch_proposal(proposal_id, project_root=str(CONFIG.project_root))
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Apply failed"))
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch applied: {proposal_id}",
            "detail": f"{len(result.get('applied_files', []))} file(s) modified",
            "severity": "info",
            "related_patch_id": proposal_id,
        })
        return result

    @app.post("/api/patches/{proposal_id}/reject")
    async def api_reject_patch(proposal_id: str, request: Request) -> Dict[str, object]:
        content = await request.json()
        reason = content.get("reason", "")
        result = patch_mod.reject_patch_proposal(proposal_id, reason=reason, project_root=str(CONFIG.project_root))
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Reject failed"))
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch rejected: {proposal_id}",
            "detail": reason or "No reason given",
            "severity": "warning",
            "related_patch_id": proposal_id,
        })
        return result

    return app


def app() -> FastAPI:
    """Factory function for uvicorn ``--factory`` mode."""
    return create_app()


def run_app(application: FastAPI, host: str = "0.0.0.0", port: int = 7778) -> None:
    """Run the FastAPI application using Uvicorn."""
    import uvicorn
    uvicorn.run(application, host=host, port=port, log_level="info")
