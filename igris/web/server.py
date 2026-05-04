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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from igris.core import anti_loop
from igris.core.task_engine import TaskEngine
from igris.core.teacher import build_teacher_payload, validate_teacher_assignment, propose_remediation_task
from igris.core import execution_report
from igris.core.chat_engine import chat as chat_llm, check_ollama_available
from igris.core import chat_streaming
from igris.core import chat_context
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
from igris.a2a import task_store as a2a_store
from igris.core.project_context import build_project_snapshot
from igris.core.memory import recent_memory_events, append_memory_event
from igris.core import mission_planner
from igris.core import decision_memory
from igris.core import diagnostics as diagnostics_mod
from igris.core import safe_policy
from igris.core import task_selection_explain
from igris.core import project_state as project_state_mod
from igris.core import decision_report as decision_report_mod
from igris.core import autonomous_loop
from igris.models.task import TaskStatus
from igris.layers.validation import validator as task_validator

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

    def _check_model_available(model_name: str) -> bool:
        """Check if a specific model is available in Ollama."""
        import urllib.request
        import urllib.error
        base_url = CONFIG.local_llm.base_url or "http://127.0.0.1:11434"
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
                import json as _json
                data = _json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(model_name in m for m in models)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError, ConnectionError, ValueError):
            return False

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

    # ---- Chat Streaming + Tier ----

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        content = await request.json()
        message = content.get("message", "")
        session_id = content.get("session_id")
        enrich = content.get("enrich", False)
        if not message:
            raise HTTPException(status_code=400, detail="message required")

        history = []
        if session_id and session_id in sessions:
            history = sessions[session_id]

        system_prompt = None
        if enrich:
            system_prompt = chat_context.build_context_system_prompt(
                task_engine=task_engine,
                project_root=str(CONFIG.project_root),
            )

        chunks = chat_streaming.chat_stream_sync(
            message=message, history=history, system_prompt=system_prompt,
        )

        # Store in session if provided
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append({"role": "user", "content": message})
            full_text = "".join(c.text for c in chunks if c.type == "content")
            sessions[session_id].append({"role": "assistant", "content": full_text})

            task_engine.append_timeline_event({
                "type": "chat", "title": "Chat stream",
                "detail": f"User: {message[:80]}",
            })

        async def event_generator():
            for chunk in chunks:
                yield chunk.to_sse()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/chat/context")
    async def api_chat_context() -> Dict[str, object]:
        return chat_context.build_chat_context(
            task_engine=task_engine,
            project_root=str(CONFIG.project_root),
        )

    @app.get("/api/chat/context/summary")
    async def api_chat_context_summary() -> Dict[str, object]:
        return chat_context.get_context_summary(
            task_engine=task_engine,
            project_root=str(CONFIG.project_root),
        )

    @app.get("/api/chat/tiers")
    async def api_chat_tiers() -> Dict[str, object]:
        return chat_streaming.get_tier_availability()

    @app.post("/api/chat/tiers")
    async def api_set_chat_tier(request: Request) -> Dict[str, object]:
        content = await request.json()
        tier = content.get("tier", "")
        if not tier:
            raise HTTPException(status_code=400, detail="tier required")
        try:
            config = chat_streaming.set_tier(tier)
            return config.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

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

    # ---- GitHub Workflow (gated) ----

    from igris.layers.git_layer import github_workflow as gh_wf

    @app.post("/api/git/commit")
    async def api_git_commit_gated(request: Request) -> Dict[str, object]:
        content = await request.json()
        message = content.get("message", "")
        approval = content.get("approval", "")
        if not message:
            raise HTTPException(status_code=400, detail="Commit message required")
        result = gh_wf.gated_commit(message=message, approval=approval)
        task_engine.append_timeline_event({
            "type": "git",
            "title": f"Gated commit: {'OK' if result.success else 'blocked'}",
            "detail": safety.redact_secrets(result.message if result.success else result.error),
            "severity": "info" if result.success else "warning",
        })
        return result.to_dict()

    @app.post("/api/github/pr/prepare")
    async def api_github_pr_prepare(request: Request) -> Dict[str, object]:
        content = await request.json()
        base = content.get("base", "main")
        title = content.get("title")
        extra = content.get("extra_context")
        prep = gh_wf.prepare_pr(base_branch=base, title=title, extra_context=extra)
        return prep.to_dict()

    @app.post("/api/github/pr/create")
    async def api_github_pr_create(request: Request) -> Dict[str, object]:
        content = await request.json()
        title = content.get("title", "")
        body = content.get("body", "")
        base = content.get("base", "main")
        approval = content.get("approval", "")
        if not title:
            raise HTTPException(status_code=400, detail="PR title required")
        result = gh_wf.gated_create_pr(
            title=title, body=body, base=base, approval=approval,
        )
        task_engine.append_timeline_event({
            "type": "github",
            "title": f"PR create: {'OK (gated)' if result.success else 'blocked'}",
            "detail": safety.redact_secrets(result.error if result.error else f"PR #{result.pr_number}"),
            "severity": "info" if result.success else "warning",
        })
        return result.to_dict()

    @app.get("/api/github/pr/status")
    async def api_github_pr_status() -> Dict[str, object]:
        return gh_wf.get_pr_status()

    # ---- Vast.ai (gated) ----

    from igris.layers.advisory.vastai_manager import VastAIManager
    vastai_mgr = VastAIManager()

    @app.get("/api/vastai/config")
    async def api_vastai_config() -> Dict[str, object]:
        return vastai_mgr.get_config()

    @app.get("/api/vastai/status")
    async def api_vastai_status() -> Dict[str, object]:
        return vastai_mgr.get_status()

    @app.post("/api/vastai/estimate")
    async def api_vastai_estimate(request: Request) -> Dict[str, object]:
        content = await request.json()
        model = content.get("model")
        hours = content.get("hours", 1.0)
        return vastai_mgr.estimate_cost(model=model, hours=hours)

    @app.post("/api/vastai/offers/search")
    async def api_vastai_offers_search(request: Request) -> Dict[str, object]:
        content = await request.json()
        model = content.get("model")
        max_cost = content.get("max_cost")
        result = vastai_mgr.search_offers(model=model, max_cost=max_cost)
        return result.to_dict()

    @app.post("/api/vastai/provision")
    async def api_vastai_provision(request: Request) -> Dict[str, object]:
        content = await request.json()
        approval = content.get("approval", "")
        model = content.get("model")
        offer_id = content.get("offer_id")
        result = vastai_mgr.provision(
            approval=approval, model=model, offer_id=offer_id,
        )
        task_engine.append_timeline_event({
            "type": "vastai",
            "title": f"Provision: {'OK (mock)' if result.get('success') else 'blocked'}",
            "detail": result.get("note", result.get("error", "")),
            "severity": "info" if result.get("success") else "warning",
        })
        return result

    @app.post("/api/vastai/destroy")
    async def api_vastai_destroy(request: Request) -> Dict[str, object]:
        content = await request.json()
        approval = content.get("approval", "")
        result = vastai_mgr.destroy(approval=approval)
        task_engine.append_timeline_event({
            "type": "vastai",
            "title": f"Destroy: {'OK (mock)' if result.get('success') else 'blocked'}",
            "detail": result.get("note", result.get("error", "")),
            "severity": "info" if result.get("success") else "warning",
        })
        return result

    @app.post("/api/vastai/set-mode")
    async def api_vastai_set_mode(request: Request) -> Dict[str, object]:
        content = await request.json()
        mode = content.get("mode", "")
        approval = content.get("approval", "")
        if not mode:
            raise HTTPException(status_code=400, detail="Mode required")
        return vastai_mgr.set_mode(mode=mode, approval=approval)

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

    @app.get("/api/routing/availability")
    async def api_routing_availability() -> Dict[str, object]:
        return provider_router.check_availability()

    @app.post("/api/routing/estimate")
    async def api_routing_estimate(body: Dict[str, object] = Body(default={})) -> Dict[str, object]:
        task_type = str(body.get("task_type", "chat")) if isinstance(body, dict) else "chat"
        complexity = str(body.get("complexity", "low")) if isinstance(body, dict) else "low"
        return provider_router.estimate_route(task_type=task_type, complexity=complexity)

    @app.get("/api/cost/budget")
    async def api_cost_budget() -> Dict[str, object]:
        return provider_router.get_budget_status()

    @app.post("/api/cost/budget")
    async def api_cost_budget_update(body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        max_cost = body.get("max_session_cost")
        warn = body.get("warn_threshold")
        return provider_router.set_budget_config(
            max_session_cost=float(max_cost) if max_cost is not None else None,
            warn_threshold=float(warn) if warn is not None else None,
        )

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
        ollama_ok = check_ollama_available()
        checks["ollama_available"] = ollama_ok
        checks["local_model_configured"] = CONFIG.local_llm.model
        checks["local_model_available"] = _check_model_available(CONFIG.local_llm.model) if ollama_ok else False
        checks["fallback_active"] = bool(CONFIG.fallback_llm.api_key)
        checks["fallback_reason"] = (
            "OpenAI API key configured" if CONFIG.fallback_llm.api_key
            else "No fallback API key — using deterministic fallback"
        )
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

    # ---- Decision Memory ----

    @app.get("/api/memory/failures")
    async def api_memory_failures(limit: int = 20) -> Dict[str, object]:
        events = decision_memory.get_recent_failures(limit, project_root=str(CONFIG.project_root))
        return {"events": events}

    @app.get("/api/memory/decisions")
    async def api_memory_decisions(limit: int = 20) -> Dict[str, object]:
        events = decision_memory.get_recent_decisions(limit, project_root=str(CONFIG.project_root))
        return {"events": events}

    @app.get("/api/memory/saturation")
    async def api_memory_saturation() -> Dict[str, object]:
        families = decision_memory.get_saturated_families(project_root=str(CONFIG.project_root))
        constraints = decision_memory.explain_memory_constraints(project_root=str(CONFIG.project_root))
        return {
            "saturated_families": families,
            "constraints": constraints,
        }

    @app.post("/api/memory/analyze")
    async def api_memory_analyze() -> Dict[str, object]:
        from igris.core import memory_analysis
        result = memory_analysis.analyze_memory(project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "memory", "title": "Memory analysis performed",
            "detail": f"LLM enhanced: {result.get('llm_enhanced', False)}",
            "severity": "info",
        })
        return result

    @app.get("/api/memory/analysis")
    async def api_memory_analysis_summary() -> Dict[str, object]:
        from igris.core import memory_analysis
        return memory_analysis.get_analysis_summary(project_root=str(CONFIG.project_root))

    @app.get("/api/memory/lessons")
    async def api_memory_lessons() -> Dict[str, object]:
        from igris.core import memory_analysis
        return memory_analysis.get_lessons_learned(project_root=str(CONFIG.project_root))

    @app.post("/api/memory/events")
    async def api_memory_record_event(request: Request) -> Dict[str, object]:
        content = await request.json()
        event_type = content.get("event_type", "")
        if event_type not in ("decision", "failure", "saturation", "remediation"):
            raise HTTPException(status_code=400, detail="event_type must be decision|failure|saturation|remediation")
        title = content.get("title", "")
        if not title and event_type != "saturation":
            raise HTTPException(status_code=400, detail="title is required")
        pr = str(CONFIG.project_root)
        if event_type == "decision":
            event = decision_memory.record_decision(
                title=title, family=content.get("family", ""),
                task_id=content.get("task_id", ""),
                description=content.get("description", ""),
                outcome=content.get("outcome", "success"),
                reason=content.get("reason", ""), project_root=pr,
            )
        elif event_type == "failure":
            event = decision_memory.record_failure(
                title=title, family=content.get("family", ""),
                task_id=content.get("task_id", ""),
                description=content.get("description", ""),
                reason=content.get("reason", ""), project_root=pr,
            )
        elif event_type == "saturation":
            family = content.get("family", "")
            if not family:
                raise HTTPException(status_code=400, detail="family is required for saturation")
            event = decision_memory.record_saturation(
                family=family, reason=content.get("reason", ""), project_root=pr,
            )
        else:
            event = decision_memory.record_remediation_attempt(
                title=title, family=content.get("family", ""),
                task_id=content.get("task_id", ""),
                description=content.get("description", ""),
                outcome=content.get("outcome", "pending"),
                reason=content.get("reason", ""), project_root=pr,
            )
        task_engine.append_timeline_event({
            "type": "memory", "title": f"Memory event: {_redact(title or event_type)}",
            "detail": _redact(content.get("description", "")[:200]),
            "severity": "info",
        })
        return event.to_dict()

    # ---- Autonomous Loop ----

    @app.post("/api/loop/step")
    async def api_loop_step() -> Dict[str, object]:
        result = autonomous_loop.execute_step(
            task_engine, project_root=str(CONFIG.project_root),
        )
        return result.to_dict()

    @app.post("/api/loop/run")
    async def api_loop_run(request: Request) -> Dict[str, object]:
        content = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        max_steps = content.get("max_steps", 1)
        if not isinstance(max_steps, int) or max_steps < 1:
            raise HTTPException(status_code=400, detail="max_steps must be a positive integer")
        status = autonomous_loop.run_loop(
            task_engine, max_steps=max_steps,
            project_root=str(CONFIG.project_root),
        )
        return status.to_dict()

    @app.get("/api/loop/status")
    async def api_loop_status() -> Dict[str, object]:
        return autonomous_loop.get_loop_status().to_dict()

    @app.get("/api/loop/recent")
    async def api_loop_recent(limit: int = 20) -> Dict[str, object]:
        return {"steps": autonomous_loop.get_recent_steps(limit)}

    # ---- Validation ----

    @app.post("/api/tasks/{task_id}/validate")
    async def api_validate_task(task_id: int, request: Request) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        content = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            content = await request.json()
        reports = execution_report.recent_reports(limit=10)
        files_changed = content.get("files_changed", [])
        manual_reason = content.get("manual_completion_reason", "")
        result = task_validator.validate_task_completion(
            task, reports=reports, files_changed=files_changed,
            manual_completion_reason=manual_reason,
            project_root=str(CONFIG.project_root),
        )
        task_engine.append_timeline_event({
            "type": "validation", "task_id": task_id,
            "title": f"Validation: {result.overall_status}",
            "detail": result.reason, "severity": "info" if result.valid else "warning",
        })
        return result.to_dict()

    @app.get("/api/tasks/{task_id}/validations")
    async def api_task_validations(task_id: int) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        validations = task_validator.get_validations_for_task(
            task_id, project_root=str(CONFIG.project_root),
        )
        return {"validations": [v.to_dict() for v in validations]}

    @app.get("/api/validations/{validation_id}")
    async def api_get_validation(validation_id: str) -> Dict[str, object]:
        result = task_validator.get_validation(
            validation_id, project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="Validation not found")
        return result.to_dict()

    @app.post("/api/tasks/{task_id}/complete")
    async def api_complete_task_validated(task_id: int, request: Request) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        content = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            content = await request.json()
        # Check existing validations
        validations = task_validator.get_validations_for_task(
            task_id, project_root=str(CONFIG.project_root),
        )
        has_valid = any(v.valid for v in validations)
        manual_reason = content.get("manual_completion_reason", "")
        if not has_valid and not manual_reason:
            raise HTTPException(
                status_code=400,
                detail="Task has no passing validation. Provide manual_completion_reason or validate first.",
            )
        if manual_reason and not has_valid:
            # Create manual validation
            task_validator.validate_task_completion(
                task, manual_completion_reason=manual_reason,
                project_root=str(CONFIG.project_root),
            )
        updated = task_engine.complete_task(task_id, result=manual_reason or "Validated completion")
        task_engine.append_timeline_event({
            "type": "validation", "task_id": task_id,
            "title": "Task completed (validated)",
            "detail": manual_reason or "passed validation",
            "severity": "info",
        })
        return updated.to_dict() if updated else {}

    # ---- Missions ----

    def _redact_mission_dict(d: Dict[str, object]) -> Dict[str, object]:
        """Redact secrets from mission response."""
        for key in ("title", "description", "plan_summary"):
            if key in d and isinstance(d[key], str):
                d[key] = _redact(d[key])
        if "steps" in d and isinstance(d["steps"], list):
            for step in d["steps"]:
                for key in ("title", "description"):
                    if key in step and isinstance(step[key], str):
                        step[key] = _redact(step[key])
        return d

    @app.post("/api/missions")
    async def api_create_mission(request: Request) -> Dict[str, object]:
        content = await request.json()
        title = content.get("title", "")
        description = content.get("description", "")
        if not title:
            raise HTTPException(status_code=400, detail="Mission title required")
        m = mission_planner.Mission(title=title, description=description)
        mission_planner.save_mission(m, project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Mission created: {_redact(title)}",
            "detail": _redact(description[:200]), "severity": "info",
        })
        return _redact_mission_dict(m.to_dict())

    @app.get("/api/missions")
    async def api_list_missions() -> Dict[str, object]:
        missions = mission_planner.list_missions(project_root=str(CONFIG.project_root))
        return {"missions": [_redact_mission_dict(m.to_dict()) for m in missions]}

    @app.get("/api/missions/{mission_id}")
    async def api_get_mission(mission_id: str) -> Dict[str, object]:
        m = mission_planner.load_mission(mission_id, project_root=str(CONFIG.project_root))
        if not m:
            raise HTTPException(status_code=404, detail="Mission not found")
        return _redact_mission_dict(m.to_dict())

    @app.post("/api/missions/{mission_id}/plan")
    async def api_plan_mission(
        mission_id: str,
        mode: str = "deterministic",
    ) -> Dict[str, object]:
        from igris.core import llm_planner
        if mode not in ("deterministic", "llm", "auto"):
            raise HTTPException(status_code=400, detail="Invalid mode. Use: deterministic, llm, auto")
        result = llm_planner.plan_mission_with_mode(
            mission_id, mode=mode, project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        task_engine.append_timeline_event({
            "type": "mission",
            "title": f"Mission planned ({result['planning']['mode']}): {_redact(result['mission'].get('title', ''))}",
            "detail": f"{len(result['mission'].get('steps', []))} steps",
            "severity": "info",
        })
        result["mission"] = _redact_mission_dict(result["mission"])
        return result

    @app.get("/api/missions/{mission_id}/plan/explain")
    async def api_plan_explain(mission_id: str) -> Dict[str, object]:
        from igris.core import llm_planner
        explanation = llm_planner.explain_plan(
            mission_id, project_root=str(CONFIG.project_root),
        )
        if not explanation:
            raise HTTPException(status_code=404, detail="Mission not found")
        return explanation

    @app.post("/api/missions/{mission_id}/materialize-tasks")
    async def api_materialize_tasks(mission_id: str) -> Dict[str, object]:
        m = mission_planner.materialize_tasks(
            mission_id, task_engine, project_root=str(CONFIG.project_root),
        )
        if not m:
            raise HTTPException(status_code=404, detail="Mission not found or no plan")
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Tasks materialized: {_redact(m.title)}",
            "detail": f"{len(m.task_ids)} tasks created", "severity": "info",
        })
        return _redact_mission_dict(m.to_dict())

    @app.get("/api/missions/{mission_id}/graph")
    async def api_mission_graph(mission_id: str) -> Dict[str, object]:
        graph = mission_planner.get_mission_graph(
            mission_id, project_root=str(CONFIG.project_root),
        )
        if not graph:
            raise HTTPException(status_code=404, detail="Mission not found")
        return graph

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

    # ---- Safety Policy ----

    @app.get("/api/safety/policy")
    async def api_safety_policy() -> Dict[str, object]:
        return safe_policy.get_policy_status()

    @app.post("/api/safety/policy/check")
    async def api_safety_policy_check(request: Request) -> Dict[str, object]:
        content = await request.json()
        command_id = content.get("command_id", "")
        if not command_id:
            raise HTTPException(status_code=400, detail="command_id required")
        context = content.get("context")
        decision = safe_policy.check_command_policy(command_id, context=context)
        return decision.to_dict()

    # ---- Explainable Task Selection ----

    @app.get("/api/tasks/selection/explain")
    async def api_explain_task_selection() -> Dict[str, object]:
        tasks = task_engine.list_tasks()
        history = [t.description for t in tasks if t.status == TaskStatus.completed]
        explanation = task_selection_explain.explain_task_selection(
            candidate_tasks=tasks,
            history=history,
            project_root=str(CONFIG.project_root),
        )
        return explanation.to_dict()

    # ---- Project State + Saturation Cooldown ----

    @app.get("/api/project-state")
    async def api_project_state() -> Dict[str, object]:
        return project_state_mod.get_project_state(project_root=str(CONFIG.project_root))

    @app.get("/api/project-state/recovery")
    async def api_recovery_summary() -> Dict[str, object]:
        return project_state_mod.get_recovery_summary(project_root=str(CONFIG.project_root))

    @app.get("/api/project-state/family/{family}")
    async def api_family_availability(family: str) -> Dict[str, object]:
        return project_state_mod.is_family_available(family, project_root=str(CONFIG.project_root))

    @app.post("/api/project-state/family/{family}/reset-cooldown")
    async def api_reset_cooldown(family: str) -> Dict[str, object]:
        ok = project_state_mod.reset_family_cooldown(family, project_root=str(CONFIG.project_root))
        if not ok:
            raise HTTPException(status_code=404, detail=f"Family '{family}' not found")
        return {"family": family, "cooldown_reset": True}

    @app.get("/api/project-state/fingerprints")
    async def api_recent_fingerprints() -> Dict[str, object]:
        fps = project_state_mod.get_recent_fingerprints(limit=20, project_root=str(CONFIG.project_root))
        return {"fingerprints": fps}

    # ---- Decision Reports ----

    @app.get("/api/decision-reports")
    async def api_list_decision_reports() -> Dict[str, object]:
        reports = decision_report_mod.list_decision_reports(
            limit=20, project_root=str(CONFIG.project_root),
        )
        return {"reports": reports}

    @app.get("/api/decision-reports/{report_id}")
    async def api_get_decision_report(report_id: str) -> Dict[str, object]:
        report = decision_report_mod.get_decision_report(
            report_id, project_root=str(CONFIG.project_root),
        )
        if not report:
            raise HTTPException(status_code=404, detail="Decision report not found")
        return report

    @app.post("/api/decision-reports")
    async def api_create_decision_report(request: Request) -> Dict[str, object]:
        content = await request.json()
        tasks = task_engine.list_tasks()
        report = decision_report_mod.create_decision_report(
            step_number=content.get("step_number", 0),
            tasks=tasks,
            action_type=content.get("action_type", ""),
            action_detail=content.get("action_detail", ""),
            outcome=content.get("outcome", ""),
            outcome_reason=content.get("outcome_reason", ""),
            next_action=content.get("next_action", ""),
            next_action_reason=content.get("next_action_reason", ""),
            safety_decisions=content.get("safety_decisions", []),
            project_root=str(CONFIG.project_root),
        )
        return report.to_dict()

    # ---- Diagnostics ----

    @app.get("/api/diagnostics")
    async def api_diagnostics() -> Dict[str, object]:
        tasks = [t.to_dict() for t in task_engine.list_tasks()]
        timeline = task_engine.recent_timeline_events(limit=50)
        report = diagnostics_mod.run_diagnostics(
            tasks, timeline, project_root=str(CONFIG.project_root),
        )
        return report.to_dict()

    @app.get("/api/diagnostics/summary")
    async def api_diagnostics_summary() -> Dict[str, object]:
        tasks = [t.to_dict() for t in task_engine.list_tasks()]
        timeline = task_engine.recent_timeline_events(limit=50)
        return diagnostics_mod.get_diagnostic_summary(
            tasks, timeline, project_root=str(CONFIG.project_root),
        )

    # ---- A2A Task Store (artifacts, events, cancel) ----

    @app.post("/api/a2a/store/tasks")
    async def api_a2a_store_create(body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        title = body.get("title", "")
        description = body.get("description", "")
        if not title and not description:
            raise HTTPException(status_code=400, detail="title or description required")
        task = a2a_store.create_a2a_task(
            title=str(title), description=str(description),
            project_root=str(CONFIG.project_root),
        )
        task_engine.append_timeline_event({
            "type": "a2a", "title": f"A2A task created: {_redact(str(title)[:80])}",
            "severity": "info",
        })
        return task

    @app.get("/api/a2a/store/tasks")
    async def api_a2a_store_list() -> Dict[str, object]:
        tasks = a2a_store.list_a2a_tasks(project_root=str(CONFIG.project_root))
        return {"tasks": tasks}

    @app.get("/api/a2a/store/tasks/{task_id}")
    async def api_a2a_store_get(task_id: str) -> Dict[str, object]:
        task = a2a_store.get_a2a_task(task_id, project_root=str(CONFIG.project_root))
        if not task:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return task

    @app.post("/api/a2a/store/tasks/{task_id}/status")
    async def api_a2a_store_update_status(task_id: str, body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        status = body.get("status", "")
        if not status or status not in a2a_store.VALID_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid status. Valid: {sorted(a2a_store.VALID_STATUSES)}")
        detail = str(body.get("detail", ""))
        result = a2a_store.update_a2a_task_status(
            task_id, str(status), detail=detail,
            project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="Task not found or cannot transition")
        return result

    @app.get("/api/a2a/tasks/{task_id}/artifacts")
    async def api_a2a_artifacts_list(task_id: str) -> Dict[str, object]:
        artifacts = a2a_store.get_artifacts(task_id, project_root=str(CONFIG.project_root))
        if artifacts is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return {"artifacts": artifacts}

    @app.post("/api/a2a/tasks/{task_id}/artifacts")
    async def api_a2a_artifacts_add(task_id: str, body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        name = body.get("name", "")
        content = body.get("content", "")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        mime_type = str(body.get("mime_type", "text/plain"))
        result = a2a_store.add_artifact(
            task_id, name=str(name), content=str(content),
            mime_type=mime_type, project_root=str(CONFIG.project_root),
        )
        if result is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        task_engine.append_timeline_event({
            "type": "a2a", "title": f"A2A artifact added: {_redact(str(name)[:80])}",
            "severity": "info",
        })
        return result

    @app.post("/api/a2a/tasks/{task_id}/cancel")
    async def api_a2a_cancel(task_id: str, body: Dict[str, object] = Body(default={})) -> Dict[str, object]:
        reason = str(body.get("reason", "")) if isinstance(body, dict) else ""
        result = a2a_store.cancel_a2a_task(
            task_id, reason=reason,
            project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="A2A task not found or already terminal")
        task_engine.append_timeline_event({
            "type": "a2a", "title": f"A2A task canceled: {task_id}",
            "severity": "warning",
        })
        return result

    @app.get("/api/a2a/tasks/{task_id}/events")
    async def api_a2a_events(task_id: str) -> Dict[str, object]:
        events = a2a_store.get_events(task_id, project_root=str(CONFIG.project_root))
        if events is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return {"events": events}

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
