"""IGRIS web server router — auto-split from server.py (#725).

Route handlers are extracted from _create_app_impl; shared app state is
received via ``deps`` (SimpleNamespace). Do not edit route logic here;
changes should first be made in the original handler before full migration.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

# Module-level path constants mirrored from server.py
_MODULE_DIR = Path(__file__).resolve().parent.parent  # igris/web/
TEMPLATES_DIR = _MODULE_DIR / "templates"
STATIC_DIR = _MODULE_DIR / "static"

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from igris.core import anti_loop
from igris.core import chat_context
from igris.core import chat_streaming
from igris.core import decision_memory
from igris.core import diagnostics as diagnostics_mod
from igris.core import execution_report
from igris.core import mission_planner
from igris.core import project_state as project_state_mod
from igris.core import safe_policy
from igris.core import safety
from igris.core import task_selection_explain
from igris.core import decision_report as decision_report_mod
from igris.core import autonomous_loop
from igris.core.chat_engine import chat as chat_llm, check_ollama_available
from igris.core import patch_proposal as patch_mod
from igris.core.memory import recent_memory_events, append_memory_event
from igris.core.memory_graph import MemoryGraph
from igris.core.outcome_router import route_outcome
from igris.core.project_context import build_project_snapshot
from igris.core.teacher import (
    build_teacher_payload, validate_teacher_assignment, propose_remediation_task,
)
from igris.core.task_engine import TaskEngine
from igris.layers.advisory import router as provider_router
from igris.layers.execution import runner as execution_runner
from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
from igris.layers.git_layer import git_ops
from igris.layers.git_layer.git_status import get_git_info
from igris.layers.validation import validator as task_validator
from igris.models.config import CONFIG
from igris.models.report import GitStatusResponse, TestRunResponse
from igris.models.task import TaskStatus
from igris.agents import build_default_registry
from igris.a2a.agent_card import build_agent_card
from igris.a2a import task_store as a2a_store


def create_router(deps) -> APIRouter:
    """Router module 3/10 — _create_app_impl chunk 3."""
    router = APIRouter()
    # Unpack shared app state (names match what route bodies use directly)
    _redact = deps.redact
    _check_model_available = deps.check_model_available
    _get_graph = deps.get_graph
    jinja_env = deps.jinja_env
    sessions = deps.sessions
    task_engine = deps.task_engine
    nonlocal_test_running = deps.nonlocal_test_running
    nonlocal_cmd_running = deps.nonlocal_cmd_running

    @router.get("/api/files/preview")
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

    @router.post("/api/tests/run", response_model=TestRunResponse)
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

    @router.get("/api/logs")
    async def api_logs(lines: int = 200) -> Dict[str, str]:
        log_path = Path("logs/igris.log")
        if not log_path.exists():
            return {"logs": "Log file not found."}
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()[-lines:]
        return {"logs": "".join(data)}

    # ---- Timeline ----

    @router.get("/api/agent/timeline")
    async def api_agent_timeline() -> Dict[str, object]:
        events = task_engine.recent_timeline_events(limit=50)
        return {"timeline": events}

    # ---- Safety ----

    @router.get("/api/safety/status")
    async def api_safety_status() -> Dict[str, object]:
        tasks = [t.description for t in task_engine.tasks]
        counts = anti_loop.compute_family_counts(tasks)
        saturated = anti_loop.saturated_families(counts)
        return {"saturated_families": saturated, "counts": counts}

    # ---- Health / Readiness ----

    @router.get("/api/health")
    async def api_health() -> Dict[str, object]:
        return {"status": "ok", "version": "0.1.0", "time": time.time()}

    @router.get("/api/smw/health")
    async def api_smw_health() -> Dict[str, object]:
        from igris.core.smw_patterns import detect_patterns
        from igris.core.smw_sensors import take_snapshot
        from igris.core.smw_weak_signals import get_weak_signal_summary
        from dataclasses import asdict
        snapshot = await take_snapshot(str(CONFIG.project_root))
        patterns = detect_patterns(snapshot)
        weak = get_weak_signal_summary(str(CONFIG.project_root))
        return {
            "snapshot": asdict(snapshot),
            "active_patterns": [
                {"name": p.pattern.name, "severity": p.pattern.severity, "evidence": p.evidence}
                for p in patterns
            ],
            "weak_signals_active": weak.get("weak_signals_active", []),
            "metrics": weak.get("metrics", {}),
        }

    @router.get("/api/readiness")
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

    @router.get("/api/project/context")
    async def api_project_context() -> Dict[str, object]:
        snapshot = build_project_snapshot(task_engine=task_engine)
        return snapshot

    # ---- Memory ----

    @router.get("/api/memory/recent")
    async def api_memory_recent(namespace: str, limit: int = 20) -> Dict[str, object]:
        events = recent_memory_events(namespace, limit)
        return {"events": events}

    # ---- Decision Memory ----

    @router.get("/api/memory/failures")
    async def api_memory_failures(limit: int = 20) -> Dict[str, object]:
        events = decision_memory.get_recent_failures(limit, project_root=str(CONFIG.project_root))
        return {"events": events}

    @router.get("/api/memory/decisions")
    async def api_memory_decisions(limit: int = 20) -> Dict[str, object]:
        events = decision_memory.get_recent_decisions(limit, project_root=str(CONFIG.project_root))
        return {"events": events}

    @router.get("/api/memory/saturation")
    async def api_memory_saturation() -> Dict[str, object]:
        families = decision_memory.get_saturated_families(project_root=str(CONFIG.project_root))
        constraints = decision_memory.explain_memory_constraints(project_root=str(CONFIG.project_root))
        return {
            "saturated_families": families,
            "constraints": constraints,
        }

    @router.post("/api/memory/analyze")
    async def api_memory_analyze() -> Dict[str, object]:
        from igris.core import memory_analysis
        result = memory_analysis.analyze_memory(project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "memory", "title": "Memory analysis performed",
            "detail": f"LLM enhanced: {result.get('llm_enhanced', False)}",
            "severity": "info",
        })
        return result

    @router.get("/api/memory/analysis")
    async def api_memory_analysis_summary() -> Dict[str, object]:
        from igris.core import memory_analysis
        return memory_analysis.get_analysis_summary(project_root=str(CONFIG.project_root))

    @router.get("/api/memory/lessons")
    async def api_memory_lessons() -> Dict[str, object]:
        from igris.core import memory_analysis
        return memory_analysis.get_lessons_learned(project_root=str(CONFIG.project_root))

    @router.get("/api/memory/summary")
    async def api_memory_summary() -> Dict[str, object]:
        g = _get_graph()
        node_count = g.conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
        edge_count = g.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
        rows = g.conn.execute("SELECT node_type, COUNT(*) as c FROM memory_nodes GROUP BY node_type").fetchall()
        migration_done = bool(g.conn.execute("SELECT 1 FROM memory_nodes WHERE node_type='environment_fact' AND content LIKE '%\"migration_done\"%' LIMIT 1").fetchone())
        # Issue #616 — dependency_graph: best-effort (never fails the endpoint)
        dependency_graph: Dict[str, object] = {}
        try:
            from igris.core.dependency_checker import DependencyChecker, load_dep_file
            _dep_map = load_dep_file(str(CONFIG.project_root))
            if _dep_map:
                _checker = DependencyChecker(str(CONFIG.project_root))
                for _issue_str, _deps in _dep_map.items():
                    try:
                        _dep_ok, _dep_unsat = _checker.check(int(_issue_str))
                        dependency_graph[_issue_str] = {
                            "deps": _deps,
                            "satisfied": _dep_ok,
                            "unsatisfied": _dep_unsat,
                        }
                    except Exception:
                        dependency_graph[_issue_str] = {"deps": _deps, "satisfied": None}
        except Exception:
            pass
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "node_types": {r[0]: r[1] for r in rows},
            "migration_done": migration_done,
            "db_size_kb": round(g.db_path.stat().st_size / 1024.0, 2) if g.db_path.exists() else 0.0,
            "dependency_graph": dependency_graph,
        }

    @router.get("/api/memory/health")
    async def api_memory_health() -> Dict[str, object]:
        """Epic #1073 — Memory system health-check endpoint.

        Returns the health status of each memory subsystem:
        - long_term: entries.json / index.json / summary.json on disk
        - memory_graph: SQLite node/edge counts
        - decision_memory: failure/decision event counts

        Returns HTTP 200 with {status: "healthy"|"degraded"|"unhealthy"}.
        """
        import time as _time
        checks: Dict[str, object] = {}
        overall = "healthy"

        # 1. Long-term memory file system check
        try:
            from igris.core.long_term_memory import LongTermMemory
            _ltm = LongTermMemory()
            _entries = _ltm.get_entries("__health_probe__", limit=1)
            checks["long_term"] = {"status": "ok", "reachable": True}
        except Exception as exc:
            _logger = logging.getLogger("igris.memory.health")
            _logger.warning("Memory health: long_term check failed: %s", exc)
            checks["long_term"] = {"status": "degraded", "error": str(exc)[:200]}
            overall = "degraded"

        # 2. Memory graph (SQLite)
        try:
            _g = _get_graph()
            _node_count = _g.conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
            _edge_count = _g.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
            checks["memory_graph"] = {"status": "ok", "nodes": _node_count, "edges": _edge_count}
        except Exception as exc:
            _logger = logging.getLogger("igris.memory.health")
            _logger.warning("Memory health: memory_graph check failed: %s", exc)
            checks["memory_graph"] = {"status": "degraded", "error": str(exc)[:200]}
            overall = "degraded"

        # 3. Decision memory (JSON store)
        try:
            _failures = decision_memory.get_recent_failures(1, project_root=str(CONFIG.project_root))
            checks["decision_memory"] = {"status": "ok", "reachable": True}
        except Exception as exc:
            _logger = logging.getLogger("igris.memory.health")
            _logger.warning("Memory health: decision_memory check failed: %s", exc)
            checks["decision_memory"] = {"status": "degraded", "error": str(exc)[:200]}
            overall = "degraded"

        return {
            "status": overall,
            "checks": checks,
            "timestamp": _time.time(),
        }

    @router.get("/api/memory/search")
    async def api_memory_search(q: str, node_type: Optional[str] = None, limit: int = 10) -> Dict[str, object]:
        results = _get_graph().query_by_intent(q, node_type=node_type, limit=limit)
        return {"results": results, "count": len(results)}

    @router.post("/api/memory/record")
    async def api_memory_record(request: Request) -> Dict[str, object]:
        body = await request.json()
        node_id = _get_graph().add_node(body["node_type"], body.get("content", {}), confidence=body.get("confidence", 1.0), tags=body.get("tags", []))
        return {"node_id": node_id}

    @router.post("/api/memory/learn-command")
    async def api_memory_learn_command(request: Request) -> Dict[str, object]:
        body = await request.json()
        node_id = _get_graph().add_node("command_recipe", {"intent": body.get("intent", ""), "command": body.get("command", ""), "risk": body.get("risk", "low")}, success_rate=1.0 if body.get("success", True) else 0.0)
        return {"node_id": node_id}

    @router.post("/api/memory/export-safe")
    async def api_memory_export_safe() -> StreamingResponse:
        payload = json.dumps({"nodes": _get_graph().export_safe()}, indent=2).encode("utf-8")
        return StreamingResponse(iter([payload]), media_type="application/json", headers={"Content-Disposition": "attachment; filename=memory_export_safe.json"})

    @router.post("/api/memory/import-safe")
    async def api_memory_import_safe(request: Request) -> Dict[str, object]:
        body = await request.json()
        return _get_graph().import_safe(body.get("nodes", []))

    @router.post("/api/memory/events")
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

    @router.post("/api/loop/step")
    async def api_loop_step() -> Dict[str, object]:
        result = autonomous_loop.execute_step(
            task_engine, project_root=str(CONFIG.project_root),
        )
        return result.to_dict()

    @router.post("/api/loop/run")
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


    return router
