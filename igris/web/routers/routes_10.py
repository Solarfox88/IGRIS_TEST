"""IGRIS web server router — auto-split from server.py (#725).

Route handlers are extracted from _create_app_impl; shared app state is
received via ``deps`` (SimpleNamespace). Do not edit route logic here;
changes should first be made in the original handler before full migration.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def _safe_redact(value: object) -> str:
    """Redact secrets from a string value (local helper for Epic #1077 endpoints)."""
    from igris.core.safety import redact_secrets
    return redact_secrets(str(value) if value is not None else "")


def create_router(deps) -> APIRouter:
    """Router module 10/10 — _create_app_impl chunk 10."""
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

    @router.post("/api/reasoning/step")
    async def api_reasoning_step(request: Request) -> Dict[str, object]:
        """Execute a single reasoning loop step (for testing/debugging)."""
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        content = await request.json()
        loop = AgentReasoningLoop(
            project_root=str(CONFIG.project_root),
            role=content.get("role", "coder"),
            max_steps=1,
        )
        result = loop.run(
            goal=content.get("goal", ""),
            mission_id=content.get("mission_id", ""),
        )
        return result.to_dict()

    @router.get("/api/reasoning/stop-reasons")
    async def api_reasoning_stop_reasons() -> Dict[str, object]:
        """List all possible loop stop reasons."""
        from igris.core.agent_reasoning_loop import STOP_REASONS
        return {"stop_reasons": list(STOP_REASONS)}

    # ------------------------------------------------------------------
    # Rank Self-Repair Supervisor
    # ------------------------------------------------------------------

    @router.post("/api/rank/run-supervised")
    async def api_rank_run_supervised(request: Request) -> Dict[str, object]:
        """Run a controlled rank mission through the self-repair supervisor."""
        from igris.core.self_repair_supervisor import start_supervised_rank_async
        content = await request.json()
        if not content.get("goal"):
            raise HTTPException(status_code=400, detail="goal required")
        run = start_supervised_rank_async(content, project_root=str(CONFIG.project_root))
        return run.to_dict()

    @router.get("/api/rank/runs/active")
    async def api_rank_runs_active() -> Dict[str, object]:
        """List active supervised rank runs with compact summaries."""
        from igris.core.self_repair_supervisor import list_active_supervised_run_summaries
        runs = list_active_supervised_run_summaries(project_root=str(CONFIG.project_root))
        return {"runs": runs}

    @router.get("/api/rank/runs/{run_id}/summary")
    async def api_rank_run_summary(run_id: str) -> Dict[str, object]:
        """Return compact supervised rank run summary."""
        from igris.core.self_repair_supervisor import get_supervised_run, summarize_supervised_run
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")
        return summarize_supervised_run(run)

    @router.get("/api/rank/runs")
    async def api_rank_runs() -> Dict[str, object]:
        """List supervised rank runs held in memory."""
        from igris.core.self_repair_supervisor import list_supervised_runs
        return {"runs": [run.to_dict() for run in list_supervised_runs()]}

    @router.get("/api/rank/runs/{run_id}")
    async def api_rank_run_detail(run_id: str) -> Dict[str, object]:
        """Return one supervised rank run.

        Falls back to the on-disk supervisor_runs.json archive when the run is
        no longer in the in-memory RUN_STORE (e.g. after a service restart).
        This prevents zombie poll loops where a caller keeps hitting 404 for a
        run that completed before the last restart.
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        run = get_supervised_run(run_id)
        if run is not None:
            return run.to_dict()
        # Fallback: check on-disk archive so callers receive a terminal status
        # (blocked/completed) instead of 404, which stops poll loops.
        try:
            _runs_path = Path(CONFIG.project_root) / ".igris" / "supervisor_runs.json"
            if _runs_path.exists():
                _payload = json.loads(_runs_path.read_text(encoding="utf-8"))
                _record = (_payload.get("runs") or {}).get(run_id)
                if _record and isinstance(_record, dict):
                    # Return the archived snapshot with an explicit archived flag
                    return {**_record, "archived": True, "run_id": run_id}
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="rank run not found")

    @router.post("/api/rank/runs/{run_id}/cancel")
    async def api_rank_run_cancel(run_id: str, request: Request) -> Dict[str, object]:
        """Cancel one supervised rank run safely."""
        from igris.core.self_repair_supervisor import cancel_supervised_run
        # Issue #723 — guard against empty body or malformed JSON (confirmed 500 in prod logs)
        try:
            raw = await request.body()
            content = json.loads(raw) if raw else {}
            if not isinstance(content, dict):
                content = {}
        except (json.JSONDecodeError, ValueError):
            content = {}
        reason = str(content.get("reason", "Cancelled by user"))
        run = cancel_supervised_run(run_id, project_root=str(CONFIG.project_root), reason=reason)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")
        return run.to_dict()

    @router.get("/api/rank/audit/summary")
    async def api_rank_audit_summary() -> Dict[str, object]:
        """Return compact supervisor audit summary."""
        from igris.core.self_repair_supervisor import get_supervisor_audit_summary
        return get_supervisor_audit_summary(project_root=str(CONFIG.project_root))

    # ------------------------------------------------------------------
    # Epic #1077 — Control Room UX: run status, risk cards, approve/block
    # ------------------------------------------------------------------

    @router.get("/api/rank/runs/{run_id}/status")
    async def api_rank_run_status(run_id: str) -> Dict[str, object]:
        """Rich run status API for the Control Room UI (Epic #1077).

        Returns:
            run_id, status, phase, failure_class, repair_cycles_used,
            recent_events (last 10), risk_card, elapsed_seconds
        """
        import time as _time
        from igris.core.self_repair_supervisor import get_supervised_run
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")

        # Build a risk card from the run's current state
        risk_card = {
            "failure_class": run.failure_class or "none",
            "repair_cycles_used": run.repair_cycles_used,
            "same_failure_count": getattr(run, "same_failure_count", 0),
            "execution_budget_used_usd": getattr(run, "execution_budget_used_usd", 0.0),
            "capability_signals": dict(getattr(run, "capability_signals", {})),
            "decomposition_pending": run.decomposition is not None,
            "cancel_requested": run.cancel_requested,
        }

        # Determine current phase from latest event
        current_phase = "unknown"
        recent_events = []
        if run.events:
            current_phase = run.events[-1].phase if hasattr(run.events[-1], "phase") else "unknown"
            recent_events = [
                {
                    "phase": e.phase if hasattr(e, "phase") else str(e.get("phase", "")),
                    "status": e.status if hasattr(e, "status") else str(e.get("status", "")),
                    "detail": (e.detail if hasattr(e, "detail") else str(e.get("detail", "")))[:200],
                    "ts": e.ts if hasattr(e, "ts") else e.get("ts", 0),
                }
                for e in run.events[-10:]
            ]

        # Elapsed time
        start_ts = getattr(run, "start_ts", None) or (
            run.events[0].ts if run.events and hasattr(run.events[0], "ts") else 0
        )
        elapsed = round(_time.time() - start_ts, 1) if start_ts else None

        return {
            "run_id": run.run_id,
            "rank_id": run.rank_id,
            "status": run.status,
            "phase": current_phase,
            "failure_class": run.failure_class,
            "goal": _safe_redact(run.goal) if run.goal else "",
            "risk_card": risk_card,
            "recent_events": recent_events,
            "elapsed_seconds": elapsed,
        }

    @router.post("/api/rank/runs/{run_id}/approve")
    async def api_rank_run_approve(run_id: str) -> Dict[str, object]:
        """Approve a blocked/pending run to continue (Epic #1077).

        Clears cancel_requested flag and updates status to 'running'.
        For runs blocked on human approval.
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")
        if run.status not in ("blocked", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"Run {run_id} is in status {run.status!r}; can only approve blocked/running runs",
            )
        run.cancel_requested = False
        run.cancel_reason = ""
        run.add("control_room", "approved", "Run approved via Control Room API")
        return {"run_id": run_id, "status": run.status, "approved": True}

    @router.post("/api/rank/runs/{run_id}/block")
    async def api_rank_run_block(run_id: str, request: Request) -> Dict[str, object]:
        """Block a running run immediately (Epic #1077).

        Equivalent to cancel but marks the run as 'blocked' rather than 'cancelled'.
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        try:
            raw = await request.body()
            content = json.loads(raw) if raw else {}
            if not isinstance(content, dict):
                content = {}
        except (json.JSONDecodeError, ValueError):
            content = {}
        reason = str(content.get("reason", "Blocked via Control Room API"))
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")
        run.cancel_requested = True
        run.cancel_reason = reason
        run.add("control_room", "blocked", f"Run blocked via Control Room: {reason}")
        return {"run_id": run_id, "status": "blocking", "reason": reason}

    # ------------------------------------------------------------------
    # Epic #1076 — DevOps/VPS operator: health check, deploy status, diagnostics
    # ------------------------------------------------------------------

    @router.get("/api/devops/health")
    async def api_devops_health() -> Dict[str, object]:
        """VPS health check — Epic #1076.

        Returns CPU/memory/disk usage and service status.
        Non-crashing: if a check fails, the error is recorded in the report.
        """
        import subprocess as _sp
        import time as _time
        checks: Dict[str, object] = {}

        # Disk usage
        try:
            _du = _sp.run(["df", "-h", "/"], capture_output=True, text=True, timeout=5)
            lines = _du.stdout.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[1].split()
                checks["disk"] = {
                    "status": "ok",
                    "size": parts[1] if len(parts) > 1 else "?",
                    "used": parts[2] if len(parts) > 2 else "?",
                    "available": parts[3] if len(parts) > 3 else "?",
                    "use_pct": parts[4] if len(parts) > 4 else "?",
                }
        except Exception as exc:
            checks["disk"] = {"status": "error", "error": str(exc)[:200]}

        # Memory usage
        try:
            _mem = _sp.run(["free", "-h"], capture_output=True, text=True, timeout=5)
            mem_lines = _mem.stdout.strip().splitlines()
            if len(mem_lines) >= 2:
                mem_parts = mem_lines[1].split()
                checks["memory"] = {
                    "status": "ok",
                    "total": mem_parts[1] if len(mem_parts) > 1 else "?",
                    "used": mem_parts[2] if len(mem_parts) > 2 else "?",
                    "free": mem_parts[3] if len(mem_parts) > 3 else "?",
                }
        except Exception as exc:
            checks["memory"] = {"status": "error", "error": str(exc)[:200]}

        # IGRIS service (port 7778)
        try:
            _nc = _sp.run(
                ["nc", "-z", "-w", "2", "localhost", "7778"],
                capture_output=True, timeout=5,
            )
            checks["igris_service"] = {
                "status": "ok" if _nc.returncode == 0 else "down",
                "port": 7778,
            }
        except Exception as exc:
            checks["igris_service"] = {"status": "error", "error": str(exc)[:200]}

        overall = "healthy" if all(
            c.get("status") == "ok" for c in checks.values() if isinstance(c, dict)
        ) else "degraded"

        return {"status": overall, "checks": checks, "timestamp": _time.time()}

    @router.get("/api/devops/deploy-status")
    async def api_devops_deploy_status() -> Dict[str, object]:
        """Deploy status check — Epic #1076.

        Returns the most recent git log, current branch, and dirty status.
        """
        import subprocess as _sp
        result: Dict[str, object] = {}
        try:
            _branch = _sp.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(CONFIG.project_root),
            )
            result["branch"] = _branch.stdout.strip() if _branch.returncode == 0 else "unknown"

            _log_out = _sp.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True, text=True, timeout=5,
                cwd=str(CONFIG.project_root),
            )
            result["recent_commits"] = _log_out.stdout.strip().splitlines() if _log_out.returncode == 0 else []

            _status = _sp.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
                cwd=str(CONFIG.project_root),
            )
            result["is_dirty"] = bool(_status.stdout.strip()) if _status.returncode == 0 else None
        except Exception as exc:
            result["error"] = str(exc)[:200]

        return result

    @router.get("/api/devops/diagnostics")
    async def api_devops_diagnostics() -> Dict[str, object]:
        """Nginx/systemd diagnostic — Epic #1076.

        Returns systemd service status for igris and nginx (if available).
        Best-effort: missing services return status=unknown.
        """
        import subprocess as _sp
        services: Dict[str, object] = {}
        for svc in ("igris", "nginx"):
            try:
                _svc = _sp.run(
                    ["systemctl", "is-active", svc],
                    capture_output=True, text=True, timeout=5,
                )
                status = _svc.stdout.strip() or "unknown"
                services[svc] = {"status": status}
            except Exception as exc:
                services[svc] = {"status": "unknown", "error": str(exc)[:100]}

        return {"services": services}

    # ------------------------------------------------------------------
    # Integration Layer — Epic #62
    # ------------------------------------------------------------------

    @router.post("/api/integration/run-mission")
    async def api_integration_run_mission(request: Request) -> Dict[str, object]:
        """Run a full governed mission through the integration pipeline."""
        from igris.core.integration_layer import IntegrationLayer
        content = await request.json()
        layer = IntegrationLayer(
            project_root=str(CONFIG.project_root),
            max_steps=content.get("max_steps", 50),
            role=content.get("role", "coder"),
        )
        report = layer.run_mission(
            goal=content.get("goal", ""),
            title=content.get("title", ""),
            description=content.get("description", ""),
            constraints=content.get("constraints"),
            success_criteria=content.get("success_criteria"),
        )
        return report.to_dict()

    @router.get("/api/integration/pipeline-status")
    async def api_integration_pipeline_status() -> Dict[str, object]:
        """Get status of all pipeline components."""
        from igris.core.integration_layer import IntegrationLayer
        layer = IntegrationLayer(project_root=str(CONFIG.project_root))
        return layer.get_pipeline_status()

    @router.get("/api/integration/action-families")
    async def api_integration_action_families() -> Dict[str, object]:
        """Get action type to family mapping."""
        from igris.core.integration_layer import IntegrationLayer
        return {"families": {
            "code_nav": ["search_code", "find_files", "list_directory",
                        "read_file_range", "repo_map", "find_symbol"],
            "code_edit": ["write_file", "propose_patch", "apply_patch"],
            "test": ["run_tests"],
            "git": ["git_status", "git_diff"],
            "shell": ["shell_template", "raw_shell_proposal"],
            "http": ["http_check"],
            "planning": ["update_plan"],
            "memory": ["record_memory"],
            "human": ["ask_user"],
            "terminal": ["finish", "blocked"],
        }}

    # ------------------------------------------------------------------
    # Command Risk Engine v2 — Epic #63
    # ------------------------------------------------------------------

    @router.post("/api/risk/evaluate")
    async def api_risk_evaluate(request: Request) -> Dict[str, object]:
        """Evaluate a raw shell command through the risk engine."""
        from igris.core.command_risk_engine import CommandRiskEngine
        content = await request.json()
        engine = CommandRiskEngine(
            project_root=str(CONFIG.project_root),
            use_llm_reviewer=content.get("use_llm_reviewer", True),
        )
        event, review = engine.evaluate_command(
            command=content.get("command", ""),
            context=content.get("context", ""),
            mission_id=content.get("mission_id", ""),
        )
        return {"event": event.to_dict(), "review": review.to_dict()}

    @router.post("/api/risk/evaluate-template")
    async def api_risk_evaluate_template(request: Request) -> Dict[str, object]:
        """Evaluate a parametrized shell template."""
        from igris.core.command_risk_engine import CommandRiskEngine
        content = await request.json()
        engine = CommandRiskEngine(
            project_root=str(CONFIG.project_root),
            use_llm_reviewer=content.get("use_llm_reviewer", True),
        )
        event, review = engine.evaluate_template(
            template_id=content.get("template_id", ""),
            parameters=content.get("parameters", {}),
            mission_id=content.get("mission_id", ""),
        )
        return {"event": event.to_dict(), "review": review.to_dict()}

    @router.post("/api/risk/parse")
    async def api_risk_parse(request: Request) -> Dict[str, object]:
        """Parse a shell command into its components."""
        from igris.core.command_risk_engine import parse_command
        content = await request.json()
        parsed = parse_command(content.get("command", ""))
        return parsed.to_dict()

    @router.get("/api/risk/levels")
    async def api_risk_levels() -> Dict[str, object]:
        """Get all risk levels."""
        from igris.core.command_risk_engine import RISK_LEVELS
        return {"risk_levels": list(RISK_LEVELS)}

    # ------------------------------------------------------------------
    # Benchmark /api/ping — Epic #64
    # ------------------------------------------------------------------

    @router.get("/api/ping")
    async def api_ping() -> Dict[str, object]:
        """Simple ping endpoint — benchmark target."""
        return {"pong": True}

    @router.post("/api/benchmark/run")
    async def api_benchmark_run(request: Request) -> Dict[str, object]:
        """Run the /api/ping operational benchmark."""
        from igris.core.benchmark_ping import BenchmarkRunner
        content = await request.json()
        runner = BenchmarkRunner(project_root=str(CONFIG.project_root))
        mode = content.get("mode", "deterministic")
        if mode == "integration":
            result = runner.run_integration(
                max_steps=content.get("max_steps", 10),
            )
        else:
            result = runner.run_deterministic()
        return result.to_dict()

    @router.get("/api/benchmark/phases")
    async def api_benchmark_phases() -> Dict[str, object]:
        """List benchmark phases."""
        from igris.core.benchmark_ping import BENCHMARK_PHASES, BENCHMARK_GOAL
        return {"phases": BENCHMARK_PHASES, "goal": BENCHMARK_GOAL}

    # Issue #729 — storage stats endpoint
    @router.get("/api/storage/stats")
    async def api_storage_stats() -> Dict[str, object]:
        """Return size and rotation stats for .igris/ JSON files."""
        from igris.core.file_rotation import get_file_stats
        igris_dir = Path(CONFIG.project_root) / ".igris"
        return {"files": get_file_stats(igris_dir)}

    return router
