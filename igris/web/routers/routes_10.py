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
    # Epic #1077 — Evidence card: diff summary + test results for a run
    # ------------------------------------------------------------------

    @router.get("/api/rank/runs/{run_id}/evidence")
    async def api_rank_run_evidence(run_id: str) -> Dict[str, object]:
        """Return evidence card for a run (Epic #1077).

        Provides:
        - diff_summary: files changed, lines added/removed (from git diff)
        - test_results: latest test phase outcome (pass/fail counts, failed tests)
        - cost_breakdown: budget used per phase from run events
        - key_events: phase → status snapshot for the full run

        Useful for the Control Room "understand what changed" view.
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        import subprocess as _sp
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")

        # 1. Diff summary from git
        diff_summary: Dict[str, object] = {"available": False}
        try:
            diff_stat = _sp.run(
                ["git", "diff", "--stat", "HEAD"],
                cwd=str(CONFIG.project_root),
                capture_output=True, text=True, timeout=10,
            )
            if diff_stat.returncode == 0 and diff_stat.stdout.strip():
                lines = diff_stat.stdout.strip().splitlines()
                summary_line = lines[-1] if lines else ""
                diff_summary = {
                    "available": True,
                    "files_changed": [l.split("|")[0].strip() for l in lines[:-1]],
                    "summary": summary_line,
                }
            else:
                diff_summary = {"available": True, "files_changed": [], "summary": "no changes"}
        except Exception as exc:
            diff_summary = {"available": False, "error": str(exc)[:200]}

        # 2. Test results from latest targeted_tests / full_tests event
        test_results: Dict[str, object] = {"available": False}
        for evt in reversed(run.events):
            phase = evt.phase if hasattr(evt, "phase") else evt.get("phase", "")
            if phase in ("targeted_tests", "full_tests", "run_tests"):
                detail = evt.detail if hasattr(evt, "detail") else str(evt.get("detail", ""))
                status = evt.status if hasattr(evt, "status") else str(evt.get("status", ""))
                test_results = {
                    "available": True,
                    "phase": phase,
                    "status": status,
                    "detail": detail[:500],
                }
                break

        # 3. Cost breakdown per phase
        cost_breakdown: Dict[str, float] = {}
        total_cost = 0.0
        for evt in run.events:
            data = evt.data if hasattr(evt, "data") else {}
            phase = evt.phase if hasattr(evt, "phase") else str(evt.get("phase", ""))
            cost = float(data.get("estimated_cost", 0) or 0)
            if cost > 0:
                cost_breakdown[phase] = cost_breakdown.get(phase, 0.0) + cost
                total_cost += cost

        # 4. Key events snapshot
        key_events = [
            {
                "phase": e.phase if hasattr(e, "phase") else str(e.get("phase", "")),
                "status": e.status if hasattr(e, "status") else str(e.get("status", "")),
            }
            for e in run.events
        ]

        return {
            "run_id": run_id,
            "diff_summary": diff_summary,
            "test_results": test_results,
            "cost_breakdown": cost_breakdown,
            "total_cost_usd": round(total_cost, 6),
            "key_events": key_events,
        }

    # ------------------------------------------------------------------
    # Epic #1077 — Control Room: timeline, risk-detail, final report
    # ------------------------------------------------------------------

    @router.get("/api/rank/runs/{run_id}/timeline")
    async def api_rank_run_timeline(run_id: str) -> Dict[str, object]:
        """Full event timeline for a run, grouped by phase (Epic #1077).

        Returns:
            timeline: list of phase groups, each with phase name, events, and
                      aggregate status (success/failure/running/skipped).
            total_events: total number of events
            phases_seen: list of unique phases in order
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        import time as _time
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")

        # Group events by phase
        phase_groups: Dict[str, Dict[str, object]] = {}
        phases_order: list = []
        for ev in (run.events or []):
            phase = getattr(ev, "phase", None) or (ev.get("phase", "unknown") if isinstance(ev, dict) else "unknown")
            status = getattr(ev, "status", None) or (ev.get("status", "") if isinstance(ev, dict) else "")
            detail = getattr(ev, "detail", None) or (ev.get("detail", "") if isinstance(ev, dict) else "")
            ts = getattr(ev, "ts", None) or (ev.get("ts", 0) if isinstance(ev, dict) else 0)
            if phase not in phase_groups:
                phase_groups[phase] = {"phase": phase, "events": [], "final_status": status, "started_ts": ts}
                phases_order.append(phase)
            entry = {"status": status, "detail": str(detail)[:300], "ts": ts}
            phase_groups[phase]["events"].append(entry)  # type: ignore[index]
            # Update final status to the last seen status for this phase
            if status:
                phase_groups[phase]["final_status"] = status

        timeline = [phase_groups[p] for p in phases_order]

        # Compute duration per phase
        for group in timeline:
            evs = group["events"]
            if len(evs) >= 2:
                first_ts = evs[0].get("ts") or 0
                last_ts = evs[-1].get("ts") or 0
                group["duration_seconds"] = round(float(last_ts) - float(first_ts), 1) if last_ts and first_ts else None

        return {
            "run_id": run_id,
            "status": run.status,
            "timeline": timeline,
            "total_events": len(run.events or []),
            "phases_seen": phases_order,
        }

    @router.get("/api/rank/runs/{run_id}/risk-detail")
    async def api_rank_run_risk_detail(run_id: str) -> Dict[str, object]:
        """Full risk detail card for a run (Epic #1077).

        Returns:
            failure_history: list of all failure events in order
            repair_attempts: list of repair cycle events
            capability_signals: dict of capability limit signals
            risk_trajectory: simplified risk level per phase
            budget_used_usd: total execution budget consumed
            decomposition_info: decomposition quality and wave structure (if applicable)
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")

        # Extract failure history
        failure_history = []
        repair_attempts = []
        risk_trajectory = []
        for ev in (run.events or []):
            phase = getattr(ev, "phase", None) or (ev.get("phase", "") if isinstance(ev, dict) else "")
            status = getattr(ev, "status", None) or (ev.get("status", "") if isinstance(ev, dict) else "")
            detail = getattr(ev, "detail", None) or (ev.get("detail", "") if isinstance(ev, dict) else "")
            ts = getattr(ev, "ts", 0) or (ev.get("ts", 0) if isinstance(ev, dict) else 0)
            if phase == "failure":
                failure_history.append({"failure_class": str(detail)[:100], "ts": ts})
            if phase == "repair_reasoning":
                repair_attempts.append({
                    "status": status,
                    "detail": str(detail)[:200],
                    "ts": ts,
                    "same_failure_count": ev.same_failure_count if hasattr(ev, "same_failure_count") else None,
                })
            # Risk trajectory: phase → success/failure
            if status in ("success", "failure", "blocked", "running"):
                risk_trajectory.append({"phase": phase, "status": status, "ts": ts})

        return {
            "run_id": run_id,
            "failure_class": run.failure_class or "none",
            "repair_cycles_used": run.repair_cycles_used,
            "same_failure_count": getattr(run, "same_failure_count", 0),
            "capability_signals": dict(getattr(run, "capability_signals", {})),
            "failure_history": failure_history,
            "repair_attempts": repair_attempts,
            "risk_trajectory": risk_trajectory[-20:],  # last 20 phase transitions
            "budget_used_usd": round(getattr(run, "execution_budget_used_usd", 0.0), 6),
            "decomposition_info": {
                "pending": run.decomposition is not None,
                "quality_score": (run.decomposition or {}).get("_quality_score"),
                "quality_valid": (run.decomposition or {}).get("_quality_valid"),
            } if run.decomposition else {"pending": False},
        }

    @router.get("/api/rank/runs/{run_id}/report")
    async def api_rank_run_report(run_id: str) -> Dict[str, object]:
        """Structured final report for a completed or blocked run (Epic #1077).

        Returns:
            A human-readable + machine-parsable summary of the run outcome:
            - outcome: 'success' | 'blocked' | 'decomposition_required' | 'cancelled' | 'in_progress'
            - goal: the original mission goal
            - failure_class: for blocked runs, the failure classification
            - acceptance_evidence: semantic gate result (if run)
            - decomposition: decomposition info (if decomposition_required)
            - key_metrics: repair_cycles, budget, elapsed, event_count
            - recommendations: list of actionable next steps
        """
        from igris.core.self_repair_supervisor import get_supervised_run
        import time as _time
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")

        # Determine outcome
        status = run.status or "unknown"
        if status in ("success", "completed", "noop"):
            outcome = "success"
        elif status in ("cancelled", "cancelling"):
            outcome = "cancelled"
        elif status == "running":
            outcome = "in_progress"
        elif run.failure_class == "decomposition_required":
            outcome = "decomposition_required"
        else:
            outcome = "blocked"

        # Recommendations based on outcome
        recommendations: list = []
        if outcome == "decomposition_required":
            recommendations.append("Review the decomposition and approve sub-missions via /api/rank/runs/{run_id}/approve")
            recommendations.append("Check sub-mission dependency order via /api/rank/runs/{run_id}/timeline")
        elif outcome == "blocked":
            fc = run.failure_class or "unknown"
            if fc in ("pytest_failure", "missing_tests"):
                recommendations.append(f"Repair cycle failed on {fc}; review repair attempts via /api/rank/runs/{run_id}/risk-detail")
            elif fc == "capability_ceiling_reached":
                recommendations.append("Mission exceeds model capability; consider manual decomposition into smaller tasks")
            elif fc == "workspace_dirty":
                recommendations.append("Clean the workspace (git reset --hard or git stash) and retry")
        elif outcome == "success":
            recommendations.append("Review evidence via /api/rank/runs/{run_id}/evidence before closing the GitHub issue")

        # Elapsed
        start_ts = getattr(run, "start_ts", None)
        if not start_ts and run.events:
            start_ts = getattr(run.events[0], "ts", None)
        elapsed = round(_time.time() - float(start_ts), 1) if start_ts else None

        return {
            "run_id": run_id,
            "outcome": outcome,
            "status": status,
            "goal": _safe_redact(run.goal) if run.goal else "",
            "failure_class": run.failure_class or "none",
            "acceptance_evidence": getattr(run, "acceptance_evidence", None),
            "decomposition": {
                "pending": True,
                "quality_score": (run.decomposition or {}).get("_quality_score"),
                "wave_count": None,
            } if run.decomposition else None,
            "key_metrics": {
                "repair_cycles_used": run.repair_cycles_used,
                "budget_used_usd": round(getattr(run, "execution_budget_used_usd", 0.0), 6),
                "elapsed_seconds": elapsed,
                "event_count": len(run.events or []),
                "same_failure_count": getattr(run, "same_failure_count", 0),
            },
            "recommendations": recommendations,
        }

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
        """Full VPS/server diagnostic — Epic #1076.

        Returns:
        - systemd service status for igris, nginx, docker (if available)
        - nginx config test result (nginx -t)
        - docker containers list (if docker is installed)
        - SSL certificate expiry for configured domain
        - open ports (ss -tlnp summary)

        Best-effort: each check records its own error if it fails.
        """
        import subprocess as _sp
        import re as _re
        import os as _os

        report: Dict[str, object] = {}

        # 1. Systemd services
        services: Dict[str, object] = {}
        for svc in ("igris", "nginx", "docker"):
            try:
                _r = _sp.run(
                    ["systemctl", "is-active", svc],
                    capture_output=True, text=True, timeout=5,
                )
                status = _r.stdout.strip() or "unknown"
                # Also grab the loaded/active/sub state for more detail
                _status_r = _sp.run(
                    ["systemctl", "show", svc, "--property=ActiveState,SubState,LoadState"],
                    capture_output=True, text=True, timeout=5,
                )
                props: Dict[str, str] = {}
                for line in _status_r.stdout.splitlines():
                    if "=" in line:
                        k, _, v = line.partition("=")
                        props[k.strip()] = v.strip()
                services[svc] = {"status": status, **props}
            except Exception as exc:
                services[svc] = {"status": "unknown", "error": str(exc)[:100]}
        report["services"] = services

        # 2. Nginx config test
        nginx_config: Dict[str, object] = {"available": False}
        try:
            _ng = _sp.run(
                ["nginx", "-t"],
                capture_output=True, text=True, timeout=10,
            )
            nginx_config = {
                "available": True,
                "ok": _ng.returncode == 0,
                "output": (_ng.stdout + _ng.stderr).strip()[:500],
            }
        except FileNotFoundError:
            nginx_config = {"available": False, "reason": "nginx not installed"}
        except Exception as exc:
            nginx_config = {"available": False, "error": str(exc)[:100]}
        report["nginx_config"] = nginx_config

        # 3. Docker containers
        docker_containers: Dict[str, object] = {"available": False}
        try:
            _dk = _sp.run(
                ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True, text=True, timeout=10,
            )
            if _dk.returncode == 0:
                containers = []
                for line in _dk.stdout.strip().splitlines():
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        containers.append({"name": parts[0], "status": parts[1]})
                docker_containers = {"available": True, "containers": containers}
            else:
                docker_containers = {"available": False, "error": _dk.stderr.strip()[:200]}
        except FileNotFoundError:
            docker_containers = {"available": False, "reason": "docker not installed"}
        except Exception as exc:
            docker_containers = {"available": False, "error": str(exc)[:100]}
        report["docker"] = docker_containers

        # 4. SSL certificate expiry (uses openssl if domain is configured)
        ssl_info: Dict[str, object] = {"available": False}
        _domain = _os.environ.get("IGRIS_VPS_DOMAIN", "")
        if _domain:
            try:
                _ssl = _sp.run(
                    ["openssl", "s_client", "-connect", f"{_domain}:443",
                     "-servername", _domain, "-showcerts"],
                    input="", capture_output=True, text=True, timeout=10,
                )
                _cert = _sp.run(
                    ["openssl", "x509", "-noout", "-dates"],
                    input=_ssl.stdout, capture_output=True, text=True, timeout=5,
                )
                _exp = {}
                for line in _cert.stdout.splitlines():
                    if "notAfter" in line:
                        _exp["expires"] = line.split("=", 1)[1].strip()
                    if "notBefore" in line:
                        _exp["valid_from"] = line.split("=", 1)[1].strip()
                ssl_info = {"available": True, "domain": _domain, **_exp}
            except Exception as exc:
                ssl_info = {"available": False, "domain": _domain, "error": str(exc)[:100]}
        report["ssl"] = ssl_info

        # 5. Open listening ports (ss -tlnp)
        ports: Dict[str, object] = {"available": False}
        try:
            _ss = _sp.run(
                ["ss", "-tlnp"],
                capture_output=True, text=True, timeout=5,
            )
            if _ss.returncode == 0:
                # Parse port numbers from Local Address:Port column
                _ports_found = list(set(_re.findall(r":(\d+)\s", _ss.stdout)))
                ports = {"available": True, "listening_ports": sorted(int(p) for p in _ports_found if p.isdigit())}
            else:
                ports = {"available": False, "error": _ss.stderr.strip()[:100]}
        except Exception as exc:
            ports = {"available": False, "error": str(exc)[:100]}
        report["ports"] = ports

        return report

    # ------------------------------------------------------------------
    # Epic #1076 extended — Host registry, policy, deploy, smoke test
    # ------------------------------------------------------------------

    @router.get("/api/devops/hosts")
    async def api_devops_hosts_list() -> Dict[str, object]:
        """List all registered deployment hosts — Epic #1076."""
        from igris.core.devops_manager import DevOpsManager
        mgr = DevOpsManager(str(CONFIG.project_root))
        return {"hosts": mgr.list_hosts()}

    @router.post("/api/devops/hosts")
    async def api_devops_hosts_register(request: Request) -> Dict[str, object]:
        """Register (or update) a deployment host — Epic #1076.

        Body: { hostname, alias?, policy?, allowed_paths?, allowed_services?,
                requires_backup?, health_url? }
        """
        from igris.core.devops_manager import DevOpsManager, HostConfig
        data = await request.json()
        hostname = data.get("hostname", "").strip()
        if not hostname:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="hostname is required")
        config = HostConfig.from_dict(data)
        mgr = DevOpsManager(str(CONFIG.project_root))
        return mgr.register_host(config)

    @router.delete("/api/devops/hosts/{hostname}")
    async def api_devops_hosts_remove(hostname: str) -> Dict[str, object]:
        """Remove a host from the registry — Epic #1076."""
        from igris.core.devops_manager import DevOpsManager
        mgr = DevOpsManager(str(CONFIG.project_root))
        result = mgr.remove_host(hostname)
        if not result.get("removed"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=result.get("error", "host not found"))
        return result

    @router.get("/api/devops/hosts/{hostname}/policy")
    async def api_devops_host_policy(hostname: str, action: str = "deploy") -> Dict[str, object]:
        """Check whether *action* is permitted on *hostname* — Epic #1076."""
        from igris.core.devops_manager import DevOpsManager
        mgr = DevOpsManager(str(CONFIG.project_root))
        return mgr.check_policy(hostname, action)

    @router.post("/api/devops/preflight")
    async def api_devops_preflight(request: Request) -> Dict[str, object]:
        """Run pre-deploy preflight checks — Epic #1076.

        Body: { hostname? (for labelling), min_disk_pct_free? (default 10) }
        """
        from igris.core.devops_manager import DevOpsManager
        data = await request.json()
        mgr = DevOpsManager(str(CONFIG.project_root))
        return mgr.run_preflight(
            hostname=data.get("hostname"),
            min_disk_pct_free=int(data.get("min_disk_pct_free", 10)),
        )

    @router.post("/api/devops/deploy")
    async def api_devops_deploy(request: Request) -> Dict[str, object]:
        """Full deploy cycle: preflight → action → postcheck — Epic #1076.

        Body: { strategy? (default git_pull_restart), hostname?, health_url?,
                dry_run? (default false) }
        """
        from igris.core.devops_manager import DevOpsManager
        data = await request.json()
        mgr = DevOpsManager(str(CONFIG.project_root))
        return mgr.run_deploy(
            strategy=data.get("strategy", "git_pull_restart"),
            hostname=data.get("hostname"),
            health_url=data.get("health_url", ""),
            dry_run=bool(data.get("dry_run", False)),
        )

    @router.get("/api/devops/smoke")
    async def api_devops_smoke(url: str = "") -> Dict[str, object]:
        """HTTP smoke test — GET a URL and return evidence — Epic #1076.

        Defaults to http://localhost:7778/api/ping if no url given.
        """
        from igris.core.devops_manager import DevOpsManager
        mgr = DevOpsManager(str(CONFIG.project_root))
        return mgr.run_smoke_test(url=url)

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
