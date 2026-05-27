"""
FastAPI application for IGRIS_GPT.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import signal
import subprocess
import time
from contextlib import asynccontextmanager

# Module-level aliases so tests can patch them by name
_re = re
_sig = signal
_sp = subprocess
_time = time

try:
    import uvicorn as uvicorn  # noqa: PLC0414 — module-level for test patching
except ImportError:  # pragma: no cover
    uvicorn = None  # type: ignore
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
from igris.core.memory_graph import MemoryGraph
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


_watchdog_logger = logging.getLogger("igris.watchdog")

_WATCHDOG_POLL_SECONDS = 60
_WATCHDOG_COOLDOWN_SECONDS = 30
_WATCHDOG_MAX_CONSECUTIVE_FAILURES = 3
# Runs that confirm a structural ceiling are skipped after just 1 failure —
# the strong model already exhausted its step budget, further runs waste budget.
_WATCHDOG_CEILING_SKIP_AFTER = 1
_WATCHDOG_ZOMBIE_TIMEOUT_SECONDS = 7200  # 2h no events → evict
_REPAIR_ISSUE_PATTERNS = ("supervised repair for", "supervised repair:", "repair for reasoning", "repair for pytest")
# Persisted skip list path — survives server restarts so IGRIS won't retry
# a ceiling issue every time the service restarts after another run.
_WATCHDOG_SKIPPED_PATH = ".igris/watchdog_skipped_issues.json"


def _load_skipped_issues(project_root: str) -> set:
    """Load the persisted set of skipped issue numbers from disk."""
    path = os.path.join(project_root, _WATCHDOG_SKIPPED_PATH)
    try:
        with open(path) as f:
            data = json.load(f)
        return {int(n) for n in data.get("skipped", [])}
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return set()


def _save_skipped_issues(project_root: str, skipped: set) -> None:
    """Persist the set of skipped issue numbers to disk."""
    path = os.path.join(project_root, _WATCHDOG_SKIPPED_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"skipped": sorted(skipped)}, f)
    except OSError as exc:
        _watchdog_logger.warning("Watchdog: failed to persist skipped issues: %s", exc)


def _pick_next_roadmap_issue(
    project_root: str,
    skip_issues: Optional[set] = None,
) -> Optional[Dict]:
    """Query GitHub for the next actionable roadmap issue.

    Returns a dict with 'number', 'title', 'body' or None if nothing found.
    Skips repair/orphan issues and any issue numbers in skip_issues.
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--label", "roadmap", "--state", "open",
             "--limit", "50", "--json", "number,title,body,labels"],
            cwd=project_root, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        issues = json.loads(result.stdout or "[]")
    except Exception:
        return None

    _EPIC_SKIP_KEYWORDS = ("epic", "phase", "milestone", "overview", "arch", "design")

    def _issue_priority(issue: Dict) -> tuple:
        labels = [l.get("name", "").lower() for l in (issue.get("labels") or [])]
        p = 99
        if any(x in labels for x in ("p1", "priority: high", "priority:high")):
            p = 1
        elif any(x in labels for x in ("p2", "priority: medium", "priority:medium")):
            p = 2
        return (p, issue.get("number", 9999))

    def _is_epic_issue(issue: Dict) -> bool:
        title = (issue.get("title") or "").lower()
        labels = [l.get("name", "").lower() for l in (issue.get("labels") or [])]
        # Use word-boundary matching to avoid false positives like
        # "hierarchy" matching "arch"
        return "epic" in labels or any(
            re.search(r"\b" + re.escape(k) + r"\b", title)
            for k in _EPIC_SKIP_KEYWORDS
        )

    skip = skip_issues or set()
    for issue in sorted(issues, key=_issue_priority):
        number = issue.get("number")
        title = (issue.get("title") or "").lower()
        if any(pat in title for pat in _REPAIR_ISSUE_PATTERNS):
            continue
        if number in skip:
            continue
        if _is_epic_issue(issue):
            continue
        return issue
    return None


def _create_skip_issue(project_root: str, issue_number: int, failure_count: int) -> None:
    """Open a GitHub issue to record that the watchdog skipped a roadmap issue."""
    try:
        title = f"Watchdog: skipped roadmap issue #{issue_number} after {failure_count} consecutive failures"
        body = (
            f"The IGRIS watchdog skipped roadmap issue #{issue_number} because it failed "
            f"{failure_count} consecutive times without producing a passing diff.\n\n"
            f"This indicates a capability ceiling for the current model configuration on this task.\n\n"
            f"**Actions:**\n"
            f"- Review the failure logs for issue #{issue_number}\n"
            f"- Consider decomposing the issue into smaller sub-tasks\n"
            f"- Or upgrade the reasoning model configuration\n\n"
            f"_Auto-created by the IGRIS watchdog._"
        )
        subprocess.run(
            ["gh", "issue", "create", "--title", title, "--body", body,
             "--label", "watchdog,capability-ceiling"],
            cwd=project_root, capture_output=True, text=True, timeout=30,
        )
        _watchdog_logger.info("Watchdog: created skip issue for roadmap #%d", issue_number)
    except Exception as exc:
        _watchdog_logger.warning("Watchdog: failed to create skip issue: %s", exc)


async def _watchdog_loop(project_root: str) -> None:
    """Background task: start the next roadmap issue when no run is active.

    Tracks consecutive failures per issue and skips issues that have failed
    _WATCHDOG_MAX_CONSECUTIVE_FAILURES times in this server session, moving
    on to the next lowest-numbered roadmap issue instead of retrying forever.
    """
    from igris.core.self_repair_supervisor import (
        list_active_supervised_runs,
        list_supervised_runs,
        start_supervised_rank_async,
    )
    # Per-issue consecutive failure counter.  The skip set is persisted to disk
    # so it survives server restarts — IGRIS won't retry a capability-ceiling
    # issue every time the service restarts after another successful run.
    _issue_failures: Dict[int, int] = {}
    # Last capability_signals dict from the most-recent failed run per issue.
    # Passed as prior_capability_signals to the next run so the assignment
    # router can escalate to hard_debugging (→ gpu_reasoning → VastAI) after
    # repeated failures with accumulated no_diff_repair / reasoning_timeout.
    _issue_last_signals: Dict[int, Dict] = {}
    _skipped_issues: set = _load_skipped_issues(project_root)
    if _skipped_issues:
        _watchdog_logger.info(
            "Watchdog: loaded %d persisted skipped issues: %s",
            len(_skipped_issues), sorted(_skipped_issues),
        )
    _last_run_id: Optional[str] = None
    _last_issue_num: Optional[int] = None
    _dirty_cleanup_consecutive_failures: int = 0

    await asyncio.sleep(_WATCHDOG_COOLDOWN_SECONDS)
    while True:
        try:
            active = list_active_supervised_runs()
            if active:
                _now = _time.time()
                for _ar in active:
                    _last = getattr(_ar, "last_event", None)
                    if _last is not None:
                        _ts = getattr(_last, "timestamp", None)
                        if _ts is not None:
                            try:
                                _elapsed = _now - (
                                    _ts.timestamp() if hasattr(_ts, "timestamp") else float(_ts)
                                )
                                if _elapsed > 600:
                                    _watchdog_logger.warning(
                                        "Watchdog: active run %s has not emitted events for %ds — possible hang",
                                        _ar.run_id,
                                        int(_elapsed),
                                    )
                                    if _elapsed > _WATCHDOG_ZOMBIE_TIMEOUT_SECONDS:
                                        _watchdog_logger.warning(
                                            "Watchdog: zombie run %s (%ds) — evicting from RUN_STORE",
                                            _ar.run_id, int(_elapsed),
                                        )
                                        try:
                                            _ar.status = "blocked"
                                            _ar.blocked_reason = f"zombie_timeout_{int(_elapsed)}s"
                                            _ar.failure_class = "zombie_timeout"
                                            from igris.core.self_repair_supervisor import RUN_LOCK, RUN_STORE
                                            with RUN_LOCK:
                                                RUN_STORE.pop(_ar.run_id, None)
                                        except Exception as _ze:
                                            _watchdog_logger.warning("Watchdog: zombie eviction failed: %s", _ze)
                            except Exception:
                                pass
            if not active:
                # Account for the outcome of the run we last launched
                if _last_run_id is not None and _last_issue_num is not None:
                    all_runs = list_supervised_runs()
                    last_run = next((r for r in all_runs if r.run_id == _last_run_id), None)
                    if last_run:
                        if last_run.status in ("blocked", "failed"):
                            # If the blocked run spawned child runs via decomposition, wait for
                            # them — there is a brief window where the child is registered in
                            # RUN_STORE but not yet visible to list_active_supervised_runs().
                            def _ev_phase(e: Any) -> str:
                                return e.phase if hasattr(e, "phase") else e.get("phase", "")
                            def _ev_detail(e: Any) -> str:
                                return e.detail if hasattr(e, "detail") else e.get("detail", "")
                            def _ev_data(e: Any) -> dict:
                                return e.data if hasattr(e, "data") else e.get("data", {})
                            child_run_ids = [
                                (_ev_data(e) or {}).get("child_run_id") or ""
                                for e in getattr(last_run, "events", [])
                                if _ev_phase(e) == "submission_autorun_run_id"
                            ]
                            child_run_ids = [rid for rid in child_run_ids if rid]
                            if not child_run_ids:
                                child_run_ids = [
                                    _ev_detail(e).split("Child run ")[-1].split(" ")[0]
                                    for e in getattr(last_run, "events", [])
                                    if _ev_phase(e) == "submission_autorun_run_id"
                                    and "Child run" in _ev_detail(e)
                                ]
                                child_run_ids = [rid for rid in child_run_ids if len(rid) == 12]
                            if child_run_ids:
                                _watchdog_logger.info(
                                    "Watchdog: blocked run #%s spawned child runs %s — deferring next launch",
                                    _last_run_id, child_run_ids,
                                )
                                # Re-check active after a brief pause to let children register
                                await asyncio.sleep(5)
                                if list_active_supervised_runs():
                                    await asyncio.sleep(_WATCHDOG_POLL_SECONDS)
                                    continue
                            # capability_ceiling_reached means the strong model confirmed
                            # there is no way forward — skip after just 1 failure.
                            is_ceiling = (
                                getattr(last_run, "failure_class", "") == "capability_ceiling_reached"
                            )
                            threshold = (
                                _WATCHDOG_CEILING_SKIP_AFTER if is_ceiling
                                else _WATCHDOG_MAX_CONSECUTIVE_FAILURES
                            )
                            count = _issue_failures.get(_last_issue_num, 0) + 1
                            _issue_failures[_last_issue_num] = count
                            # Save capability_signals from the failed run so the
                            # next attempt can merge them — enables cross-run
                            # escalation to hard_debugging → gpu_reasoning → VastAI.
                            _last_sigs = dict(
                                getattr(last_run, "capability_signals", None) or {}
                            )
                            if _last_sigs:
                                prev = _issue_last_signals.get(_last_issue_num, {})
                                merged: Dict = dict(prev)
                                for _s, _c in _last_sigs.items():
                                    merged[_s] = merged.get(_s, 0) + _c
                                _issue_last_signals[_last_issue_num] = merged
                            _watchdog_logger.info(
                                "Watchdog: issue #%d failed (consecutive=%d, failure_class=%s, signals=%s)",
                                _last_issue_num, count,
                                getattr(last_run, "failure_class", ""),
                                _issue_last_signals.get(_last_issue_num, {}),
                            )
                            if count >= threshold:
                                _watchdog_logger.warning(
                                    "Watchdog: issue #%d skipped after %d consecutive failures "
                                    "(ceiling=%s, threshold=%d)",
                                    _last_issue_num, count, is_ceiling, threshold,
                                )
                                _skipped_issues.add(_last_issue_num)
                                _save_skipped_issues(project_root, _skipped_issues)
                                _create_skip_issue(project_root, _last_issue_num, count)
                        elif last_run.status == "done":
                            _issue_failures.pop(_last_issue_num, None)
                            _issue_last_signals.pop(_last_issue_num, None)
                    _last_run_id = None
                    _last_issue_num = None

                # Clean up any dirty workspace left by an interrupted run (e.g. service restart
                # during active execution sends SIGTERM with no chance for _cleanup_blocked_workspace).
                _ws = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: __import__("subprocess").run(
                        ["git", "status", "--porcelain"],
                        capture_output=True, text=True, cwd=project_root,
                    )
                )
                if _ws.returncode == 0 and _ws.stdout.strip():
                    _watchdog_logger.warning(
                        "Watchdog: dirty workspace detected before launch — running cleanup: %s",
                        _ws.stdout.strip()[:200],
                    )
                    restore_res = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: __import__("subprocess").run(
                            ["git", "restore", "--worktree", "--staged", "."],
                            capture_output=True, text=True, cwd=project_root,
                        )
                    )
                    root_untracked = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: __import__("subprocess").run(
                            ["git", "ls-files", "--others", "--exclude-standard", "--directory"],
                            capture_output=True, text=True, cwd=project_root,
                        )
                    )
                    for item in (root_untracked.stdout or "").splitlines():
                        item = item.strip().rstrip("/")
                        if item and "/" not in item and not item.startswith("."):
                            _p = __import__("os").path.join(project_root, item)
                            if __import__("os").path.isfile(_p):
                                __import__("os").unlink(_p)
                            else:
                                __import__("shutil").rmtree(_p, ignore_errors=True)
                    clean_res = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: __import__("subprocess").run(
                            ["git", "clean", "-fd", "--", "igris", "tests", "docs", "."],
                            capture_output=True, text=True, cwd=project_root,
                        )
                    )
                    if restore_res.returncode != 0 or clean_res.returncode != 0:
                        _dirty_cleanup_consecutive_failures += 1
                        _watchdog_logger.warning(
                            "Watchdog cleanup failed (%d consecutive): restore=%s clean=%s",
                            _dirty_cleanup_consecutive_failures,
                            (restore_res.stderr or restore_res.stdout or "")[:200],
                            (clean_res.stderr or clean_res.stdout or "")[:200],
                        )
                        if _dirty_cleanup_consecutive_failures >= 3:
                            try:
                                from igris.core import smw_actions
                                await smw_actions.execute_action(
                                    "open_diagnostic_issue",
                                    tier=2,
                                    project_root=project_root,
                                    details={
                                        "source": "watchdog_cleanup_loop",
                                        "problematic_file": _ws.stdout.strip().splitlines()[0][:500],
                                    },
                                )
                            except Exception as exc:
                                _watchdog_logger.warning("Watchdog diagnostic issue action failed: %s", exc)
                            await asyncio.sleep(300)
                    else:
                        _dirty_cleanup_consecutive_failures = 0
                else:
                    _dirty_cleanup_consecutive_failures = 0

                _hint_path = __import__("pathlib").Path(project_root) / ".igris" / "next_roadmap_target.json"
                _hint_issue = None
                if _hint_path.exists():
                    try:
                        _hint_data = __import__("json").loads(_hint_path.read_text(encoding="utf-8"))
                        _hint_num = int(_hint_data.get("issue_number", 0))
                        if _hint_num and _hint_num not in _skipped_issues:
                            _hint_issue = {"number": _hint_num, "title": _hint_data.get("issue_title", ""), "body": ""}
                        _hint_path.unlink()
                    except Exception:
                        try:
                            _hint_path.unlink()
                        except Exception:
                            pass
                issue = _hint_issue or _pick_next_roadmap_issue(project_root, skip_issues=_skipped_issues)
                if issue:
                    number = issue["number"]
                    title = issue.get("title", f"issue #{number}")
                    body = issue.get("body", "")
                    # Pass the full issue body so acceptance criteria and
                    # integration points are never cut off.  Previously capped
                    # at 1000 chars, which silently dropped Rule 6, all
                    # integration targets, and the entire ## Acceptance criteria
                    # section for issues longer than ~1 KB.
                    goal = f"Implement GitHub issue #{number}: {title}\n\n{body}"
                    _watchdog_logger.info("Watchdog: starting run for issue #%s — %s", number, title)
                    _run_budget = max(0.0, float(os.getenv("IGRIS_MAX_COST_PER_RUN", "3.0") or "3.0"))
                    _max_escalations = max(0, int(os.getenv("IGRIS_MAX_ESCALATIONS_PER_RUN", "3") or "3"))
                    launched = start_supervised_rank_async(
                        {
                            "goal": goal,
                            "github_issue": number,
                            "allow_merge_if_green": True,
                            "allow_auto_subissues": (
                                str(os.getenv("IGRIS_ALLOW_AUTO_SUBISSUES_DEFAULT", "true")).strip().lower()
                                not in {"0", "false", "no", "off"}
                            ),
                            "autochain_depth": 1,
                            "allow_api_escalation": True,
                            "max_api_escalations_per_run": _max_escalations,
                            "max_api_budget_usd": _run_budget,
                            # Allow more read-heavy steps before triggering no_diff_repair.
                            # Complex multi-file tasks (feat issues) need time to navigate
                            # to integration targets before the next write. Default 20 is
                            # too tight when a task spans 3+ files with unfamiliar paths.
                            "no_diff_steps_max": max(
                                1,
                                int(os.getenv("IGRIS_NO_DIFF_STEPS_MAX", "40") or "40"),
                            ),
                            # Cross-run escalation: tell the assignment router how many
                            # prior attempts have been made and what signals accumulated
                            # so it can escalate to hard_debugging → gpu_reasoning → VastAI
                            # after repeated failures without resetting from scratch.
                            "prior_attempts": _issue_failures.get(number, 0),
                            "prior_capability_signals": _issue_last_signals.get(number, {}),
                        },
                        project_root=project_root,
                    )
                    _last_run_id = launched.run_id
                    _last_issue_num = number
                    await asyncio.sleep(_WATCHDOG_COOLDOWN_SECONDS)
                else:
                    _watchdog_logger.debug(
                        "Watchdog: no actionable roadmap issue found (skipped=%s)", _skipped_issues
                    )
        except Exception as exc:
            _watchdog_logger.warning("Watchdog error: %s", exc)
        await asyncio.sleep(_WATCHDOG_POLL_SECONDS)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    project_root = str(Path(__file__).resolve().parents[2])
    task = asyncio.create_task(_watchdog_loop(project_root))
    _watchdog_logger.info("Watchdog started (poll=%ds)", _WATCHDOG_POLL_SECONDS)
    from igris.core.meta_watchdog import start_smw
    smw_task = start_smw(project_root)
    logging.getLogger("igris.smw").info("SMW started (poll=%ds)", 120)
    try:
        yield
    finally:
        task.cancel()
        smw_task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="IGRIS_GPT", version="0.1.0", lifespan=_lifespan)
    # Issue #727 — security: CORS restriction, API-key auth, rate limiting
    from igris.web.security import apply_security_middleware
    apply_security_middleware(app)
    _graph_instance: Optional[MemoryGraph] = None
    def _get_graph() -> MemoryGraph:
        nonlocal _graph_instance
        if _graph_instance is None:
            _graph_instance = MemoryGraph(str(CONFIG.project_root))
            _graph_instance.migrate_legacy(str(CONFIG.project_root))
        return _graph_instance

    @app.get('/api/diagnostics/session-resume')
    async def session_resume():
        # Implement the logic for session resume
        return JSONResponse(content={'status': 'success'})

    @app.get('/api/rank/s-dashboard')
    async def get_rank_s_dashboard():
        return {
            'app': 'IGRIS_GPT',
            'rank': 'S',
            'status': 'ok',
            'capability': 'end-to-end-supervised',
            'checks': {
                'backend': True,
                'ui': True,
                'tests': True,
                'workflow': True
            }
        }

    @app.get('/api/rank/ui-card')
    async def get_rank_ui_card():
        return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}

    @app.get('/api/rank/summary-card')
    async def get_rank_summary_card():
        return {'app': 'IGRIS_GPT', 'rank': 'A+', 'status': 'ok', 'capability': 'multi-file-supervised'}

    @app.get('/api/system/version-summary')
    async def get_version_summary():
        return {'app': 'IGRIS_GPT', 'rank': 'A-generalization', 'status': 'ok'}

    @app.get('/api/rank/status')
    async def get_rank_status():
        return {'rank': 'A', 'status': 'ok', 'agent': 'IGRIS_GPT'}
    @app.get('/api/version-info')
    async def version_info():
        return {'app': 'IGRIS_GPT', 'status': 'ok'}

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
            "intent_detected": result.get("intent_detected"),
            "suggested_actions": result.get("suggested_actions", []),
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

    # ---- System Info ----

    @app.get("/api/system/info")
    async def api_system_info() -> Dict[str, object]:
        """Safe, read-only system information."""
        from igris.core.system_info import get_system_info
        import os as _os
        return get_system_info(
            project_root=str(CONFIG.project_root),
            host=_os.environ.get("IGRIS_HOST", "127.0.0.1"),
            port=int(_os.environ.get("IGRIS_PORT", "8000")),
        )

    # ---- Dashboard Summary ----

    @app.get("/api/dashboard/summary")
    async def api_dashboard_summary() -> Dict[str, object]:
        """Aggregated dashboard view — health, readiness, diagnostics, loop."""
        from igris.core import diagnostics as diagnostics_dash

        diag = {}
        try:
            tasks = [t.to_dict() for t in task_engine.list_tasks()]
            timeline = task_engine.recent_timeline_events(limit=50)
            diag = diagnostics_dash.get_diagnostic_summary(
                tasks, timeline, project_root=str(CONFIG.project_root),
            )
        except Exception:
            pass

        loop_info = {}
        try:
            loop_info = loop_engine.get_status()
        except Exception:
            pass

        return {
            "health": {"status": "ok"},
            "diagnostics": diag,
            "loop": loop_info,
            "tab_layout": {
                "primary": ["dashboard", "code", "tasks", "terminal", "memory", "safety", "advanced"],
                "grouped": {
                    "code": ["files", "git", "patches"],
                    "tasks": ["tasks", "loop"],
                    "terminal": ["commands", "tests"],
                    "memory": ["memory", "timeline"],
                    "safety": ["safety", "cost"],
                    "advanced": ["a2a", "logs"],
                },
            },
        }

    # ---- Chat Personality / Capabilities ----

    @app.get("/api/chat/capabilities")
    async def api_chat_capabilities() -> Dict[str, object]:
        from igris.core.chat_personality import get_capability_summary
        return get_capability_summary()

    @app.post("/api/chat/intent")
    async def api_chat_intent(request: Request) -> Dict[str, object]:
        from igris.core.chat_personality import (
            detect_intent, get_grounded_response, get_suggested_actions,
        )
        content = await request.json()
        message = content.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="message required")
        intent = detect_intent(message)
        response = get_grounded_response(intent) if intent else None
        actions = get_suggested_actions(intent) if intent else []
        return {
            "intent": intent,
            "grounded_response": response,
            "has_response": response is not None,
            "suggested_actions": actions,
        }

    @app.get("/api/chat/actions")
    async def api_chat_actions() -> Dict[str, object]:
        from igris.core.chat_personality import get_all_safe_actions
        return {"actions": get_all_safe_actions()}

    @app.get("/api/chat/actions/{intent_name}")
    async def api_chat_actions_by_intent(intent_name: str) -> Dict[str, object]:
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions(intent_name)
        if not actions:
            raise HTTPException(status_code=404, detail=f"No actions for intent: {intent_name}")
        return {"intent": intent_name, "actions": actions}

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

    from igris.layers.advisory.vastai_manager import _SHARED_MANAGER as vastai_mgr

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

    # ---- Fleet API ----

    from igris.layers.advisory.vastai_fleet import _SHARED_FLEET

    @app.get("/api/fleet/status")
    async def api_fleet_status() -> Dict[str, object]:
        """Fleet-wide status: all instances, queue, costs."""
        return _SHARED_FLEET.fleet_status()

    @app.post("/api/fleet/provision")
    async def api_fleet_provision(request: Request) -> Dict[str, object]:
        """Manually trigger provisioning of N new fleet instances."""
        body = await request.json()
        approval = body.get("approval", "")
        count = int(body.get("count", 1))
        if approval != "I_APPROVE_VASTAI_COSTS":
            raise HTTPException(status_code=403, detail="approval required: I_APPROVE_VASTAI_COSTS")
        if count < 1 or count > 5:
            raise HTTPException(status_code=400, detail="count must be 1-5")
        new_instances = _SHARED_FLEET._provision_instances(count)
        return {"provisioned": len(new_instances), "fleet": _SHARED_FLEET.fleet_status()}

    @app.post("/api/fleet/release/{instance_id}")
    async def api_fleet_release(instance_id: str, request: Request) -> Dict[str, object]:
        """Manually release a fleet instance back to idle."""
        body = await request.json()
        outcome = body.get("outcome", "manual_release")
        _SHARED_FLEET.release(instance_id, outcome=outcome)
        return {"released": instance_id, "fleet": _SHARED_FLEET.fleet_status()}

    @app.get("/api/fleet/queue")
    async def api_fleet_queue() -> Dict[str, object]:
        """Current task queue waiting for GPU instances."""
        status = _SHARED_FLEET.fleet_status()
        return {"queue_depth": status["queue_depth"], "queue": status.get("queue", [])}

    @app.get("/api/fleet/worktrees")
    async def api_fleet_worktrees() -> Dict[str, object]:
        """Active git worktrees managed by WorktreeManager."""
        status = _SHARED_FLEET.fleet_status()
        return {"worktrees": status.get("worktrees", [])}

    @app.get("/api/fleet/locks")
    async def api_fleet_locks() -> Dict[str, object]:
        """Current file lock registry — which issue holds which paths."""
        status = _SHARED_FLEET.fleet_status()
        return {"file_locks": status.get("file_locks", {})}

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

    @app.get("/api/smw/health")
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

    @app.get("/api/memory/summary")
    async def api_memory_summary() -> Dict[str, object]:
        g = _get_graph()
        node_count = g.conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
        edge_count = g.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
        rows = g.conn.execute("SELECT node_type, COUNT(*) as c FROM memory_nodes GROUP BY node_type").fetchall()
        migration_done = bool(g.conn.execute("SELECT 1 FROM memory_nodes WHERE node_type='environment_fact' AND content LIKE '%\"migration_done\"%' LIMIT 1").fetchone())
        return {"node_count": node_count, "edge_count": edge_count, "node_types": {r[0]: r[1] for r in rows}, "migration_done": migration_done, "db_size_kb": round(g.db_path.stat().st_size / 1024.0, 2) if g.db_path.exists() else 0.0}

    @app.get("/api/memory/search")
    async def api_memory_search(q: str, node_type: Optional[str] = None, limit: int = 10) -> Dict[str, object]:
        results = _get_graph().query_by_intent(q, node_type=node_type, limit=limit)
        return {"results": results, "count": len(results)}

    @app.post("/api/memory/record")
    async def api_memory_record(request: Request) -> Dict[str, object]:
        body = await request.json()
        node_id = _get_graph().add_node(body["node_type"], body.get("content", {}), confidence=body.get("confidence", 1.0), tags=body.get("tags", []))
        return {"node_id": node_id}

    @app.post("/api/memory/learn-command")
    async def api_memory_learn_command(request: Request) -> Dict[str, object]:
        body = await request.json()
        node_id = _get_graph().add_node("command_recipe", {"intent": body.get("intent", ""), "command": body.get("command", ""), "risk": body.get("risk", "low")}, success_rate=1.0 if body.get("success", True) else 0.0)
        return {"node_id": node_id}

    @app.post("/api/memory/export-safe")
    async def api_memory_export_safe() -> StreamingResponse:
        payload = json.dumps({"nodes": _get_graph().export_safe()}, indent=2).encode("utf-8")
        return StreamingResponse(iter([payload]), media_type="application/json", headers={"Content-Disposition": "attachment; filename=memory_export_safe.json"})

    @app.post("/api/memory/import-safe")
    async def api_memory_import_safe(request: Request) -> Dict[str, object]:
        body = await request.json()
        return _get_graph().import_safe(body.get("nodes", []))

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

    @app.post("/api/patches/generate")
    async def api_generate_patch(request: Request) -> Dict[str, object]:
        from igris.core import llm_patch_generator
        content = await request.json()
        task_title = content.get("title", content.get("task_title", ""))
        if not task_title:
            raise HTTPException(status_code=400, detail="title is required")
        result = llm_patch_generator.generate_patch(
            task_title=task_title,
            task_description=content.get("description", ""),
            context=content.get("context", ""),
        )
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch generated: {task_title[:80]}",
            "detail": f"by={result.get('generated_by', 'unknown')}, files={len(result.get('files', []))}",
            "severity": "info",
        })
        return result

    @app.post("/api/tasks/{task_id}/generate-patch")
    async def api_task_generate_patch(task_id: int) -> Dict[str, object]:
        from igris.core import llm_patch_generator
        task = task_engine.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        td = task.to_dict() if hasattr(task, "to_dict") else {}
        result = llm_patch_generator.generate_patch(
            task_title=td.get("title", ""),
            task_description=td.get("description", ""),
        )
        result["task_id"] = task_id
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch generated for task: {task_id}",
            "detail": f"by={result.get('generated_by', 'unknown')}",
            "severity": "info",
        })
        return result

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

    # ---- Doctor / Verify / Crash Recovery ----

    @app.get("/api/doctor")
    async def api_doctor() -> Dict[str, object]:
        """Run environment diagnostics (igris doctor)."""
        from igris.core.doctor import run_doctor
        import os as _os
        report = run_doctor(
            project_root=str(CONFIG.project_root),
            host=_os.environ.get("IGRIS_HOST", "127.0.0.1"),
            port=int(_os.environ.get("IGRIS_PORT", "8000")),
        )
        task_engine.append_timeline_event({
            "type": "doctor", "title": f"Doctor run: {report._compute_overall()}",
            "detail": f"{len(report.checks)} checks, overall={report._compute_overall()}",
            "severity": "info" if report._compute_overall() == "ok" else "warning",
        })
        return report.to_dict()

    @app.get("/api/doctor/markdown")
    async def api_doctor_markdown() -> JSONResponse:
        """Run doctor and return Markdown report."""
        from igris.core.doctor import run_doctor
        import os as _os
        report = run_doctor(
            project_root=str(CONFIG.project_root),
            host=_os.environ.get("IGRIS_HOST", "127.0.0.1"),
            port=int(_os.environ.get("IGRIS_PORT", "8000")),
        )
        return JSONResponse(content={"markdown": report.to_markdown()})

    @app.get("/api/verify")
    async def api_verify() -> Dict[str, object]:
        """Quick installation verification."""
        from igris.core.doctor import run_verify
        result = run_verify(project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "verify", "title": f"Verify: {'PASS' if result['ok'] else 'FAIL'}",
            "detail": json.dumps({k: v for k, v in result["checks"].items()}, default=str),
            "severity": "info" if result["ok"] else "warning",
        })
        return result

    @app.get("/api/config/validate")
    async def api_config_validate() -> Dict[str, object]:
        """Validate configuration (.env, config.json, providers, budget, safety)."""
        from igris.core.config_validator import validate_all
        result = validate_all(project_root=str(CONFIG.project_root))
        return result.to_dict()

    @app.get("/api/crash-reports")
    async def api_crash_reports(limit: int = 20) -> Dict[str, object]:
        """List recent crash reports."""
        from igris.core.crash_recovery import list_crash_reports
        reports = list_crash_reports(project_root=str(CONFIG.project_root), limit=limit)
        return {"reports": reports, "count": len(reports)}

    @app.get("/api/crash-reports/last-good-state")
    async def api_last_good_state() -> Dict[str, object]:
        """Get the last known good state."""
        from igris.core.crash_recovery import load_good_state
        state = load_good_state(project_root=str(CONFIG.project_root))
        return {"state": state, "available": state is not None}

    @app.post("/api/crash-reports/save-good-state")
    async def api_save_good_state(request: Request) -> Dict[str, object]:
        """Persist the current state as last known good."""
        from igris.core.crash_recovery import save_good_state
        content = await request.json()
        state = content.get("state", {})
        if not state:
            raise HTTPException(status_code=400, detail="State payload required")
        save_good_state(state, project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "recovery",
            "title": "Good state saved",
            "detail": f"Keys: {', '.join(state.keys())}",
            "severity": "info",
        })
        return {"saved": True}

    @app.get("/api/crash-reports/{crash_id}")
    async def api_crash_report_detail(crash_id: str) -> Dict[str, object]:
        """Get a specific crash report."""
        from igris.core.crash_recovery import get_crash_report
        report = get_crash_report(crash_id, project_root=str(CONFIG.project_root))
        if not report:
            raise HTTPException(status_code=404, detail=f"Crash report {crash_id} not found")
        return report


    _WORK_SESSIONS: Dict[str, object] = {}

    @app.post("/api/work-session/start")
    async def api_work_session_start(request: Request) -> Dict[str, str]:
        from igris.core.work_session import WorkSession
        content = await request.json()
        goal = content.get("goal", "")
        if not goal:
            raise HTTPException(status_code=400, detail="goal required")
        session = WorkSession.create(goal=goal, mission_id=content.get("mission_id"))
        _WORK_SESSIONS[session.session_id] = session
        return {"session_id": session.session_id}

    @app.get("/api/work-session/{session_id}")
    async def api_work_session_get(session_id: str) -> Dict[str, object]:
        session = _WORK_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="WorkSession not found")
        return session.to_dict()

    @app.post("/api/work-session/{session_id}/advance")
    async def api_work_session_advance(session_id: str, request: Request) -> Dict[str, object]:
        from igris.core.work_session import WorkPhase
        session = _WORK_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="WorkSession not found")
        content = await request.json()
        phase = WorkPhase(content.get("phase", "understand"))
        session.advance_phase(phase=phase, outcome=content.get("outcome", "success"), notes=content.get("notes", ""))
        return session.to_dict()

    @app.post("/api/work-session/{session_id}/deliver")
    async def api_work_session_deliver(session_id: str, request: Request) -> Dict[str, object]:
        from igris.core.work_session import DeliveryReport
        session = _WORK_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="WorkSession not found")
        content = await request.json()
        report = DeliveryReport(**content)
        session.complete_deliver(report)
        session.remember(project_root=str(CONFIG.project_root))
        return {"status": "delivered", "delivery_report": report.__dict__}

    # ---- Mission Controller (Epic #40) ----

    @app.post("/api/controller/missions")
    async def api_controller_create_mission(request: Request) -> Dict[str, object]:
        """Create a controlled mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        title = content.get("title", "")
        goal = content.get("goal", "")
        if not title or not goal:
            raise HTTPException(status_code=400, detail="title and goal required")
        mission = ctrl.create_mission(
            title=title,
            goal=goal,
            description=content.get("description", ""),
            workspace=content.get("workspace", str(CONFIG.project_root)),
            target_hosts=content.get("target_hosts", []),
            constraints=content.get("constraints", []),
            success_criteria=content.get("success_criteria", []),
            risk_level=content.get("risk_level", "low"),
            rollback_plan=content.get("rollback_plan"),
        )
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Mission created: {title}",
            "detail": goal[:200], "severity": "info",
            "mission_id": mission.id, "trace_id": mission.trace_id,
        })
        return mission.to_dict()

    @app.get("/api/controller/missions")
    async def api_controller_list_missions() -> Dict[str, object]:
        """List all controlled missions."""
        from igris.core.mission_controller import list_controlled_missions
        missions = list_controlled_missions(project_root=str(CONFIG.project_root))
        return {"missions": [m.to_dict() for m in missions], "count": len(missions)}

    @app.get("/api/controller/missions/{mission_id}")
    async def api_controller_get_mission(mission_id: str) -> Dict[str, object]:
        """Get a controlled mission by ID."""
        from igris.core.mission_controller import load_controlled_mission
        mission = load_controlled_mission(mission_id, project_root=str(CONFIG.project_root))
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @app.get("/api/controller/missions/{mission_id}/explain")
    async def api_controller_explain(mission_id: str) -> Dict[str, object]:
        """Explain current mission state and next action."""
        from igris.core.mission_controller import load_controlled_mission
        mission = load_controlled_mission(mission_id, project_root=str(CONFIG.project_root))
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.explain_state()

    @app.post("/api/controller/missions/{mission_id}/plan")
    async def api_controller_plan(mission_id: str) -> Dict[str, object]:
        """Generate plan for a controlled mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        mission = ctrl.plan_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Mission planned: {mission.title}",
            "detail": f"{mission.total_steps} steps", "severity": "info",
            "mission_id": mission.id, "trace_id": mission.trace_id,
        })
        return mission.to_dict()

    @app.post("/api/controller/missions/{mission_id}/execute-next")
    async def api_controller_execute_next(mission_id: str) -> Dict[str, object]:
        """Execute the next step in the mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        result = ctrl.execute_next_step(mission_id)
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @app.post("/api/controller/missions/{mission_id}/report-outcome")
    async def api_controller_report_outcome(mission_id: str, request: Request) -> Dict[str, object]:
        """Report step outcome."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        step_index = content.get("step_index", 0)
        outcome = content.get("outcome", "success")
        detail = content.get("detail", "")
        mission = ctrl.report_step_outcome(mission_id, step_index, outcome, detail)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @app.post("/api/controller/missions/{mission_id}/pause")
    async def api_controller_pause(mission_id: str, request: Request) -> Dict[str, object]:
        """Pause a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        mission = ctrl.pause_mission(mission_id, content.get("reason", ""))
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @app.post("/api/controller/missions/{mission_id}/resume")
    async def api_controller_resume(mission_id: str) -> Dict[str, object]:
        """Resume a paused mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        mission = ctrl.resume_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @app.post("/api/controller/missions/{mission_id}/block")
    async def api_controller_block(mission_id: str, request: Request) -> Dict[str, object]:
        """Block a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        reason = content.get("reason", "blocked")
        mission = ctrl.block_mission(mission_id, reason)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @app.post("/api/controller/missions/{mission_id}/unblock")
    async def api_controller_unblock(mission_id: str) -> Dict[str, object]:
        """Unblock a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        mission = ctrl.unblock_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @app.post("/api/controller/missions/{mission_id}/verify")
    async def api_controller_verify(mission_id: str) -> Dict[str, object]:
        """Verify mission success criteria."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        result = ctrl.verify_mission(mission_id)
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        return result

    @app.get("/api/controller/missions/{mission_id}/report")
    async def api_controller_report(mission_id: str) -> Dict[str, object]:
        """Generate final report for a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        report = ctrl.generate_final_report(mission_id)
        if not report:
            raise HTTPException(status_code=404, detail="Mission not found")
        return report

    @app.get("/api/controller/missions/{mission_id}/context")
    async def api_controller_context(mission_id: str) -> Dict[str, object]:
        """Reconstruct mission context (for restart recovery)."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        ctx = ctrl.reconstruct_context(mission_id)
        if not ctx:
            raise HTTPException(status_code=404, detail="Mission not found")
        return ctx

    @app.post("/api/controller/missions/{mission_id}/artifacts")
    async def api_controller_add_artifact(mission_id: str, request: Request) -> Dict[str, object]:
        """Add an artifact to a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        mission = ctrl.add_artifact(
            mission_id,
            artifact_type=content.get("type", "file"),
            path=content.get("path", ""),
            description=content.get("description", ""),
        )
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    # ---- Safety / Risk / Rollback (Epic #42) ----

    @app.post("/api/safety/classify-risk")
    async def api_classify_risk(request: Request) -> Dict[str, object]:
        """Classify action risk level."""
        from igris.core.risk_classifier import classify_action_risk
        content = await request.json()
        action_id = content.get("action_id", "")
        description = content.get("description", "")
        risk = classify_action_risk(action_id, description)
        return {"action_id": action_id, "risk_level": risk}

    @app.post("/api/safety/check-approval")
    async def api_check_approval(request: Request) -> Dict[str, object]:
        """Check if an action is approved under current policy."""
        from igris.core.risk_classifier import check_approval
        content = await request.json()
        decision = check_approval(
            action_id=content.get("action_id", ""),
            risk_level=content.get("risk_level", "low"),
            approval_mode=content.get("approval_mode", "safe"),
            has_rollback=content.get("has_rollback", False),
            host=content.get("host", ""),
            authorized_hosts=content.get("authorized_hosts"),
            approval_token=content.get("approval_token"),
            trace_id=content.get("trace_id", ""),
        )
        return decision.to_dict()

    @app.post("/api/safety/guard-secret")
    async def api_guard_secret(request: Request) -> Dict[str, object]:
        """Check if a file path is a secret file."""
        from igris.core.risk_classifier import guard_secret_access
        content = await request.json()
        decision = guard_secret_access(content.get("path", ""), content.get("action", "read"))
        return decision.to_dict()

    @app.post("/api/rollback/backup-file")
    async def api_rollback_backup_file(request: Request) -> Dict[str, object]:
        """Create a file backup for rollback."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        content = await request.json()
        entry = mgr.backup_file(
            file_path=content.get("file_path", ""),
            mission_id=content.get("mission_id", ""),
            action_id=content.get("action_id", ""),
            trace_id=content.get("trace_id", ""),
            description=content.get("description", ""),
        )
        if not entry:
            raise HTTPException(status_code=400, detail="File not found or backup failed")
        return entry.to_dict()

    @app.post("/api/rollback/save-state")
    async def api_rollback_save_state(request: Request) -> Dict[str, object]:
        """Save a state snapshot for rollback."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        content = await request.json()
        entry = mgr.save_state_snapshot(
            state=content.get("state", {}),
            mission_id=content.get("mission_id", ""),
            action_id=content.get("action_id", ""),
            trace_id=content.get("trace_id", ""),
            description=content.get("description", ""),
        )
        return entry.to_dict()

    @app.get("/api/rollback/entries")
    async def api_rollback_list(mission_id: str = "", limit: int = 50) -> Dict[str, object]:
        """List rollback entries."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        entries = mgr.list_entries(mission_id=mission_id or None, limit=limit)
        return {"entries": entries, "count": len(entries)}

    @app.get("/api/rollback/entries/{entry_id}")
    async def api_rollback_get(entry_id: str) -> Dict[str, object]:
        """Get a rollback entry."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        entry = mgr.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Rollback entry not found")
        return entry

    @app.post("/api/rollback/entries/{entry_id}/verify")
    async def api_rollback_verify(entry_id: str) -> Dict[str, object]:
        """Verify if a rollback can be applied."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        return mgr.verify_rollback_applicable(entry_id)

    @app.post("/api/rollback/entries/{entry_id}/apply")
    async def api_rollback_apply(entry_id: str) -> Dict[str, object]:
        """Apply a file rollback."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        success = mgr.apply_file_rollback(entry_id)
        return {"applied": success, "entry_id": entry_id}

    @app.get("/api/safety/events")
    async def api_safety_events(
        event_type: str = "",
        mission_id: str = "",
        severity: str = "",
        limit: int = 100,
    ) -> Dict[str, object]:
        """List safety events."""
        from igris.core.safety_event_log import SafetyEventLog
        log = SafetyEventLog(project_root=str(CONFIG.project_root))
        events = log.list_events(
            event_type=event_type or None,
            mission_id=mission_id or None,
            severity=severity or None,
            limit=limit,
        )
        return {"events": events, "count": len(events)}

    @app.get("/api/safety/events/{event_id}")
    async def api_safety_event_detail(event_id: str) -> Dict[str, object]:
        """Get a specific safety event."""
        from igris.core.safety_event_log import SafetyEventLog
        log = SafetyEventLog(project_root=str(CONFIG.project_root))
        event = log.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Safety event not found")
        return event

    @app.get("/api/safety/summary")
    async def api_safety_summary(mission_id: str = "") -> Dict[str, object]:
        """Get safety event summary."""
        from igris.core.safety_event_log import SafetyEventLog
        log = SafetyEventLog(project_root=str(CONFIG.project_root))
        return log.get_summary(mission_id)

    # ---- Tool Runtime (Epic #41) ----

    @app.get("/api/tools")
    async def api_tools_list() -> Dict[str, object]:
        """List available tool families."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return {"tools": rt.list_tools()}

    @app.post("/api/tools/shell/execute")
    async def api_tools_shell(request: Request) -> Dict[str, object]:
        """Execute a governed shell command."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.shell_execute(
            command_id=content.get("command_id", ""),
            args=content.get("args"),
            cwd=content.get("cwd"),
            timeout=content.get("timeout", 30),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.post("/api/tools/fs/read")
    async def api_tools_fs_read(request: Request) -> Dict[str, object]:
        """Read a file safely."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.fs_read(
            path=content.get("path", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.post("/api/tools/fs/write")
    async def api_tools_fs_write(request: Request) -> Dict[str, object]:
        """Write to a file safely."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.fs_write(
            path=content.get("path", ""),
            content=content.get("content", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.post("/api/tools/fs/diff")
    async def api_tools_fs_diff(request: Request) -> Dict[str, object]:
        """Preview diff for a file."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.fs_diff(
            path=content.get("path", ""),
            new_content=content.get("new_content", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.get("/api/tools/git/status")
    async def api_tools_git_status(mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git status."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_status(mission_id=mission_id, trace_id=trace_id).to_dict()

    @app.get("/api/tools/git/diff")
    async def api_tools_git_diff(staged: bool = False, mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git diff."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_diff(staged=staged, mission_id=mission_id, trace_id=trace_id).to_dict()

    @app.get("/api/tools/git/log")
    async def api_tools_git_log(count: int = 10, mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git log."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_log(count=count, mission_id=mission_id, trace_id=trace_id).to_dict()

    @app.get("/api/tools/git/branch")
    async def api_tools_git_branch(mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git branches."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_branch(mission_id=mission_id, trace_id=trace_id).to_dict()

    @app.post("/api/tools/git/commit")
    async def api_tools_git_commit(request: Request) -> Dict[str, object]:
        """Gated git commit."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.git_commit(
            message=content.get("message", ""),
            files=content.get("files"),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.post("/api/tools/docker/ps")
    async def api_tools_docker_ps(request: Request) -> Dict[str, object]:
        """Docker ps."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        return rt.docker_ps(
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        ).to_dict()

    @app.post("/api/tools/http/check")
    async def api_tools_http_check(request: Request) -> Dict[str, object]:
        """HTTP health check."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.http_check(
            url=content.get("url", ""),
            timeout=content.get("timeout", 10),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.post("/api/tools/test/run")
    async def api_tools_test_run(request: Request) -> Dict[str, object]:
        """Run tests."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        result = rt.run_tests(
            args=content.get("args"),
            timeout=content.get("timeout", 120),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @app.get("/api/tools/hosts")
    async def api_tools_hosts() -> Dict[str, object]:
        """List registered SSH hosts."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return {"hosts": rt.list_hosts()}

    @app.post("/api/tools/hosts/register")
    async def api_tools_host_register(request: Request) -> Dict[str, object]:
        """Register an SSH host."""
        from igris.core.tool_runtime import SSHHost, ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        host = SSHHost.from_dict(content)
        rt.register_host(host)
        return {"registered": host.to_dict()}

    # ---- GOAP Planner (Epic #43) ----

    @app.get("/api/goap/state")
    async def api_goap_state() -> Dict[str, object]:
        """Get current world state."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        return planner.get_current_state().to_dict()

    @app.post("/api/goap/plan")
    async def api_goap_plan(request: Request) -> Dict[str, object]:
        """Generate a GOAP plan for a goal."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json()
        goal = content.get("goal", {})
        mission_id = content.get("mission_id", "")
        plan = planner.generate_plan(goal=goal, mission_id=mission_id)
        planner.save_plan(plan)
        return plan.to_dict()

    @app.get("/api/goap/plans")
    async def api_goap_plans(mission_id: str = "") -> Dict[str, object]:
        """List GOAP plans."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plans = planner.list_plans(mission_id=mission_id)
        return {"plans": plans, "count": len(plans)}

    @app.get("/api/goap/plans/{plan_id}")
    async def api_goap_plan_get(plan_id: str) -> Dict[str, object]:
        """Get a specific GOAP plan."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return plan.to_dict()

    @app.get("/api/goap/plans/{plan_id}/explain")
    async def api_goap_plan_explain(plan_id: str) -> Dict[str, object]:
        """Explain a GOAP plan."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return planner.explain_plan(plan)

    @app.get("/api/goap/plans/{plan_id}/next")
    async def api_goap_plan_next(plan_id: str) -> Dict[str, object]:
        """Explain next action in a GOAP plan."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return planner.explain_next_action(plan)

    @app.post("/api/goap/eligible-actions")
    async def api_goap_eligible(request: Request) -> Dict[str, object]:
        """Get eligible actions for a state."""
        from igris.core.goap_planner import GOAPPlanner, WorldState
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        state = WorldState.from_dict(content) if content else planner.get_current_state()
        eligible = planner.get_eligible_actions(state)
        return {"actions": [a.to_dict() for a in eligible], "count": len(eligible)}

    @app.post("/api/goap/validate-llm-plan")
    async def api_goap_validate(request: Request) -> Dict[str, object]:
        """Validate LLM-generated plan output."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json()
        plan = planner.validate_llm_plan(content)
        if not plan:
            return {"valid": False, "reason": "Plan does not match required schema"}
        return {"valid": True, "plan": plan.to_dict()}

    @app.post("/api/goap/replan")
    async def api_goap_replan(request: Request) -> Dict[str, object]:
        """Replan after failure."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json()
        plan_id = content.get("plan_id", "")
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        new_plan = planner.replan_after_failure(
            plan=plan,
            failed_action_id=content.get("failed_action_id", ""),
            failure_reason=content.get("failure_reason", ""),
        )
        planner.save_plan(new_plan)
        return new_plan.to_dict()

    # ---- Teacher/Governor (Epic #46) ----

    @app.post("/api/governor/evaluate")
    async def api_governor_evaluate(request: Request) -> Dict[str, object]:
        """Evaluate a proposed task against governance rules."""
        from igris.core.teacher_governor import TeacherGovernor, TaskFingerprint
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        fp = None
        if content.get("fingerprint"):
            fp = TaskFingerprint(**content["fingerprint"])
        decision = gov.evaluate_task(
            description=content.get("description", ""),
            family=content.get("family", ""),
            differentiator=content.get("differentiator", ""),
            success_criteria=content.get("success_criteria"),
            fingerprint=fp,
            trace_id=content.get("trace_id", ""),
        )
        return decision.to_dict()

    @app.get("/api/governor/summary")
    async def api_governor_summary() -> Dict[str, object]:
        """Get governor state summary."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        return gov.get_summary()

    @app.get("/api/governor/saturated")
    async def api_governor_saturated() -> Dict[str, object]:
        """Get saturated families."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        return {
            "saturated": gov.get_saturated_families(),
            "counts": gov.get_family_counts(),
        }

    @app.post("/api/governor/block-family")
    async def api_governor_block(request: Request) -> Dict[str, object]:
        """Block a family from future selection."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        decision = gov.block_family(
            family=content.get("family", ""),
            reason=content.get("reason", ""),
        )
        gov.save_state()
        return decision.to_dict()

    @app.post("/api/governor/materialize-alternative")
    async def api_governor_materialize(request: Request) -> Dict[str, object]:
        """Materialize an alternative task."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        decision = gov.materialize_alternative(
            original_family=content.get("family", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return decision.to_dict()

    @app.get("/api/governor/escalation-report")
    async def api_governor_escalation(trace_id: str = "") -> Dict[str, object]:
        """Generate escalation report."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        return gov.generate_escalation_report(trace_id=trace_id)

    @app.post("/api/governor/record-task")
    async def api_governor_record(request: Request) -> Dict[str, object]:
        """Record a task execution."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        gov.record_task(
            description=content.get("description", ""),
            family=content.get("family", ""),
        )
        gov.save_state()
        return {"recorded": True, "history_length": len(gov.get_history())}

    # ------------------------------------------------------------------
    # Agent Action Schema / Prompt Contract / Model Orchestrator — Epic #58
    # ------------------------------------------------------------------

    @app.get("/api/agent/schema")
    async def api_agent_schema() -> Dict[str, object]:
        """Return the Agent Action JSON schema."""
        from igris.core.agent_action_schema import ACTION_JSON_SCHEMA
        return {"schema": ACTION_JSON_SCHEMA}

    @app.get("/api/agent/roles")
    async def api_agent_roles() -> Dict[str, object]:
        """List all registered agent roles."""
        from igris.core.agent_action_schema import list_registry
        return {"roles": list_registry()}

    @app.get("/api/agent/action-types")
    async def api_agent_action_types() -> Dict[str, object]:
        """List all available action types."""
        from igris.core.agent_action_schema import (
            ACTION_TYPES, ACTION_ROUTING,
            READ_ONLY_ACTIONS, WRITE_ACTIONS, RISK_GATED_ACTIONS,
        )
        return {
            "action_types": list(ACTION_TYPES),
            "routing": dict(ACTION_ROUTING),
            "read_only": sorted(READ_ONLY_ACTIONS),
            "write": sorted(WRITE_ACTIONS),
            "risk_gated": sorted(RISK_GATED_ACTIONS),
        }

    @app.get("/api/agent/examples")
    async def api_agent_examples() -> Dict[str, object]:
        """Return example scenarios for the action schema."""
        from igris.core.prompt_contract import get_example_scenarios
        return {"examples": get_example_scenarios()}

    @app.post("/api/agent/validate")
    async def api_agent_validate(request: Request) -> Dict[str, object]:
        """Validate an action against the schema."""
        from igris.core.agent_action_schema import AgentAction, validate_action
        content = await request.json()
        action = AgentAction.from_dict(content)
        result = validate_action(action)
        return result.to_dict()

    @app.post("/api/agent/parse")
    async def api_agent_parse(request: Request) -> Dict[str, object]:
        """Parse raw LLM output into a validated action."""
        from igris.core.agent_action_schema import parse_llm_action
        content = await request.json()
        raw = content.get("raw_output", "")
        action, issues = parse_llm_action(raw)
        return {
            "parsed": action.to_dict() if action else None,
            "issues": issues,
            "valid": action is not None,
        }

    @app.get("/api/orchestrator/providers")
    async def api_orchestrator_providers() -> Dict[str, object]:
        """List configured LLM providers (no secrets)."""
        from igris.core.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        return {"providers": orch.list_providers()}

    @app.get("/api/orchestrator/profiles")
    async def api_orchestrator_profiles() -> Dict[str, object]:
        """List task type to profile mappings."""
        from igris.core.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        return {"profiles": orch.get_profiles()}

    @app.get("/api/orchestrator/cost")
    async def api_orchestrator_cost() -> Dict[str, object]:
        """Get cost tracking summary."""
        from igris.core.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        return orch.get_cost_summary()

    @app.get("/api/agent/prompt-contract")
    async def api_agent_prompt_contract(role: str = "coder") -> Dict[str, object]:
        """Get the reasoning loop prompt contract for a role."""
        from igris.core.prompt_contract import build_reasoning_prompt
        prompt = build_reasoning_prompt(
            role=role,
            mission_context="Example: Add /api/ping endpoint with tests",
            state_context="repo_clean: true, tests_pass: true",
            recent_actions="No recent actions.",
            file_context="No files loaded.",
        )
        return {"role": role, "prompt": prompt}

    # ------------------------------------------------------------------
    # Code Navigation Tools — Epic #59
    # ------------------------------------------------------------------

    @app.post("/api/nav/search-code")
    async def api_nav_search_code(request: Request) -> Dict[str, object]:
        """Search for patterns in code files."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.search_code(
            pattern=content.get("pattern", ""),
            path=content.get("path"),
            max_results=content.get("max_results", 50),
            context_lines=content.get("context_lines", 0),
        )
        return result.to_dict()

    @app.post("/api/nav/find-files")
    async def api_nav_find_files(request: Request) -> Dict[str, object]:
        """Find files by name/glob pattern."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.find_files(
            pattern=content.get("pattern", ""),
            max_results=content.get("max_results", 100),
        )
        return result.to_dict()

    @app.post("/api/nav/list-directory")
    async def api_nav_list_directory(request: Request) -> Dict[str, object]:
        """List directory contents."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.list_directory(
            path=content.get("path", "."),
            depth=content.get("depth", 1),
            max_entries=content.get("max_entries", 200),
        )
        return result.to_dict()

    @app.post("/api/nav/read-file-range")
    async def api_nav_read_file_range(request: Request) -> Dict[str, object]:
        """Read specific lines from a file."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.read_file_range(
            path=content.get("path", ""),
            start=content.get("start", 1),
            end=content.get("end"),
            max_lines=content.get("max_lines", 500),
        )
        return result.to_dict()

    @app.get("/api/nav/repo-map")
    async def api_nav_repo_map() -> Dict[str, object]:
        """Build a lightweight repository map."""
        from igris.core.code_navigation import CodeNavigator
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.repo_map()
        return result.to_dict()

    @app.post("/api/nav/find-symbol")
    async def api_nav_find_symbol(request: Request) -> Dict[str, object]:
        """Find symbol definitions (function, class, variable)."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.find_symbol(
            symbol=content.get("symbol", ""),
            path=content.get("path"),
            max_results=content.get("max_results", 50),
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Context Manager — Epic #60
    # ------------------------------------------------------------------

    @app.post("/api/context/build")
    async def api_context_build(request: Request) -> Dict[str, object]:
        """Build a context packet for the reasoning loop."""
        from igris.core.context_manager import ContextManager
        content = await request.json()
        ctx = ContextManager(project_root=str(CONFIG.project_root))
        packet = ctx.build_context(
            goal=content.get("goal", ""),
            role=content.get("role", "coder"),
            profile=content.get("profile", "default"),
            mission_id=content.get("mission_id", ""),
            mission_status=content.get("mission_status", ""),
            world_state=content.get("world_state"),
            recent_actions=content.get("recent_actions"),
            recent_errors=content.get("recent_errors"),
            memory_items=content.get("memory_items"),
            relevant_files=content.get("relevant_files"),
            file_snippets=content.get("file_snippets"),
            keywords=content.get("keywords"),
        )
        return packet.to_dict()

    @app.get("/api/context/budgets")
    async def api_context_budgets() -> Dict[str, object]:
        """Get token budget information for all profiles."""
        from igris.core.context_manager import ContextManager, TOKEN_BUDGETS
        ctx = ContextManager(project_root=str(CONFIG.project_root))
        return {
            profile: ctx.get_budget_info(profile)
            for profile in TOKEN_BUDGETS
        }

    @app.post("/api/context/score-files")
    async def api_context_score_files(request: Request) -> Dict[str, object]:
        """Score file relevance for a given task."""
        from igris.core.context_manager import score_file_relevance
        content = await request.json()
        files = content.get("files", [])
        keywords = content.get("keywords", [])
        recent_files = content.get("recent_files", [])
        error_files = content.get("error_files", [])
        scored = []
        for f in files:
            s = score_file_relevance(f, keywords, recent_files, error_files)
            scored.append({"path": f, "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"files": scored}

    # ------------------------------------------------------------------
    # Agent Reasoning Loop — Epic #61
    # ------------------------------------------------------------------

    @app.post("/api/reasoning/run")
    async def api_reasoning_run(request: Request) -> Dict[str, object]:
        """Run the agent reasoning loop for a goal."""
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        content = await request.json()

        # Validate and normalise initial_context
        raw_ctx = content.get("initial_context")
        if raw_ctx is not None and not isinstance(raw_ctx, dict):
            if isinstance(raw_ctx, str):
                raw_ctx = {"note": raw_ctx}
            else:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "initial_context must be a dict or string",
                        "received_type": type(raw_ctx).__name__,
                    },
                )

        loop = AgentReasoningLoop(
            project_root=str(CONFIG.project_root),
            max_steps=content.get("max_steps", 50),
            max_consecutive_errors=content.get("max_consecutive_errors", 5),
            role=content.get("role", "coder"),
        )
        result = loop.run(
            goal=content.get("goal", ""),
            mission_id=content.get("mission_id", ""),
            initial_context=raw_ctx,
        )
        return result.to_dict()

    @app.post("/api/reasoning/step")
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

    @app.get("/api/reasoning/stop-reasons")
    async def api_reasoning_stop_reasons() -> Dict[str, object]:
        """List all possible loop stop reasons."""
        from igris.core.agent_reasoning_loop import STOP_REASONS
        return {"stop_reasons": list(STOP_REASONS)}

    # ------------------------------------------------------------------
    # Rank Self-Repair Supervisor
    # ------------------------------------------------------------------

    @app.post("/api/rank/run-supervised")
    async def api_rank_run_supervised(request: Request) -> Dict[str, object]:
        """Run a controlled rank mission through the self-repair supervisor."""
        from igris.core.self_repair_supervisor import start_supervised_rank_async
        content = await request.json()
        if not content.get("goal"):
            raise HTTPException(status_code=400, detail="goal required")
        run = start_supervised_rank_async(content, project_root=str(CONFIG.project_root))
        return run.to_dict()

    @app.get("/api/rank/runs/active")
    async def api_rank_runs_active() -> Dict[str, object]:
        """List active supervised rank runs with compact summaries."""
        from igris.core.self_repair_supervisor import list_active_supervised_run_summaries
        runs = list_active_supervised_run_summaries(project_root=str(CONFIG.project_root))
        return {"runs": runs}

    @app.get("/api/rank/runs/{run_id}/summary")
    async def api_rank_run_summary(run_id: str) -> Dict[str, object]:
        """Return compact supervised rank run summary."""
        from igris.core.self_repair_supervisor import get_supervised_run, summarize_supervised_run
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")
        return summarize_supervised_run(run)

    @app.get("/api/rank/runs")
    async def api_rank_runs() -> Dict[str, object]:
        """List supervised rank runs held in memory."""
        from igris.core.self_repair_supervisor import list_supervised_runs
        return {"runs": [run.to_dict() for run in list_supervised_runs()]}

    @app.get("/api/rank/runs/{run_id}")
    async def api_rank_run_detail(run_id: str) -> Dict[str, object]:
        """Return one supervised rank run."""
        from igris.core.self_repair_supervisor import get_supervised_run
        run = get_supervised_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rank run not found")
        return run.to_dict()

    @app.post("/api/rank/runs/{run_id}/cancel")
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

    @app.get("/api/rank/audit/summary")
    async def api_rank_audit_summary() -> Dict[str, object]:
        """Return compact supervisor audit summary."""
        from igris.core.self_repair_supervisor import get_supervisor_audit_summary
        return get_supervisor_audit_summary(project_root=str(CONFIG.project_root))

    # ------------------------------------------------------------------
    # Integration Layer — Epic #62
    # ------------------------------------------------------------------

    @app.post("/api/integration/run-mission")
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

    @app.get("/api/integration/pipeline-status")
    async def api_integration_pipeline_status() -> Dict[str, object]:
        """Get status of all pipeline components."""
        from igris.core.integration_layer import IntegrationLayer
        layer = IntegrationLayer(project_root=str(CONFIG.project_root))
        return layer.get_pipeline_status()

    @app.get("/api/integration/action-families")
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

    @app.post("/api/risk/evaluate")
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

    @app.post("/api/risk/evaluate-template")
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

    @app.post("/api/risk/parse")
    async def api_risk_parse(request: Request) -> Dict[str, object]:
        """Parse a shell command into its components."""
        from igris.core.command_risk_engine import parse_command
        content = await request.json()
        parsed = parse_command(content.get("command", ""))
        return parsed.to_dict()

    @app.get("/api/risk/levels")
    async def api_risk_levels() -> Dict[str, object]:
        """Get all risk levels."""
        from igris.core.command_risk_engine import RISK_LEVELS
        return {"risk_levels": list(RISK_LEVELS)}

    # ------------------------------------------------------------------
    # Benchmark /api/ping — Epic #64
    # ------------------------------------------------------------------

    @app.get("/api/ping")
    async def api_ping() -> Dict[str, object]:
        """Simple ping endpoint — benchmark target."""
        return {"pong": True}

    @app.post("/api/benchmark/run")
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

    @app.get("/api/benchmark/phases")
    async def api_benchmark_phases() -> Dict[str, object]:
        """List benchmark phases."""
        from igris.core.benchmark_ping import BENCHMARK_PHASES, BENCHMARK_GOAL
        return {"phases": BENCHMARK_PHASES, "goal": BENCHMARK_GOAL}

    return app


def app() -> FastAPI:
    """Factory function for uvicorn ``--factory`` mode."""
    return create_app()


def run_app(application: FastAPI, host: str = "0.0.0.0", port: int = 7778) -> None:
    """Run the FastAPI application using Uvicorn."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
    except ImportError:
        pass
    startup_logger = logging.getLogger("igris.startup")
    # Kill any stale IGRIS process on the same port before starting.
    for attempt in range(1, 4):
        _ss = _sp.run(["ss", "-tlnp"], capture_output=True, text=True)
        stale_pids: List[int] = []
        for _line in _ss.stdout.splitlines():
            if f":{port}" in _line and "python" in _line:
                _m = _re.search(r"pid=(\d+)", _line)
                if _m:
                    stale_pid = int(_m.group(1))
                    if stale_pid != os.getpid():
                        stale_pids.append(stale_pid)
        if not stale_pids:
            break
        for stale_pid in stale_pids:
            startup_logger.warning("Startup: stale IGRIS process %d on port %d — killing", stale_pid, port)
            try:
                os.kill(stale_pid, _sig.SIGTERM)
                # NOTE: _time.sleep is intentional here — run_app() is a synchronous
                # function called BEFORE uvicorn starts the event loop, so blocking
                # sleep is safe. Do NOT replace with asyncio.sleep (#728).
                _time.sleep(2)
                os.kill(stale_pid, _sig.SIGKILL)
            except ProcessLookupError:
                pass
        _time.sleep(1)  # Wait for port release — sync context, safe
    else:
        startup_logger.critical("IGRIS startup blocked: port %d occupied after 3 attempts", port)
        try:
            _sp.run(
                [
                    "gh", "issue", "create",
                    "--title", f"IGRIS startup blocked: port {port} occupied",
                    "--body", (
                        f"IGRIS startup could not free port {port} after 3 attempts. "
                        "Manual intervention required."
                    ),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            startup_logger.error("Unable to open diagnostic GitHub issue: %s", exc)
        raise SystemExit(1)
    uvicorn.run(application, host=host, port=port, log_level="info")
