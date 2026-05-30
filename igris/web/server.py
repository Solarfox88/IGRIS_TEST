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
                    # Issues with "no-decompose" label are leaf sub-issues created by
                    # a prior decomposition pass — they must be implemented directly,
                    # never decomposed further. Respect this by disabling auto-subissues.
                    _issue_labels = [
                        l.get("name", "").lower()
                        for l in (issue.get("labels") or [])
                    ]
                    _is_leaf_subissue = "no-decompose" in _issue_labels
                    if _is_leaf_subissue:
                        _watchdog_logger.info(
                            "Watchdog: issue #%d has no-decompose label — disabling auto-subissues",
                            number,
                        )
                    launched = start_supervised_rank_async(
                        {
                            "goal": goal,
                            "github_issue": number,
                            "allow_merge_if_green": True,
                            "allow_auto_subissues": (
                                not _is_leaf_subissue
                                and str(os.getenv("IGRIS_ALLOW_AUTO_SUBISSUES_DEFAULT", "true")).strip().lower()
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
    project_root = str(CONFIG.project_root)
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
    """Create and configure the FastAPI application.

    Shared app state (jinja_env, sessions, task_engine, etc.) is built here
    and injected into each router via a ``deps`` namespace, preserving
    per-call isolation for tests.  Routes live in igris/web/routers/.
    """
    from types import SimpleNamespace
    import urllib.request
    import urllib.error
    from igris.web.security import apply_security_middleware

    app = FastAPI(title="IGRIS_GPT", version="0.1.0", lifespan=_lifespan)
    apply_security_middleware(app)

    # ---- Static files ----
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---- Shared state (isolated per create_app() call for test safety) ----
    _graph_instance: list = [None]

    def _get_graph() -> MemoryGraph:
        if _graph_instance[0] is None:
            _graph_instance[0] = MemoryGraph(str(CONFIG.project_root))
            _graph_instance[0].migrate_legacy(str(CONFIG.project_root))
        return _graph_instance[0]

    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    sessions: Dict[str, List[Dict[str, str]]] = {}
    task_engine = TaskEngine()
    build_default_registry()
    nonlocal_test_running: Dict[str, bool] = {"running": False}
    nonlocal_cmd_running: Dict[str, bool] = {"running": False}

    def _redact(text: str) -> str:
        return safety.redact_secrets(text)

    def _check_model_available(model_name: str) -> bool:
        """Check if a specific model is available in Ollama."""
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

    deps = SimpleNamespace(
        jinja_env=jinja_env,
        sessions=sessions,
        task_engine=task_engine,
        get_graph=_get_graph,
        nonlocal_test_running=nonlocal_test_running,
        nonlocal_cmd_running=nonlocal_cmd_running,
        check_model_available=_check_model_available,
        redact=_redact,
    )

    # ---- Register route modules ----
    from igris.web.routers import (
        routes_01, routes_02, routes_03, routes_04, routes_05,
        routes_06, routes_07, routes_08, routes_09, routes_10,
    )
    for _mod in (
        routes_01, routes_02, routes_03, routes_04, routes_05,
        routes_06, routes_07, routes_08, routes_09, routes_10,
    ):
        app.include_router(_mod.create_router(deps))

    # ---- Register modular API routers (igris/api/) ----
    try:
        from igris.api.routes.github_admin import router as _github_admin_router
        app.include_router(_github_admin_router)
    except Exception:
        pass  # best-effort — never block app startup

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
