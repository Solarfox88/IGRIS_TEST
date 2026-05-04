"""
Autonomous execution loop MVP for IGRIS_GPT.

Implements a safety-first semi-autonomous loop:
  mission/task -> select -> propose action -> execute safe command
  or patch proposal -> report -> outcome -> next recommendation

All actions are bounded by max_steps, safety checks and family
saturation.  No auto-commit, no auto-push, no unsafe patch apply.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from igris.core import decision_memory
from igris.core.outcome_router import route_outcome
from igris.core.safety import redact_secrets
from igris.core.task_selection import select_next_task
from igris.core.teacher import build_teacher_payload, propose_remediation_task
from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
from igris.models.task import TaskStatus


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LoopStepResult:
    step_number: int = 0
    task_id: Optional[int] = None
    task_title: str = ""
    action_type: str = ""  # select_task | execute_command | propose_patch | remediation | skip | stop
    action_detail: str = ""
    outcome: str = ""  # success | failure | blocked | skipped | stopped
    reason: str = ""
    recommendation: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["task_title"] = redact_secrets(d.get("task_title", ""))
        d["action_detail"] = redact_secrets(d.get("action_detail", ""))
        d["reason"] = redact_secrets(d.get("reason", ""))
        return d


@dataclass
class LoopStatus:
    running: bool = False
    current_step: int = 0
    max_steps: int = 0
    steps_completed: int = 0
    stopped_reason: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loop state (singleton per process)
# ---------------------------------------------------------------------------

_loop_status = LoopStatus()
_recent_results: List[Dict[str, Any]] = []
_MAX_RECENT = 50


def get_loop_status() -> LoopStatus:
    return _loop_status


def get_recent_steps(limit: int = 20) -> List[Dict[str, Any]]:
    return _recent_results[-limit:]


# ---------------------------------------------------------------------------
# Single step execution
# ---------------------------------------------------------------------------

def execute_step(
    task_engine: Any,
    step_number: int = 0,
    project_root: Optional[str] = None,
) -> LoopStepResult:
    """Execute a single loop step.

    1. Select next task
    2. Decide safe action (command_id or patch proposal)
    3. Execute if safe
    4. Create report
    5. Route outcome
    6. Record memory events
    7. Return result
    """
    result = LoopStepResult(step_number=step_number)
    pr = project_root

    # 1. Select next task
    pending_tasks = task_engine.list_tasks(status="pending")
    if not pending_tasks:
        result.action_type = "stop"
        result.outcome = "stopped"
        result.reason = "No pending tasks available"
        _record_step(result)
        return result

    history = [t.description for t in task_engine.tasks]
    selection = select_next_task(
        candidate_tasks=pending_tasks,
        history=history,
        project_root=pr,
    )

    if not selection.selected_task:
        result.action_type = "stop"
        result.outcome = "stopped"
        result.reason = selection.fallback_reason or "No suitable task found"
        _record_step(result)
        return result

    task = selection.selected_task
    result.task_id = task.id
    result.task_title = task.title or task.description

    # 2. Check safety
    if task.risk == "high":
        result.action_type = "skip"
        result.outcome = "skipped"
        result.reason = "Task risk is high — requires manual review"
        decision_memory.record_decision(
            title=f"Skipped high-risk task: {task.title or task.description}",
            family=task.family or "", task_id=str(task.id),
            outcome="skipped", reason="high risk", project_root=pr,
        )
        _record_step(result)
        return result

    # 3. Check family saturation via memory
    family = task.family or "other"
    if decision_memory.should_avoid_family(family, project_root=pr):
        result.action_type = "skip"
        result.outcome = "blocked"
        result.reason = f"Family '{family}' should be avoided (saturated or repeated failures)"
        task_engine.block_task(task.id, reason=result.reason)
        decision_memory.record_decision(
            title=f"Blocked task in saturated family: {family}",
            family=family, task_id=str(task.id),
            outcome="blocked", reason=result.reason, project_root=pr,
        )
        _record_step(result)
        return result

    # 4. Decide action type based on task family
    action_type, command_id = _decide_action(task)
    result.action_type = action_type

    if action_type == "execute_command" and command_id:
        result.action_detail = f"command_id={command_id}"
        # Execute via safe command runner
        from igris.layers.execution.runner import run_safe_command
        cmd_result = run_safe_command(command_id)
        success = cmd_result.get("returncode", 1) == 0
        stdout = redact_secrets(cmd_result.get("stdout", ""))
        stderr = redact_secrets(cmd_result.get("stderr", ""))

        report = {
            "command_id": command_id,
            "success": success,
            "stdout_truncated": stdout[:500],
            "stderr_truncated": stderr[:500],
            "task_id": task.id,
        }
        task_engine.append_timeline_event({
            "type": "loop", "title": f"Loop step {step_number}: {command_id}",
            "detail": f"success={success}", "severity": "info" if success else "warning",
        })

        recommendation = route_outcome(report, task.description, history)
        result.recommendation = recommendation
        result.outcome = "success" if success else "failure"

        if success:
            task_engine.complete_task(task.id, result=f"Command {command_id} succeeded")
            decision_memory.record_decision(
                title=f"Executed {command_id} successfully",
                family=family, task_id=str(task.id),
                outcome="success", project_root=pr,
            )
        else:
            decision_memory.record_failure(
                title=f"Command {command_id} failed",
                family=family, task_id=str(task.id),
                reason=stderr[:200], project_root=pr,
            )
            if recommendation.get("should_call_teacher"):
                teacher_payload = build_teacher_payload(
                    history, project_root=pr,
                    last_execution_report=report,
                )
                remediation = propose_remediation_task(teacher_payload)
                task_engine.create_task(
                    description=remediation.get("task_description", "Remediation needed"),
                    title=remediation.get("task_title", "Remediation"),
                    family=remediation.get("family", "fix"),
                    source="loop_remediation",
                    success_criteria=remediation.get("success_criteria", []),
                )
                result.action_detail += " → remediation created"

    elif action_type == "propose_patch":
        result.action_detail = "Patch proposal recommended (manual review required)"
        result.outcome = "skipped"
        result.reason = "Auto-patch not supported in loop — propose via UI"
        decision_memory.record_decision(
            title=f"Deferred patch proposal for task {task.id}",
            family=family, task_id=str(task.id),
            outcome="skipped", reason="patch proposals require manual review",
            project_root=pr,
        )

    else:
        result.action_type = "skip"
        result.outcome = "skipped"
        result.reason = "No safe automatic action available for this task"
        decision_memory.record_decision(
            title=f"No auto action for task: {task.title or task.description}",
            family=family, task_id=str(task.id),
            outcome="skipped", reason="no safe auto action", project_root=pr,
        )

    _record_step(result)
    return result


# ---------------------------------------------------------------------------
# Multi-step run
# ---------------------------------------------------------------------------

def run_loop(
    task_engine: Any,
    max_steps: int = 1,
    project_root: Optional[str] = None,
) -> LoopStatus:
    """Run the loop for up to max_steps steps."""
    global _loop_status

    if max_steps < 1:
        max_steps = 1
    if max_steps > 100:
        max_steps = 100

    _loop_status = LoopStatus(
        running=True,
        max_steps=max_steps,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    for step_num in range(1, max_steps + 1):
        _loop_status.current_step = step_num
        step_result = execute_step(task_engine, step_number=step_num, project_root=project_root)
        _loop_status.steps.append(step_result.to_dict())
        _loop_status.steps_completed = step_num

        if step_result.outcome in ("stopped",):
            _loop_status.stopped_reason = step_result.reason
            break

        # Safety: stop on consecutive failures
        recent = _loop_status.steps[-3:]
        if len(recent) >= 3 and all(s.get("outcome") == "failure" for s in recent):
            _loop_status.stopped_reason = "Stopped: 3 consecutive failures"
            decision_memory.record_failure(
                title="Loop stopped: consecutive failures",
                reason="3 consecutive failures in loop",
                project_root=project_root,
            )
            break

        # Safety: stop if all tasks blocked/skipped
        if step_result.outcome in ("blocked", "skipped"):
            skip_count = sum(1 for s in _loop_status.steps if s.get("outcome") in ("blocked", "skipped"))
            if skip_count >= 3:
                _loop_status.stopped_reason = "Stopped: too many blocked/skipped steps"
                break

    _loop_status.running = False
    _loop_status.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    task_engine.append_timeline_event({
        "type": "loop",
        "title": f"Loop completed: {_loop_status.steps_completed}/{max_steps} steps",
        "detail": _loop_status.stopped_reason or "completed normally",
        "severity": "info",
    })

    return _loop_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decide_action(task: Any) -> tuple:
    """Decide the best safe action for a task.

    Returns (action_type, command_id_or_none).
    """
    family = task.family or "other"
    desc = (task.description or "").lower()

    if family == "test" or "test" in desc:
        return ("execute_command", "run_tests")

    if family in ("analyze", "other") and ("list" in desc or "file" in desc):
        return ("execute_command", "list_files")

    if family in ("git", "config") and ("status" in desc or "git" in desc):
        return ("execute_command", "git_status")

    if family in ("code", "fix", "refactor", "docs"):
        return ("propose_patch", None)

    return ("skip", None)


def _record_step(result: LoopStepResult) -> None:
    """Record step to recent results list."""
    global _recent_results
    _recent_results.append(result.to_dict())
    if len(_recent_results) > _MAX_RECENT:
        _recent_results = _recent_results[-_MAX_RECENT:]
