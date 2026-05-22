"""Mission Controller for IGRIS_GPT — Epic #40.

Manages persistent, long-running, replayable missions with full lifecycle:

    create → plan → materialize → execute → observe → replan → verify → report

Missions survive restarts, prevent duplicate execution, support pause/resume,
and always produce an explainable state + final report.

Uses the existing :mod:`igris.core.mission_planner` for plan generation and
:mod:`igris.core.task_engine` for task execution, adding controller
orchestration on top.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets
from igris.core.work_session import DeliveryReport


# ---------------------------------------------------------------------------
# Mission statuses
# ---------------------------------------------------------------------------

MISSION_STATUSES = (
    "created",
    "planning",
    "planned",
    "executing",
    "blocked",
    "verifying",
    "paused",
    "done",
    "failed",
)


# ---------------------------------------------------------------------------
# Enhanced Mission schema
# ---------------------------------------------------------------------------

@dataclass
class MissionArtifact:
    """An artifact produced during a mission."""
    id: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:8]}")
    type: str = ""  # file | patch | report | log | test_result
    path: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "path": redact_secrets(self.path),
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionArtifact":
        return cls(
            id=data.get("id", f"art-{uuid.uuid4().hex[:8]}"),
            type=data.get("type", ""),
            path=data.get("path", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class ControlledMission:
    """Enhanced mission with full controller fields.

    Extends the basic Mission from mission_planner with workspace, targets,
    constraints, success criteria, risk level, artifacts, rollback, final
    report and trace ID.
    """
    id: str = field(default_factory=lambda: f"mission-{uuid.uuid4().hex[:12]}")
    title: str = ""
    goal: str = ""
    description: str = ""
    status: str = "created"
    workspace: str = ""
    target_hosts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low | medium | high | critical
    plan: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[MissionArtifact] = field(default_factory=list)
    rollback_plan: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    final_report: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    trace_id: str = field(default_factory=lambda: f"trace-{uuid.uuid4().hex[:8]}")
    paused_at: Optional[str] = None
    blocked_reason: Optional[str] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    work_session_id: Optional[str] = None

    def _touch(self) -> None:
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "description": self.description,
            "status": self.status,
            "workspace": self.workspace,
            "target_hosts": self.target_hosts,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "risk_level": self.risk_level,
            "plan": self.plan,
            "tasks": self.tasks,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "rollback_plan": self.rollback_plan,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "final_report": self.final_report,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "trace_id": self.trace_id,
            "paused_at": self.paused_at,
            "blocked_reason": self.blocked_reason,
            "execution_log": self.execution_log[-50:],
            "work_session_id": self.work_session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlledMission":
        artifacts = [MissionArtifact.from_dict(a) for a in data.get("artifacts", [])]
        return cls(
            id=data.get("id", f"mission-{uuid.uuid4().hex[:12]}"),
            title=data.get("title", ""),
            goal=data.get("goal", ""),
            description=data.get("description", ""),
            status=data.get("status", "created"),
            workspace=data.get("workspace", ""),
            target_hosts=data.get("target_hosts", []),
            constraints=data.get("constraints", []),
            success_criteria=data.get("success_criteria", []),
            risk_level=data.get("risk_level", "low"),
            plan=data.get("plan", []),
            tasks=data.get("tasks", []),
            artifacts=artifacts,
            rollback_plan=data.get("rollback_plan"),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            final_report=data.get("final_report"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            trace_id=data.get("trace_id", f"trace-{uuid.uuid4().hex[:8]}"),
            paused_at=data.get("paused_at"),
            blocked_reason=data.get("blocked_reason"),
            execution_log=data.get("execution_log", []),
            work_session_id=data.get("work_session_id"),
        )

    def explain_state(self) -> Dict[str, Any]:
        """Return a human-readable explanation of current state."""
        current_task = None
        if self.tasks and 0 <= self.current_step < len(self.tasks):
            current_task = self.tasks[self.current_step]

        completed = sum(1 for t in self.tasks if t.get("status") == "done")
        failed = sum(1 for t in self.tasks if t.get("status") == "failed")
        pending = sum(1 for t in self.tasks if t.get("status") in ("pending", "created"))

        return {
            "mission_id": self.id,
            "status": self.status,
            "progress": f"{completed}/{self.total_steps} steps completed",
            "current_step": self.current_step,
            "current_task": current_task,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "blocked_reason": self.blocked_reason,
            "next_action_explanation": self._explain_next_action(),
        }

    def _explain_next_action(self) -> str:
        if self.status == "done":
            return "Mission completed successfully."
        if self.status == "failed":
            return "Mission failed. Check final report for details."
        if self.status == "paused":
            return "Mission is paused. Resume to continue."
        if self.status == "blocked":
            return f"Mission blocked: {self.blocked_reason or 'unknown reason'}"
        if self.status == "verifying":
            return "Verifying success criteria before marking complete."
        if self.status in ("created", "planning"):
            return "Mission needs planning before execution."
        if not self.tasks:
            return "No tasks materialized yet. Generate plan first."
        if self.current_step >= len(self.tasks):
            return "All tasks completed. Ready for verification."
        task = self.tasks[self.current_step]
        return f"Next: step {self.current_step + 1}/{self.total_steps} — {task.get('title', 'untitled')}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _controller_dir(project_root: Optional[str] = None) -> Path:
    import os
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    d = root / ".igris" / "controller" / "missions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_controlled_mission(mission: ControlledMission, project_root: Optional[str] = None) -> Path:
    d = _controller_dir(project_root)
    path = d / f"{mission.id}.json"
    path.write_text(json.dumps(mission.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_controlled_mission(mission_id: str, project_root: Optional[str] = None) -> Optional[ControlledMission]:
    d = _controller_dir(project_root)
    path = d / f"{mission_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ControlledMission.from_dict(data)
    except Exception:
        return None


def list_controlled_missions(project_root: Optional[str] = None) -> List[ControlledMission]:
    d = _controller_dir(project_root)
    missions: List[ControlledMission] = []
    for fp in sorted(d.glob("mission-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            missions.append(ControlledMission.from_dict(data))
        except Exception:
            continue
    return missions


def delete_controlled_mission(mission_id: str, project_root: Optional[str] = None) -> bool:
    d = _controller_dir(project_root)
    path = d / f"{mission_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Mission Controller — lifecycle operations
# ---------------------------------------------------------------------------

class MissionController:
    """Orchestrates mission lifecycle.

    Provides create/plan/execute/pause/resume/block/verify/report operations.
    All state changes are persisted immediately.
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root

    def _save(self, mission: ControlledMission) -> None:
        save_controlled_mission(mission, self.project_root)

    def _log(self, mission: ControlledMission, event: str, detail: str = "") -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "detail": redact_secrets(detail),
            "step": mission.current_step,
            "status": mission.status,
        }
        mission.execution_log.append(entry)

    # -- Create --

    def create_mission(
        self,
        title: str,
        goal: str,
        description: str = "",
        workspace: str = "",
        target_hosts: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        risk_level: str = "low",
        rollback_plan: Optional[str] = None,
    ) -> ControlledMission:
        mission = ControlledMission(
            title=title,
            goal=goal,
            description=description or goal,
            workspace=workspace,
            target_hosts=target_hosts or [],
            constraints=constraints or [],
            success_criteria=success_criteria or [],
            risk_level=risk_level,
            rollback_plan=rollback_plan,
        )
        self._log(mission, "created", f"Mission '{title}' created")
        self._save(mission)
        return mission

    # -- Plan --

    def plan_mission(self, mission_id: str) -> Optional[ControlledMission]:
        """Generate a plan for the mission using deterministic planner."""
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        if mission.status not in ("created", "planned", "failed"):
            return mission  # Already in progress, don't re-plan

        mission.status = "planning"
        mission._touch()
        self._log(mission, "planning_started")

        # Use deterministic planner
        from igris.core.mission_planner import generate_plan, Mission as PlannerMission
        planner_mission = PlannerMission(
            id=mission.id,
            title=mission.title,
            description=mission.description,
        )
        steps = generate_plan(planner_mission)

        mission.plan = [s.to_dict() for s in steps]
        mission.tasks = []
        for s in steps:
            mission.tasks.append({
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "family": s.family,
                "status": "pending",
                "dependencies": s.dependencies,
                "success_criteria": s.success_criteria,
                "risk": s.risk,
                "order": s.order,
            })
        mission.total_steps = len(steps)
        mission.current_step = 0
        mission.status = "planned"
        mission._touch()
        self._log(mission, "planned", f"{len(steps)} steps generated")
        self._save(mission)
        return mission

    # -- Execute next step --

    def execute_next_step(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Advance to next step. Returns step info or None.

        This is intentionally limited: it marks the current step as
        'executing' and returns it.  Actual execution is handled by the
        caller (tool runtime / autonomous loop).
        """
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        if mission.status == "paused":
            return {"error": "Mission is paused. Resume first.", "mission_id": mission_id}
        if mission.status in ("done", "failed"):
            return {"error": f"Mission is {mission.status}.", "mission_id": mission_id}
        if mission.status == "blocked":
            return {"error": f"Mission is blocked: {mission.blocked_reason}", "mission_id": mission_id}
        if not mission.tasks:
            return {"error": "No tasks. Plan mission first.", "mission_id": mission_id}

        # Find next pending task (guard against duplicates)
        while mission.current_step < len(mission.tasks):
            task = mission.tasks[mission.current_step]
            if task.get("status") == "executing":
                return {
                    "warning": "Step already executing",
                    "step": mission.current_step,
                    "task": task,
                }
            if task.get("status") in ("pending", "created"):
                break
            mission.current_step += 1
        else:
            # All tasks completed
            mission.status = "verifying"
            mission._touch()
            self._log(mission, "all_steps_completed", "Moving to verification")
            self._save(mission)
            return {"status": "verifying", "mission_id": mission_id, "message": "All steps completed. Verifying."}

        task = mission.tasks[mission.current_step]
        task["status"] = "executing"
        task["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mission.status = "executing"
        mission._touch()
        self._log(mission, "step_started", f"Step {mission.current_step}: {task.get('title', '')}")
        self._save(mission)

        return {
            "status": "executing",
            "step": mission.current_step,
            "total_steps": mission.total_steps,
            "task": task,
            "mission_id": mission_id,
            "trace_id": mission.trace_id,
        }

    # -- Report step outcome --

    def report_step_outcome(
        self,
        mission_id: str,
        step_index: int,
        outcome: str,  # "success" | "failure" | "skipped"
        detail: str = "",
    ) -> Optional[ControlledMission]:
        """Record the outcome of a step and advance."""
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        if step_index < 0 or step_index >= len(mission.tasks):
            return mission

        task = mission.tasks[step_index]
        task["status"] = "done" if outcome == "success" else "failed" if outcome == "failure" else "skipped"
        task["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        task["outcome_detail"] = redact_secrets(detail)

        self._log(mission, f"step_{outcome}", f"Step {step_index}: {detail[:200]}")

        if outcome == "failure":
            consecutive_failures = 0
            for t in reversed(mission.tasks[:step_index + 1]):
                if t.get("status") == "failed":
                    consecutive_failures += 1
                else:
                    break
            if consecutive_failures >= 3:
                mission.status = "blocked"
                mission.blocked_reason = f"3 consecutive failures at step {step_index}"
                self._log(mission, "blocked", mission.blocked_reason)
            else:
                # Move to next step
                mission.current_step = step_index + 1
        else:
            mission.current_step = step_index + 1

        # Check if all done
        if mission.current_step >= len(mission.tasks) and mission.status != "blocked":
            mission.status = "verifying"
            self._log(mission, "all_steps_completed")

        mission._touch()
        self._save(mission)
        return mission

    # -- Pause / Resume --

    def pause_mission(self, mission_id: str, reason: str = "") -> Optional[ControlledMission]:
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        if mission.status in ("done", "failed"):
            return mission
        mission.status = "paused"
        mission.paused_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mission._touch()
        self._log(mission, "paused", reason)
        self._save(mission)
        return mission

    def resume_mission(self, mission_id: str) -> Optional[ControlledMission]:
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        if mission.status != "paused":
            return mission
        previous_status = "executing" if mission.tasks else "planned"
        mission.status = previous_status
        mission.paused_at = None
        mission._touch()
        self._log(mission, "resumed")
        self._save(mission)
        return mission

    # -- Block / Unblock --

    def block_mission(self, mission_id: str, reason: str) -> Optional[ControlledMission]:
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        mission.status = "blocked"
        mission.blocked_reason = reason
        mission._touch()
        self._log(mission, "blocked", reason)
        self._save(mission)
        return mission

    def unblock_mission(self, mission_id: str) -> Optional[ControlledMission]:
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        if mission.status != "blocked":
            return mission
        mission.status = "executing"
        mission.blocked_reason = None
        mission._touch()
        self._log(mission, "unblocked")
        self._save(mission)
        return mission

    # -- Verify --

    def verify_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Check success criteria. Returns verification result."""
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None

        completed = [t for t in mission.tasks if t.get("status") == "done"]
        failed = [t for t in mission.tasks if t.get("status") == "failed"]
        total = len(mission.tasks)

        all_done = len(completed) + len(failed) == total
        success_rate = len(completed) / total if total > 0 else 0
        criteria_met = success_rate >= 0.8 and len(failed) == 0

        result = {
            "mission_id": mission_id,
            "total_tasks": total,
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": round(success_rate, 2),
            "criteria_met": criteria_met,
            "all_tasks_done": all_done,
            "success_criteria": mission.success_criteria,
        }

        if criteria_met and all_done:
            mission.status = "done"
            self._log(mission, "verified_success", f"Success rate: {success_rate:.0%}")
        elif all_done and not criteria_met:
            mission.status = "failed"
            self._log(mission, "verified_failure", f"{len(failed)} tasks failed")
        # else: still verifying

        mission._touch()
        self._save(mission)
        result["final_status"] = mission.status
        return result

    # -- Final report --

    def generate_final_report(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Generate a final report for the mission."""
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None

        completed = [t for t in mission.tasks if t.get("status") == "done"]
        failed = [t for t in mission.tasks if t.get("status") == "failed"]
        skipped = [t for t in mission.tasks if t.get("status") == "skipped"]

        file_artifacts = [a.path for a in mission.artifacts if a.type == "file" and a.path]

        report: Dict[str, Any] = {
            "mission_id": mission.id,
            "title": mission.title,
            "goal": mission.goal,
            "status": mission.status,
            "trace_id": mission.trace_id,
            "created_at": mission.created_at,
            "completed_at": mission.updated_at,
            "total_tasks": len(mission.tasks),
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "skipped_tasks": len(skipped),
            "success_rate": round(len(completed) / len(mission.tasks), 2) if mission.tasks else 0,
            "artifacts": [a.to_dict() for a in mission.artifacts],
            "execution_summary": [],
            "risk_level": mission.risk_level,
            "rollback_plan": mission.rollback_plan,
            "constraints_respected": mission.constraints,
        }

        for t in mission.tasks:
            report["execution_summary"].append({
                "title": t.get("title", ""),
                "status": t.get("status", "unknown"),
                "family": t.get("family", ""),
                "outcome_detail": redact_secrets(t.get("outcome_detail", "")),
            })

        from dataclasses import asdict as _asdict
        report["delivery_report"] = _asdict(DeliveryReport(
            work_session_id=mission.work_session_id or "",
            goal=mission.goal,
            files_modified=file_artifacts,
            diff_summary="",
            test_output="",
            ci_status="unknown",
            pr_url="",
            pr_number=0,
            healthcheck_url="",
            residual_risks=[],
            rollback_available=bool(mission.rollback_plan),
            run_id=mission.trace_id,
            last_failure_class="",
            repair_cycles_used=0,
            capability_signals={},
        ))

        mission.final_report = report
        mission._touch()
        self._log(mission, "report_generated")
        self._save(mission)
        return report

    # -- Add artifact --

    def add_artifact(
        self,
        mission_id: str,
        artifact_type: str,
        path: str = "",
        description: str = "",
    ) -> Optional[ControlledMission]:
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None
        artifact = MissionArtifact(
            type=artifact_type,
            path=path,
            description=description,
        )
        mission.artifacts.append(artifact)
        mission._touch()
        self._log(mission, "artifact_added", f"{artifact_type}: {description[:100]}")
        self._save(mission)
        return mission

    # -- Context reconstruction --

    def reconstruct_context(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct mission context after a restart.

        Returns everything needed to resume work: current state, next step,
        recent log, pending tasks.
        """
        mission = load_controlled_mission(mission_id, self.project_root)
        if not mission:
            return None

        pending_tasks = [t for t in mission.tasks if t.get("status") in ("pending", "created")]
        executing_tasks = [t for t in mission.tasks if t.get("status") == "executing"]

        # If we find tasks stuck in 'executing', they were interrupted
        interrupted = len(executing_tasks) > 0

        return {
            "mission_id": mission.id,
            "status": mission.status,
            "interrupted": interrupted,
            "current_step": mission.current_step,
            "total_steps": mission.total_steps,
            "pending_tasks": len(pending_tasks),
            "executing_tasks": [t.get("title", "") for t in executing_tasks],
            "recent_log": mission.execution_log[-10:],
            "trace_id": mission.trace_id,
            "state_explanation": mission.explain_state(),
            "can_resume": mission.status in ("paused", "planned", "executing", "blocked"),
            "needs_replan": interrupted,
        }
