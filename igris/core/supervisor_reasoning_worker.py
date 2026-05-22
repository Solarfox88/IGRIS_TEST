"""Subprocess entrypoint for bounded supervisor reasoning runs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from igris.core.agent_reasoning_loop import AgentReasoningLoop
from igris.core.work_session import DeliveryReport, WorkPhase, WorkSession

_HEARTBEAT_INTERVAL_SECONDS = 30


def _phase_for_step(step_num: int, max_steps: int, action_type: str) -> WorkPhase:
    if step_num == 1:
        return WorkPhase.UNDERSTAND
    if step_num == 2:
        return WorkPhase.PLAN
    if step_num >= max_steps - 1:
        return WorkPhase.VERIFY
    if action_type in ("ci_fix", "repair", "fix"):
        return WorkPhase.FIX
    if action_type in ("observe", "check", "read"):
        return WorkPhase.OBSERVE
    return WorkPhase.ACT


def _heartbeat_writer(path: str, state: dict, stop_event: threading.Event) -> None:
    """Write periodic heartbeat so the supervisor can detect stalled workers."""
    while not stop_event.wait(_HEARTBEAT_INTERVAL_SECONDS):
        try:
            with open(path, "w") as f:
                json.dump({**state, "heartbeat_at": time.time()}, f)
        except OSError:
            pass


def main() -> int:
    payload = json.load(sys.stdin)
    project_root = str(payload["project_root"])
    heartbeat_path: str = str(payload.get("heartbeat_path") or "")

    state = {
        "pid": os.getpid(),
        "started_at": time.time(),
        "goal": str(payload.get("goal", ""))[:200],
        "max_steps": int(payload["max_steps"]),
        "steps_completed": 0,
        "heartbeat_at": time.time(),
    }
    ws = WorkSession.create(goal=str(payload["goal"]), mission_id=payload.get("mission_id") or None)

    stop_event = threading.Event()
    if heartbeat_path:
        hb_thread = threading.Thread(
            target=_heartbeat_writer,
            args=(heartbeat_path, state, stop_event),
            daemon=True,
        )
        hb_thread.start()
    else:
        hb_thread = None

    loop = AgentReasoningLoop(
        project_root=project_root,
        max_steps=int(payload["max_steps"]),
        task_type=str(payload.get("task_type") or "code_reasoning"),
        preferred_profile=payload.get("preferred_profile") or None,
    )
    progress_path = str(Path(project_root) / ".igris" / "reasoning_progress.json")

    def _write_progress(step_num: int, action_type: str) -> None:
        state["current_step"] = int(step_num)
        state["last_action_type"] = str(action_type or "unknown")
        try:
            phase = _phase_for_step(int(step_num), int(payload["max_steps"]), str(action_type or ""))
            ws.advance_phase(phase)
            if int(step_num) >= int(payload["max_steps"]) - 1:
                ws.advance_phase(WorkPhase.OBSERVE)
                ws.advance_phase(WorkPhase.VERIFY)
        except Exception:
            pass
        try:
            Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(progress_path).with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "loop_id": getattr(loop, "loop_id", ""),
                "goal": str(payload.get("goal", ""))[:200],
                "current_step": int(step_num),
                "last_action_type": str(action_type or "unknown"),
                "timestamp": time.time(),
                "max_steps": int(payload["max_steps"]),
            }, indent=2), encoding="utf-8")
            tmp.replace(progress_path)
        except OSError:
            pass

    result = loop.run(
        goal=str(payload["goal"]),
        initial_context=dict(payload.get("initial_context") or {}),
        step_callback=_write_progress,
    )
    result_dict = result.to_dict()
    report = DeliveryReport(
        work_session_id=ws.session_id,
        goal=ws.goal,
        files_modified=list(getattr(result, "files_modified", []) or []),
        diff_summary=getattr(result, "diff_summary", "") or "",
        test_output=getattr(result, "test_output", "") or "",
        ci_status=getattr(result, "ci_status", "") or "unknown",
        pr_url=getattr(result, "pr_url", "") or "",
        pr_number=int(getattr(result, "pr_number", 0) or 0),
        healthcheck_url="",
        residual_risks=list(getattr(result, "residual_risks", []) or []),
        rollback_available=bool(getattr(result, "rollback_available", False)),
        run_id=result_dict.get("run_id", ""),
        last_failure_class=result_dict.get("last_failure_class", ""),
        repair_cycles_used=int(result_dict.get("repair_cycles_used", 0)),
        capability_signals=dict(result_dict.get("capability_signals") or {}),
    )
    ws.advance_phase(WorkPhase.DELIVER, outcome="success")
    ws.complete_deliver(report)
    ws.advance_phase(WorkPhase.REMEMBER)
    ws.remember(project_root)
    try:
        from igris.core.memory import _get_graph
        _mg = _get_graph()
        _mg.flush_session_memory(result.loop_id, getattr(loop, "_memory_items", []))
    except Exception:
        pass
    try:
        cfg_path = Path(project_root) / ".igris" / "memory_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        from igris.core.memory_validator import MemoryValidator

        MemoryValidator(project_root).run(
            half_life_days=float(cfg.get("half_life_days", 14.0)),
            max_age_days=float(cfg.get("max_age_days", 30.0)),
        )
    except Exception:
        pass

    stop_event.set()
    if hb_thread:
        hb_thread.join(timeout=2)
    if result.status == "finished":
        try:
            Path(progress_path).unlink(missing_ok=True)
        except OSError:
            pass

    result_dict["heartbeat_path"] = heartbeat_path
    result_dict["steps_completed"] = result.total_steps
    result_dict["work_session_id"] = ws.session_id
    print(json.dumps(result_dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
