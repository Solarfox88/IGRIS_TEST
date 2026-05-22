"""Subprocess entrypoint for bounded supervisor reasoning runs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from igris.core.agent_reasoning_loop import AgentReasoningLoop

_HEARTBEAT_INTERVAL_SECONDS = 30


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
    try:
        from igris.core.memory_graph import MemoryGraph
        _mg = MemoryGraph(project_root)
        _mg.flush_session_memory(result.loop_id, getattr(loop, "_memory_items", []))
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

    result_dict = result.to_dict()
    result_dict["heartbeat_path"] = heartbeat_path
    result_dict["steps_completed"] = result.total_steps
    print(json.dumps(result_dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
