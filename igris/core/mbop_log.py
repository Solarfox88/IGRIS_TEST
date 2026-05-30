"""MBOP event log reader for .igris/mbop_events.jsonl."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def log_path(project_root: str) -> Path:
    return Path(project_root) / ".igris" / "mbop_events.jsonl"

def read_all(project_root: str) -> List[Dict[str, Any]]:
    path = log_path(project_root)
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return events

def read_for_run(project_root: str, run_id: str) -> List[Dict[str, Any]]:
    return [e for e in read_all(project_root) if e.get("run_id") == run_id]

def read_for_issue(project_root: str, issue_number: int) -> List[Dict[str, Any]]:
    return [e for e in read_all(project_root) if e.get("issue_number") == issue_number]

def last_n(project_root: str, n: int = 20) -> List[Dict[str, Any]]:
    return read_all(project_root)[-n:]

def phases_for_run(project_root: str, run_id: str) -> List[str]:
    return [e.get("phase", "") for e in read_for_run(project_root, run_id)]

def summary_for_run(project_root: str, run_id: str) -> Optional[Dict[str, Any]]:
    events = read_for_run(project_root, run_id)
    if not events:
        return None
    phases = {e.get("phase", ""): e.get("status", "") for e in events}
    return {"run_id": run_id, "issue_number": events[0].get("issue_number"),
            "event_count": len(events), "first_ts": events[0].get("ts"),
            "last_ts": events[-1].get("ts"), "phases": phases}
