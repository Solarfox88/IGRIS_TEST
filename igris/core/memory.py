from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from igris.core.memory_graph import MemoryGraph

_graph_instance: MemoryGraph | None = None

def _get_graph() -> MemoryGraph:
    global _graph_instance
    if _graph_instance is None:
        root = os.environ.get("PROJECT_ROOT", str(Path.cwd()))
        _graph_instance = MemoryGraph(root)
        _graph_instance.migrate_legacy(root)
    return _graph_instance

def read_memory(namespace: str) -> Any:
    for n in _get_graph().get_project_facts():
        c = n.get("content", {})
        if c.get("namespace") == namespace:
            return c.get("data")
    return None

def write_memory(namespace: str, data: Any) -> None:
    g = _get_graph()
    facts = g.get_project_facts()
    for n in facts:
        c = n.get("content", {})
        if c.get("namespace") == namespace:
            g.update_node(n["node_id"], content={"namespace": namespace, "data": data})
            return
    g.add_node("project_fact", {"namespace": namespace, "data": data})

def append_memory_event(namespace: str, event: Dict[str, Any]) -> None:
    _get_graph().add_node("run_event", {"namespace": namespace, **event})

def recent_memory_events(namespace: str, limit: int = 20) -> List[Dict[str, Any]]:
    rows = _get_graph().query_by_intent(namespace, node_type="run_event", limit=200)
    filtered = [r for r in rows if r.get("content", {}).get("namespace") == namespace]
    # Take most recent N, return in ascending order (oldest first) — backward compat
    recent = sorted(filtered, key=lambda x: x.get("created_at", 0), reverse=True)[:limit]
    recent.sort(key=lambda x: x.get("created_at", 0))
    return [{k: v for k, v in r["content"].items() if k != "namespace"} for r in recent]
