from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

NODE_TYPES = {
    "identity_fact", "project_fact", "command_recipe", "lesson",
    "decision", "run_event", "capability", "environment_fact",
    "world_state_snapshot",
}

EDGE_TYPES = {
    "learned_from", "applies_to_project", "uses_command", "fixed_by",
    "supersedes", "related_to", "requires_approval", "same_category",
    "triggered_by_intent",
}

_SECRET_RE = re.compile(r"(?i)(token|secret|password|api_key)\s*[=:]\s*\S+")
_LEGACY_SECRET_RE = re.compile(r"(?i)(token|secret|password|key).*?=\S+")


class MemoryGraph:
    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root)
        mem_dir = self.project_root / ".igris" / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = mem_dir / "graph.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
CREATE TABLE IF NOT EXISTS memory_nodes (
    node_id     TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,
    content     TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    success_rate REAL NOT NULL DEFAULT 1.0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS memory_edges (
    edge_id     TEXT PRIMARY KEY,
    src_node    TEXT NOT NULL REFERENCES memory_nodes(node_id),
    dst_node    TEXT NOT NULL REFERENCES memory_nodes(node_id),
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON memory_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_edges_src  ON memory_edges(src_node);
CREATE INDEX IF NOT EXISTS idx_edges_dst  ON memory_edges(dst_node);
"""
        )
        self.conn.commit()

    def _contains_secret(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(_SECRET_RE.search(value)) or bool(_LEGACY_SECRET_RE.search(value))
        if isinstance(value, dict):
            return any(self._contains_secret(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_secret(v) for v in value)
        return False

    def _row_to_node(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["content"] = json.loads(d["content"])
        d["tags"] = json.loads(d["tags"])
        return d

    def add_node(self, node_type, content: dict, confidence=1.0, success_rate=1.0, tags=None) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError("invalid node_type")
        if self._contains_secret(content):
            raise ValueError("secret-like content blocked")
        node_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory_nodes (node_id,node_type,content,confidence,success_rate,created_at,updated_at,tags) VALUES (?,?,?,?,?,?,?,?)",
                (node_id, node_type, json.dumps(content), float(confidence), float(success_rate), now, now, json.dumps(tags or [])),
            )
            self.conn.commit()
        return node_id

    def add_edge(self, src_node, dst_node, edge_type, weight=1.0) -> str:
        if edge_type not in EDGE_TYPES:
            raise ValueError("invalid edge_type")
        edge_id = uuid.uuid4().hex
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory_edges (edge_id,src_node,dst_node,edge_type,weight,created_at) VALUES (?,?,?,?,?,?)",
                (edge_id, src_node, dst_node, edge_type, float(weight), time.time()),
            )
            self.conn.commit()
        return edge_id

    def update_node(self, node_id, content=None, confidence=None, success_rate=None) -> None:
        node = self.get_node(node_id)
        if not node:
            return
        new_content = node["content"] if content is None else content
        if self._contains_secret(new_content):
            raise ValueError("secret-like content blocked")
        with self._lock:
            self.conn.execute(
                "UPDATE memory_nodes SET content=?, confidence=?, success_rate=?, updated_at=? WHERE node_id=?",
                (
                    json.dumps(new_content),
                    float(node["confidence"] if confidence is None else confidence),
                    float(node["success_rate"] if success_rate is None else success_rate),
                    time.time(),
                    node_id,
                ),
            )
            self.conn.commit()

    def get_node(self, node_id) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM memory_nodes WHERE node_id=?", (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def get_related(self, node_id, edge_type=None, direction="out") -> List[dict]:
        if direction not in ("out", "in"):
            direction = "out"
        if direction == "out":
            q = "SELECT n.* FROM memory_edges e JOIN memory_nodes n ON n.node_id=e.dst_node WHERE e.src_node=?"
        else:
            q = "SELECT n.* FROM memory_edges e JOIN memory_nodes n ON n.node_id=e.src_node WHERE e.dst_node=?"
        params: List[Any] = [node_id]
        if edge_type:
            q += " AND e.edge_type=?"
            params.append(edge_type)
        return [self._row_to_node(r) for r in self.conn.execute(q, tuple(params)).fetchall()]

    def query_by_intent(self, intent: str, node_type=None, limit=10) -> List[dict]:
        tokens = {t for t in re.findall(r"\w+", (intent or "").lower()) if t}
        q = "SELECT * FROM memory_nodes"
        params: List[Any] = []
        if node_type:
            q += " WHERE node_type=?"
            params.append(node_type)
        rows = [self._row_to_node(r) for r in self.conn.execute(q, tuple(params)).fetchall()]
        now = time.time()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for n in rows:
            txt = json.dumps(n.get("content", {})).lower()
            overlap = sum(1 for t in tokens if t in txt)
            recency = 1.0 / (1.0 + max(0.0, now - float(n.get("updated_at", now))) / 86400.0)
            score = overlap * float(n.get("confidence", 1.0)) * float(n.get("success_rate", 1.0)) * recency
            scored.append((score, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for s, n in scored if s > 0][: int(limit)]

    def get_command_recipe(self, intent: str) -> Optional[dict]:
        for node in self.query_by_intent(intent, node_type="command_recipe", limit=50):
            c = node.get("content", {})
            if str(c.get("risk", "low")).lower() in ("high", "destructive"):
                continue
            guard = str(c.get("os_guard", "")).strip().lower()
            if guard and guard not in os.sys.platform.lower() and guard not in os.name.lower():
                continue
            return node
        return None

    def get_project_facts(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM memory_nodes WHERE node_type='project_fact' ORDER BY updated_at DESC").fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_lessons_for_goal(self, goal: str, limit=5) -> List[dict]:
        return self.query_by_intent(goal, node_type="lesson", limit=limit)

    def query_lessons_for_failure_class(self, failure_class: str) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM memory_nodes WHERE node_type='lesson'").fetchall()
        nodes = [self._row_to_node(r) for r in rows]
        return [n for n in nodes if n.get("content", {}).get("failure_class") == failure_class]

    def get_action_history(self, goal_type: str, action_family: str) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM memory_nodes WHERE node_type='decision' ORDER BY created_at DESC LIMIT 20").fetchall()
        out = []
        for n in [self._row_to_node(r) for r in rows]:
            c = n.get("content", {})
            if c.get("goal_type") == goal_type and c.get("action_family") == action_family:
                out.append(n)
        return out

    def flush_session_memory(self, loop_id: str, memory_items: List[dict]) -> None:
        rows = self.conn.execute("SELECT * FROM memory_nodes WHERE node_type='run_event'").fetchall()
        loop_node = next((self._row_to_node(r) for r in rows if self._row_to_node(r).get("content", {}).get("loop_id") == loop_id), None)
        loop_node_id = loop_node["node_id"] if loop_node else self.add_node("run_event", {"loop_id": loop_id, "event_type": "loop_session"})
        for item in memory_items or []:
            content = item.get("content")
            if not content:
                continue
            nt = "lesson" if item.get("event_type") == "lesson" else "run_event"
            nid = self.add_node(nt, {"loop_id": loop_id, **item})
            self.add_edge(nid, loop_node_id, "learned_from")

    def unsaturate_family(self, family: str) -> None:
        rows = self.conn.execute("SELECT * FROM memory_nodes WHERE node_type='capability'").fetchall()
        for n in [self._row_to_node(r) for r in rows]:
            c = dict(n.get("content", {}))
            if c.get("family") == family and c.get("saturated") is True:
                c["saturated"] = False
                self.update_node(n["node_id"], content=c)

    def migrate_legacy(self, project_root: str) -> None:
        done = self.conn.execute(
            "SELECT 1 FROM memory_nodes WHERE node_type='environment_fact' AND content LIKE '%\"migration_done\"%' LIMIT 1"
        ).fetchone()
        if done:
            return

        root = Path(project_root)
        to_insert: List[tuple[Any, ...]] = []
        to_edges: List[tuple[Any, ...]] = []

        def append_node(node_type: str, content: Dict[str, Any], ts: float) -> str:
            if self._contains_secret(content):
                return ""
            nid = uuid.uuid4().hex
            to_insert.append((nid, node_type, json.dumps(content), 1.0, float(content.get("success_rate", 1.0)), ts, ts, "[]"))
            return nid

        decisions = root / ".igris" / "memory" / "decisions.json"
        if decisions.exists():
            for e in json.loads(decisions.read_text(encoding="utf-8") or "[]"):
                et = e.get("event_type")
                nt = "decision" if et == "decision" else "lesson" if et == "failure" else "capability" if et == "saturation" else "run_event"
                append_node(nt, e, float(e.get("timestamp", time.time())))

        failures = root / ".igris" / "memory" / "failures.json"
        if failures.exists():
            for e in json.loads(failures.read_text(encoding="utf-8") or "[]"):
                append_node("lesson", e, float(e.get("timestamp", time.time())))

        smw = root / ".igris" / "smw_knowledge_base.json"
        if smw.exists():
            for inc in json.loads(smw.read_text(encoding="utf-8") or "[]"):
                ts = float(inc.get("detected_at", time.time()))
                nid = append_node("lesson", inc, ts)
                pat = inc.get("resolution_pattern")
                if nid and pat:
                    pid = append_node("command_recipe", {"pattern": pat}, ts)
                    if pid:
                        to_edges.append((uuid.uuid4().hex, nid, pid, "fixed_by", 1.0, ts))

        outcomes = root / ".igris" / "assignment_outcomes.json"
        if outcomes.exists():
            for o in json.loads(outcomes.read_text(encoding="utf-8") or "[]"):
                c = dict(o)
                c["success_rate"] = float(o.get("success", 1.0))
                append_node("command_recipe", c, float(o.get("timestamp", time.time())))

        with self._lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO memory_nodes (node_id,node_type,content,confidence,success_rate,created_at,updated_at,tags) VALUES (?,?,?,?,?,?,?,?)",
                to_insert,
            )
            if to_edges:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO memory_edges (edge_id,src_node,dst_node,edge_type,weight,created_at) VALUES (?,?,?,?,?,?)",
                    to_edges,
                )
            self.conn.commit()

        self.add_node("environment_fact", {"key": "migration_done", "version": "1", "migrated_at": time.time()})

    def export_safe(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM memory_nodes").fetchall()
        out: List[dict] = []
        for r in rows:
            n = self._row_to_node(r)
            if n.get("node_type") == "environment_fact":
                continue
            if self._contains_secret(n.get("content", {})):
                continue
            out.append(n)
        return out

    def import_safe(self, nodes: List[dict]) -> Dict[str, int]:
        imported = 0
        skipped = 0
        with self._lock:
            for n in nodes:
                node_id = n.get("node_id")
                if not node_id:
                    skipped += 1
                    continue
                if self.conn.execute("SELECT 1 FROM memory_nodes WHERE node_id=?", (node_id,)).fetchone():
                    skipped += 1
                    continue
                if n.get("node_type") not in NODE_TYPES or self._contains_secret(n.get("content", {})):
                    skipped += 1
                    continue
                self.conn.execute(
                    "INSERT INTO memory_nodes (node_id,node_type,content,confidence,success_rate,created_at,updated_at,tags) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        node_id,
                        n["node_type"],
                        json.dumps(n.get("content", {})),
                        float(n.get("confidence", 1.0)),
                        float(n.get("success_rate", 1.0)),
                        float(n.get("created_at", time.time())),
                        float(n.get("updated_at", time.time())),
                        json.dumps(n.get("tags", [])),
                    ),
                )
                imported += 1
            self.conn.commit()
        return {"imported": imported, "skipped": skipped}

    def decay_confidence(self, max_age_days: float = 30.0, half_life_days: float = 14.0) -> int:
        import math

        now = time.time()
        cutoff = now - (max_age_days / 10) * 86400
        rows = self.conn.execute(
            "SELECT node_id, confidence, created_at FROM memory_nodes "
            "WHERE node_type IN ('lesson','world_state_snapshot') AND created_at < ?",
            (cutoff,),
        ).fetchall()
        updated = 0
        with self._lock:
            for row in rows:
                age_days = (now - float(row["created_at"])) / 86400
                conf = float(row["confidence"])
                new_conf = max(0.05, conf * math.exp(-math.log(2) * age_days / half_life_days))
                if abs(new_conf - conf) > 0.001:
                    self.conn.execute(
                        "UPDATE memory_nodes SET confidence=?, updated_at=? WHERE node_id=?",
                        (new_conf, now, row["node_id"]),
                    )
                    updated += 1
            self.conn.commit()
        return updated

    def deprecate_stale_lessons(self, project_root: str) -> int:
        now = time.time()
        rows = self.conn.execute(
            "SELECT node_id, content, confidence, tags FROM memory_nodes "
            "WHERE node_type = 'lesson' AND confidence > 0.1"
        ).fetchall()
        marked = 0
        with self._lock:
            for row in rows:
                try:
                    content = json.loads(row["content"])
                    tags = json.loads(row["tags"] or "[]")
                except Exception:
                    continue
                if "stale" in tags:
                    continue
                files = content.get("files_modified", [])
                if not files:
                    continue
                missing = [f for f in files if not os.path.exists(os.path.join(project_root, f))]
                if missing:
                    conf = float(row["confidence"])
                    new_conf = max(0.05, conf * 0.3)
                    new_tags = list(set(tags + ["stale"]))
                    self.conn.execute(
                        "UPDATE memory_nodes SET confidence=?, tags=?, updated_at=? WHERE node_id=?",
                        (new_conf, json.dumps(new_tags), now, row["node_id"]),
                    )
                    marked += 1
            self.conn.commit()
        return marked

    def detect_and_mark_contradictions(self, window_nodes: int = 100) -> int:
        now = time.time()
        rows = self.conn.execute(
            "SELECT node_id, content, confidence, tags FROM memory_nodes "
            "WHERE node_type = 'lesson' ORDER BY created_at DESC LIMIT ?",
            (window_nodes,),
        ).fetchall()
        by_goal: Dict[str, List[tuple[str, str, float, List[str]]]] = {}
        for row in rows:
            try:
                content = json.loads(row["content"])
                tags = json.loads(row["tags"] or "[]")
            except Exception:
                continue
            goal_key = str(content.get("goal") or "")[:50]
            outcome = str(content.get("outcome") or "")
            if goal_key:
                by_goal.setdefault(goal_key, []).append((row["node_id"], outcome, float(row["confidence"]), tags))

        pairs = 0
        with self._lock:
            for _, nodes in by_goal.items():
                successes = [(nid, c, t) for nid, o, c, t in nodes if o == "success"]
                failures = [(nid, c, t) for nid, o, c, t in nodes if o == "failure"]
                if successes and failures:
                    for nid, conf, tags in successes + failures:
                        if "contradicted" not in tags:
                            new_conf = max(0.05, conf * 0.6)
                            new_tags = list(set(tags + ["contradicted"]))
                            self.conn.execute(
                                "UPDATE memory_nodes SET confidence=?, tags=?, updated_at=? WHERE node_id=?",
                                (new_conf, json.dumps(new_tags), now, nid),
                            )
                    pairs += 1
            self.conn.commit()
        return pairs
