import time
from unittest.mock import patch

from igris.core.memory_graph import MemoryGraph
from igris.core.memory_validator import MemoryValidator


def _set_created_at(mg: MemoryGraph, node_id: str, created_at: float) -> None:
    mg.conn.execute("UPDATE memory_nodes SET created_at=?, updated_at=? WHERE node_id=?", (created_at, created_at, node_id))
    mg.conn.commit()


def test_decay_reduces_old_node_confidence(tmp_path):
    mg = MemoryGraph(str(tmp_path))
    node_id = mg.add_node("lesson", {"goal": "g"}, confidence=1.0)
    _set_created_at(mg, node_id, time.time() - 20 * 86400)
    mg.decay_confidence()
    assert mg.get_node(node_id)["confidence"] < 0.9


def test_decay_does_not_touch_recent_node(tmp_path):
    mg = MemoryGraph(str(tmp_path))
    node_id = mg.add_node("lesson", {"goal": "g"}, confidence=1.0)
    _set_created_at(mg, node_id, time.time() - 3600)
    mg.decay_confidence()
    assert mg.get_node(node_id)["confidence"] == 1.0


def test_deprecate_stale_removes_missing_file(tmp_path):
    mg = MemoryGraph(str(tmp_path))
    node_id = mg.add_node("lesson", {"files_modified": ["no_such.py"]}, confidence=1.0)
    mg.deprecate_stale_lessons(str(tmp_path))
    node = mg.get_node(node_id)
    assert node["confidence"] < 0.5
    assert "stale" in node["tags"]


def test_deprecate_stale_skips_existing_file(tmp_path):
    existing = tmp_path / "exists.py"
    existing.write_text("x=1", encoding="utf-8")
    mg = MemoryGraph(str(tmp_path))
    node_id = mg.add_node("lesson", {"files_modified": ["exists.py"]}, confidence=1.0)
    mg.deprecate_stale_lessons(str(tmp_path))
    node = mg.get_node(node_id)
    assert node["confidence"] == 1.0
    assert "stale" not in node["tags"]


def test_detect_contradictions_marks_both_sides(tmp_path):
    mg = MemoryGraph(str(tmp_path))
    goal = "same goal prefix for contradiction"
    n1 = mg.add_node("lesson", {"goal": goal, "outcome": "success"}, confidence=1.0)
    n2 = mg.add_node("lesson", {"goal": goal, "outcome": "failure"}, confidence=1.0)
    mg.detect_and_mark_contradictions()
    assert "contradicted" in mg.get_node(n1)["tags"]
    assert "contradicted" in mg.get_node(n2)["tags"]


def test_detect_contradictions_no_false_positive(tmp_path):
    mg = MemoryGraph(str(tmp_path))
    n1 = mg.add_node("lesson", {"goal": "goal A", "outcome": "success"}, confidence=1.0)
    mg.add_node("lesson", {"goal": "goal B", "outcome": "failure"}, confidence=1.0)
    mg.detect_and_mark_contradictions()
    assert "contradicted" not in mg.get_node(n1)["tags"]


def test_memory_validator_run_returns_summary(tmp_path):
    with patch("igris.core.memory_validator.MemoryGraph") as mg:
        inst = mg.return_value
        inst.decay_confidence.return_value = 1
        inst.deprecate_stale_lessons.return_value = 2
        inst.detect_and_mark_contradictions.return_value = 3
        out = MemoryValidator(str(tmp_path)).run()
    assert out == {"decayed": 1, "stale": 2, "contradictions": 3}
