import json
import os
import sys
from unittest.mock import patch

from igris.core.memory_graph import MemoryGraph
from igris.core.goap_planner import GOAPPlanner, GOAPAction, WorldState


def _mg(tmp_path):
    return MemoryGraph(str(tmp_path))

def test_add_query_node_roundtrip(tmp_path):
    g = _mg(tmp_path); nid = g.add_node("lesson", {"a": 1}); assert g.get_node(nid)["content"]["a"] == 1

def test_add_edge_get_related(tmp_path):
    g=_mg(tmp_path); a=g.add_node("lesson", {"x":1}); b=g.add_node("lesson", {"y":2}); g.add_edge(a,b,"related_to"); assert g.get_related(a)[0]["node_id"]==b

def test_query_by_intent_keyword_match(tmp_path):
    g=_mg(tmp_path); g.add_node("lesson", {"text":"fix docker nginx"}); g.add_node("lesson", {"text":"unrelated"}); assert "docker" in json.dumps(g.query_by_intent("docker fix", "lesson", 1)[0]["content"]).lower()

def test_command_recipe_retrieval_no_llm(tmp_path):
    g=_mg(tmp_path); g.add_node("command_recipe", {"intent":"run tests","command":"pytest","risk":"low"}); assert g.get_command_recipe("run tests")["content"]["command"]=="pytest"

def test_command_recipe_skips_high_risk(tmp_path):
    g=_mg(tmp_path); g.add_node("command_recipe", {"intent":"clean","command":"rm -rf /","risk":"high"}); assert g.get_command_recipe("clean") is None

def test_os_specific_guard(tmp_path):
    g=_mg(tmp_path); g.add_node("command_recipe", {"intent":"x","command":"cmd","risk":"low","os_guard":"definitely_not_this_os"}); assert g.get_command_recipe("x") is None

def test_secrets_never_persisted(tmp_path):
    g=_mg(tmp_path)
    try:
        g.add_node("lesson", {"k": "token=abc"})
        assert False
    except ValueError:
        assert True

def test_migrate_legacy_decisions(tmp_path):
    d=tmp_path/'.igris/memory'; d.mkdir(parents=True); (d/'decisions.json').write_text(json.dumps([{"event_type":"decision","timestamp":1.0}]))
    g=_mg(tmp_path); g.migrate_legacy(str(tmp_path)); assert g.conn.execute("SELECT COUNT(*) FROM memory_nodes WHERE node_type='decision'").fetchone()[0] >= 1

def test_migrate_legacy_idempotent(tmp_path):
    d=tmp_path/'.igris/memory'; d.mkdir(parents=True); (d/'decisions.json').write_text(json.dumps([{"event_type":"decision","timestamp":1.0}]))
    g=_mg(tmp_path); g.migrate_legacy(str(tmp_path)); c1=g.conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]; g.migrate_legacy(str(tmp_path)); c2=g.conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]; assert c1==c2

def test_migrate_legacy_missing_files_ok(tmp_path):
    g=_mg(tmp_path); g.migrate_legacy(str(tmp_path)); assert True

def test_export_excludes_secrets(tmp_path):
    g=_mg(tmp_path); g.add_node("lesson", {"ok":"v"}); out=g.export_safe(); assert len(out)==1 and out[0]["node_type"]!="environment_fact"

def test_import_skips_existing(tmp_path):
    g=_mg(tmp_path); nid=g.add_node("lesson", {"ok":"v"}); node=g.get_node(nid); r=g.import_safe([node,node]); assert r["skipped"]>=1

def test_flush_session_memory(tmp_path):
    g=_mg(tmp_path); g.flush_session_memory("loop1", [{"event_type":"lesson","content":"x","timestamp":1}]); assert len(g.query_lessons_for_failure_class("none"))==0; assert g.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]>=1

def test_unsaturate_family(tmp_path):
    g=_mg(tmp_path); nid=g.add_node("capability", {"family":"f","saturated":True}); g.unsaturate_family("f"); assert g.get_node(nid)["content"]["saturated"] is False

def test_get_action_history_filters_by_family(tmp_path):
    g=_mg(tmp_path); g.add_node("decision", {"goal_type":"g","action_family":"a","outcome":"failure"}); g.add_node("decision", {"goal_type":"g","action_family":"b"}); assert len(g.get_action_history("g","a"))==1

def test_goap_penalizes_failed_actions(tmp_path):
    act=GOAPAction(id='a1', title='x', family='fam', effects={'done':True}, cost=1.0)
    planner=GOAPPlanner(project_root=str(tmp_path), action_library=[act])
    g=_mg(tmp_path); g.add_node("decision", {"goal_type":"goal","action_family":"fam","outcome":"failure"}); g.add_node("decision", {"goal_type":"goal","action_family":"fam","outcome":"failure"})
    with patch.dict(os.environ, {"PROJECT_ROOT": str(tmp_path)}):
        planner.generate_plan({"done":True, "type":"goal"}, state=WorldState(properties={}))
    assert act.cost > 1.0

def test_query_lessons_for_failure_class(tmp_path):
    g=_mg(tmp_path); g.add_node("lesson", {"failure_class":"net"}); assert len(g.query_lessons_for_failure_class("net"))==1
