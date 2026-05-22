import json
from pathlib import Path
from unittest.mock import patch

from igris.core import supervisor_reasoning_worker as w
from igris.core.agent_reasoning_loop import LoopResult


def test_progress_file_deleted_on_clean_completion(tmp_path, monkeypatch, capsys):
    payload = {
        "project_root": str(tmp_path),
        "goal": "x",
        "max_steps": 1,
    }
    progress = tmp_path / ".igris" / "reasoning_progress.json"

    def fake_run(self, goal, initial_context, step_callback=None):
        if step_callback:
            step_callback(1, "read_file_range")
        return LoopResult(status="finished", total_steps=1)

    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    with patch("igris.core.agent_reasoning_loop.AgentReasoningLoop.run", fake_run):
        assert w.main() == 0
    assert not progress.exists()


def test_worksession_created_and_phases_advance(tmp_path, monkeypatch):
    payload = {"project_root": str(tmp_path), "goal": "x", "max_steps": 4}

    def fake_run(self, goal, initial_context, step_callback=None):
        if step_callback:
            step_callback(1, "read")
            step_callback(2, "plan")
            step_callback(3, "fix")
        return LoopResult(status="finished", total_steps=3)

    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    with patch("igris.core.agent_reasoning_loop.AgentReasoningLoop.run", fake_run), patch("igris.core.supervisor_reasoning_worker.WorkSession") as ws:
        ws_inst = ws.create.return_value
        ws_inst.session_id = "sid"
        ws_inst.goal = "x"
        assert w.main() == 0
    phases = [c.args[0] for c in ws_inst.advance_phase.call_args_list]
    assert w.WorkPhase.UNDERSTAND in phases
    assert w.WorkPhase.DELIVER in phases


def test_result_dict_includes_work_session_id(tmp_path, monkeypatch, capsys):
    payload = {"project_root": str(tmp_path), "goal": "x", "max_steps": 1}
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    with patch("igris.core.supervisor_reasoning_worker.WorkSession") as ws:
        ws_inst = ws.create.return_value
        ws_inst.session_id = "abc123"
        ws_inst.goal = "x"
        assert w.main() == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["work_session_id"] == "abc123"


def test_remember_called_after_deliver(tmp_path, monkeypatch):
    payload = {"project_root": str(tmp_path), "goal": "x", "max_steps": 1}
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    with patch("igris.core.supervisor_reasoning_worker.WorkSession") as ws:
        ws_inst = ws.create.return_value
        ws_inst.session_id = "abc123"
        ws_inst.goal = "x"
        assert w.main() == 0
    ws_inst.remember.assert_called_once_with(str(tmp_path))
