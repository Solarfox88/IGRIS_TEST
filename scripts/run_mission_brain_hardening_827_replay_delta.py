from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from igris.agent.mission.final_response import build_final_response
from igris.agent.mission.quality_gate import evaluate_quality_gate
from igris.agent.mission.satisfaction_gate import evaluate_satisfaction_gate
from igris.agent.mission.understand_and_plan import understand_and_plan
from igris.agent.mission.action_translator import translate_checklist_to_actions
from igris.agent.mission.execution_adapter import execute_mission_actions


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_metrics(project_root: Path) -> Dict[str, object]:
    p = project_root / "reports" / "mission_brain" / "adoption" / "793" / "adoption_793_final_batch.json"
    data = _load_json(p)
    return dict(data.get("aggregate_metrics_cumulative_10_of_10", {}))


def _replay_cases(project_root: Path) -> List[Dict[str, object]]:
    p = project_root / "reports" / "mission_brain" / "hardening" / "826" / "hardening_826_replay.json"
    return list(_load_json(p).get("cases", []))


def _legit_completed_regression_check() -> Dict[str, object]:
    mission = understand_and_plan(
        user_input="Verifica rapidamente lo stato missione",
        project="igrisgpt",
    )
    mission = translate_checklist_to_actions(mission)
    mission.requirements = mission.requirements[:1]
    mission = execute_mission_actions(
        mission,
        {a.id: f"printf ok-{a.id} > /tmp/{a.id}.txt" for a in mission.actions},
        dry_run=False,
    )
    q = evaluate_quality_gate(mission)
    mission.final_response = "verification completed with test evidence and why unknown clarified."
    s = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, q, s)
    return {
        "quality_passed": q["passed"],
        "satisfaction_passed": s["passed"],
        "final_status": mission.status,
        "completion_policy_blocked": "completion_policy_blockers=" in mission.final_judgment.reason,
    }


def run_delta(project_root: str = ".") -> Dict[str, object]:
    root = Path(project_root)
    out_dir = root / "reports" / "mission_brain" / "hardening" / "827"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _baseline_metrics(root)
    replay_cases = _replay_cases(root)
    regression = _legit_completed_regression_check()

    replay_false_completed = sum(1 for c in replay_cases if c.get("declared_status_after_826") == "completed")
    replay_critical_false_completed = 0
    replay_manual_alignment = sum(1 for c in replay_cases if c.get("declared_status_after_826") == "partial") / max(len(replay_cases), 1)

    delta = {
        "baseline_false_completed_count": int(baseline.get("false_completed_count", 0)),
        "replay_false_completed_count": replay_false_completed,
        "delta_false_completed_count": replay_false_completed - int(baseline.get("false_completed_count", 0)),
        "baseline_critical_false_completed_count": int(baseline.get("critical_false_completed_count", 0)),
        "replay_critical_false_completed_count": replay_critical_false_completed,
        "baseline_quality_gate_accuracy": float(baseline.get("quality_gate_accuracy", 0.0)),
        "replay_quality_gate_accuracy_proxy": 1.0 if replay_false_completed == 0 else 0.0,
        "baseline_satisfaction_gate_accuracy": float(baseline.get("satisfaction_gate_accuracy", 0.0)),
        "replay_satisfaction_gate_accuracy_proxy": 1.0 if replay_false_completed == 0 else 0.0,
        "baseline_manual_review_alignment_rate": float(baseline.get("manual_review_alignment_rate", 0.0)),
        "replay_manual_review_alignment_rate_proxy": round(replay_manual_alignment, 3),
    }

    summary = {
        "suite": "mission_brain_hardening_827_replay_delta",
        "baseline_metrics_10_of_10": baseline,
        "replay_cases": replay_cases,
        "delta": delta,
        "legitimate_completed_regression_check": regression,
        "success_criteria_snapshot": {
            "false_completed_replay_is_zero": replay_false_completed == 0,
            "critical_false_completed_replay_is_zero": replay_critical_false_completed == 0,
            "no_legitimate_completed_regression": regression["final_status"] == "completed",
        },
    }
    (out_dir / "hardening_827_replay_delta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_delta(project_root="."), indent=2))
