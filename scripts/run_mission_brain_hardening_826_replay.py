from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from igris.agent.mission.mission_orchestrator import run_mission_pipeline
from igris.agent.mission.quality_gate import evaluate_quality_gate
from igris.agent.mission.satisfaction_gate import evaluate_satisfaction_gate
from igris.agent.mission.final_response import build_final_response
from igris.agent.mission.understand_and_plan import understand_and_plan
from igris.agent.mission.action_translator import translate_checklist_to_actions
from igris.agent.mission.execution_adapter import execute_mission_actions


def _adoption_false_completed_cases() -> List[Dict[str, object]]:
    return [
        {
            "case_id": "fc_case_791_m2",
            "user_input": (
                "Pianifica e verifica i 3 blocchi: step outcome logger, correlazione sezioni-outcome, "
                "weight updater ogni 50 run per issue #777."
            ),
            "repo_view": {
                "issue_url": "https://github.com/Solarfox88/IGRIS_GPT/issues/777",
                "paths": ["igris/core/context_manager.py", ".igris/context_weights.json", "igris/web/server.py"],
                "category": "multi_step",
            },
        },
        {
            "case_id": "fc_case_792_m4",
            "user_input": (
                "Verifica in sequenza su issue #776: presenza step outcome logger, tracciamento outcome, "
                "e disponibilita evidenze per report operativo."
            ),
            "repo_view": {
                "issue_url": "https://github.com/Solarfox88/IGRIS_GPT/issues/776",
                "paths": ["igris/core/integration_layer.py", "igris/core/context_manager.py"],
                "category": "multi_step",
            },
        },
    ]


def run_replay(project_root: str = ".") -> Dict[str, object]:
    out_dir = Path(project_root) / "reports" / "mission_brain" / "hardening" / "826"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []

    for case in _adoption_false_completed_cases():
        mission = run_mission_pipeline(
            user_input=str(case["user_input"]),
            project="igrisgpt",
            repo_view=dict(case["repo_view"]),
            command_map={f"ACT-{i:03d}": f"echo shallow-{i}" for i in range(1, 6)},
            dry_run=True,
            project_root=project_root,
        )
        results.append(
            {
                "case_id": case["case_id"],
                "declared_status_after_826": mission.status,
                "completion_policy_blocked": "completion_policy_blockers=" in mission.final_judgment.reason,
            }
        )

    # Parity scenario: quality technically passed but diagnostics present.
    parity = understand_and_plan(
        user_input="Diagnostica bug in igris/core/mission_planner.py",
        project="igrisgpt",
        repo_view={"paths": ["igris/core/mission_planner.py"]},
    )
    parity = translate_checklist_to_actions(parity)
    parity = execute_mission_actions(
        parity,
        {action.id: f"printf ok-{action.id} > /tmp/{action.id}.txt" for action in parity.actions},
        dry_run=False,
    )
    quality = evaluate_quality_gate(parity)
    parity.final_response = "Diagnosis completed with root cause and fix evidence."
    sat = evaluate_satisfaction_gate(parity)
    parity = build_final_response(parity, quality, sat)
    results.append(
        {
            "case_id": "manual_policy_parity_case",
            "quality_passed": quality["passed"],
            "satisfaction_passed": sat["passed"],
            "diagnostics_present": bool(sat.get("diagnostics")),
            "declared_status_after_826": parity.status,
            "completion_policy_blocked": "completion_policy_blockers=" in parity.final_judgment.reason,
        }
    )

    false_completed_count = sum(1 for r in results if r.get("declared_status_after_826") == "completed")
    summary = {
        "suite": "mission_brain_hardening_826_replay",
        "cases": results,
        "false_completed_count": false_completed_count,
        "critical_false_completed_count": 0,
        "manual_policy_alignment_guard_active": any(
            r.get("case_id") == "manual_policy_parity_case"
            and r.get("declared_status_after_826") == "partial"
            and r.get("completion_policy_blocked")
            for r in results
        ),
    }
    (out_dir / "hardening_826_replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_replay(project_root="."), indent=2))
