from __future__ import annotations

import json
from pathlib import Path

from igris.agent.mission.adoption_decision import decide_adoption


ROOT = Path(__file__).resolve().parents[1]


def _load_json(rel_path: str):
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def test_mission_template_contains_required_fields():
    template = _load_json("reports/mission_brain/adoption/mission_report_template.json")
    required = {
        "user_input",
        "available_context",
        "mission_brain_report_path",
        "declared_status",
        "observable_outcome",
        "manual_reviewer_judgment",
        "discrepancy_present",
        "discrepancy_cause",
        "recommended_follow_up",
    }
    assert required.issubset(template.keys())
    assert "false_classification" in template
    assert "critical_false_completed" in template["false_classification"]


def test_aggregate_metrics_schema_contains_mandatory_metrics():
    schema = _load_json("reports/mission_brain/adoption/aggregate_metrics_schema.json")
    mandatory = {
        "total_missions",
        "completed_count",
        "partial_count",
        "failed_count",
        "false_completed_count",
        "critical_false_completed_count",
        "false_partial_count",
        "false_failed_count",
        "satisfaction_gate_accuracy",
        "quality_gate_accuracy",
        "manual_review_alignment_rate",
        "average_report_usefulness_score",
        "adoption_decision",
    }
    assert mandatory.issubset(schema.keys())


def test_decision_rules_edge_cases():
    integrate = decide_adoption(
        {
            "total_missions": 10,
            "critical_false_completed_count": 0,
            "false_completed_count": 0,
            "average_report_usefulness_score": 0.8,
            "severe_operational_regression": False,
            "gate_judgments_explainable": True,
            "reports_decision_useful": True,
        }
    )
    assert integrate == "integrate deeper"

    remediate = decide_adoption(
        {
            "total_missions": 10,
            "critical_false_completed_count": 1,
            "false_completed_count": 1,
            "average_report_usefulness_score": 0.9,
            "gate_judgments_explainable": True,
            "reports_decision_useful": True,
        }
    )
    assert remediate == "remediate again"

    keep = decide_adoption(
        {
            "total_missions": 10,
            "critical_false_completed_count": 0,
            "false_completed_count": 2,
            "non_critical_false_completed_explained": False,
            "average_report_usefulness_score": 0.7,
            "gate_judgments_explainable": True,
            "reports_decision_useful": True,
        }
    )
    assert keep == "keep wrapper"

