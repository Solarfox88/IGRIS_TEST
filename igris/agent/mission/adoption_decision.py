from __future__ import annotations

from typing import Any, Dict


def decide_adoption(metrics: Dict[str, Any]) -> str:
    total_missions = int(metrics.get("total_missions", 0) or 0)
    critical_false_completed = int(metrics.get("critical_false_completed_count", 0) or 0)
    false_completed = int(metrics.get("false_completed_count", 0) or 0)
    avg_usefulness = float(metrics.get("average_report_usefulness_score", 0.0) or 0.0)
    severe_regression = bool(metrics.get("severe_operational_regression", False))
    explainable_judgments = bool(metrics.get("gate_judgments_explainable", True))
    reports_useful = bool(metrics.get("reports_decision_useful", avg_usefulness >= 0.6))

    if (
        total_missions >= 10
        and critical_false_completed == 0
        and (false_completed == 0 or bool(metrics.get("non_critical_false_completed_explained", False)))
        and not severe_regression
        and explainable_judgments
        and reports_useful
    ):
        return "integrate deeper"

    if (
        critical_false_completed >= 1
        or not explainable_judgments
        or not reports_useful
        or bool(metrics.get("real_failures_masked", False))
    ):
        return "remediate again"

    return "keep wrapper"

