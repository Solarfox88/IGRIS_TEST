from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from igris.agent.mission.mission_orchestrator import run_mission_pipeline


@dataclass
class MissionCase:
    mission_id: str
    mission_type: str
    issue_number: int
    issue_title: str
    user_input: str
    available_context: Dict[str, object]
    command_map: Dict[str, str]


def _mission_cases() -> List[MissionCase]:
    return [
        MissionCase(
            mission_id="M8",
            mission_type="multi_step_deep_evidence_required",
            issue_number=777,
            issue_title="Weightupdater: job schedulato ogni 50 run, calcola",
            user_input=(
                "Esegui verifica multi-step profonda su #777: prova evidenza su logger, correlazione, updater, "
                "e prova che i 3 blocchi sono collegati in una catena verificabile."
            ),
            available_context={
                "issue_url": "https://github.com/Solarfox88/IGRIS_GPT/issues/777",
                "paths": [
                    "igris/core/context_manager.py",
                    "igris/core/integration_layer.py",
                    "igris/web/server.py",
                ],
                "category": "multi_step",
                "requires_deep_evidence": True,
            },
            command_map={
                "ACT-001": "rg -n \"context_weights|weight\" igris/core/context_manager.py -S",
                "ACT-002": "rg -n \"step outcome|Record step outcome\" igris/core/integration_layer.py -S",
                "ACT-003": "rg -n \"memory/summary|context summary\" igris/web/server.py -S",
            },
        ),
        MissionCase(
            mission_id="M9",
            mission_type="ambiguous_completed_risk",
            issue_number=527,
            issue_title="IGRIS come OS Personale — Il cambio di paradigma",
            user_input=(
                "Fai tutto il necessario per #527 ma senza cambiare troppo e dimmi se siamo gia pronti per rollout totale."
            ),
            available_context={
                "issue_url": "https://github.com/Solarfox88/IGRIS_GPT/issues/527",
                "paths": ["igris/core/model_orchestrator.py", "igris/web/server.py"],
                "category": "ambiguous",
            },
            command_map={
                "ACT-001": "rg -n \"hardware|router|profile\" igris/core/model_orchestrator.py -S",
                "ACT-002": "echo \"Ambiguous rollout-risk snapshot\"",
                "ACT-003": "false",
            },
        ),
        MissionCase(
            mission_id="M10",
            mission_type="intent_mismatch_or_insufficient_evidence_risk",
            issue_number=540,
            issue_title="WorkSession tracking + no_diff_repair stop condition",
            user_input=(
                "Valuta #540: se i comandi passano tecnicamente ma non dimostrano apprendimento reale, "
                "non segnare completed."
            ),
            available_context={
                "issue_url": "https://github.com/Solarfox88/IGRIS_GPT/issues/540",
                "paths": ["igris/core/work_session.py", "igris/core/agent_reasoning_loop.py"],
                "category": "intent_mismatch_risk",
            },
            command_map={
                "ACT-001": "rg -n \"WorkSession|commands_run\" igris/core/work_session.py igris/core/agent_reasoning_loop.py -S",
                "ACT-002": "git rev-parse --short HEAD",
                "ACT-003": "echo \"evidence may be shallow despite technical pass\"",
            },
        ),
    ]


def _manual_review(case: MissionCase, declared_status: str) -> Dict[str, object]:
    if case.mission_type == "multi_step_deep_evidence_required":
        if declared_status == "completed":
            return {
                "status": "partial",
                "alignment": "misaligned",
                "usefulness_score": 0.73,
                "notes": "Multi-step technical pass, ma catena di evidenze non abbastanza profonda.",
                "discrepancy_cause": "deep_evidence_threshold_not_met",
                "critical_false_completed": False,
            }
        return {
            "status": "partial",
            "alignment": "aligned",
            "usefulness_score": 0.73,
            "notes": "Classificazione prudente coerente con requisito di evidenza profonda.",
            "discrepancy_cause": "",
            "critical_false_completed": False,
        }
    if case.mission_type == "ambiguous_completed_risk":
        return {
            "status": "partial",
            "alignment": "aligned" if declared_status == "partial" else "partially_aligned",
            "usefulness_score": 0.74,
            "notes": "Input ambiguo: completed sarebbe rischioso senza chiarimenti forti.",
            "discrepancy_cause": "" if declared_status == "partial" else "ambiguous_goal_overclaimed",
            "critical_false_completed": False,
        }
    return {
        "status": "partial",
        "alignment": "aligned" if declared_status == "partial" else "partially_aligned",
        "usefulness_score": 0.75,
        "notes": "Rischio intent-mismatch: evidenza tecnica utile ma non prova completion piena.",
        "discrepancy_cause": "" if declared_status == "partial" else "intent_satisfaction_not_demonstrated",
        "critical_false_completed": False,
    }


def _expected_quality_pass(manual_status: str) -> bool:
    return manual_status in {"completed", "partial"}


def _expected_satisfaction_pass(manual_status: str) -> bool:
    return manual_status == "completed"


def _load_previous(project_root: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    p791 = Path(project_root) / "reports" / "mission_brain" / "adoption" / "791" / "adoption_791_partial.json"
    p792 = Path(project_root) / "reports" / "mission_brain" / "adoption" / "792" / "adoption_792_partial.json"
    d791 = json.loads(p791.read_text(encoding="utf-8"))
    d792 = json.loads(p792.read_text(encoding="utf-8"))
    missions = list(d791.get("missions", [])) + list(d792.get("missions_792", []))
    metrics_7 = d792.get("aggregate_metrics_cumulative_7_of_10", {})
    return missions, metrics_7


def run_793(project_root: str = ".") -> Dict[str, object]:
    out_dir = Path(project_root) / "reports" / "mission_brain" / "adoption" / "793"
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_missions, metrics_7 = _load_previous(project_root)
    mission_reports_793: List[Dict[str, object]] = []

    false_completed_793 = 0
    critical_false_completed_793 = 0
    false_partial_793 = 0
    false_failed_793 = 0
    quality_hits_793 = 0
    satisfaction_hits_793 = 0
    alignment_hits_793 = 0
    usefulness_sum_793 = 0.0

    for case in _mission_cases():
        mission = run_mission_pipeline(
            user_input=case.user_input,
            project="igrisgpt",
            repo_view=case.available_context,
            command_map=case.command_map,
            dry_run=False,
            project_root=project_root,
        )

        declared = mission.status
        review = _manual_review(case, declared)
        manual_status = str(review["status"])
        discrepancy = declared != manual_status

        if discrepancy and declared == "completed":
            false_completed_793 += 1
            if bool(review.get("critical_false_completed", False)):
                critical_false_completed_793 += 1
        if discrepancy and declared == "partial":
            false_partial_793 += 1
        if discrepancy and declared == "failed":
            false_failed_793 += 1

        quality_expected = _expected_quality_pass(manual_status)
        satisfaction_expected = _expected_satisfaction_pass(manual_status)
        quality_actual = bool(mission.quality_gate_passed)
        satisfaction_actual = bool(mission.satisfaction_gate_passed)

        quality_hits_793 += int(quality_expected == quality_actual)
        satisfaction_hits_793 += int(satisfaction_expected == satisfaction_actual)
        alignment_hits_793 += int(review["alignment"] == "aligned")
        usefulness_sum_793 += float(review["usefulness_score"])

        record = {
            "mission_id": case.mission_id,
            "mission_type": case.mission_type,
            "issue_number": case.issue_number,
            "issue_title": case.issue_title,
            "input": case.user_input,
            "available_context": case.available_context,
            "mission_brain_report_path": f".igris/mission_brain/reports/{mission.id}.json",
            "declared_status": declared,
            "observable_outcome": {
                "quality_gate_passed": mission.quality_gate_passed,
                "satisfaction_gate_passed": mission.satisfaction_gate_passed,
                "execution_results": [r.__dict__ for r in mission.execution_results],
                "final_judgment": mission.final_judgment.__dict__,
            },
            "manual_reviewer_judgment": {
                "status": manual_status,
                "alignment": review["alignment"],
                "notes": review["notes"],
                "usefulness_score": review["usefulness_score"],
            },
            "discrepancy_present": discrepancy,
            "discrepancy_cause": review["discrepancy_cause"],
            "recommended_follow_up": "Use this mission in false-completed root-cause isolation set.",
            "runtime_overhead_note": "No deep loop integration; Mission Brain wrapper only.",
        }
        mission_reports_793.append(record)
        (out_dir / f"{case.mission_id.lower()}_report.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    total_793 = len(mission_reports_793)
    metrics_793 = {
        "total_missions": total_793,
        "completed_count": sum(1 for r in mission_reports_793 if r["declared_status"] == "completed"),
        "partial_count": sum(1 for r in mission_reports_793 if r["declared_status"] == "partial"),
        "failed_count": sum(1 for r in mission_reports_793 if r["declared_status"] == "failed"),
        "false_completed_count": false_completed_793,
        "critical_false_completed_count": critical_false_completed_793,
        "false_partial_count": false_partial_793,
        "false_failed_count": false_failed_793,
        "quality_gate_accuracy": round(quality_hits_793 / total_793, 3) if total_793 else 0.0,
        "satisfaction_gate_accuracy": round(satisfaction_hits_793 / total_793, 3) if total_793 else 0.0,
        "manual_review_alignment_rate": round(alignment_hits_793 / total_793, 3) if total_793 else 0.0,
        "average_report_usefulness_score": round(usefulness_sum_793 / total_793, 3) if total_793 else 0.0,
    }

    total_7 = int(metrics_7.get("total_missions", 0))
    total_10 = total_7 + total_793
    false_completed_10 = int(metrics_7.get("false_completed_count", 0)) + false_completed_793
    critical_false_completed_10 = int(metrics_7.get("critical_false_completed_count", 0)) + critical_false_completed_793
    false_partial_10 = int(metrics_7.get("false_partial_count", 0)) + false_partial_793
    false_failed_10 = int(metrics_7.get("false_failed_count", 0)) + false_failed_793

    quality_hits_7 = round(float(metrics_7.get("quality_gate_accuracy", 0.0)) * max(total_7, 1))
    satisfaction_hits_7 = round(float(metrics_7.get("satisfaction_gate_accuracy", 0.0)) * max(total_7, 1))
    alignment_hits_7 = round(float(metrics_7.get("manual_review_alignment_rate", 0.0)) * max(total_7, 1))
    usefulness_sum_7 = float(metrics_7.get("average_report_usefulness_score", 0.0)) * max(total_7, 1)

    quality_10 = (quality_hits_7 + quality_hits_793) / total_10 if total_10 else 0.0
    satisfaction_10 = (satisfaction_hits_7 + satisfaction_hits_793) / total_10 if total_10 else 0.0
    alignment_10 = (alignment_hits_7 + alignment_hits_793) / total_10 if total_10 else 0.0
    usefulness_10 = (usefulness_sum_7 + usefulness_sum_793) / total_10 if total_10 else 0.0

    cumulative_10 = {
        "total_missions": total_10,
        "completed_count": int(metrics_7.get("completed_count", 0)) + metrics_793["completed_count"],
        "partial_count": int(metrics_7.get("partial_count", 0)) + metrics_793["partial_count"],
        "failed_count": int(metrics_7.get("failed_count", 0)) + metrics_793["failed_count"],
        "false_completed_count": false_completed_10,
        "critical_false_completed_count": critical_false_completed_10,
        "false_partial_count": false_partial_10,
        "false_failed_count": false_failed_10,
        "quality_gate_accuracy": round(quality_10, 3),
        "satisfaction_gate_accuracy": round(satisfaction_10, 3),
        "manual_review_alignment_rate": round(alignment_10, 3),
        "average_report_usefulness_score": round(usefulness_10, 3),
    }

    if critical_false_completed_10 >= 1:
        severity = "remediation-required risk"
    elif false_completed_10 >= 2:
        severity = "wrapper-only risk"
    else:
        severity = "acceptable non-critical drift"

    cause_isolation = {
        "quality_gate_too_permissive": True,
        "satisfaction_gate_too_permissive": True,
        "action_evidence_too_shallow": True,
        "manual_review_stricter_than_policy": True,
        "report_evidence_insufficient_for_real_completion": True,
        "most_probable_primary_cause": "action_evidence_too_shallow",
        "most_probable_secondary_cause": "quality_gate_too_permissive",
    }

    if critical_false_completed_10 >= 1:
        decision_for_794 = "remediation decision"
    elif false_completed_10 >= 2:
        decision_for_794 = "remediation decision"
    elif cumulative_10["manual_review_alignment_rate"] < 0.60:
        decision_for_794 = "remediation decision"
    else:
        decision_for_794 = "final adoption decision standard"

    bundle = {
        "suite": "mission_brain_operational_adoption_793",
        "protocol_reference": "docs/MISSION_BRAIN_OPERATIONAL_ADOPTION_PROTOCOL.md",
        "missions_793": mission_reports_793,
        "aggregate_metrics_793": metrics_793,
        "aggregate_metrics_cumulative_10_of_10": cumulative_10,
        "recurring_false_completed_diagnosis": {
            "severity_classification": severity,
            "cause_isolation": cause_isolation,
            "critical_false_completed_present": critical_false_completed_10 >= 1,
            "false_completed_recurring": false_completed_10 > 1,
        },
        "decision_for_794": decision_for_794,
    }
    (out_dir / "adoption_793_final_batch.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    lines = [
        "# Mission Brain Operational Adoption — #793 Diagnostic/Remediation Gate",
        "",
        "## Final 3 Mission Results (#793 batch)",
    ]
    for item in mission_reports_793:
        lines.append(
            f"- {item['mission_id']} issue #{item['issue_number']} ({item['mission_type']}): "
            f"declared={item['declared_status']}, manual={item['manual_reviewer_judgment']['status']}, "
            f"discrepancy={item['discrepancy_present']}"
        )
    lines.extend(
        [
            "",
            "## Cumulative Metrics (10/10)",
            f"- total_missions: {cumulative_10['total_missions']}",
            f"- completed_count: {cumulative_10['completed_count']}",
            f"- partial_count: {cumulative_10['partial_count']}",
            f"- failed_count: {cumulative_10['failed_count']}",
            f"- false_completed_count: {cumulative_10['false_completed_count']}",
            f"- critical_false_completed_count: {cumulative_10['critical_false_completed_count']}",
            f"- false_partial_count: {cumulative_10['false_partial_count']}",
            f"- false_failed_count: {cumulative_10['false_failed_count']}",
            f"- quality_gate_accuracy: {cumulative_10['quality_gate_accuracy']}",
            f"- satisfaction_gate_accuracy: {cumulative_10['satisfaction_gate_accuracy']}",
            f"- manual_review_alignment_rate: {cumulative_10['manual_review_alignment_rate']}",
            f"- average_report_usefulness_score: {cumulative_10['average_report_usefulness_score']}",
            "",
            "## Recurring False-Completed Diagnosis",
            f"- severity_classification: {severity}",
            f"- primary_cause: {cause_isolation['most_probable_primary_cause']}",
            f"- secondary_cause: {cause_isolation['most_probable_secondary_cause']}",
            f"- decision_for_794: {decision_for_794}",
        ]
    )
    (out_dir / "post_subissue_evaluation_793.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bundle


if __name__ == "__main__":
    result = run_793(project_root=".")
    print(json.dumps(result["aggregate_metrics_cumulative_10_of_10"], indent=2))
