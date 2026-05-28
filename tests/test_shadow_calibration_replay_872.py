"""Tests for #872 — calibration replay on all 30 shadow cycles.

Verifies:
- Script output fields are correct
- 30 cycles loaded and processed
- agreement_rate is invariant
- scope_mismatch and ambiguous_context counts correct
- legacy_unclassified = 10 (baseline cycles without goal_class)
- taxonomy_changed_cycles = 20 (only new cycles can have goal_class mapping)
- safety gates pass
- per_cycle_replay records present
- report files created
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPORT_PATH = Path("reports/mission_brain/calibration/872/calibration_replay_872.json")


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip("Report not generated yet; run run_shadow_calibration_replay_872.py first.")
    return json.loads(REPORT_PATH.read_text())


class TestReportFields:
    def test_report_exists(self):
        assert REPORT_PATH.exists(), f"Missing: {REPORT_PATH}"

    def test_subissue(self, report):
        assert report["subissue"] == 872

    def test_epic(self, report):
        assert report["epic"] == 868

    def test_evaluation_passed(self, report):
        assert report["evaluation"] == "passed"

    def test_stop_reason_none(self, report):
        assert report["stop_reason"] is None

    def test_next_subissue(self, report):
        assert report["next_subissue"] == 873


class TestCycleCounts:
    def test_total_cycles(self, report):
        assert report["total_cycles"] == 30

    def test_baseline_cycles(self, report):
        assert report["baseline_cycles"] == 10

    def test_new_cycles(self, report):
        assert report["new_cycles"] == 20

    def test_per_cycle_replay_length(self, report):
        assert len(report["per_cycle_replay"]) == 30


class TestAgreementRateInvariant:
    def test_agreement_rate_zero(self, report):
        assert report["agreement_rate"] == 0.0

    def test_agreement_rate_invariant(self, report):
        assert report["agreement_rate_invariant"] is True


class TestTaxonomyCounts:
    def test_scope_mismatch_count(self, report):
        assert report["metrics_after"]["scope_mismatch_count"] == 17

    def test_ambiguous_context_count(self, report):
        assert report["metrics_after"]["ambiguous_context_count"] == 3

    def test_legacy_unclassified_count(self, report):
        # 10 baseline cycles have no goal_class → fall back to legacy
        assert report["metrics_after"]["legacy_unclassified_count"] == 10

    def test_taxonomy_changed_cycles(self, report):
        # Only new cycles (20) have goal_class → can be reclassified
        assert report["taxonomy_changed_cycles"] == 20

    def test_taxonomy_unchanged_cycles(self, report):
        # Baseline cycles (10) have no goal_class → unchanged
        assert report["taxonomy_unchanged_cycles"] == 10

    def test_scope_plus_ambiguous_plus_legacy_equals_disagreements(self, report):
        # All 30 cycles disagree (agreement_rate=0.0) → all classified
        total_disagreements = report["total_cycles"]  # 30 since rate=0.0
        after = report["metrics_after"]
        assert (
            after["scope_mismatch_count"]
            + after["ambiguous_context_count"]
            + after["legacy_unclassified_count"]
            == total_disagreements
        )


class TestSafetyGate:
    def test_risk_introduced_candidates_zero(self, report):
        assert report["risk_introduced_candidates"] == 0

    def test_potential_critical_false_completed_zero(self, report):
        assert report["potential_critical_false_completed"] == 0

    def test_rollback_path_status_ok(self, report):
        assert report["rollback_path_status"] == "ok"


class TestTaxonomyVersions:
    def test_taxonomy_version_before(self, report):
        assert report["taxonomy_version_before"] == "legacy_v0"

    def test_taxonomy_version_after(self, report):
        assert report["taxonomy_version_after"] == "calibrated_v1"


class TestMetricsBeforeAfter:
    def test_metrics_before_present(self, report):
        assert "metrics_before" in report
        assert "disagreement_by_class" in report["metrics_before"]
        assert "dominant_mismatch_classes" in report["metrics_before"]

    def test_metrics_after_present(self, report):
        assert "metrics_after" in report
        assert "calibrated_disagreement_by_class" in report["metrics_after"]
        assert "calibrated_dominant_mismatch_classes" in report["metrics_after"]


class TestPerCycleReplay:
    def test_all_records_have_required_fields(self, report):
        required = {
            "cycle_id", "goal_class", "mission_brain_decision",
            "current_loop_decision", "agreement",
            "mismatch_class_before", "mismatch_class_after", "taxonomy_changed",
        }
        for rec in report["per_cycle_replay"]:
            missing = required - set(rec.keys())
            assert not missing, f"cycle {rec.get('cycle_id')}: missing fields {missing}"

    def test_new_cycles_have_calibrated_mismatch_class(self, report):
        calibrated_classes = {
            "scope_mismatch_goal_vs_run_assessment",
            "expected_divergence_ambiguous_context",
        }
        new_changed = [
            r for r in report["per_cycle_replay"]
            if r["taxonomy_changed"]
        ]
        assert len(new_changed) == 20
        for rec in new_changed:
            assert rec["mismatch_class_after"] in calibrated_classes, (
                f"cycle {rec['cycle_id']}: unexpected class {rec['mismatch_class_after']!r}"
            )

    def test_baseline_cycles_unchanged(self, report):
        unchanged = [r for r in report["per_cycle_replay"] if not r["taxonomy_changed"]]
        assert len(unchanged) == 10

    def test_no_cycle_has_empty_mismatch_class_before_in_disagreeing(self, report):
        # Every disagreeing cycle must have had a before class
        for rec in report["per_cycle_replay"]:
            if not rec["agreement"]:
                assert rec["mismatch_class_before"] != "", (
                    f"cycle {rec['cycle_id']}: mismatch_class_before is empty but agreement=False"
                )


class TestGuardrails:
    def test_shadow_mode_only(self, report):
        assert report["guardrails"]["shadow_mode_only"] is True

    def test_default_behavior_unchanged(self, report):
        assert report["guardrails"]["default_behavior_unchanged"] is True

    def test_agreement_rate_invariant_guardrail(self, report):
        assert report["guardrails"]["agreement_rate_invariant"] is True


class TestReportFilesExist:
    def test_json_report(self):
        assert Path("reports/mission_brain/calibration/872/calibration_replay_872.json").exists()

    def test_md_report(self):
        assert Path("reports/mission_brain/calibration/872/calibration_replay_872.md").exists()
