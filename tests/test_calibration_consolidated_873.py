"""Tests for #873 — consolidated calibration report and final decision.

Verifies:
- All required report fields present and correct
- Final decision is 'calibration_complete' (from allowed set)
- Final decision is NOT in forbidden set
- Gate chain passed
- Safety metrics all zero
- Key counts correct
- Subissues all listed
- Epic status complete
- Guardrails enforced
- Findings and recommendations present and well-formed
- Report files created (JSON + MD)
- MD file contains expected sections
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPORT_PATH = Path("reports/mission_brain/calibration/873/calibration_consolidated_873.json")
MD_PATH = Path("reports/mission_brain/calibration/873/calibration_consolidated_873.md")

ALLOWED_DECISIONS = frozenset({
    "continue_calibration",
    "calibration_complete",
    "recommend_further_analysis",
    "stop_calibration",
})

FORBIDDEN_DECISIONS = frozenset({
    "enable_mission_brain",
    "integrate_mission_brain",
    "deploy",
    "rollout",
    "enable_by_default",
    "mandatory_gate",
})


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip("Report not generated yet; run run_calibration_consolidated_873.py first.")
    return json.loads(REPORT_PATH.read_text())


# ---------------------------------------------------------------------------
# Report file existence
# ---------------------------------------------------------------------------

class TestReportFilesExist:
    def test_json_exists(self):
        assert REPORT_PATH.exists(), f"Missing: {REPORT_PATH}"

    def test_md_exists(self):
        assert MD_PATH.exists(), f"Missing: {MD_PATH}"


# ---------------------------------------------------------------------------
# Core report fields
# ---------------------------------------------------------------------------

class TestReportFields:
    def test_epic(self, report):
        assert report["epic"] == 868

    def test_subissue(self, report):
        assert report["subissue"] == 873

    def test_evaluation_passed(self, report):
        assert report["evaluation"] == "passed"

    def test_stop_reason_none(self, report):
        assert report["stop_reason"] is None

    def test_epic_status_complete(self, report):
        assert report["epic_status"] == "complete"


# ---------------------------------------------------------------------------
# Final decision validation
# ---------------------------------------------------------------------------

class TestFinalDecision:
    def test_final_decision_present(self, report):
        assert "final_decision" in report

    def test_final_decision_is_calibration_complete(self, report):
        assert report["final_decision"] == "calibration_complete"

    def test_final_decision_in_allowed_set(self, report):
        assert report["final_decision"] in ALLOWED_DECISIONS

    def test_final_decision_not_in_forbidden_set(self, report):
        assert report["final_decision"] not in FORBIDDEN_DECISIONS

    def test_final_decision_rationale_present(self, report):
        assert "final_decision_rationale" in report
        assert len(report["final_decision_rationale"]) > 50


# ---------------------------------------------------------------------------
# Gate chain
# ---------------------------------------------------------------------------

class TestGateChain:
    def test_gate_chain_passed(self, report):
        assert report["gate_chain_passed"] is True

    def test_all_subissues_completed(self, report):
        assert set(report["subissues_completed"]) == {869, 870, 871, 872, 873}


# ---------------------------------------------------------------------------
# Safety metrics
# ---------------------------------------------------------------------------

class TestSafetyMetrics:
    def test_risk_introduced_candidates_zero(self, report):
        assert report["risk_introduced_candidates"] == 0

    def test_potential_critical_false_completed_zero(self, report):
        assert report["potential_critical_false_completed"] == 0

    def test_risky_more_optimistic_count_zero(self, report):
        assert report["risky_more_optimistic_count"] == 0


# ---------------------------------------------------------------------------
# Key calibration metrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    def test_total_shadow_cycles(self, report):
        assert report["total_shadow_cycles"] == 30

    def test_agreement_rate(self, report):
        assert report["agreement_rate"] == 0.0

    def test_scope_mismatch_count(self, report):
        assert report["scope_mismatch_count"] == 17

    def test_ambiguous_context_count(self, report):
        assert report["ambiguous_context_count"] == 3

    def test_legacy_unclassified_count(self, report):
        assert report["legacy_unclassified_count"] == 10

    def test_invariant_failed_count_zero(self, report):
        assert report["invariant_failed_count"] == 0

    def test_taxonomy_changed_cycles(self, report):
        assert report["taxonomy_changed_cycles"] == 20

    def test_taxonomy_version(self, report):
        assert report["taxonomy_version"] == "calibrated_v1"

    def test_scope_plus_ambiguous_plus_legacy_equals_total(self, report):
        assert (
            report["scope_mismatch_count"]
            + report["ambiguous_context_count"]
            + report["legacy_unclassified_count"]
            == report["total_shadow_cycles"]
        )


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class TestFindings:
    def test_findings_present(self, report):
        assert "findings" in report
        assert len(report["findings"]) >= 3

    def test_findings_have_required_fields(self, report):
        for f in report["findings"]:
            for field in ("id", "finding", "evidence", "impact"):
                assert field in f, f"Finding missing field {field!r}: {f}"

    def test_scope_mismatch_finding_present(self, report):
        texts = [f["finding"].lower() for f in report["findings"]]
        assert any("scope mismatch" in t for t in texts)

    def test_no_invariant_failed_finding_present(self, report):
        texts = [f["finding"].lower() for f in report["findings"]]
        assert any("invariant_failed" in t or "never dangerously" in t for t in texts)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_recommendations_present(self, report):
        assert "recommendations" in report
        assert len(report["recommendations"]) >= 2

    def test_recommendations_have_required_fields(self, report):
        for r in report["recommendations"]:
            for field in ("id", "recommendation", "rationale", "scope"):
                assert field in r, f"Recommendation missing field {field!r}: {r}"

    def test_shadow_only_recommendation_present(self, report):
        scopes = [r["scope"] for r in report["recommendations"]]
        assert "shadow_only" in scopes

    def test_no_recommendation_enables_default_behavior(self, report):
        for r in report["recommendations"]:
            text = r["recommendation"].lower()
            for forbidden in ("enable by default", "enable_by_default", "mandatory gate", "rollout"):
                assert forbidden not in text, (
                    f"Recommendation {r['id']} mentions forbidden action: {text!r}"
                )


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_shadow_mode_only(self, report):
        assert report["guardrails"]["shadow_mode_only"] is True

    def test_default_behavior_unchanged(self, report):
        assert report["guardrails"]["default_behavior_unchanged"] is True

    def test_no_enable_by_default(self, report):
        assert report["guardrails"]["no_enable_by_default"] is True

    def test_no_mandatory_gate(self, report):
        assert report["guardrails"]["no_mandatory_gate"] is True

    def test_no_rollout(self, report):
        assert report["guardrails"]["no_rollout"] is True

    def test_no_integration_without_approval(self, report):
        assert report["guardrails"]["no_integration_without_approval"] is True


# ---------------------------------------------------------------------------
# MD report content
# ---------------------------------------------------------------------------

class TestMdReport:
    def _md(self):
        return MD_PATH.read_text()

    def test_md_contains_final_decision(self):
        assert "CALIBRATION_COMPLETE" in self._md()

    def test_md_contains_gate_chain(self):
        assert "Gate Chain" in self._md()

    def test_md_contains_epic_complete(self):
        assert "complete" in self._md().lower()

    def test_md_contains_guardrails_section(self):
        assert "Guardrails" in self._md()
