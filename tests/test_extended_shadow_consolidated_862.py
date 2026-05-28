"""Tests for Extended Shadow Monitoring Consolidated Report — Epic #857, #862.

Verifies:
- Consolidated report exists and is valid JSON
- Final decision is from the allowed set
- Final decision is NOT from the forbidden set
- All guardrails are correctly set
- Cumulative metrics are complete and consistent
- Epic closure is declared
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/mission_brain/shadow_monitoring/862/extended_shadow_consolidated_862.json"


def _load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


class TestConsolidatedArtifacts:

    def test_json_report_exists(self):
        assert REPORT.exists()

    def test_md_report_exists(self):
        md = ROOT / "reports/mission_brain/shadow_monitoring/862/extended_shadow_consolidated_862.md"
        assert md.exists()


class TestFinalDecision:

    def test_final_decision_is_allowed(self):
        from igris.agent.mission.shadow_monitoring_decision import ALLOWED_DECISIONS
        d = _load()
        assert d["final_decision"] in ALLOWED_DECISIONS, \
            f"Decision '{d['final_decision']}' not in allowed set"

    def test_final_decision_not_forbidden(self):
        from igris.agent.mission.shadow_monitoring_decision import FORBIDDEN_DECISIONS
        d = _load()
        assert d["final_decision"] not in FORBIDDEN_DECISIONS, \
            f"Forbidden decision returned: {d['final_decision']}"

    def test_decision_rationale_nonempty(self):
        d = _load()
        assert d.get("decision_rationale"), "Decision rationale must not be empty"

    def test_evaluation_passed(self):
        d = _load()
        assert d["evaluation"] == "passed"


class TestConsolidatedGuardrails:

    def test_shadow_mode_only(self):
        d = _load()
        assert d["guardrails"]["shadow_mode_only"] is True

    def test_default_behavior_unchanged(self):
        d = _load()
        assert d["guardrails"]["default_behavior_unchanged"] is True

    def test_no_enable_by_default(self):
        d = _load()
        assert d["guardrails"]["no_enable_by_default"] is True

    def test_no_mandatory_gate(self):
        d = _load()
        assert d["guardrails"]["no_mandatory_gate"] is True

    def test_rollback_path_ok(self):
        d = _load()
        assert d["guardrails"]["rollback_path_status"] == "ok"


class TestConsolidatedMetrics:

    def test_total_cycles_30(self):
        d = _load()
        assert d["total_shadow_cycles"] == 30

    def test_cumulative_no_critical_false_completed(self):
        d = _load()
        assert d["cumulative_metrics"]["potential_critical_false_completed_all"] == 0

    def test_cumulative_no_risk_introduced(self):
        d = _load()
        assert d["cumulative_metrics"]["risk_introduced_candidates_all"] == 0

    def test_cumulative_cost_zero(self):
        d = _load()
        assert d["cumulative_metrics"]["cost_overhead_total_usd"] == 0.0

    def test_stability_verdict_present(self):
        d = _load()
        assert d["cumulative_metrics"].get("stability_verdict")

    def test_epic_closure_declared(self):
        d = _load()
        assert d.get("epic_closure") is True

    def test_all_3_batches_present(self):
        d = _load()
        assert "baseline_845" in d["batches"]
        assert "batch1_859" in d["batches"]
        assert "batch2_860" in d["batches"]
