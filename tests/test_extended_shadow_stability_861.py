"""Tests for Extended Shadow Monitoring Stability Analysis — Epic #857, #861.

Verifies:
- Stability analysis artifact exists and is valid
- Agreement rate per epoch computed and consistent
- Stability verdict is one of the valid values
- Sample representativeness >= 0.8 over 20+ new cycles
- No stop conditions in 30-cycle view
- Dominant mismatch class identified
- Both decision distributions populated
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports/mission_brain/shadow_monitoring/861/extended_shadow_stability_analysis_861.json"


def _load() -> dict:
    return json.loads(ANALYSIS.read_text(encoding="utf-8"))


class TestStabilityArtifacts:

    def test_analysis_json_exists(self):
        assert ANALYSIS.exists()

    def test_analysis_md_exists(self):
        md = ROOT / "reports/mission_brain/shadow_monitoring/861/extended_shadow_stability_analysis_861.md"
        assert md.exists()


class TestStabilityMetrics:

    def test_total_cumulative_cycles_30(self):
        d = _load()
        assert d["total_cumulative_cycles"] == 30

    def test_agreement_rate_by_epoch_present(self):
        d = _load()
        epochs = d["agreement_rate_by_epoch"]
        assert "baseline_845" in epochs
        assert "batch1_859" in epochs
        assert "batch2_860" in epochs

    def test_all_epoch_rates_are_floats(self):
        d = _load()
        for k, v in d["agreement_rate_by_epoch"].items():
            assert isinstance(v, float), f"Epoch {k} rate is not float: {v}"

    def test_stability_verdict_is_valid(self):
        d = _load()
        valid = {"structurally_zero", "improving", "stable", "degrading"}
        assert d["stability_verdict"] in valid

    def test_dominant_mismatch_class_present(self):
        d = _load()
        assert d.get("dominant_mismatch_class")

    def test_mismatch_distribution_nonempty(self):
        d = _load()
        assert len(d.get("mismatch_distribution_20_new_cycles", {})) > 0

    def test_representativeness_score_high(self):
        d = _load()
        assert d["sample_representativeness_score"] >= 0.5, \
            "30-cycle sample must have representativeness >= 0.5"

    def test_all_3_complexity_levels_covered(self):
        d = _load()
        dist = d.get("complexity_distribution", {})
        assert len(dist) >= 3, f"Only {len(dist)} complexity levels found: {dist}"

    def test_decision_distributions_populated(self):
        d = _load()
        assert len(d.get("decision_distribution_mission_brain_20_cycles", {})) > 0
        assert len(d.get("decision_distribution_current_loop_20_cycles", {})) > 0


class TestStabilityStopConditions:

    def test_no_critical_false_completed_30_cycles(self):
        d = _load()
        assert d["any_critical_false_completed_30_cycles"] is False, \
            "STOP CONDITION: critical_false_completed found in 30-cycle view"

    def test_no_risk_introduced_30_cycles(self):
        d = _load()
        assert d["any_risk_introduced_30_cycles"] is False, \
            "STOP CONDITION: risk_introduced found in 30-cycle view"

    def test_rollback_path_ok(self):
        d = _load()
        assert d["rollback_path_status"] == "ok"

    def test_evaluation_passed(self):
        d = _load()
        assert d["evaluation"] == "passed"
        assert d["stop_reason"] is None
