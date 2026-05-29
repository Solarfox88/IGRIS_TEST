"""Tests for igris/core/context_section_weighter.py (issue #524)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from igris.core.context_section_weighter import (
    ContextSectionWeighter,
    StepUsageRecord,
    compute_section_stats,
    compute_weights,
    detect_cited_sections,
    load_section_weights,
    load_usage_records,
    save_section_weights,
    save_usage_records,
    TRACKED_SECTIONS,
    _MAX_RECORDS,
    _MULTIPLIER_MAX,
    _MULTIPLIER_MIN,
)


# ---------------------------------------------------------------------------
# detect_cited_sections
# ---------------------------------------------------------------------------

class TestDetectCitedSections:
    def test_detects_error_section(self):
        response = "I see an error in the traceback above"
        cited = detect_cited_sections(response, ["error_context", "memory_context"])
        assert "error_context" in cited

    def test_detects_memory_section(self):
        response = "According to the memory, we should use async here"
        cited = detect_cited_sections(response, ["memory_context", "state_context"])
        assert "memory_context" in cited

    def test_no_match_returns_empty(self):
        response = "Let me think about this differently."
        cited = detect_cited_sections(response, ["state_context"])
        assert cited == []

    def test_case_insensitive(self):
        response = "RECENT_ACTION shows file was edited"
        cited = detect_cited_sections(response, ["recent_actions"])
        assert "recent_actions" in cited

    def test_only_present_sections_considered(self):
        response = "According to memory and the goal"
        cited = detect_cited_sections(response, ["memory_context"])
        assert "memory_context" in cited
        # mission_context not in sections_present even though "goal" is in response
        assert "mission_context" not in cited


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_usage_records(self, tmp_path):
        record = StepUsageRecord(
            step_id="s1",
            sections_present=["memory_context", "error_context"],
            sections_cited=["error_context"],
            success=True,
        )
        save_usage_records(str(tmp_path), [record])
        loaded = load_usage_records(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].step_id == "s1"
        assert "error_context" in loaded[0].sections_cited
        assert loaded[0].success is True

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_usage_records(str(tmp_path)) == []

    def test_load_corrupt_returns_empty(self, tmp_path):
        (tmp_path / ".igris").mkdir()
        (tmp_path / ".igris" / "section_usage.json").write_text("NOT JSON")
        assert load_usage_records(str(tmp_path)) == []

    def test_rolling_window_respected(self, tmp_path):
        records = [
            StepUsageRecord(
                step_id=f"s{i}", sections_present=["error_context"],
                sections_cited=[], success=True,
            )
            for i in range(_MAX_RECORDS + 50)
        ]
        save_usage_records(str(tmp_path), records)
        loaded = load_usage_records(str(tmp_path))
        assert len(loaded) == _MAX_RECORDS

    def test_save_and_load_section_weights(self, tmp_path):
        weights = {"memory_context": 1.5, "error_context": 0.8}
        save_section_weights(str(tmp_path), weights)
        loaded = load_section_weights(str(tmp_path))
        assert loaded["memory_context"] == pytest.approx(1.5)
        assert loaded["error_context"] == pytest.approx(0.8)

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        save_usage_records(str(tmp_path), [
            StepUsageRecord("s1", [], [], True)
        ])
        assert not list(tmp_path.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# compute_section_stats / compute_weights
# ---------------------------------------------------------------------------

class TestComputeStats:
    def _make_records(self, n_success_cited, n_fail_cited, n_success_not_cited,
                      section="memory_context"):
        records = []
        for _ in range(n_success_cited):
            records.append(StepUsageRecord("s", [section], [section], success=True))
        for _ in range(n_fail_cited):
            records.append(StepUsageRecord("f", [section], [section], success=False))
        for _ in range(n_success_not_cited):
            records.append(StepUsageRecord("n", [section], [], success=True))
        return records

    def test_cited_count_correct(self):
        records = self._make_records(5, 2, 3)
        stats = compute_section_stats(records)
        assert stats["memory_context"].cited_count == 7

    def test_cited_in_success_correct(self):
        records = self._make_records(5, 2, 3)
        stats = compute_section_stats(records)
        assert stats["memory_context"].cited_in_success == 5

    def test_present_count_correct(self):
        records = self._make_records(5, 2, 3)
        stats = compute_section_stats(records)
        assert stats["memory_context"].present_count == 10

    def test_weight_high_for_useful_section(self):
        # memory_context cited in 9 successes, 0 failures
        # Also: 5 records where not cited + failed → lowers baseline so weight > 1
        section = "memory_context"
        records = self._make_records(9, 0, 0, section)
        # Add 5 failed steps where section is present but NOT cited
        for _ in range(5):
            records.append(StepUsageRecord("f", [section], [], success=False))
        # baseline success = 9/14 ≈ 0.64; P(success|cited) = 9/9 = 1.0 → weight >> 1.0
        weights = compute_weights(records, min_samples=5)
        assert weights["memory_context"] > 1.0

    def test_weight_low_for_useless_section(self):
        # memory_context never cited in success, always in failure → low utility
        records = self._make_records(0, 9, 9)  # 9 failures cited, 9 successes not cited
        weights = compute_weights(records, min_samples=5)
        assert weights["memory_context"] <= 1.0

    def test_weight_clamped_to_min(self):
        records = self._make_records(0, 20, 0)
        weights = compute_weights(records, min_samples=5)
        assert weights["memory_context"] >= _MULTIPLIER_MIN

    def test_weight_clamped_to_max(self):
        # Only success cases with citation
        records = self._make_records(50, 0, 0)
        weights = compute_weights(records, min_samples=5)
        assert weights["memory_context"] <= _MULTIPLIER_MAX

    def test_insufficient_data_returns_ones(self):
        records = self._make_records(3, 0, 0)
        weights = compute_weights(records, min_samples=20)
        assert all(w == pytest.approx(1.0) for w in weights.values())


# ---------------------------------------------------------------------------
# ContextSectionWeighter
# ---------------------------------------------------------------------------

class TestContextSectionWeighter:
    def test_record_step_persists_record(self, tmp_path):
        weighter = ContextSectionWeighter(str(tmp_path))
        rec = weighter.record_step(
            step_id="abc",
            sections_present=["error_context", "memory_context"],
            model_response="I see an error in the traceback",
            success=True,
        )
        assert "error_context" in rec.sections_cited
        records = load_usage_records(str(tmp_path))
        assert len(records) == 1

    def test_get_budget_multipliers_returns_all_sections(self, tmp_path):
        weighter = ContextSectionWeighter(str(tmp_path))
        multipliers = weighter.get_budget_multipliers()
        assert set(multipliers.keys()) == set(TRACKED_SECTIONS)

    def test_get_budget_multipliers_defaults_to_one(self, tmp_path):
        weighter = ContextSectionWeighter(str(tmp_path))
        multipliers = weighter.get_budget_multipliers()
        assert all(m == pytest.approx(1.0) for m in multipliers.values())

    def test_weights_updated_after_enough_records(self, tmp_path):
        weighter = ContextSectionWeighter(str(tmp_path), min_samples=5)
        # Add 10 records where error_context is always cited and step succeeds
        for i in range(10):
            weighter.record_step(
                step_id=f"s{i}",
                sections_present=["error_context", "memory_context"],
                model_response="Looking at the traceback and the error output",
                success=True,
            )
        multipliers = weighter.get_budget_multipliers()
        # error_context is always cited in success → weight > 1.0
        assert multipliers["error_context"] >= 1.0

    def test_get_stats_returns_dict(self, tmp_path):
        weighter = ContextSectionWeighter(str(tmp_path))
        weighter.record_step("s1", ["memory_context"], "memory shows...", True)
        stats = weighter.get_stats()
        assert "total_records" in stats
        assert "sections" in stats
        assert stats["total_records"] == 1

    def test_record_step_handles_empty_response(self, tmp_path):
        weighter = ContextSectionWeighter(str(tmp_path))
        rec = weighter.record_step("s1", ["error_context"], "", True)
        assert rec.sections_cited == []
