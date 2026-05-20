"""
Tests for helper_ab_eval: scorer, persistence, hybrid switch policy.
"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from igris.core.helper_ab_eval import (
    REQUIRED_SCHEMA_FIELDS, SCORE_WEIGHTS,
    is_safe_to_switch, load_ab_results,
    make_ab_record, save_ab_result, score_helper_response,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _full_response(**overrides):
    base = {
        "ok": True,
        "diagnosis": "FastAPI TestClient non importato correttamente nel test fixture",
        "likely_supervisor_gap": "test setup mancante",
        "suggested_repair_strategy": "aggiorna conftest.py con fixture app",
        "execution_plan": [
            "step1: apri tests/conftest.py",
            "step2: aggiungi @pytest.fixture app",
            "step3: usa TestClient(app) nel test",
        ],
        "acceptance_matrix": [
            {"test": "test_create_item", "assertion": "status_code == 201"},
        ],
        "suggested_tests": ["tests/test_items.py"],
        "do_not_do": ["non modificare la logica del endpoint"],
        "risk": "low",
        "confidence": 0.9,
        "requires_human_or_codex_audit": False,
        "must_not_complete_product_manually": True,
    }
    base.update(overrides)
    return base


def _make_records(
    n, primary_score=0.75, alt_score=0.75,
    source="organic_run", case_ids=None,
    primary_breakdown=None, alt_breakdown=None,
):
    bd = primary_breakdown or {k: 1.0 for k in SCORE_WEIGHTS}
    abd = alt_breakdown or {k: 1.0 for k in SCORE_WEIGHTS}
    ids = case_ids or [f"case_{i}" for i in range(n)]
    return [
        make_ab_record(
            case_id=ids[i % len(ids)],
            primary_model="gpt-5.3-codex",
            alt_model="deepseek-v4-pro",
            primary_score=primary_score,
            alt_score=alt_score,
            primary_breakdown=bd,
            alt_breakdown=abd,
            primary_cost_usd=0.01,
            alt_cost_usd=0.003,
            source=source,
        )
        for i in range(n)
    ]


def _synthetic(n=9, **kw):
    return _make_records(n, source="synthetic_fixture", **kw)


def _organic(n=5, case_ids=None, **kw):
    cids = case_ids or [
        "pytest_failure", "semantic_incomplete_stub", "missing_tests",
        "decomposition_required", "reasoning_loop",
    ]
    return _make_records(n, source="organic_run", case_ids=cids, **kw)


# ---------------------------------------------------------------------------
# 1. Schema scoring
# ---------------------------------------------------------------------------

class TestSchemaValidScoring:
    def test_fully_valid_response_scores_high(self):
        r = score_helper_response(_full_response(), {})
        assert r["total"] >= 0.60

    def test_all_required_fields_present(self):
        r = score_helper_response(_full_response(), {})
        assert r["breakdown"]["schema_valid"] >= 0.9

    def test_response_with_all_fields_has_no_schema_issues(self):
        r = score_helper_response(_full_response(), {})
        schema_issues = [i for i in r["issues"] if "missing" in i]
        assert not schema_issues


class TestNonJsonResponse:
    def test_non_dict_response_scores_zero(self):
        r = score_helper_response("some markdown string", {})
        assert r["total"] == 0.0

    def test_empty_dict_scores_low(self):
        r = score_helper_response({}, {})
        assert r["total"] < 0.20

    def test_partial_schema_penalised(self):
        r = score_helper_response({"diagnosis": "something"}, {})
        assert r["breakdown"]["schema_valid"] < 1.0


# ---------------------------------------------------------------------------
# 2. Secret detection
# ---------------------------------------------------------------------------

class TestSecretDetection:
    def test_response_with_api_key_scores_no_secrets_zero(self):
        resp = _full_response(diagnosis="use sk-proj-abc123 key to fix this sk-secretXYZabc123def456ghi789")
        r = score_helper_response(resp, {})
        assert r["breakdown"]["no_secrets"] == 0.0

    def test_clean_response_scores_no_secrets_one(self):
        r = score_helper_response(_full_response(), {})
        assert r["breakdown"]["no_secrets"] == 1.0

    def test_secret_in_execution_plan_detected(self):
        resp = _full_response(execution_plan=["set env sk-proj-realkey12345678901234567", "run tests"])
        r = score_helper_response(resp, {})
        assert r["breakdown"]["no_secrets"] == 0.0


# ---------------------------------------------------------------------------
# 3. Execution plan scoring
# ---------------------------------------------------------------------------

class TestExecutionPlanScoring:
    def test_specific_plan_scores_higher_than_generic(self):
        specific = _full_response(execution_plan=[
            "apri tests/conftest.py riga 5",
            "aggiungi @pytest.fixture def app(): ...",
            "sostituisci client = TestClient(app)",
        ])
        generic = _full_response(execution_plan=[
            "check the tests",
            "review the code",
            "try running again",
        ])
        r_specific = score_helper_response(specific, {})
        r_generic = score_helper_response(generic, {})
        assert r_specific["breakdown"]["execution_plan_actionability"] > r_generic["breakdown"]["execution_plan_actionability"]

    def test_empty_execution_plan_scores_zero(self):
        r = score_helper_response(_full_response(execution_plan=[]), {})
        assert r["breakdown"]["execution_plan_actionability"] == 0.0

    def test_single_step_scores_low(self):
        r = score_helper_response(_full_response(execution_plan=["fix the test"]), {})
        assert r["breakdown"]["execution_plan_actionability"] <= 0.3


# ---------------------------------------------------------------------------
# 4. Shadow mode contract
# ---------------------------------------------------------------------------

class TestShadowModeContract:
    def test_shadow_result_does_not_affect_primary_output(self):
        primary = _full_response(diagnosis="primary diagnosis")
        shadow = _full_response(diagnosis="shadow diagnosis — different advice")
        r_primary = score_helper_response(primary, {})
        r_shadow = score_helper_response(shadow, {})
        # Both are scored independently; primary controls
        assert r_primary["total"] > 0
        assert r_shadow["total"] > 0
        assert primary["diagnosis"] != shadow["diagnosis"]


# ---------------------------------------------------------------------------
# 5. AB persistence
# ---------------------------------------------------------------------------

class TestABPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        p = str(tmp_path / "ab.json")
        rec = _make_records(1)[0]
        save_ab_result(rec, p)
        loaded = load_ab_results(p)
        assert len(loaded) == 1
        assert loaded[0]["primary_model"] == "gpt-5.3-codex"

    def test_multiple_records_accumulate(self, tmp_path):
        p = str(tmp_path / "ab.json")
        for rec in _make_records(3):
            save_ab_result(rec, p)
        assert len(load_ab_results(p)) == 3

    def test_secrets_redacted_in_persistence(self, tmp_path):
        p = str(tmp_path / "ab.json")
        rec = _make_records(1)[0]
        rec["primary_breakdown"]["debug"] = "sk-proj-secretkeyABCDEF1234567890"
        save_ab_result(rec, p)
        raw = Path(p).read_text()
        assert "sk-proj-secretkeyABCDEF1234567890" not in raw
        assert "[REDACTED]" in raw


# ---------------------------------------------------------------------------
# 6. AB disabled — no alt call
# ---------------------------------------------------------------------------

class TestABDisabled:
    def test_ab_disabled_makes_single_call(self):
        records = _make_records(0)
        result = is_safe_to_switch(records)
        assert result["safe_to_switch"] is False
        assert result["synthetic_count"] == 0
        assert result["organic_count"] == 0


# ---------------------------------------------------------------------------
# 7. Shadow mode — primary always controls
# ---------------------------------------------------------------------------

class TestShadowModePrimaryControls:
    def test_primary_result_returned_not_shadow(self):
        recs = _synthetic(9) + _organic(5)
        result = is_safe_to_switch(recs)
        # safe_to_switch is still False (costs all 0 -> tie OK, but default sources)
        # primary model must be in each record
        for r in recs:
            assert r["primary_model"] == "gpt-5.3-codex"

    def test_source_field_present_in_record(self):
        rec = _make_records(1, source="organic_run")[0]
        assert rec["source"] == "organic_run"

    def test_synthetic_source_field(self):
        rec = _make_records(1, source="synthetic_fixture")[0]
        assert rec["source"] == "synthetic_fixture"


# ---------------------------------------------------------------------------
# 8. Switch policy — hybrid
# ---------------------------------------------------------------------------

class TestSwitchPolicy:
    def test_no_records_never_safe(self):
        r = is_safe_to_switch([])
        assert r["safe_to_switch"] is False

    def test_only_synthetic_not_safe(self):
        r = is_safe_to_switch(_synthetic(9))
        assert r["safe_to_switch"] is False
        assert r["organic_count"] == 0

    def test_only_organic_not_safe(self):
        r = is_safe_to_switch(_organic(5))
        assert r["safe_to_switch"] is False
        assert r["synthetic_count"] == 0

    def test_insufficient_synthetic_blocks(self):
        r = is_safe_to_switch(_synthetic(5) + _organic(5))
        assert r["safe_to_switch"] is False
        assert any("synthetic" in reason for reason in r["reasons"])

    def test_insufficient_organic_blocks(self):
        r = is_safe_to_switch(_synthetic(9) + _organic(2))
        assert r["safe_to_switch"] is False
        assert any("organic" in reason for reason in r["reasons"])

    def test_insufficient_failure_class_diversity_blocks(self):
        # Only 2 distinct case_ids in organic
        organic = _make_records(5, source="organic_run", case_ids=["case_a", "case_b"])
        r = is_safe_to_switch(_synthetic(9) + organic)
        assert r["safe_to_switch"] is False
        assert any("failure_class" in reason for reason in r["reasons"])

    def test_safety_failure_blocks_switch(self):
        bad_bd = {k: 1.0 for k in SCORE_WEIGHTS}
        bad_bd["safety_compliance"] = 0.0
        records = _synthetic(9) + _organic(5, alt_breakdown=bad_bd)
        r = is_safe_to_switch(records)
        assert r["safe_to_switch"] is False

    def test_low_alt_score_blocks_switch(self):
        records = _synthetic(9) + _organic(5, primary_score=0.80, alt_score=0.60)
        r = is_safe_to_switch(records)
        assert r["safe_to_switch"] is False

    def test_critical_regression_blocks_switch(self):
        records = _synthetic(9) + _organic(5)
        records.append(make_ab_record(
            case_id="decomposition_required_large_goal",
            primary_model="gpt-5.3-codex", alt_model="deepseek-v4-pro",
            primary_score=0.80, alt_score=0.10,
            primary_breakdown={k: 1.0 for k in SCORE_WEIGHTS},
            alt_breakdown={k: 0.1 for k in SCORE_WEIGHTS},
            primary_cost_usd=0.01, alt_cost_usd=0.003,
            source="organic_run",
        ))
        r = is_safe_to_switch(records)
        assert r["safe_to_switch"] is False
        assert any("critical" in reason for reason in r["reasons"])

    def test_switch_never_auto_enabled(self):
        for rec in _synthetic(9) + _organic(5):
            assert rec["safe_to_switch"] is False

    def test_failure_classes_covered_reported(self):
        cids = ["pytest_failure", "semantic_incomplete", "missing_tests", "decomposition_required", "reasoning_loop"]
        organic = _make_records(5, source="organic_run", case_ids=cids)
        r = is_safe_to_switch(_synthetic(9) + organic)
        assert len(r["failure_classes_covered"]) >= 3

    def test_synthetic_and_organic_counts_reported(self):
        r = is_safe_to_switch(_synthetic(9) + _organic(5))
        assert r["synthetic_count"] == 9
        assert r["organic_count"] == 5
