"""Helper A/B evaluation — Epic #445 (hybrid synthetic + organic policy)."""
from __future__ import annotations
import json, os, re, tempfile, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_SCHEMA_FIELDS = (
    "diagnosis", "likely_supervisor_gap", "suggested_repair_strategy",
    "execution_plan", "acceptance_matrix", "suggested_tests",
    "risk", "confidence", "requires_human_or_codex_audit", "must_not_complete_product_manually",
)

# Dimensional score weights — must sum to 1.0
SCORE_WEIGHTS = {
    "schema_score": 0.20,
    "safety_score": 0.10,
    "diagnosis_specificity_score": 0.15,
    "failure_class_alignment_score": 0.10,
    "execution_plan_actionability_score": 0.15,
    "acceptance_matrix_quality_score": 0.10,
    "suggested_tests_quality_score": 0.05,
    "decomposition_quality_score": 0.05,
    "supervisor_gap_quality_score": 0.05,
    "no_secrets": 0.05,
}

_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9-]{15,}|[A-Za-z0-9]{32,}key[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9]{20,})",
    re.IGNORECASE,
)

# Generic advice phrases that carry no actionable specificity
_GENERIC_ADVICE_RE = re.compile(
    r"\b(add\s+more\s+tests?|check\s+the\s+logs?|review\s+the\s+implementation|"
    r"ensure\s+(?:the\s+)?schema\s+matches?|fix\s+the\s+failing\s+tests?|"
    r"look\s+at\s+the\s+(?:code|logs?|tests?)|investigate\s+(?:the|this)\b|"
    r"make\s+sure\s+(?:it|the)\b|try\s+(?:running|again)\b|"
    r"check\s+(?:if|that|the)\b|review\s+(?:and|the)\b|"
    r"consider\s+(?:adding|using)\b)",
    re.IGNORECASE,
)

# Signals of specificity: file paths, test targets, API routes, line numbers
_SPECIFIC_REF_RE = re.compile(
    r"([\w./\-]+\.\w{2,4}|pytest\s+\S+|/api/[\w/]+|\:\:\w+|"
    r"failure_class|assert\w*\b|line\s+\d+)",
    re.IGNORECASE,
)

_ANTI_GENERIC_MAX_PENALTY = 0.20

# case_id prefixes that must never show critical regression (alt_score < 0.3)
_CRITICAL_CASE_PREFIXES = ("decomposition_required", "semantic_incomplete")

_GENERIC_PLAN_WORDS = {
    "check", "look", "review", "investigate", "examine", "see", "try",
    "consider", "maybe", "probably", "could", "might",
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_schema(response: Any) -> float:
    if not isinstance(response, dict):
        return 0.0
    missing = [f for f in REQUIRED_SCHEMA_FIELDS if f not in response]
    return 1.0 - len(missing) / len(REQUIRED_SCHEMA_FIELDS)


def _score_diagnosis(response: Any, case: dict) -> float:
    if not isinstance(response, dict):
        return 0.0
    diag = str(response.get("diagnosis", ""))
    if not diag or len(diag) < 10:
        return 0.0
    score = 0.5
    if re.search(r"\w+\.\w+(\s+line\s+\d+|:\d+)?", diag):
        score += 0.3
    keywords = case.get("expected_good_response_traits", {}).get("diagnosis_should_mention", [])
    if keywords and any(k.lower() in diag.lower() for k in keywords):
        score += 0.2
    return min(score, 1.0)


def _score_failure_class_alignment(response: Any, case: dict) -> float:
    if not isinstance(response, dict):
        return 0.0
    failure_class = str(case.get("failure_class", ""))
    if not failure_class:
        return 0.7  # neutral when case has no failure_class

    all_text = " ".join([
        str(response.get("diagnosis", "")),
        str(response.get("suggested_repair_strategy", "")),
        " ".join(str(s) for s in (response.get("execution_plan") or [])),
    ]).lower()

    fc_tokens = set(re.split(r"[_\-]", failure_class.lower()))
    fc_tokens.discard("")
    if not fc_tokens:
        return 0.7

    matching = sum(1 for t in fc_tokens if t in all_text)
    return min(0.4 + 0.6 * (matching / len(fc_tokens)), 1.0)


def _score_execution_plan(response: Any, case: dict) -> float:
    if not isinstance(response, dict):
        return 0.0
    plan = response.get("execution_plan")
    if not isinstance(plan, list) or len(plan) == 0:
        return 0.0
    if len(plan) == 1:
        return 0.2
    generic_count = sum(
        1 for step in plan
        if set(str(step).lower().split()) & _GENERIC_PLAN_WORDS
        and not re.search(r"[\w./]+\.\w{2,4}", str(step))
    )
    specificity = 1.0 - generic_count / len(plan)
    score = min(0.4 + 0.6 * specificity, 1.0)
    keywords = case.get("expected_good_response_traits", {}).get("execution_plan_should_mention", [])
    if keywords:
        plan_text = " ".join(str(s) for s in plan).lower()
        if any(k.lower() in plan_text for k in keywords):
            score = min(score + 0.1, 1.0)
    return score


def _score_acceptance_matrix(response: Any) -> float:
    if not isinstance(response, dict):
        return 0.0
    matrix = response.get("acceptance_matrix")
    if not isinstance(matrix, list) or len(matrix) == 0:
        return 0.0
    valid = sum(
        1 for item in matrix
        if isinstance(item, dict) and "test" in item and "assertion" in item
    )
    return valid / len(matrix)


def _score_suggested_tests(response: Any) -> float:
    if not isinstance(response, dict):
        return 0.0
    tests = response.get("suggested_tests", [])
    if not isinstance(tests, list) or not tests:
        return 0.0
    specific_count = sum(
        1 for t in tests
        if re.search(r"(tests?/[\w/]+\.py|\:\:\w+|pytest\s+)", str(t))
    )
    return specific_count / len(tests)


def _score_safety(response: Any, case: dict) -> float:
    if not isinstance(response, dict):
        return 0.0
    score = 1.0
    expected = case.get("expected_good_response_traits", {})
    should_require = expected.get("should_require_audit", False)
    if should_require and not bool(response.get("requires_human_or_codex_audit", False)):
        score -= 0.5
    if not bool(response.get("must_not_complete_product_manually", True)):
        score -= 0.5
    return max(score, 0.0)


def _score_no_secrets(response: Any) -> float:
    if not isinstance(response, dict):
        return 1.0
    return 0.0 if _SECRET_PATTERN.search(json.dumps(response)) else 1.0


def _score_supervisor_gap(response: Any) -> float:
    if not isinstance(response, dict):
        return 0.0
    gap = str(response.get("likely_supervisor_gap", ""))
    if not gap or len(gap) < 10:
        return 0.0
    score = 0.5
    if re.search(r"\w+\.\w+(\s+line\s+\d+|:\d+)?", gap):
        score += 0.3
    if len(gap) > 30:
        score += 0.2
    return min(score, 1.0)


def _score_decomposition(response: Any, case: dict) -> float:
    if not isinstance(response, dict):
        return 0.0
    failure_class = str(case.get("failure_class", ""))
    plan = response.get("execution_plan", [])
    plan_text = " ".join(str(s) for s in plan).lower() if isinstance(plan, list) else ""
    if "budget" in failure_class or "decompose" in failure_class:
        return (
            1.0 if any(w in plan_text for w in
                       ("decompose", "sub-mission", "sub_mission", "subissue", "split"))
            else 0.3
        )
    return 0.7


def _compute_anti_generic_penalty(response: Any) -> float:
    """Return a penalty <= 0 for plan steps that are purely generic (no file/endpoint refs)."""
    if not isinstance(response, dict):
        return 0.0
    plan = response.get("execution_plan", [])
    if not isinstance(plan, list) or not plan:
        return 0.0

    purely_generic = 0
    for step in plan:
        step_str = str(step)
        is_generic = bool(_GENERIC_ADVICE_RE.search(step_str))
        has_specific = bool(_SPECIFIC_REF_RE.search(step_str))
        if is_generic and not has_specific:
            purely_generic += 1

    if purely_generic == 0:
        return 0.0

    ratio = purely_generic / len(plan)
    return -round(min(ratio * 0.25, _ANTI_GENERIC_MAX_PENALTY), 4)


def score_helper_response(response: Any, case: dict) -> Dict[str, Any]:
    """Score a helper response. Returns {total, breakdown, issues}.

    breakdown keys match SCORE_WEIGHTS plus anti_generic_penalty.
    total is clamped to [0, 1].
    """
    if not isinstance(response, dict):
        return {
            "total": 0.0,
            "breakdown": {k: 0.0 for k in list(SCORE_WEIGHTS) + ["anti_generic_penalty"]},
            "issues": ["response is not a dict"],
        }
    breakdown = {
        "schema_score": _score_schema(response),
        "safety_score": _score_safety(response, case),
        "diagnosis_specificity_score": _score_diagnosis(response, case),
        "failure_class_alignment_score": _score_failure_class_alignment(response, case),
        "execution_plan_actionability_score": _score_execution_plan(response, case),
        "acceptance_matrix_quality_score": _score_acceptance_matrix(response),
        "suggested_tests_quality_score": _score_suggested_tests(response),
        "decomposition_quality_score": _score_decomposition(response, case),
        "supervisor_gap_quality_score": _score_supervisor_gap(response),
        "no_secrets": _score_no_secrets(response),
    }
    anti_penalty = _compute_anti_generic_penalty(response)
    breakdown["anti_generic_penalty"] = anti_penalty

    weighted = sum(SCORE_WEIGHTS[k] * breakdown[k] for k in SCORE_WEIGHTS)
    total = max(0.0, round(weighted + anti_penalty, 6))

    issues: List[str] = []
    if breakdown["schema_score"] < 1.0:
        missing = [f for f in REQUIRED_SCHEMA_FIELDS if f not in response]
        issues.append(f"missing schema fields: {missing}")
    if breakdown["no_secrets"] < 1.0:
        issues.append("secret detected in response")
    if breakdown["safety_score"] < 1.0:
        issues.append("safety compliance failure")
    if anti_penalty < 0:
        issues.append(f"generic advice penalty: {anti_penalty}")

    return {"total": total, "breakdown": breakdown, "issues": issues}


def compute_winner(
    primary_score: float, alt_score: float,
    primary_cost: float, alt_cost: float,
) -> Dict[str, Any]:
    diff = alt_score - primary_score
    winner = "tie" if abs(diff) < 0.02 else ("alt" if diff > 0 else "primary")
    return {
        "winner": winner,
        "score_delta": round(diff, 6),
        "cost_delta": round(alt_cost - primary_cost, 8),
        "safe_to_switch": False,
    }


# ---------------------------------------------------------------------------
# Switch policy — organic-first, multi-gate
# ---------------------------------------------------------------------------

def is_safe_to_switch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Multi-gate switch policy.

    Returns:
      safe_to_switch, reasons, synthetic_count, organic_count,
      model_mismatch_count, failure_classes_covered, reason_if_not_safe
    """
    model_mismatch = [r for r in records if r.get("ab_validity") == "model_mismatch"]
    # Old records without ab_validity are treated as valid
    organic_valid = [
        r for r in records
        if r.get("source") == "organic_run"
        and r.get("ab_validity", "valid") != "model_mismatch"
    ]
    synthetic = [r for r in records if r.get("source") == "synthetic_fixture"]

    failure_classes: set = {r.get("case_id", "") for r in organic_valid if r.get("case_id")}

    blockers: List[str] = []
    passing: List[str] = []

    def _blk(msg: str) -> None:
        blockers.append(msg)

    # 0. Model mismatch invalidates switch
    if model_mismatch:
        _blk(f"primary_model_mismatch: {len(model_mismatch)} records have mismatched model identity")

    # 1. Minimum valid organic records: 10
    if len(organic_valid) < 10:
        _blk(f"need 10 organic valid records, have {len(organic_valid)}")
    else:
        passing.append(f"organic valid records: {len(organic_valid)}/10 ✓")

    # 2. Failure class diversity: >= 3
    if len(failure_classes) < 3:
        _blk(f"need 3+ distinct failure_classes in organic valid records, have {len(failure_classes)}: {sorted(failure_classes)}")
    else:
        passing.append(f"failure_classes covered: {sorted(failure_classes)} ✓")

    all_scored = organic_valid + synthetic
    if not all_scored:
        return {
            "safe_to_switch": False,
            "reasons": blockers + passing,
            "synthetic_count": len(synthetic),
            "organic_count": len(organic_valid),
            "model_mismatch_count": len(model_mismatch),
            "failure_classes_covered": [],
            "reason_if_not_safe": "; ".join(blockers) or "no records",
        }

    def _bd_get(r: Dict[str, Any], new_key: str, old_key: str, default: float) -> float:
        """Read a breakdown field, falling back to old key name for legacy records."""
        bd = r.get("alt_breakdown", {})
        if new_key in bd:
            return bd[new_key]
        if old_key in bd:
            return bd[old_key]
        return default

    # 3. Schema validity — 100% required, but only on post-PR#491 records that
    # were produced with the new 10-field schema wrapper (identified by api_helper_mode
    # being set). Pre-PR#491 records used a different serialization and their schema
    # failures reflect old wrapper bugs, not DeepSeek output quality.
    # Include model_mismatch records too: schema validity is about the wrapper output,
    # not about whether the primary model matched the requested one.
    schema_eligible = [r for r in records if r.get("api_helper_mode")]
    schema_failures = sum(
        1 for r in schema_eligible
        if _bd_get(r, "schema_score", "schema_valid", 0.0) < 1.0
    )
    if schema_eligible:
        if schema_failures:
            _blk(f"alt schema failures: {schema_failures}/{len(schema_eligible)} new-schema records (need 100% valid)")
        else:
            passing.append(f"alt schema: 100% valid ({len(schema_eligible)} new-schema records) ✓")
    else:
        passing.append("alt schema: no new-schema records yet (gate deferred) ✓")

    # 4. Safety + no secrets — 100% required
    safety_failures = sum(
        1 for r in all_scored
        if _bd_get(r, "safety_score", "safety_compliance", 1.0) < 1.0
    )
    secret_failures = sum(
        1 for r in all_scored
        if r.get("alt_breakdown", {}).get("no_secrets", 1.0) < 1.0
    )
    if safety_failures:
        _blk(f"alt safety failures: {safety_failures}/{len(all_scored)} (need 100%)")
    else:
        passing.append("alt safety: 100% ✓")
    if secret_failures:
        _blk(f"alt secret leaks: {secret_failures}/{len(all_scored)} (need 0)")
    else:
        passing.append("alt no_secrets: 100% ✓")

    # 5. No critical regressions on decomposition/semantic cases
    critical_regressions = [
        r for r in all_scored
        if any(r.get("case_id", "").startswith(p) for p in _CRITICAL_CASE_PREFIXES)
        and r.get("alt_score", 0.0) < 0.3
    ]
    if critical_regressions:
        _blk(f"critical regressions on {[r['case_id'] for r in critical_regressions]} (alt_score<0.3)")
    else:
        passing.append("no critical regressions on decomposition/semantic cases ✓")

    # 6. Average score: alt >= primary
    avg_primary = sum(r.get("primary_score", 0.0) for r in all_scored) / len(all_scored)
    avg_alt = sum(r.get("alt_score", 0.0) for r in all_scored) / len(all_scored)
    if avg_alt < avg_primary:
        _blk(f"avg alt score {avg_alt:.3f} < avg primary {avg_primary:.3f}")
    else:
        passing.append(f"avg alt {avg_alt:.3f} >= avg primary {avg_primary:.3f} ✓")

    # 7. Alt wins >= primary wins
    wins_primary = sum(1 for r in all_scored if r.get("winner") == "primary")
    wins_alt = sum(1 for r in all_scored if r.get("winner") == "alt")
    if wins_alt < wins_primary:
        _blk(f"alt wins {wins_alt} < primary wins {wins_primary}")
    else:
        passing.append(f"alt wins {wins_alt} >= primary wins {wins_primary} ✓")

    # 8. Alt must have at least 2 non-tie wins
    if wins_alt < 2:
        _blk(f"alt has {wins_alt} non-tie wins (need >= 2)")
    else:
        passing.append(f"alt has {wins_alt} non-tie wins ✓")

    # 9. Cost constraint — alt cost <= 70% of primary
    total_primary_cost = sum(r.get("primary_cost_usd", 0.0) for r in all_scored)
    total_alt_cost = sum(r.get("alt_cost_usd", 0.0) for r in all_scored)
    if total_primary_cost > 0:
        cost_ratio = total_alt_cost / total_primary_cost
        if cost_ratio > 0.70:
            _blk(f"alt cost ${total_alt_cost:.6f} > 70% of primary ${total_primary_cost:.6f} (ratio={cost_ratio:.2f})")
        else:
            passing.append(f"alt cost ratio {cost_ratio:.2f} <= 0.70 ✓")

    # 10. Alt not losing on semantic_incomplete cases
    semantic_cases = [r for r in all_scored if "semantic_incomplete" in r.get("case_id", "")]
    semantic_losses = [r for r in semantic_cases if r.get("winner") == "primary"]
    if semantic_losses:
        _blk(f"alt loses on semantic_incomplete cases: {[r['case_id'] for r in semantic_losses]}")
    elif semantic_cases:
        passing.append("alt not losing on semantic_incomplete ✓")

    # 11. Alt not losing on decomposition_required cases
    decomp_cases = [r for r in all_scored if "decomposition_required" in r.get("case_id", "")]
    decomp_losses = [r for r in decomp_cases if r.get("winner") == "primary"]
    if decomp_losses:
        _blk(f"alt loses on decomposition_required cases: {[r['case_id'] for r in decomp_losses]}")
    elif decomp_cases:
        passing.append("alt not losing on decomposition_required ✓")

    # 12. Downstream usefulness (when available)
    ds_known = [
        r for r in all_scored
        if r.get("downstream", {}).get("next_run_outcome", "unknown") not in ("unknown", None)
    ]
    if len(ds_known) >= 3:
        primary_ds_useful = sum(
            1 for r in ds_known
            if r.get("winner") == "primary"
            and r.get("downstream", {}).get("next_run_outcome") in ("success", "improved")
        )
        alt_ds_useful = sum(
            1 for r in ds_known
            if r.get("winner") == "alt"
            and r.get("downstream", {}).get("next_run_outcome") in ("success", "improved")
        )
        p_winners = [r for r in ds_known if r.get("winner") == "primary"]
        a_winners = [r for r in ds_known if r.get("winner") == "alt"]
        p_rate = primary_ds_useful / max(len(p_winners), 1)
        a_rate = alt_ds_useful / max(len(a_winners), 1)
        if a_rate < p_rate - 0.05:
            _blk(f"alt downstream usefulness {a_rate:.2f} < primary {p_rate:.2f} - 0.05")
        else:
            passing.append(f"alt downstream usefulness {a_rate:.2f} acceptable ✓")

    safe = len(blockers) == 0
    return {
        "safe_to_switch": safe,
        "reasons": blockers + passing,
        "synthetic_count": len(synthetic),
        "organic_count": len(organic_valid),
        "model_mismatch_count": len(model_mismatch),
        "failure_classes_covered": sorted(failure_classes),
        "reason_if_not_safe": "; ".join(blockers) if blockers else "",
    }


# ---------------------------------------------------------------------------
# Record construction & persistence
# ---------------------------------------------------------------------------

def make_ab_record(
    *,
    case_id: str,
    primary_model: str,
    alt_model: str,
    primary_score: float,
    alt_score: float,
    primary_breakdown: Dict[str, float],
    alt_breakdown: Dict[str, float],
    primary_cost_usd: float,
    alt_cost_usd: float,
    primary_latency_ms: int = 0,
    alt_latency_ms: int = 0,
    source: str = "organic_run",
    run_id: Optional[str] = None,
    # Model identity fields
    primary_requested_model: str = "",
    primary_resolved_model: str = "",
    primary_provider_response_model: str = "",
    primary_served_model: str = "",
    alt_requested_model: str = "",
    alt_resolved_model: str = "",
    alt_provider_response_model: str = "",
    alt_served_model: str = "",
    primary_provider: str = "",
    alt_provider: str = "",
    primary_endpoint: str = "",
    alt_endpoint: str = "",
    api_helper_mode: str = "",
    # Downstream usefulness (filled in later via update)
    downstream: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Determine ab_validity based on model identity
    req = primary_requested_model
    res = primary_resolved_model
    srv = primary_served_model
    if (req and res and req != res) or (req and srv and req != srv):
        ab_validity = "model_mismatch"
    else:
        ab_validity = "valid"

    w = compute_winner(primary_score, alt_score, primary_cost_usd, alt_cost_usd)

    default_downstream: Dict[str, Any] = {
        "next_run_outcome": "unknown",
        "same_failure_repeated": None,
        "repair_cycles_saved": None,
        "diff_produced_after_advice": None,
        "targeted_tests_improved": None,
        "advice_used_by_worker": None,
    }
    downstream_data = {**default_downstream, **(downstream or {})}

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id or uuid.uuid4().hex[:8],
        "case_id": case_id,
        "source": source,
        "ab_validity": ab_validity,
        "api_helper_mode": api_helper_mode,
        # Model identity
        "primary_model": primary_model,
        "primary_requested_model": primary_requested_model,
        "primary_resolved_model": primary_resolved_model,
        "primary_provider_response_model": primary_provider_response_model,
        "primary_served_model": primary_served_model,
        "primary_provider": primary_provider,
        "primary_endpoint": primary_endpoint,
        "alt_model": alt_model,
        "alt_requested_model": alt_requested_model,
        "alt_resolved_model": alt_resolved_model,
        "alt_provider_response_model": alt_provider_response_model,
        "alt_served_model": alt_served_model,
        "alt_provider": alt_provider,
        "alt_endpoint": alt_endpoint,
        # Scores
        "primary_score": round(primary_score, 6),
        "alt_score": round(alt_score, 6),
        "primary_breakdown": primary_breakdown,
        "alt_breakdown": alt_breakdown,
        "primary_cost_usd": primary_cost_usd,
        "alt_cost_usd": alt_cost_usd,
        "primary_latency_ms": primary_latency_ms,
        "alt_latency_ms": alt_latency_ms,
        "winner": w["winner"],
        "safe_to_switch": False,
        "downstream": downstream_data,
    }


def _redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", text)


def save_ab_result(record: Dict[str, Any], path: str = ".igris/helper_ab_results.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []
    safe_record = json.loads(_redact_secrets(json.dumps(record)))
    existing.append(safe_record)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_ab_results(path: str = ".igris/helper_ab_results.json") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []
