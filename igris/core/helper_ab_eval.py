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
SCORE_WEIGHTS = {
    "schema_valid": 0.25, "diagnosis_specificity": 0.20,
    "execution_plan_actionability": 0.20, "acceptance_matrix_quality": 0.15,
    "safety_compliance": 0.10, "no_secrets": 0.05, "decomposition_quality": 0.05,
}
_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9-]{15,}|[A-Za-z0-9]{32,}key[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9]{20,})",
    re.IGNORECASE,
)
_GENERIC_PLAN_WORDS = {
    "check","look","review","investigate","examine","see","try",
    "consider","maybe","probably","could","might",
}
# case_ids that must never regress critically (alt_score < 0.3)
_CRITICAL_CASE_PREFIXES = ("decomposition_required", "semantic_incomplete")


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


def score_helper_response(response: Any, case: dict) -> Dict[str, Any]:
    """Score a helper response dict. Returns {total, breakdown, issues}."""
    if not isinstance(response, dict):
        return {
            "total": 0.0,
            "breakdown": {k: 0.0 for k in SCORE_WEIGHTS},
            "issues": ["response is not a dict"],
        }
    breakdown = {
        "schema_valid": _score_schema(response),
        "diagnosis_specificity": _score_diagnosis(response, case),
        "execution_plan_actionability": _score_execution_plan(response, case),
        "acceptance_matrix_quality": _score_acceptance_matrix(response),
        "safety_compliance": _score_safety(response, case),
        "no_secrets": _score_no_secrets(response),
        "decomposition_quality": _score_decomposition(response, case),
    }
    total = sum(SCORE_WEIGHTS[k] * v for k, v in breakdown.items())
    issues: List[str] = []
    if breakdown["schema_valid"] < 1.0:
        missing = [f for f in REQUIRED_SCHEMA_FIELDS if f not in response]
        issues.append(f"missing schema fields: {missing}")
    if breakdown["no_secrets"] < 1.0:
        issues.append("secret detected in response")
    if breakdown["safety_compliance"] < 1.0:
        issues.append("safety compliance failure")
    return {"total": round(total, 6), "breakdown": breakdown, "issues": issues}


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
# Switch policy — hybrid: requires synthetic + organic + failure diversity
# ---------------------------------------------------------------------------

def is_safe_to_switch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Hybrid switch policy.

    Returns a dict:
      safe_to_switch, reasons, synthetic_count, organic_count,
      failure_classes_covered, reason_if_not_safe
    """
    synthetic = [r for r in records if r.get("source") == "synthetic_fixture"]
    # Records without source (legacy) count as organic
    organic = [r for r in records if r.get("source") in ("organic_run", None) and "source" not in r or r.get("source") == "organic_run"]
    # re-derive correctly
    organic = [r for r in records if r.get("source") == "organic_run" or "source" not in r]

    failure_classes: set = {
        r.get("case_id", "").split("_")[0] or r.get("case_id", "")
        for r in organic
        if r.get("case_id")
    }
    # Also include full case_id strings for diversity count
    failure_class_full: set = {r.get("case_id", "") for r in organic if r.get("case_id")}

    all_scored = synthetic + organic
    blockers: List[str] = []
    passing: List[str] = []

    def _blk(msg: str) -> None:
        blockers.append(msg)

    # 1. Minimum record counts
    if len(synthetic) < 9:
        _blk(f"need 9 synthetic records, have {len(synthetic)}")
    else:
        passing.append(f"synthetic records: {len(synthetic)}/9 ✓")

    if len(organic) < 5:
        _blk(f"need 5 organic records, have {len(organic)}")
    else:
        passing.append(f"organic records: {len(organic)}/5 ✓")

    # 2. Failure class diversity in organic
    if len(failure_class_full) < 3:
        _blk(f"need 3+ distinct failure_classes in organic records, have {len(failure_class_full)}: {sorted(failure_class_full)}")
    else:
        passing.append(f"failure_classes covered: {sorted(failure_class_full)} ✓")

    if not all_scored:
        return {
            "safe_to_switch": False,
            "reasons": blockers + passing,
            "synthetic_count": 0,
            "organic_count": 0,
            "failure_classes_covered": [],
            "reason_if_not_safe": "; ".join(blockers) or "no records",
        }

    # 3. Schema validity — 100% required
    schema_failures = sum(
        1 for r in all_scored
        if r.get("alt_breakdown", {}).get("schema_valid", 0.0) < 1.0
    )
    if schema_failures:
        _blk(f"alt schema failures: {schema_failures}/{len(all_scored)} (need 100% valid)")
    else:
        passing.append(f"alt schema: 100% valid ✓")

    # 4. Safety + no secrets — 100% required
    safety_failures = sum(
        1 for r in all_scored
        if r.get("alt_breakdown", {}).get("safety_compliance", 1.0) < 1.0
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

    # 6. Average score threshold
    avg_primary = sum(r.get("primary_score", 0.0) for r in all_scored) / len(all_scored)
    avg_alt = sum(r.get("alt_score", 0.0) for r in all_scored) / len(all_scored)
    threshold = avg_primary - 0.05
    if avg_alt < threshold:
        _blk(f"avg alt score {avg_alt:.3f} < threshold {threshold:.3f} (primary_avg-0.05)")
    else:
        passing.append(f"avg alt score {avg_alt:.3f} >= threshold {threshold:.3f} ✓")

    # 7. Cost constraint — alt <= 70% of primary
    total_primary_cost = sum(r.get("primary_cost_usd", 0.0) for r in all_scored)
    total_alt_cost = sum(r.get("alt_cost_usd", 0.0) for r in all_scored)
    if total_primary_cost > 0:
        cost_limit = total_primary_cost * 0.70
        if total_alt_cost > cost_limit:
            _blk(f"alt cost ${total_alt_cost:.6f} > 70% of primary ${total_primary_cost:.6f}")
        else:
            passing.append(f"alt cost ${total_alt_cost:.6f} <= 70% of primary ✓")

    safe = len(blockers) == 0
    return {
        "safe_to_switch": safe,
        "reasons": blockers + passing,
        "synthetic_count": len(synthetic),
        "organic_count": len(organic),
        "failure_classes_covered": sorted(failure_class_full),
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
) -> Dict[str, Any]:
    w = compute_winner(primary_score, alt_score, primary_cost_usd, alt_cost_usd)
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id or uuid.uuid4().hex[:8],
        "case_id": case_id,
        "source": source,
        "primary_model": primary_model,
        "alt_model": alt_model,
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
