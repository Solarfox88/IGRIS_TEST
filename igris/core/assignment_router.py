"""AssignmentRouter: pre-flight decision engine for IGRIS reasoning tasks.

Decides agent_role, task_type, preferred_profile, execution_strategy and
budget ONCE before the reasoning subprocess is launched.  ModelOrchestrator
remains a pure provider/model dispatcher; all semantic routing lives here.

Formula:
    estimated_expected_cost = cost_per_attempt * avg_attempts / max(p_success, 0.05)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from igris.core.agent_registry import PROFILE_RELATIVE_COST
from igris.core.assignment_outcomes import (
    compute_task_signature,
    load_assignment_outcomes,
    save_assignment_outcome,
)

_log = logging.getLogger(__name__)

# Minimum success-rate thresholds per task type
_SUCCESS_THRESHOLD: Dict[str, float] = {
    "default": 0.70,
    "backend_endpoint": 0.80,
    "security_review": 0.90,
    "devops_runtime": 0.90,
}

# Minimum history records before learned rates are trusted
_MIN_HISTORY_FOR_LEARNING = 5

# Budget tiers in USD
_BUDGET_TIER: Dict[str, float] = {
    "low": 0.50,
    "medium": 2.00,
    "high": 5.00,
    "very_high": 10.00,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssignmentRequest:
    goal_text: str
    issue_number: Optional[int] = None
    issue_title: str = ""
    issue_body: str = ""
    issue_labels: List[str] = field(default_factory=list)
    required_tests: List[str] = field(default_factory=list)
    allowed_file_scopes: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    failure_class: str = ""
    capability_signals: Dict[str, int] = field(default_factory=dict)
    failure_memory: List[Dict] = field(default_factory=list)
    prior_attempts: int = 0
    local_model_available: bool = True
    budget_remaining_usd: float = 10.0
    provider_circuit_breaker_state: Dict[str, bool] = field(default_factory=dict)
    is_repair: bool = False
    outcomes_path: str = ""


@dataclass
class AssignmentDecision:
    agent_role: str
    task_type: str
    preferred_profile: str
    execution_strategy: str
    preferred_model: str
    fallback_model_path: List[str]
    should_call_codex_helper_first: bool
    should_decompose_first: bool
    max_attempts: int
    budget_limit: float
    confidence: float
    reasons: List[str]
    estimated_success_probability: float = 0.0
    estimated_expected_cost: float = 0.0
    history_matches: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "task_type": self.task_type,
            "preferred_profile": self.preferred_profile,
            "execution_strategy": self.execution_strategy,
            "preferred_model": self.preferred_model,
            "fallback_model_path": list(self.fallback_model_path),
            "should_call_codex_helper_first": self.should_call_codex_helper_first,
            "should_decompose_first": self.should_decompose_first,
            "max_attempts": self.max_attempts,
            "budget_limit": round(self.budget_limit, 4),
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
            "estimated_success_probability": round(self.estimated_success_probability, 3),
            "estimated_expected_cost": round(self.estimated_expected_cost, 4),
            "history_matches": self.history_matches,
        }


# ---------------------------------------------------------------------------
# Keyword classifiers
# ---------------------------------------------------------------------------

_DOCS_KEYWORDS = frozenset([
    "docstring", "readme", "documentation", "comment", "annotate",
    "changelog", "typo", "rename variable", "rename function",
])
_TEST_KEYWORDS = frozenset([
    "test", "pytest", "coverage", "assert", "fixture", "mock",
])
_BACKEND_KEYWORDS = frozenset([
    "endpoint", "/api/", "api route", "backend", "handler", "controller",
    "implement get", "implement post", "implement put", "implement delete",
    "implement patch", "router", "fastapi", "flask",
])
_MEMORY_KEYWORDS = frozenset([
    "memory", "synapse", "recall", "vector store", "embedding",
    "knowledge base", "long-term",
])
_SECURITY_KEYWORDS = frozenset([
    "secret", "api key", "credential", "auth", "jwt", "token",
    "vulnerability", "injection", "xss", "csrf", "permission",
])
_DEVOPS_KEYWORDS = frozenset([
    "deploy", "restart", "ci ", "cd ", "docker", "kubernetes", "smoke",
    "health check", "migration", "infrastructure",
])
_EPIC_KEYWORDS = frozenset([
    "refactor", "rework", "rewrite", "architecture", "redesign",
    "epic", "system-wide", "overhaul",
])
_REPAIR_KEYWORDS = frozenset([
    "fix", "repair", "debug", "diagnose", "broken", "failing", "error",
])


def _contains_any(text: str, keywords: frozenset) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _classify_goal(request: AssignmentRequest) -> Tuple[str, str, List[str]]:
    """Return (agent_role, task_type, reasons)."""
    goal = request.goal_text
    failure_class = request.failure_class
    signals = request.capability_signals
    labels = [la.lower() for la in request.issue_labels]
    reasons: List[str] = []

    # Failure-class overrides take highest priority
    if failure_class in ("semantic_incomplete", "stub_detected") or signals.get("stub_detected", 0) >= 1:
        reasons.append(f"failure_class={failure_class or 'stub_detected signal'}")
        return "backend_coder", "semantic_repair", reasons

    if failure_class == "pytest_failure":
        reasons.append("failure_class=pytest_failure")
        return "test_debugger", "pytest_failure", reasons

    if failure_class in ("no_diff", "no_diff_repair") or signals.get("no_diff", 0) >= 2:
        reasons.append("repeated no_diff failure")
        return "backend_coder", "no_diff_repair", reasons

    if failure_class in ("max_steps", "reasoning_timeout") or signals.get("max_steps_ceiling", 0) >= 1:
        reasons.append("max_steps or reasoning_timeout signal")
        return "backend_coder", "complex_implementation", reasons

    # Security / devops by risk or keywords
    if request.risk_level in ("high", "very_high") or _contains_any(goal, _SECURITY_KEYWORDS):
        reasons.append("high risk or security keywords")
        return "security_reviewer", "security_review", reasons

    if _contains_any(goal, _DEVOPS_KEYWORDS) or "devops" in labels:
        reasons.append("devops keywords")
        return "devops", "devops_runtime", reasons

    # Memory system
    if _contains_any(goal, _MEMORY_KEYWORDS) or "memory" in labels:
        reasons.append("memory system keywords")
        return "memory_architect", "memory_system", reasons

    # Large epic / architecture
    if _contains_any(goal, _EPIC_KEYWORDS) and len(goal) > 200:
        reasons.append("large epic keywords + long goal")
        return "planner", "planning", reasons

    # Backend endpoint
    if _contains_any(goal, _BACKEND_KEYWORDS) or request.required_tests:
        reasons.append("backend/endpoint keywords or required_tests")
        return "backend_coder", "backend_endpoint", reasons

    # Test-only (no backend change)
    if _contains_any(goal, _TEST_KEYWORDS) and not _contains_any(goal, _BACKEND_KEYWORDS):
        reasons.append("test keywords without backend change")
        return "tester", "test_only", reasons

    # Docs / small refactor
    if _contains_any(goal, _DOCS_KEYWORDS) or _contains_any(goal, _REPAIR_KEYWORDS):
        reasons.append("docs or small repair keywords")
        return "planner", "documentation", reasons

    reasons.append("no specific signals — default backend_coder")
    return "backend_coder", "code_reasoning", reasons


# ---------------------------------------------------------------------------
# History statistics
# ---------------------------------------------------------------------------

def _compute_history_stats(
    outcomes: List[Dict],
    agent_role: str,
    task_type: str,
) -> Dict[str, Any]:
    matching = [
        o for o in outcomes
        if o.get("agent_role") == agent_role and o.get("task_type") == task_type
    ]
    if not matching:
        return {}
    total = len(matching)
    successes = sum(1 for o in matching if o.get("outcome") == "success")
    stub_count = sum(1 for o in matching if o.get("failure_class") in ("semantic_incomplete", "stub_detected"))
    no_diff = sum(1 for o in matching if o.get("failure_class") in ("no_diff", "no_diff_repair"))
    timeouts = sum(1 for o in matching if o.get("failure_class") in ("reasoning_timeout", "max_steps"))
    costs = [o.get("cost_usd", 0.0) for o in matching if isinstance(o.get("cost_usd"), (int, float))]
    attempts_list = [o.get("attempts", 1) for o in matching if isinstance(o.get("attempts"), int)]
    return {
        "total": total,
        "success_rate": successes / total,
        "stub_rate": stub_count / total,
        "no_diff_rate": no_diff / total,
        "timeout_rate": timeouts / total,
        "avg_cost": sum(costs) / len(costs) if costs else 0.0,
        "avg_attempts": sum(attempts_list) / len(attempts_list) if attempts_list else 1.0,
    }


def _compute_profile_stats(
    outcomes: List[Dict],
    agent_role: str,
    task_type: str,
    profile: str,
) -> Dict[str, Any]:
    matching = [
        o for o in outcomes
        if o.get("agent_role") == agent_role
        and o.get("task_type") == task_type
        and o.get("preferred_profile") == profile
    ]
    if not matching:
        return {}
    total = len(matching)
    successes = sum(1 for o in matching if o.get("outcome") == "success")
    costs = [o.get("cost_usd", 0.0) for o in matching if isinstance(o.get("cost_usd"), (int, float))]
    attempts_list = [o.get("attempts", 1) for o in matching if isinstance(o.get("attempts"), int)]
    return {
        "total": total,
        "success_rate": successes / total,
        "avg_cost": sum(costs) / len(costs) if costs else 0.0,
        "avg_attempts": sum(attempts_list) / len(attempts_list) if attempts_list else 1.0,
    }


# ---------------------------------------------------------------------------
# Candidate strategies
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    profile: str
    strategy: str
    preferred_model: str
    fallback_model_path: List[str]
    codex_helper: bool
    decompose: bool
    bootstrap_success_prob: float
    bootstrap_cost_per_attempt: float
    budget_tier: str


def _build_candidates(agent_role: str, task_type: str, request: AssignmentRequest) -> List[_Candidate]:
    """Build ordered list of candidate strategies from cheapest to strongest."""
    signals = request.capability_signals
    failure_class = request.failure_class
    is_repair = request.is_repair
    prior = request.prior_attempts

    has_max_steps_ceiling = signals.get("max_steps_ceiling", 0) >= 1
    has_stub = signals.get("stub_detected", 0) >= 1 or failure_class in ("semantic_incomplete", "stub_detected")
    has_repeated_no_diff = signals.get("no_diff", 0) >= 2 or failure_class in ("no_diff", "no_diff_repair")

    force_decompose = (
        task_type in ("planning", "memory_system")
        or (has_repeated_no_diff and len(request.goal_text) > 300)
        or (has_max_steps_ceiling and len(request.goal_text) > 500)
    )
    force_strong = (
        has_max_steps_ceiling
        or (is_repair and prior >= 1)
        or task_type in ("security_review", "devops_runtime")
        or request.risk_level in ("high", "very_high")
    )
    needs_helper = (
        task_type in ("backend_endpoint", "semantic_repair", "complex_implementation", "no_diff_repair")
        or has_stub
        or force_strong
    )

    candidates: List[_Candidate] = []

    if task_type == "documentation" and not is_repair:
        candidates.append(_Candidate(
            profile="cheap_cloud_reasoning",
            strategy="direct_cheap",
            preferred_model="deepseek-v4-flash",
            fallback_model_path=["gpt-4o-mini"],
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.80,
            bootstrap_cost_per_attempt=0.10,
            budget_tier="low",
        ))

    if task_type == "test_only" and not force_strong:
        candidates.append(_Candidate(
            profile="mini_execution",
            strategy="mini_direct",
            preferred_model="gpt-4o-mini",
            fallback_model_path=["deepseek-v4-flash"],
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.75,
            bootstrap_cost_per_attempt=0.20,
            budget_tier="low",
        ))

    if task_type == "pytest_failure":
        candidates.append(_Candidate(
            profile="mini_execution",
            strategy="debug_mini",
            preferred_model="gpt-4o-mini",
            fallback_model_path=["deepseek-v4-flash", "gpt-4o"],
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.70,
            bootstrap_cost_per_attempt=0.25,
            budget_tier="low",
        ))

    if task_type == "backend_endpoint" and not force_strong:
        candidates.append(_Candidate(
            profile="mini_execution",
            strategy="helper_advice_then_mini_execution",
            preferred_model="gpt-4o-mini",
            fallback_model_path=["gpt-4o"],
            codex_helper=True,
            decompose=False,
            bootstrap_success_prob=0.72,
            bootstrap_cost_per_attempt=0.40,
            budget_tier="medium",
        ))

    if force_decompose:
        candidates.append(_Candidate(
            profile="cheap_cloud_reasoning",
            strategy="decompose_first",
            preferred_model="deepseek-v4-flash",
            fallback_model_path=["gpt-4o"],
            codex_helper=needs_helper,
            decompose=True,
            bootstrap_success_prob=0.65,
            bootstrap_cost_per_attempt=0.50,
            budget_tier="medium",
        ))

    # Strong execution — always available as last resort or when forced
    strong_prob = 0.85 if task_type in ("security_review", "devops_runtime") else 0.78
    candidates.append(_Candidate(
        profile="strong_execution",
        strategy="helper_advice_then_strong_execution" if needs_helper else "strong_execution_direct",
        preferred_model="deepseek-v4-pro",
        fallback_model_path=["gpt-4o"],
        codex_helper=needs_helper,
        decompose=force_decompose,
        bootstrap_success_prob=strong_prob,
        bootstrap_cost_per_attempt=1.20,
        budget_tier="high",
    ))

    return candidates


def _profile_to_provider(profile: str) -> str:
    mapping = {
        "cheap_cloud_reasoning": "deepseek",
        "mini_execution": "openai",
        "strong_execution": "deepseek_strong",
        "strong_cloud_reasoning": "deepseek_strong",
        "endpoint_implementation": "deepseek",
        "local_light": "ollama",
        "local_coder": "ollama",
        "risk_reviewer": "deepseek",
    }
    return mapping.get(profile, "openai")


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

class AssignmentRouter:
    """Pre-flight decision engine. Call decide() once before run_reasoning()."""

    def __init__(self, outcomes_path: str = "") -> None:
        self._outcomes_path = outcomes_path

    def decide(self, request: AssignmentRequest) -> AssignmentDecision:
        agent_role, task_type, reasons = _classify_goal(request)

        outcomes: List[Dict] = []
        path = request.outcomes_path or self._outcomes_path
        if path:
            outcomes = load_assignment_outcomes(path)

        history_stats = _compute_history_stats(outcomes, agent_role, task_type)
        history_matches = history_stats.get("total", 0)

        candidates = _build_candidates(agent_role, task_type, request)
        threshold = _SUCCESS_THRESHOLD.get(task_type, _SUCCESS_THRESHOLD["default"])

        best: Optional[_Candidate] = None
        best_expected_cost = float("inf")
        best_success_prob = 0.0
        best_reasons = list(reasons)

        for cand in candidates:
            profile_stats = _compute_profile_stats(outcomes, agent_role, task_type, cand.profile)

            if profile_stats and profile_stats["total"] >= _MIN_HISTORY_FOR_LEARNING:
                p_success = profile_stats["success_rate"]
                avg_cost = profile_stats["avg_cost"]
                avg_attempts = profile_stats["avg_attempts"]
                source = "history"
            else:
                p_success = cand.bootstrap_success_prob
                avg_cost = cand.bootstrap_cost_per_attempt
                avg_attempts = 1.5
                source = "bootstrap"

            # Budget constraint
            budget_limit = _BUDGET_TIER.get(cand.budget_tier, 2.0)
            if request.budget_remaining_usd < budget_limit * 0.5:
                continue

            # Circuit breaker check
            provider_key = _profile_to_provider(cand.profile)
            if request.provider_circuit_breaker_state.get(provider_key) is False:
                continue

            if p_success < threshold and cand is not candidates[-1]:
                continue

            expected_cost = avg_cost * avg_attempts / max(p_success, 0.05)

            if best is None or expected_cost < best_expected_cost:
                best = cand
                best_expected_cost = expected_cost
                best_success_prob = p_success
                best_reasons = list(reasons) + [
                    f"profile={cand.profile} source={source} p_success={p_success:.2f}"
                ]

        if best is None:
            # All candidates filtered — force strong_execution
            best = candidates[-1]
            best_success_prob = best.bootstrap_success_prob
            best_expected_cost = best.bootstrap_cost_per_attempt * 1.5 / max(best_success_prob, 0.05)
            best_reasons = list(reasons) + ["all candidates filtered — forced strong_execution"]

        budget_limit = min(_BUDGET_TIER.get(best.budget_tier, 2.0), request.budget_remaining_usd)
        confidence = min(0.95, best_success_prob * (1.0 + 0.1 * min(history_matches, 10)))

        _log.info(
            "AssignmentRouter: role=%s type=%s profile=%s strategy=%s p=%.2f cost=%.3f history=%d",
            agent_role, task_type, best.profile, best.strategy,
            best_success_prob, best_expected_cost, history_matches,
        )

        return AssignmentDecision(
            agent_role=agent_role,
            task_type=task_type,
            preferred_profile=best.profile,
            execution_strategy=best.strategy,
            preferred_model=best.preferred_model,
            fallback_model_path=list(best.fallback_model_path),
            should_call_codex_helper_first=best.codex_helper,
            should_decompose_first=best.decompose,
            max_attempts=2,
            budget_limit=budget_limit,
            confidence=confidence,
            reasons=best_reasons,
            estimated_success_probability=best_success_prob,
            estimated_expected_cost=best_expected_cost,
            history_matches=history_matches,
        )
