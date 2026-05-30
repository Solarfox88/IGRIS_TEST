"""AssignmentRouter: pre-flight decision engine for IGRIS reasoning tasks.

Decides agent_role, task_type, preferred_profile, execution_strategy and
budget ONCE before the reasoning subprocess is launched.  ModelOrchestrator
remains a pure provider/model dispatcher; all semantic routing lives here.

Formula:
    estimated_expected_cost = cost_per_attempt * avg_attempts / max(p_success, 0.05)
"""
from __future__ import annotations

import logging
import re
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
    "code_reasoning": 0.60,  # generic tasks: prefer cheaper model first
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
_SUPERVISOR_KEYWORDS = frozenset([
    "supervisor", "watchdog", "orchestrator", "routing", "gate",
    "repair cycle", "reasoning worker", "subprocess", "timeout",
    "wall-clock", "pipeline", "escalation", "capability", "profile",
])
_BACKEND_KEYWORDS = frozenset([
    "endpoint", "/api/", "api route", "backend", "handler", "controller",
    "implement get", "implement post", "implement put", "implement delete",
    "implement patch", "router", "fastapi", "flask",
])
# Feature-implementation signals: goals that start with "feat(" or "Implement GitHub issue"
# with a feat() prefix are new-module implementations, NOT test tasks.
_FEAT_IMPL_PATTERNS = (
    "implement github issue #",   # watchdog goal format: "Implement GitHub issue #N: feat(…)"
    "feat(core)",
    "feat(supervisor)",
    "feat(memory)",
    "feat(watchdog)",
    "feat(context)",
    "feat(web)",
    "feat(layers)",
)
_MEMORY_KEYWORDS = frozenset([
    "memory", "synapse", "recall", "vector store", "embedding",
    "knowledge base", "long-term",
])
_SECURITY_KEYWORDS = frozenset([
    "secret", "api key", "credential", "jwt",
    "vulnerability", "xss", "csrf",
])
# Whole-word / phrase matches for ambiguous terms:
# - "auth"      would match "Authorization" in non-security titles
# - "injection" would match Italian "dell'injection nel contesto" (LLM context)
# - "token"     would match "token budget", "token count"
_SECURITY_KEYWORDS_WHOLE_WORD = frozenset([
    "auth token", "auth key", "api token", "access token",
    "sql injection", "command injection", "prompt injection attack",
    "permission denied", "privilege escalation",
])
_DEVOPS_KEYWORDS = frozenset([
    "deploy", "restart", "ci ", "cd ", "docker", "kubernetes", "smoke",
    "health check", "migration", "infrastructure",
])
_EPIC_KEYWORDS = frozenset([
    "refactor", "rework", "rewrite", "architecture", "redesign",
    "epic", "system-wide", "overhaul",
    "parallelism", "parallel agents", "multi-agent", "concurrent tasks",
    "voice layer", "tts", "speech synthesis",
    "interlocutor", "authorization model",
])
_REPAIR_KEYWORDS = frozenset([
    "fix", "repair", "debug", "diagnose", "broken", "failing", "error",
])


def _contains_any(text: str, keywords: frozenset) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _contains_security(text: str) -> bool:
    """Security check: substring for clear terms + phrase match for ambiguous ones."""
    t = text.lower()
    return _contains_any(t, _SECURITY_KEYWORDS) or _contains_any(t, _SECURITY_KEYWORDS_WHOLE_WORD)


def _contains_test_signal(text: str) -> bool:
    """Test-keyword check. Uses word-boundary for bare 'test' to avoid matching
    Italian/Spanish 'contesto', 'protesto' etc. Other test keywords are fine as substring."""
    t = text.lower()
    if re.search(r'\btest\b', t):
        return True
    return _contains_any(t, _TEST_KEYWORDS - {"test"})


def _classify_goal(request: AssignmentRequest) -> Tuple[str, str, List[str]]:
    """Return (agent_role, task_type, reasons)."""
    goal = request.goal_text
    failure_class = request.failure_class
    signals = request.capability_signals
    labels = [la.lower() for la in request.issue_labels]
    reasons: List[str] = []

    # Escalation override: accumulated no_diff_repair / combined signals → hard_debugging.
    # Must run before keyword classification so "fix …" text doesn't fall to documentation.
    _ndr = signals.get("no_diff_repair", 0)
    _rto = signals.get("reasoning_timeout", 0)
    if _ndr >= 3 or (_ndr + _rto) >= 4 or (_ndr >= 2 and request.prior_attempts >= 2):
        reasons.append(
            f"escalation: no_diff_repair={_ndr} reasoning_timeout={_rto} prior={request.prior_attempts}"
        )
        return "backend_coder", "hard_debugging", reasons

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

    # Security: only route to security_reviewer on explicit security content —
    # NOT on risk_level alone. risk_level=high means "historically hard task",
    # not "security task". Routing a feature implementation to security_reviewer
    # because it failed 23 times causes no_diff_repair (reviewer never writes code).
    if _contains_security(goal) or "security" in labels:
        reasons.append("security keywords in goal")
        return "security_reviewer", "security_review", reasons

    if _contains_any(goal, _DEVOPS_KEYWORDS) or "devops" in labels:
        reasons.append("devops keywords")
        return "devops", "devops_runtime", reasons

    # Memory system
    if _contains_any(goal, _MEMORY_KEYWORDS) or "memory" in labels:
        reasons.append("memory system keywords")
        return "memory_architect", "memory_system", reasons

    # Large epic / architecture — triggered by keywords+length OR keywords+epic label
    if _contains_any(goal, _EPIC_KEYWORDS) and (len(goal) > 200 or "epic" in labels):
        reasons.append("large epic keywords + (long goal or epic label)")
        return "planner", "planning", reasons

    # Epic label alone (no specific keyword match) → complex_implementation, not code_reasoning
    if "epic" in labels:
        reasons.append("epic label → complex_implementation")
        return "backend_coder", "complex_implementation", reasons

    # Backend endpoint
    if _contains_any(goal, _BACKEND_KEYWORDS) or request.required_tests:
        reasons.append("backend/endpoint keywords or required_tests")
        return "backend_coder", "backend_endpoint", reasons

    # feat(…) issues that come from the watchdog goal format ("Implement GitHub issue #N: feat(…)")
    # are new-module implementations — even if they mention unit tests in their AC.
    # Classify them as complex_implementation before the test_only check consumes them.
    goal_lower = goal.lower()
    if any(pat in goal_lower for pat in _FEAT_IMPL_PATTERNS):
        # Long feat(*) goals (> 800 chars) carry full acceptance criteria, multi-file
        # integration targets and benchmark specs — they benefit from a large reasoning
        # model on first attempt rather than going through cheap_cloud → escalation loop.
        # Route directly to architecture_review (→ gpu_reasoning → vastai_ollama first)
        # so VastAI auto-provision is triggered immediately.  Falls through gracefully to
        # deepseek_strong / openai if the instance is not ready yet.
        if len(goal) > 800:
            reasons.append(
                f"heavy feat(*) goal (len={len(goal)}) — architecture_review / gpu_reasoning"
            )
            return "backend_coder", "architecture_review", reasons
        reasons.append("feat(*) implementation goal — complex_implementation override")
        return "backend_coder", "complex_implementation", reasons

    # Test-only (no backend change, no supervisor/fix work)
    if (_contains_test_signal(goal)
            and not _contains_any(goal, _BACKEND_KEYWORDS)
            and not _contains_any(goal, _REPAIR_KEYWORDS)
            and not _contains_any(goal, _SUPERVISOR_KEYWORDS)):
        reasons.append("test keywords without backend/repair/supervisor change")
        return "tester", "test_only", reasons

    # Supervisor/infra fix — repair keywords + supervisor context → complex_implementation
    if _contains_any(goal, _REPAIR_KEYWORDS) and _contains_any(goal, _SUPERVISOR_KEYWORDS):
        reasons.append("repair + supervisor keywords → complex_implementation")
        return "backend_coder", "complex_implementation", reasons

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
    project_root: Optional[str] = None,
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
    success_rate = successes / total

    # Issue #522 — quality-weighted success rate
    quality_weighted: Optional[float] = None
    if project_root:
        try:
            from igris.core.outcome_quality_tracker import (
                load_quality_scores, avg_quality_for_profile,
            )
            scores = load_quality_scores(project_root)
            avg_q = avg_quality_for_profile(matching, profile, scores, min_history=3)
            if avg_q is not None:
                quality_weighted = success_rate * avg_q
        except Exception:
            pass  # quality tracking is best-effort, never blocks routing

    return {
        "total": total,
        "success_rate": success_rate,
        "quality_weighted_success_rate": quality_weighted if quality_weighted is not None else success_rate,
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

    # Accumulated escalation signals — gpu_reasoning fires BEFORE decomposition
    no_diff_repair_count = signals.get("no_diff_repair", 0)
    reasoning_timeout_count = signals.get("reasoning_timeout", 0)
    needs_gpu_reasoning = (
        no_diff_repair_count >= 3
        or (no_diff_repair_count + reasoning_timeout_count) >= 4
        or (no_diff_repair_count >= 2 and prior >= 2)
        or task_type in ("hard_debugging", "architecture_review")
    )
    # After gpu_reasoning has been tried at least twice, fall back to decomposition.
    # Threshold is 2 (not 1) because the first run may be blocked in baseline tests
    # before reasoning ever starts — VastAI would never have been attempted despite
    # prior_attempts=1.  Two runs gives gpu_reasoning a real chance to fire at least
    # once through _decide_action before escalating to decomposition.
    gpu_already_tried = needs_gpu_reasoning and prior >= 2

    force_decompose = (
        task_type in ("planning", "memory_system")
        or gpu_already_tried
        or (has_max_steps_ceiling and len(request.goal_text) > 500 and not needs_gpu_reasoning)
    )
    force_strong = (
        has_max_steps_ceiling
        or (is_repair and prior >= 1)
        or task_type in ("security_review", "devops_runtime", "hard_debugging")
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
            preferred_model="deepseek-v4-flash",   # primary: DeepSeek V4 Flash
            fallback_model_path=["gpt-4o-mini"],   # fallback: OpenAI mini
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.75,
            bootstrap_cost_per_attempt=0.08,       # deepseek much cheaper than gpt-4o-mini
            budget_tier="low",
        ))

    if task_type == "pytest_failure":
        candidates.append(_Candidate(
            profile="mini_execution",
            strategy="debug_mini",
            preferred_model="deepseek-v4-flash",   # primary: DeepSeek V4 Flash
            fallback_model_path=["gpt-4o-mini", "gpt-4o"],
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.70,
            bootstrap_cost_per_attempt=0.10,
            budget_tier="low",
        ))

    if task_type == "backend_endpoint" and not force_strong:
        candidates.append(_Candidate(
            profile="mini_execution",
            strategy="helper_advice_then_mini_execution",
            preferred_model="deepseek-v4-flash",   # primary: DeepSeek V4 Flash
            fallback_model_path=["gpt-4o-mini", "gpt-4o"],
            codex_helper=True,
            decompose=False,
            bootstrap_success_prob=0.72,
            bootstrap_cost_per_attempt=0.15,
            budget_tier="medium",
        ))

    # GPU reasoning — lowest expected cost when Vast.ai instance is available.
    # Only offered on the FIRST assignment attempt; after failure decomposition takes over.
    if needs_gpu_reasoning and not gpu_already_tried:
        candidates.append(_Candidate(
            profile="gpu_reasoning",
            strategy="gpu_reasoning_direct",
            preferred_model="deepseek-r1:32b",
            fallback_model_path=["deepseek-v4-pro", "gpt-4o"],
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.75,
            bootstrap_cost_per_attempt=0.02,
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

    # Medium cloud — for generic code_reasoning tasks before escalating to strong
    if task_type == "code_reasoning" and not force_strong and not force_decompose and not needs_gpu_reasoning:
        candidates.append(_Candidate(
            profile="cheap_cloud_reasoning",
            strategy="cloud_reasoning_direct",
            preferred_model="deepseek-v4-flash",
            fallback_model_path=["gpt-4o-mini"],
            codex_helper=False,
            decompose=False,
            bootstrap_success_prob=0.62,
            bootstrap_cost_per_attempt=0.25,
            budget_tier="low",
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
        "gpu_reasoning": "vastai_ollama",
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
            profile_stats = _compute_profile_stats(
                outcomes, agent_role, task_type, cand.profile,
                project_root=request.project_root if hasattr(request, "project_root") else None,
            )

            if profile_stats and profile_stats["total"] >= _MIN_HISTORY_FOR_LEARNING:
                # Issue #522 — use quality-weighted success rate when available
                p_success = profile_stats.get(
                    "quality_weighted_success_rate",
                    profile_stats["success_rate"],
                )
                avg_cost = profile_stats["avg_cost"]
                avg_attempts = profile_stats["avg_attempts"]
                source = "history" if profile_stats.get("quality_weighted_success_rate") is None else "history+quality"
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
