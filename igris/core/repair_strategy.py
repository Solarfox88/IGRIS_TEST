"""
repair_strategy.py — Epic #1074

Standalone, testable repair-strategy logic extracted from SelfRepairSupervisor.

Before this module the repair strategy was determined by a long chain of
if/elif blocks inline inside _repair_cycle, making it untestable and hard to
reason about.  This module makes the decision explicit, pure and auditable.

Design:
  RepairContext   — carries all information needed to pick a strategy
  RepairStrategy  — the chosen strategy + derived parameters
  decide_strategy — pure function: RepairContext → RepairStrategy
  select_profile  — pure function: pick the LLM profile for repair
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

# ---------------------------------------------------------------------------
# Constants (mirrors the supervisor's REPAIRABLE_FAILURES)
# ---------------------------------------------------------------------------

REPAIRABLE: FrozenSet[str] = frozenset({
    "pytest_failure", "reasoning_loop_blocked", "max_steps", "ask_user",
    "missing_tests", "missing_ui_visibility", "wrong_file_edit",
    "infrastructure_bug", "invalid_bootstrap", "syntax_error",
    "semantic_incomplete", "test_runner_timeout",
})

# Failures that benefit from the strong model (not cheap model)
STRONG_MODEL_FAILURES: FrozenSet[str] = frozenset({
    "syntax_error", "infrastructure_bug", "semantic_incomplete",
    "wrong_file_edit",
})

# Failures where same-failure repetition strongly suggests capability ceiling
CEILING_SIGNAL_FAILURES: FrozenSet[str] = frozenset({
    "pytest_failure", "max_steps", "reasoning_loop_blocked",
})

# Base task types for repair
_TASK_TYPES: Dict[str, str] = {
    "syntax_error": "code_repair",
    "infrastructure_bug": "code_repair",
    "pytest_failure": "code_reasoning",
    "wrong_file_edit": "code_reasoning",
    "semantic_incomplete": "semantic_repair",
    "missing_tests": "test_generation",
    "missing_ui_visibility": "ui_repair",
    "reasoning_loop_blocked": "code_reasoning",
    "max_steps": "code_reasoning",
    "ask_user": "code_reasoning",
    "test_runner_timeout": "code_reasoning",
    "invalid_bootstrap": "code_repair",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairContext:
    """Everything the strategy engine needs to pick a repair approach.

    All fields are simple types so RepairContext is easy to construct in tests.
    """
    failure_class: str
    cycle: int                          # 0-indexed repair cycle number
    same_failure_count: int             # consecutive identical failures
    max_repair_cycles: int
    base_timeout_seconds: int
    has_high_risk_advice: bool = False  # advisory escalation signal
    has_execution_plan: bool = False    # whether a plan exists
    environment: str = "dev"           # "dev" | "staging" | "production"
    # Capability signals accumulated so far
    capability_signals: Dict[str, int] = field(default_factory=dict)


@dataclass
class RepairStrategy:
    """The chosen repair strategy with all derived parameters."""
    task_type: str
    profile: str                # LLM profile to use
    timeout_seconds: int
    goal_prefix: str            # prepended to the repair goal
    use_targeted_tests: bool    # run targeted tests after repair (not full)
    skip_repair: bool = False   # True when repair is not warranted
    skip_reason: str = ""
    escalate_to_decomposition: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Core strategy decision function
# ---------------------------------------------------------------------------

def decide_repair_strategy(ctx: RepairContext) -> RepairStrategy:
    """Pure function: given RepairContext, return the best RepairStrategy.

    Decision tree (in order of priority):
      1. Non-repairable failure → skip
      2. Same failure repeated > threshold → signal capability ceiling, skip
      3. High-risk advisory + high failure count → skip (avoid burning budget)
      4. Strong-model failure → use strong profile with extended timeout
      5. Default → standard profile with base timeout
    """
    failure = ctx.failure_class

    # 1. Non-repairable
    if failure not in REPAIRABLE:
        return RepairStrategy(
            task_type="code_reasoning",
            profile="default",
            timeout_seconds=ctx.base_timeout_seconds,
            goal_prefix="",
            use_targeted_tests=False,
            skip_repair=True,
            skip_reason=f"failure_class={failure!r} is not in REPAIRABLE set",
        )

    # 2. Capability ceiling signal
    _ceiling_threshold = int(os.getenv("IGRIS_CAPABILITY_LIMIT_THRESHOLD", "2"))
    if ctx.same_failure_count >= _ceiling_threshold and failure in CEILING_SIGNAL_FAILURES:
        return RepairStrategy(
            task_type="code_reasoning",
            profile="default",
            timeout_seconds=ctx.base_timeout_seconds,
            goal_prefix="",
            use_targeted_tests=False,
            skip_repair=True,
            skip_reason=(
                f"Capability ceiling: same failure {failure!r} repeated "
                f"{ctx.same_failure_count}× (threshold={_ceiling_threshold})"
            ),
            escalate_to_decomposition=True,
        )

    # 3. High-risk advisory with repeated failure
    if ctx.has_high_risk_advice and ctx.same_failure_count >= 1:
        return RepairStrategy(
            task_type="code_reasoning",
            profile="default",
            timeout_seconds=ctx.base_timeout_seconds,
            goal_prefix="",
            use_targeted_tests=False,
            skip_repair=True,
            skip_reason=(
                "High-risk advisory + repeated failure: skipping repair to "
                "preserve budget; manual review recommended"
            ),
        )

    # 4. Select task type and profile
    task_type = _TASK_TYPES.get(failure, "code_reasoning")
    profile = select_repair_profile(ctx)
    timeout = _compute_timeout(ctx, profile)
    goal_prefix = _build_goal_prefix(ctx)
    use_targeted = failure not in ("missing_tests", "missing_ui_visibility")

    return RepairStrategy(
        task_type=task_type,
        profile=profile,
        timeout_seconds=timeout,
        goal_prefix=goal_prefix,
        use_targeted_tests=use_targeted,
        notes=(
            f"cycle={ctx.cycle}, same_failure_count={ctx.same_failure_count}, "
            f"has_plan={ctx.has_execution_plan}"
        ),
    )


def select_repair_profile(ctx: RepairContext) -> str:
    """Pure function: pick the LLM profile for this repair attempt.

    Logic:
      - Cycle 0 + strong failure → strong_execution
      - Cycle 1+ with previous failure → strong_cloud_reasoning
      - Default → None (let backend choose)
    """
    failure = ctx.failure_class

    if failure in STRONG_MODEL_FAILURES:
        if ctx.cycle == 0:
            return "strong_execution"
        else:
            return "strong_cloud_reasoning"

    if ctx.cycle >= 1 and ctx.same_failure_count >= 1:
        # Escalate to stronger model on repeat
        return "strong_execution"

    return ""   # default backend profile


def _compute_timeout(ctx: RepairContext, profile: str) -> int:
    """Compute repair timeout from base, profile, and environment."""
    base = ctx.base_timeout_seconds
    _STRONG_PROFILES = {"strong_execution", "strong_cloud_reasoning", "gpu_reasoning"}

    if profile in _STRONG_PROFILES:
        # Strong models need 3× the base timeout, at least 2400s
        extended = int(os.getenv(
            "IGRIS_STRONG_REASONING_TIMEOUT_SECONDS",
            str(max(base * 3, 2400)),
        ))
        return extended

    return base


def _build_goal_prefix(ctx: RepairContext) -> str:
    """Build a contextual prefix that guides the repair reasoning."""
    parts: list = []

    if ctx.same_failure_count > 0:
        parts.append(
            f"[Repair cycle {ctx.cycle+1}, same failure seen {ctx.same_failure_count} times]"
        )
    else:
        parts.append(f"[Repair cycle {ctx.cycle+1}]")

    if ctx.has_execution_plan:
        parts.append(
            "An execution plan was produced in a previous step — follow it."
        )

    if ctx.failure_class == "syntax_error":
        parts.append(
            "Priority: fix syntax errors first. Do not change test files."
        )
    elif ctx.failure_class == "missing_tests":
        parts.append(
            "Priority: add missing test coverage. "
            "Tests must be in tests/ and follow existing conventions."
        )

    return " ".join(parts)
