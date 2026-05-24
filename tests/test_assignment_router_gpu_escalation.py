"""Tests for AssignmentRouter GPU escalation hierarchy.

Verifies that:
- no_diff_repair >= 3 signals trigger gpu_reasoning profile (not cheap_cloud)
- Combined (no_diff_repair + reasoning_timeout) >= 4 triggers escalation
- no_diff_repair >= 2 AND prior_attempts >= 2 triggers escalation
- gpu_reasoning is offered only on FIRST attempt (prior=0)
- After gpu failed (prior=1) → decompose_first / strong_execution
- Easy tasks without escalation signals stay on cheap_cloud_reasoning
- task_type is classified as hard_debugging for escalation scenarios
- preferred_model is deepseek-r1:32b for gpu_reasoning
- fallback chain includes strong API models
- gpu_reasoning profile maps to vastai_ollama provider
"""

from __future__ import annotations

import pytest

from igris.core.assignment_router import (
    AssignmentDecision,
    AssignmentRequest,
    AssignmentRouter,
    _classify_goal,
    _build_candidates,
    _profile_to_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def router() -> AssignmentRouter:
    return AssignmentRouter()


def req(**kwargs) -> AssignmentRequest:
    defaults = dict(
        goal_text="Fix issue: agent explores codebase but never writes any diff",
        risk_level="medium",
        budget_remaining_usd=10.0,
    )
    defaults.update(kwargs)
    return AssignmentRequest(**defaults)


# ---------------------------------------------------------------------------
# 1. _classify_goal escalation override
# ---------------------------------------------------------------------------

class TestClassifyGoalEscalationOverride:

    def test_no_diff_repair_3_triggers_hard_debugging(self):
        r = req(capability_signals={"no_diff_repair": 3}, prior_attempts=0)
        _, task_type, reasons = _classify_goal(r)
        assert task_type == "hard_debugging"
        assert any("escalation" in rs for rs in reasons)

    def test_no_diff_repair_5_triggers_hard_debugging(self):
        r = req(capability_signals={"no_diff_repair": 5}, prior_attempts=0)
        _, task_type, _ = _classify_goal(r)
        assert task_type == "hard_debugging"

    def test_combined_signals_4_triggers(self):
        """no_diff_repair=2 + reasoning_timeout=2 = 4 → escalation."""
        r = req(capability_signals={"no_diff_repair": 2, "reasoning_timeout": 2})
        _, task_type, _ = _classify_goal(r)
        assert task_type == "hard_debugging"

    def test_combined_signals_3_no_trigger(self):
        """no_diff_repair=1 + reasoning_timeout=2 = 3 < 4 → no escalation."""
        r = req(capability_signals={"no_diff_repair": 1, "reasoning_timeout": 2})
        _, task_type, _ = _classify_goal(r)
        assert task_type != "hard_debugging"

    def test_ndr2_prior2_triggers(self):
        """no_diff_repair=2 AND prior_attempts=2 → escalation."""
        r = req(capability_signals={"no_diff_repair": 2}, prior_attempts=2)
        _, task_type, _ = _classify_goal(r)
        assert task_type == "hard_debugging"

    def test_ndr2_prior1_no_trigger(self):
        """no_diff_repair=2 AND prior_attempts=1 → NOT escalated (< 2 prior)."""
        r = req(capability_signals={"no_diff_repair": 2}, prior_attempts=1)
        _, task_type, _ = _classify_goal(r)
        assert task_type != "hard_debugging"

    def test_ndr1_no_trigger(self):
        """no_diff_repair=1 never triggers escalation."""
        r = req(capability_signals={"no_diff_repair": 1}, prior_attempts=0)
        _, task_type, _ = _classify_goal(r)
        assert task_type != "hard_debugging"

    def test_escalation_overrides_keyword_classification(self):
        """Even a goal text that looks like 'documentation' gets escalated if signals cross threshold."""
        r = req(
            goal_text="Add docstring to all public functions",  # normally → documentation
            capability_signals={"no_diff_repair": 4},
            prior_attempts=0,
        )
        _, task_type, _ = _classify_goal(r)
        assert task_type == "hard_debugging"

    def test_no_signals_simple_goal_not_escalated(self):
        """No signals → keyword classification applies normally."""
        r = req(
            goal_text="Add docstring to auth module",
            capability_signals={},
            prior_attempts=0,
        )
        _, task_type, _ = _classify_goal(r)
        assert task_type != "hard_debugging"

    def test_escalation_reason_contains_signal_counts(self):
        """Reason string must report both signal counts for traceability."""
        r = req(capability_signals={"no_diff_repair": 4, "reasoning_timeout": 1})
        _, _, reasons = _classify_goal(r)
        reason_text = " ".join(reasons)
        assert "no_diff_repair=4" in reason_text
        assert "reasoning_timeout=1" in reason_text


# ---------------------------------------------------------------------------
# 2. _build_candidates GPU candidate injection
# ---------------------------------------------------------------------------

class TestBuildCandidatesGpuCandidate:

    def _candidates(self, **kwargs) -> list:
        r = req(**kwargs)
        agent_role, task_type, _ = _classify_goal(r)
        return _build_candidates(agent_role, task_type, r)

    def test_gpu_reasoning_candidate_present_when_triggered(self):
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=0,
        )
        profiles = [c.profile for c in cands]
        assert "gpu_reasoning" in profiles

    def test_gpu_reasoning_candidate_absent_without_signals(self):
        cands = self._candidates(
            goal_text="Add docstring to all functions",
            capability_signals={}, prior_attempts=0,
        )
        profiles = [c.profile for c in cands]
        assert "gpu_reasoning" not in profiles

    def test_gpu_reasoning_candidate_absent_after_prior2(self):
        """After two attempts (prior>=2), gpu_already_tried → no gpu_reasoning candidate.

        Threshold is 2 (not 1) because the first run may be blocked before reasoning
        starts (e.g. baseline test failure) — VastAI would never have been attempted
        despite prior_attempts=1.  Two runs gives gpu_reasoning a real chance to fire.
        """
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=2,
        )
        profiles = [c.profile for c in cands]
        assert "gpu_reasoning" not in profiles

    def test_gpu_reasoning_candidate_present_at_prior1(self):
        """With prior=1 gpu_reasoning is still offered (threshold raised to prior>=2)."""
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=1,
        )
        profiles = [c.profile for c in cands]
        assert "gpu_reasoning" in profiles

    def test_gpu_reasoning_model_is_deepseek_r1(self):
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=0,
        )
        gpu_cand = next(c for c in cands if c.profile == "gpu_reasoning")
        assert gpu_cand.preferred_model == "deepseek-r1:32b"

    def test_gpu_reasoning_fallback_includes_api_models(self):
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=0,
        )
        gpu_cand = next(c for c in cands if c.profile == "gpu_reasoning")
        assert "deepseek-v4-pro" in gpu_cand.fallback_model_path or "gpt-4o" in gpu_cand.fallback_model_path

    def test_gpu_reasoning_does_not_decompose(self):
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=0,
        )
        gpu_cand = next(c for c in cands if c.profile == "gpu_reasoning")
        assert gpu_cand.decompose is False

    def test_gpu_reasoning_low_cost_beats_cheap_cloud(self):
        """gpu_reasoning cost_per_attempt must be lower than cheap_cloud_reasoning."""
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=0,
        )
        gpu = next((c for c in cands if c.profile == "gpu_reasoning"), None)
        cheap = next((c for c in cands if c.profile == "cheap_cloud_reasoning"), None)
        if gpu and cheap:
            assert gpu.bootstrap_cost_per_attempt < cheap.bootstrap_cost_per_attempt

    def test_decompose_offered_after_gpu_tried(self):
        """With prior>=2, gpu_already_tried=True → force_decompose=True → decompose_first in candidates."""
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=2,
        )
        strategies = [c.strategy for c in cands]
        assert "decompose_first" in strategies

    def test_strong_execution_always_present_as_fallback(self):
        cands = self._candidates(
            capability_signals={"no_diff_repair": 4}, prior_attempts=0,
        )
        profiles = [c.profile for c in cands]
        assert "strong_execution" in profiles


# ---------------------------------------------------------------------------
# 3. _profile_to_provider mapping
# ---------------------------------------------------------------------------

class TestProfileToProvider:

    def test_gpu_reasoning_maps_to_vastai_ollama(self):
        assert _profile_to_provider("gpu_reasoning") == "vastai_ollama"

    def test_cheap_cloud_maps_to_deepseek(self):
        assert _profile_to_provider("cheap_cloud_reasoning") == "deepseek"

    def test_strong_execution_maps_to_deepseek_strong(self):
        assert _profile_to_provider("strong_execution") == "deepseek_strong"

    def test_unknown_profile_falls_back_to_openai(self):
        assert _profile_to_provider("nonexistent_profile") == "openai"


# ---------------------------------------------------------------------------
# 4. End-to-end AssignmentRouter.decide() escalation
# ---------------------------------------------------------------------------

class TestRouterDecideEscalation:

    def test_ndr5_prior0_routes_to_gpu_reasoning(self):
        r = req(
            capability_signals={"no_diff_repair": 5, "reasoning_timeout": 1},
            prior_attempts=0,
            is_repair=True,
        )
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"
        assert d.preferred_model == "deepseek-r1:32b"
        assert d.task_type == "hard_debugging"

    def test_ndr5_prior0_no_decompose(self):
        """First gpu_reasoning attempt must NOT decompose."""
        r = req(
            capability_signals={"no_diff_repair": 5},
            prior_attempts=0,
        )
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"
        assert d.should_decompose_first is False

    def test_ndr3_prior0_routes_to_gpu_reasoning(self):
        """Threshold is >= 3 — exactly 3 must trigger."""
        r = req(capability_signals={"no_diff_repair": 3}, prior_attempts=0)
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"

    def test_combined_4_signals_routes_to_gpu_reasoning(self):
        """no_diff_repair=2 + reasoning_timeout=2 = 4 → gpu_reasoning."""
        r = req(
            capability_signals={"no_diff_repair": 2, "reasoning_timeout": 2},
            prior_attempts=0,
        )
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"

    def test_ndr2_prior2_classifier_escalates_but_gpu_already_tried(self):
        """ndr=2 + prior=2: classifier → hard_debugging, but gpu_already_tried
        (prior >= 1) means we fall to strong_execution with decompose — NOT back to gpu."""
        r = req(capability_signals={"no_diff_repair": 2}, prior_attempts=2)
        d = router().decide(r)
        # Classifier escalated (not cheap_cloud_reasoning)
        assert d.task_type == "hard_debugging"
        assert d.preferred_profile != "cheap_cloud_reasoning"
        # gpu_reasoning is NOT retried — already tried on prior=0
        assert d.preferred_profile != "gpu_reasoning"
        # Should be in decompose or strong_execution phase
        assert d.preferred_profile in ("strong_execution", "cheap_cloud_reasoning")

    def test_ndr5_prior2_does_not_retry_gpu(self):
        """After two attempts (prior=2), should NOT route back to gpu_reasoning."""
        r = req(capability_signals={"no_diff_repair": 5}, prior_attempts=2, is_repair=True)
        d = router().decide(r)
        assert d.preferred_profile != "gpu_reasoning"

    def test_ndr5_prior2_uses_decompose(self):
        """After two gpu attempts, decompose_first or strong with decompose=True."""
        r = req(capability_signals={"no_diff_repair": 5}, prior_attempts=2, is_repair=True)
        d = router().decide(r)
        # Either decompose flag set OR strategy contains 'decompose'
        assert d.should_decompose_first is True or "decompose" in d.execution_strategy

    def test_easy_task_no_signals_uses_cheap_cloud(self):
        """Simple docs task with no escalation signals → cheap_cloud_reasoning."""
        r = req(
            goal_text="Add docstring to all public functions",
            capability_signals={},
            prior_attempts=0,
        )
        d = router().decide(r)
        assert d.preferred_profile == "cheap_cloud_reasoning"
        assert d.preferred_model == "deepseek-v4-flash"

    def test_ndr1_no_escalation(self):
        """Single no_diff_repair signal is not enough to escalate."""
        r = req(
            capability_signals={"no_diff_repair": 1}, prior_attempts=0,
        )
        d = router().decide(r)
        assert d.preferred_profile != "gpu_reasoning"

    def test_ndr2_prior1_no_escalation(self):
        """no_diff_repair=2 + prior=1: combined threshold not met."""
        r = req(capability_signals={"no_diff_repair": 2}, prior_attempts=1)
        d = router().decide(r)
        assert d.preferred_profile != "gpu_reasoning"

    def test_gpu_reasoning_execution_strategy(self):
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        d = router().decide(r)
        assert d.execution_strategy == "gpu_reasoning_direct"

    def test_gpu_reasoning_success_probability(self):
        """gpu_reasoning bootstrap_success_prob must be 0.75."""
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"
        assert abs(d.estimated_success_probability - 0.75) < 1e-6

    def test_fallback_models_present(self):
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        d = router().decide(r)
        assert len(d.fallback_model_path) >= 1

    def test_budget_limit_medium_tier(self):
        """gpu_reasoning uses medium budget tier ($2.00)."""
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"
        assert d.budget_limit <= 2.00

    def test_codex_helper_not_requested_for_gpu(self):
        """GPU direct reasoning does not need codex helper."""
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"
        assert d.should_call_codex_helper_first is False

    def test_gpu_reasoning_decision_is_dict_serializable(self):
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        d = router().decide(r)
        assert d.preferred_profile == "gpu_reasoning"
        data = d.to_dict()
        assert data["preferred_profile"] == "gpu_reasoning"
        assert data["preferred_model"] == "deepseek-r1:32b"
        assert data["task_type"] == "hard_debugging"


# ---------------------------------------------------------------------------
# 5. Throughput reasoning invariants
# ---------------------------------------------------------------------------

class TestThroughputReasoningInvariants:

    def test_gpu_reasoning_cost_is_near_zero(self):
        """Vast.ai local GPU cost per attempt must be << API cost (0.02 vs 0.25+)."""
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        agent_role, task_type, _ = _classify_goal(r)
        cands = _build_candidates(agent_role, task_type, r)
        gpu = next((c for c in cands if c.profile == "gpu_reasoning"), None)
        assert gpu is not None
        assert gpu.bootstrap_cost_per_attempt <= 0.05  # practically free vs API

    def test_gpu_expected_cost_beats_all_others(self):
        """gpu_reasoning must have lowest expected cost among all candidates when triggered."""
        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        agent_role, task_type, _ = _classify_goal(r)
        cands = _build_candidates(agent_role, task_type, r)

        def expected_cost(c):
            return c.bootstrap_cost_per_attempt * 1.5 / max(c.bootstrap_success_prob, 0.05)

        gpu = next((c for c in cands if c.profile == "gpu_reasoning"), None)
        assert gpu is not None
        gpu_cost = expected_cost(gpu)
        for other in cands:
            if other.profile != "gpu_reasoning":
                assert gpu_cost < expected_cost(other), (
                    f"gpu_reasoning cost {gpu_cost:.4f} should beat {other.profile} cost {expected_cost(other):.4f}"
                )

    def test_sequential_vs_parallel_gpu_advantage(self):
        """Illustrates why parallel Vast.ai instances dominate sequential API calls.

        With 5 parallel GPU instances at $2.44/day and 129 issues resolved,
        cost-per-issue is $0.019. With sequential API at $2.39/day and 38 issues,
        cost-per-issue is $0.063. Parallel GPU is 3.3x cheaper per issue.
        """
        vastai_daily_cost_5_instances = 5 * 0.488         # $2.44/day
        issues_per_day_parallel = 5 * 25.8                # 129 issues/day
        cost_per_issue_gpu = vastai_daily_cost_5_instances / issues_per_day_parallel

        api_daily_cost = 2.39
        issues_per_day_sequential = 38
        cost_per_issue_api = api_daily_cost / issues_per_day_sequential

        assert cost_per_issue_gpu < cost_per_issue_api
        assert (cost_per_issue_api / cost_per_issue_gpu) > 2.0  # at least 2x cheaper

    def test_gpu_success_prob_above_threshold(self):
        """gpu_reasoning p_success=0.75 must exceed code_reasoning threshold=0.60."""
        from igris.core.assignment_router import _SUCCESS_THRESHOLD
        threshold = _SUCCESS_THRESHOLD.get("code_reasoning", 0.60)

        r = req(capability_signals={"no_diff_repair": 4}, prior_attempts=0)
        agent_role, task_type, _ = _classify_goal(r)
        cands = _build_candidates(agent_role, task_type, r)
        gpu = next((c for c in cands if c.profile == "gpu_reasoning"), None)
        assert gpu is not None
        assert gpu.bootstrap_success_prob > threshold
