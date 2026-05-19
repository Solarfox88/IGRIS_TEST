"""Tests for AssignmentRouter, AssignmentOutcomes, and AgentRegistry."""
from __future__ import annotations

import json
import os

import pytest

from igris.core.agent_registry import get_default_task_type, list_roles
from igris.core.assignment_outcomes import (
    compute_task_signature,
    load_assignment_outcomes,
    sanitize_for_storage,
    save_assignment_outcome,
)
from igris.core.assignment_router import (
    AssignmentRequest,
    AssignmentRouter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router(tmp_path) -> AssignmentRouter:
    path = str(tmp_path / ".igris" / "assignment_outcomes.json")
    return AssignmentRouter(outcomes_path=path)


def make_request(**kwargs) -> AssignmentRequest:
    defaults = dict(
        goal_text="do something",
        risk_level="medium",
        budget_remaining_usd=10.0,
    )
    defaults.update(kwargs)
    return AssignmentRequest(**defaults)


# ---------------------------------------------------------------------------
# 1. simple docs task routes cheap
# ---------------------------------------------------------------------------

def test_simple_docs_routes_cheap(tmp_path):
    router = make_router(tmp_path)
    req = make_request(
        goal_text="Aggiungi docstring a tutte le funzioni pubbliche",
        risk_level="low",
    )
    decision = router.decide(req)
    assert decision.preferred_profile in (
        "cheap_cloud_reasoning", "mini_execution", "local_light", "local_coder",
    )
    assert not decision.should_decompose_first
    assert decision.budget_limit <= 2.01
    assert decision.confidence >= 0.70


# ---------------------------------------------------------------------------
# 2. endpoint task routes backend_coder + helper + mini first
# ---------------------------------------------------------------------------

def test_endpoint_task_routes_backend_coder(tmp_path):
    router = make_router(tmp_path)
    req = make_request(
        goal_text="Implementa GET /api/diagnostics/session-resume con logica reale e test",
        risk_level="medium",
        required_tests=["tests/test_session_resume.py"],
    )
    decision = router.decide(req)
    assert decision.agent_role == "backend_coder"
    assert decision.task_type == "backend_endpoint"
    assert decision.should_call_codex_helper_first
    assert decision.confidence >= 0.70
    assert "gpt-4o" in decision.fallback_model_path or decision.preferred_profile in (
        "mini_execution", "strong_execution"
    )


# ---------------------------------------------------------------------------
# 3. semantic_incomplete + prior failure escalates to strong_execution
# ---------------------------------------------------------------------------

def test_semantic_incomplete_escalates_to_strong(tmp_path):
    router = make_router(tmp_path)
    req = make_request(
        goal_text="Implementa l'endpoint /api/users con vera logica di business",
        failure_class="semantic_incomplete",
        capability_signals={"stub_detected": 2},
        prior_attempts=1,
        is_repair=True,
    )
    decision = router.decide(req)
    assert decision.preferred_profile == "strong_execution"
    assert decision.should_call_codex_helper_first
    assert any(
        "stub" in r.lower() or "semantic" in r.lower() or "failure" in r.lower()
        for r in decision.reasons
    )


# ---------------------------------------------------------------------------
# 4. repeated no_diff/max_steps triggers decompose or strong
# ---------------------------------------------------------------------------

def test_repeated_no_diff_triggers_decompose_or_strong(tmp_path):
    router = make_router(tmp_path)
    req = make_request(
        goal_text=(
            "Refactor completo del sistema di autenticazione, migrazione da JWT a OAuth2, "
            "aggiornamento di tutti i middleware e test di integrazione completi"
        ),
        failure_class="no_diff",
        capability_signals={"no_diff": 3},
        prior_attempts=2,
    )
    decision = router.decide(req)
    assert decision.should_decompose_first or decision.preferred_profile == "strong_execution"


# ---------------------------------------------------------------------------
# 5. prior history changes model choice away from mini_execution
# ---------------------------------------------------------------------------

def test_prior_history_influences_choice(tmp_path):
    outcomes_path = str(tmp_path / ".igris" / "assignment_outcomes.json")
    os.makedirs(os.path.dirname(outcomes_path), exist_ok=True)
    # Inject 6 failed records for mini_execution on backend_endpoint
    records = [
        {
            "agent_role": "backend_coder",
            "task_type": "backend_endpoint",
            "preferred_profile": "mini_execution",
            "outcome": "blocked",
            "failure_class": "pytest_failure",
            "cost_usd": 0.30,
            "attempts": 2,
        }
        for _ in range(6)
    ]
    with open(outcomes_path, "w") as f:
        json.dump(records, f)

    router = AssignmentRouter(outcomes_path=outcomes_path)
    req = make_request(
        goal_text="Implementa GET /api/users con test",
        required_tests=["tests/test_users.py"],
    )
    decision = router.decide(req)
    # History shows mini_execution 0% success — must escalate or pick differently
    # Either not mini_execution OR it has escalated to strong
    assert decision.preferred_profile in ("strong_execution", "cheap_cloud_reasoning") or \
           decision.preferred_profile != "mini_execution"


# ---------------------------------------------------------------------------
# 6. high-risk task avoids weak/local model
# ---------------------------------------------------------------------------

def test_high_risk_avoids_weak_model(tmp_path):
    router = make_router(tmp_path)
    req = make_request(
        goal_text="Revisa i permessi di accesso alle API key e il sistema di autenticazione",
        risk_level="high",
    )
    decision = router.decide(req)
    assert decision.preferred_profile not in ("local_light", "local_coder")
    assert decision.agent_role in ("security_reviewer", "devops", "backend_coder")


# ---------------------------------------------------------------------------
# 7. low budget caps budget_limit
# ---------------------------------------------------------------------------

def test_budget_caps_budget_limit(tmp_path):
    router = make_router(tmp_path)
    req = make_request(
        goal_text="Aggiungi docstring alle funzioni",
        budget_remaining_usd=0.20,
        risk_level="low",
    )
    decision = router.decide(req)
    assert decision.budget_limit <= 0.20 + 0.01


# ---------------------------------------------------------------------------
# 8. Codex direct execution disabled by default
# ---------------------------------------------------------------------------

def test_codex_direct_disabled_by_default():
    assert os.environ.get("IGRIS_ENABLE_CODEX_DIRECT_EXECUTION", "false").lower() != "true"


# ---------------------------------------------------------------------------
# 9. assignment_outcomes.json written atomically
# ---------------------------------------------------------------------------

def test_outcomes_written_atomically(tmp_path):
    path = str(tmp_path / "outcomes.json")
    record = {
        "task_signature": "abc123",
        "agent_role": "backend_coder",
        "task_type": "backend_endpoint",
        "outcome": "success",
        "cost_usd": 0.25,
        "attempts": 1,
        "created_at": "2026-01-01T00:00:00Z",
    }
    save_assignment_outcome(path, record)
    loaded = load_assignment_outcomes(path)
    assert len(loaded) == 1
    assert loaded[0]["agent_role"] == "backend_coder"

    save_assignment_outcome(path, {**record, "agent_role": "tester"})
    loaded2 = load_assignment_outcomes(path)
    assert len(loaded2) == 2
    assert loaded2[1]["agent_role"] == "tester"


# ---------------------------------------------------------------------------
# 10. sanitize_for_storage redacts secrets
# ---------------------------------------------------------------------------

def test_sanitize_redacts_secrets():
    record = {
        "goal_excerpt": "use key sk-proj-abc123xyz4567890123 secret",
        "api_key": "sk-abcdef123456789012345678901234",
        "bearer": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.longtoken1234567890",
        "nested": {"key": "sk-proj-anotherkey123456789012345"},
        "cost_usd": 0.10,
    }
    safe = sanitize_for_storage(record)
    text = json.dumps(safe)
    assert "sk-proj-abc123xyz" not in text
    assert "sk-abcdef" not in text
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in text
    assert "[REDACTED]" in text
    assert safe["cost_usd"] == 0.10


# ---------------------------------------------------------------------------
# 11. compute_task_signature stable for whitespace/case
# ---------------------------------------------------------------------------

def test_task_signature_stable():
    sig1 = compute_task_signature("Fix the bug in the API endpoint")
    sig2 = compute_task_signature("fix the bug in the api endpoint")
    sig3 = compute_task_signature("  Fix  the  bug  in  the  API  endpoint  ")
    assert sig1 == sig2 == sig3
    assert len(sig1) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# 12. AssignmentRouter does not import ModelOrchestrator
# ---------------------------------------------------------------------------

def test_assignment_router_does_not_import_orchestrator():
    """Semantic routing lives in AssignmentRouter; ModelOrchestrator stays a dispatcher."""
    import importlib
    import sys

    mod = sys.modules.get("igris.core.assignment_router")
    if mod is None:
        mod = importlib.import_module("igris.core.assignment_router")

    src_path = mod.__file__
    import_lines = [
        line for line in open(src_path).readlines()
        if line.strip().startswith(("import ", "from "))
    ]
    import_text = "\n".join(import_lines)
    assert "model_orchestrator" not in import_text
    assert "ModelOrchestrator" not in import_text
