"""Agent role registry for IGRIS assignment routing."""
from __future__ import annotations

from typing import Any, Dict, Optional

ROLES: Dict[str, Dict[str, Any]] = {
    "planner": {
        "description": "Decomposes large missions and plans execution.",
        "default_task_type": "planning",
        "risk_tolerance": "low",
    },
    "backend_coder": {
        "description": "Implements backend/API changes.",
        "default_task_type": "backend_endpoint",
        "risk_tolerance": "medium",
    },
    "tester": {
        "description": "Writes and repairs tests.",
        "default_task_type": "test_only",
        "risk_tolerance": "low",
    },
    "test_debugger": {
        "description": "Diagnoses pytest failures.",
        "default_task_type": "pytest_failure",
        "risk_tolerance": "low",
    },
    "devops": {
        "description": "Handles runtime, CI, deploy, smoke.",
        "default_task_type": "devops_runtime",
        "risk_tolerance": "high",
    },
    "security_reviewer": {
        "description": "Reviews secrets, destructive diffs and safety gates.",
        "default_task_type": "security_review",
        "risk_tolerance": "very_high",
    },
    "memory_architect": {
        "description": "Works on memory/synapse systems.",
        "default_task_type": "memory_system",
        "risk_tolerance": "medium",
    },
    "cost_guardian": {
        "description": "Optimizes routing and budget decisions.",
        "default_task_type": "cost_control",
        "risk_tolerance": "low",
    },
}

# Relative cost per output token (normalized, deepseek_flash=1.0)
PROFILE_RELATIVE_COST: Dict[str, float] = {
    "local_light": 0.0,
    "local_coder": 0.0,
    "cheap_cloud_reasoning": 1.0,
    "mini_execution": 2.1,
    "endpoint_implementation": 2.1,
    "risk_reviewer": 1.0,
    "strong_cloud_reasoning": 3.1,
    "strong_execution": 3.1,
}


def get_role(name: str) -> Optional[Dict[str, Any]]:
    return ROLES.get(name)


def get_default_task_type(role_name: str) -> str:
    role = ROLES.get(role_name, {})
    return str(role.get("default_task_type", "code_reasoning"))


def list_roles() -> Dict[str, Dict[str, Any]]:
    return dict(ROLES)
