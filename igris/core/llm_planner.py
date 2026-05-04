"""LLM-based mission planner with safe schema validation.

The LLM produces a JSON plan that is validated against a strict schema.
Invalid or unsafe plans fall back to the deterministic planner.

Modes:
- deterministic: keyword-based, no LLM (default)
- llm: try LLM first, fallback to deterministic on failure
- auto: use LLM if available, otherwise deterministic
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from igris.core.chat_engine import chat
from igris.core.mission_planner import (
    Mission,
    PlanStep,
    generate_plan as deterministic_plan,
    load_mission,
    save_mission,
    _classify_family,
)
from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_STEP_FIELDS = {"title", "description", "family", "success_criteria", "risk"}
VALID_FAMILIES = {
    "analyze", "code", "test", "docs", "config", "refactor", "deploy",
    "review", "debug", "other",
}
VALID_RISKS = {"low", "medium", "high"}
UNSAFE_CAPABILITIES = {
    "shell_exec", "auto_push", "force_push", "delete_repo",
    "auto_merge", "write_env", "write_secrets", "sudo",
}
SAFE_CAPABILITIES = {
    "read", "write", "patch_propose", "patch_validate", "patch_apply",
    "test_run", "lint_run", "analyze", "search", "diff_view",
}


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are a mission planner for IGRIS_GPT. Given a mission title and description, produce a JSON plan.

Output ONLY valid JSON — no markdown, no code fences, no explanation.

Schema:
{
  "steps": [
    {
      "title": "short step title",
      "description": "what this step does",
      "family": "analyze|code|test|docs|config|refactor|deploy|review|debug|other",
      "dependencies": [],
      "success_criteria": ["criterion 1", "criterion 2"],
      "safe_capabilities": ["read", "write", "patch_propose"],
      "risk": "low|medium|high"
    }
  ]
}

Rules:
- Every step MUST have success_criteria (at least 1)
- Every step MUST have risk ("low", "medium", or "high")
- safe_capabilities must only use: read, write, patch_propose, patch_validate, patch_apply, test_run, lint_run, analyze, search, diff_view
- Do NOT include: shell_exec, auto_push, force_push, delete_repo, auto_merge, write_env, write_secrets, sudo
- Steps should be ordered logically with dependencies
- Keep plans concise (3-8 steps typical)
- No secrets or credentials in any field
"""


def _build_plan_prompt(mission: Mission) -> str:
    return (
        f"Mission: {mission.title}\n\n"
        f"Description:\n{mission.description}\n\n"
        "Produce a JSON plan following the schema above."
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class PlanValidationResult:
    def __init__(
        self,
        valid: bool = False,
        steps: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.valid = valid
        self.steps = steps or []
        self.errors = errors or []
        self.warnings = warnings or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "step_count": len(self.steps),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_plan_schema(raw_json: str) -> "PlanValidationResult":
    """Validate LLM output against the plan schema."""
    errors: List[str] = []
    warnings: List[str] = []

    # Parse JSON
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return PlanValidationResult(
            valid=False,
            errors=[f"Invalid JSON: {e}"],
        )

    # Must have steps array
    if not isinstance(data, dict) or "steps" not in data:
        return PlanValidationResult(
            valid=False,
            errors=["Missing 'steps' key in plan"],
        )

    steps = data["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        return PlanValidationResult(
            valid=False,
            errors=["'steps' must be a non-empty array"],
        )

    validated_steps: List[Dict[str, Any]] = []

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step {i}: not a dict")
            continue

        # Check required fields
        missing = REQUIRED_STEP_FIELDS - set(step.keys())
        if missing:
            errors.append(f"Step {i}: missing fields: {sorted(missing)}")
            continue

        # Validate family
        family = step.get("family", "other")
        if family not in VALID_FAMILIES:
            warnings.append(f"Step {i}: unknown family '{family}', defaulting to 'other'")
            step["family"] = "other"

        # Validate risk
        risk = step.get("risk", "low")
        if risk not in VALID_RISKS:
            warnings.append(f"Step {i}: unknown risk '{risk}', defaulting to 'low'")
            step["risk"] = "low"

        # Validate success_criteria
        criteria = step.get("success_criteria", [])
        if not isinstance(criteria, list) or len(criteria) == 0:
            errors.append(f"Step {i}: success_criteria must be non-empty list")
            continue

        # Validate safe_capabilities
        caps = step.get("safe_capabilities", [])
        if isinstance(caps, list):
            unsafe = set(caps) & UNSAFE_CAPABILITIES
            if unsafe:
                errors.append(f"Step {i}: unsafe capabilities rejected: {sorted(unsafe)}")
                continue
            # Filter to known safe capabilities
            step["safe_capabilities"] = [c for c in caps if c in SAFE_CAPABILITIES]

        # Check for secrets in text fields
        text_fields = [
            step.get("title", ""),
            step.get("description", ""),
            *[str(c) for c in criteria],
        ]
        for txt in text_fields:
            redacted = redact_secrets(txt)
            if redacted != txt:
                errors.append(f"Step {i}: secret-like content detected")
                break

        validated_steps.append(step)

    if errors:
        return PlanValidationResult(
            valid=False,
            steps=validated_steps,
            errors=errors,
            warnings=warnings,
        )

    return PlanValidationResult(
        valid=True,
        steps=validated_steps,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Convert validated JSON to PlanSteps
# ---------------------------------------------------------------------------

def _json_to_plan_steps(
    steps_data: List[Dict[str, Any]],
    mission_id: str,
) -> List[PlanStep]:
    """Convert validated JSON steps to PlanStep objects."""
    plan_steps: List[PlanStep] = []

    for i, s in enumerate(steps_data):
        step = PlanStep(
            mission_id=mission_id,
            title=s.get("title", "")[:120],
            description=s.get("description", ""),
            family=s.get("family", "other"),
            dependencies=s.get("dependencies", []),
            success_criteria=s.get("success_criteria", []),
            safe_capabilities=s.get("safe_capabilities", []),
            risk=s.get("risk", "low"),
            order=i,
        )
        # Wire up dependencies to previous step if not specified
        if not step.dependencies and plan_steps:
            step.dependencies = [plan_steps[-1].id]
        plan_steps.append(step)

    return plan_steps


# ---------------------------------------------------------------------------
# LLM planning
# ---------------------------------------------------------------------------

def plan_with_llm(mission: Mission) -> Dict[str, Any]:
    """Try LLM-based planning, return result with metadata.

    Returns:
    {
        "steps": List[PlanStep],
        "mode": "llm" | "deterministic",
        "fallback_used": bool,
        "fallback_reason": str,
        "validation": {...},
        "provider": str,
        "model": str,
        "latency_ms": int,
    }
    """
    prompt = _build_plan_prompt(mission)

    t0 = time.monotonic()
    response = chat(
        message=prompt,
        system_prompt=PLAN_SYSTEM_PROMPT,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    raw_text = response.get("text", "")
    provider = response.get("provider", "unknown")
    model = response.get("model", "unknown")

    # Extract JSON from response (handle markdown fences)
    json_text = _extract_json(raw_text)

    # Validate schema
    validation = validate_plan_schema(json_text)

    if validation.valid:
        steps = _json_to_plan_steps(validation.steps, mission.id)
        return {
            "steps": steps,
            "mode": "llm",
            "fallback_used": False,
            "fallback_reason": "",
            "validation": validation.to_dict(),
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
        }

    # Fallback to deterministic
    det_steps = deterministic_plan(mission)
    return {
        "steps": det_steps,
        "mode": "deterministic",
        "fallback_used": True,
        "fallback_reason": f"LLM plan invalid: {'; '.join(validation.errors[:3])}",
        "validation": validation.to_dict(),
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
    }


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]

    return text


# ---------------------------------------------------------------------------
# Plan mission with mode selection
# ---------------------------------------------------------------------------

def plan_mission_with_mode(
    mission_id: str,
    mode: str = "deterministic",
    project_root: str = ".",
) -> Optional[Dict[str, Any]]:
    """Plan a mission with the specified mode.

    Modes:
    - deterministic: keyword-based, no LLM
    - llm: try LLM, fallback to deterministic
    - auto: use LLM if available, otherwise deterministic
    """
    mission = load_mission(mission_id, project_root)
    if not mission:
        return None

    if mode == "deterministic":
        steps = deterministic_plan(mission)
        result = {
            "steps": steps,
            "mode": "deterministic",
            "fallback_used": False,
            "fallback_reason": "",
            "validation": {"valid": True, "step_count": len(steps), "errors": [], "warnings": []},
            "provider": "deterministic",
            "model": "keyword-based",
            "latency_ms": 0,
        }
    elif mode in ("llm", "auto"):
        result = plan_with_llm(mission)
    else:
        return None

    # Save to mission
    mission.steps = result["steps"]
    mission.status = "planned"
    mission.plan_summary = (
        f"{len(result['steps'])} steps ({result['mode']})"
    )
    mission.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_mission(mission, project_root)

    return {
        "mission": mission.to_dict(),
        "planning": {
            "mode": result["mode"],
            "fallback_used": result["fallback_used"],
            "fallback_reason": redact_secrets(result.get("fallback_reason", "")),
            "validation": result["validation"],
            "provider": result["provider"],
            "model": result["model"],
            "latency_ms": result["latency_ms"],
        },
    }


# ---------------------------------------------------------------------------
# Plan explanation
# ---------------------------------------------------------------------------

def explain_plan(mission_id: str, project_root: str = ".") -> Optional[Dict[str, Any]]:
    """Explain the current plan for a mission."""
    mission = load_mission(mission_id, project_root)
    if not mission:
        return None

    if not mission.steps:
        return {
            "mission_id": mission_id,
            "status": "no plan",
            "steps": [],
            "explanation": "Mission has not been planned yet.",
        }

    steps_info = []
    for step in mission.steps:
        d = step.to_dict()
        steps_info.append({
            "order": d["order"],
            "title": redact_secrets(d["title"]),
            "family": d["family"],
            "risk": d["risk"],
            "success_criteria": d["success_criteria"],
            "safe_capabilities": d["safe_capabilities"],
            "dependencies": d["dependencies"],
            "status": d["status"],
        })

    families = list(set(s["family"] for s in steps_info))
    risks = [s["risk"] for s in steps_info]
    max_risk = "high" if "high" in risks else ("medium" if "medium" in risks else "low")

    return {
        "mission_id": mission_id,
        "title": redact_secrets(mission.title),
        "status": mission.status,
        "plan_summary": mission.plan_summary,
        "step_count": len(steps_info),
        "families": families,
        "max_risk": max_risk,
        "steps": steps_info,
        "explanation": (
            f"Plan has {len(steps_info)} steps across {len(families)} families. "
            f"Maximum risk level: {max_risk}. "
            f"All steps require explicit execution — no auto-execution."
        ),
    }
