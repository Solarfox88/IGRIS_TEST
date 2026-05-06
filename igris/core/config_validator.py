"""Configuration validation for IGRIS_GPT.

Validates:
- ``.env`` presence and expected keys (without reading secret values)
- ``config/config.sample.json`` schema
- Provider configuration (local LLM, fallback, Vast.ai)
- Budget / cost limits
- Safety policy settings

Returns structured validation results that feed into doctor/verify.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ConfigIssue:
    """A single configuration issue."""
    field: str
    severity: str  # info | warning | error
    message: str
    fix_suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "field": self.field,
            "severity": self.severity,
            "message": redact_secrets(self.message),
        }
        if self.fix_suggestion:
            d["fix_suggestion"] = self.fix_suggestion
        return d


@dataclass
class ConfigValidationResult:
    """Full config validation report."""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    valid: bool = True
    issues: List[ConfigIssue] = field(default_factory=list)
    validated_sections: List[str] = field(default_factory=list)

    def add_issue(self, issue: ConfigIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "valid": self.valid,
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "validated_sections": self.validated_sections,
            "has_errors": any(i.severity == "error" for i in self.issues),
            "has_warnings": any(i.severity == "warning" for i in self.issues),
        }


# ---------------------------------------------------------------------------
# .env validation (existence and expected keys, NOT values)
# ---------------------------------------------------------------------------

# Keys that .env.example defines — we check env vars exist, not file content
EXPECTED_ENV_KEYS: List[str] = [
    "LOCAL_LLM_PROVIDER",
    "LOCAL_LLM_MODEL",
    "LOCAL_LLM_BASE_URL",
]

OPTIONAL_ENV_KEYS: List[str] = [
    "OPENAI_API_KEY",
    "FALLBACK_LLM_PROVIDER",
    "FALLBACK_LLM_MODEL",
    "VASTAI_API_KEY",
    "VASTAI_MODEL",
    "WORKSPACE_ROOT",
    "PROJECT_ROOT",
]


def validate_env(project_root: Optional[str] = None) -> List[ConfigIssue]:
    """Check env configuration. Never reads .env file contents."""
    issues: List[ConfigIssue] = []
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))

    # .env file existence
    env_path = root / ".env"
    example_path = root / ".env.example"
    if not env_path.exists() and not any(os.environ.get(k) for k in EXPECTED_ENV_KEYS):
        sev = "warning"
        msg = ".env not found and no env vars set for local LLM config"
        fix = "Copy .env.example to .env: cp .env.example .env"
        if example_path.exists():
            msg += " (.env.example available)"
        issues.append(ConfigIssue(field=".env", severity=sev, message=msg, fix_suggestion=fix))

    # Check expected env vars are set (from environment, not from file)
    for key in EXPECTED_ENV_KEYS:
        if not os.environ.get(key):
            issues.append(ConfigIssue(
                field=key, severity="info",
                message=f"{key} not set — default will be used",
            ))

    return issues


# ---------------------------------------------------------------------------
# config.json validation
# ---------------------------------------------------------------------------

EXPECTED_CONFIG_KEYS = [
    "local_llm_provider",
    "local_llm_model",
    "local_llm_base_url",
]


def validate_config_json(project_root: Optional[str] = None) -> List[ConfigIssue]:
    """Validate config/config.sample.json structure."""
    issues: List[ConfigIssue] = []
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    sample = root / "config" / "config.sample.json"

    if not sample.exists():
        issues.append(ConfigIssue(
            field="config.sample.json", severity="warning",
            message="config/config.sample.json not found",
        ))
        return issues

    try:
        data = json.loads(sample.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(ConfigIssue(
            field="config.sample.json", severity="error",
            message=f"Invalid JSON: {exc}",
            fix_suggestion="Fix JSON syntax in config/config.sample.json",
        ))
        return issues

    if not isinstance(data, dict):
        issues.append(ConfigIssue(
            field="config.sample.json", severity="error",
            message="Config must be a JSON object (dict)",
        ))
        return issues

    for key in EXPECTED_CONFIG_KEYS:
        if key not in data:
            issues.append(ConfigIssue(
                field=f"config.{key}", severity="info",
                message=f"Key '{key}' not in config sample (may use env var instead)",
            ))

    return issues


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------

VALID_PROVIDERS = {"ollama", "openai", "deterministic"}


def validate_provider() -> List[ConfigIssue]:
    """Validate LLM provider configuration."""
    issues: List[ConfigIssue] = []

    local_provider = os.environ.get("LOCAL_LLM_PROVIDER", "ollama")
    if local_provider not in VALID_PROVIDERS:
        issues.append(ConfigIssue(
            field="LOCAL_LLM_PROVIDER", severity="warning",
            message=f"Unknown local provider: '{local_provider}' (expected: {VALID_PROVIDERS})",
        ))

    fallback_provider = os.environ.get("FALLBACK_LLM_PROVIDER", "openai")
    if fallback_provider not in VALID_PROVIDERS:
        issues.append(ConfigIssue(
            field="FALLBACK_LLM_PROVIDER", severity="warning",
            message=f"Unknown fallback provider: '{fallback_provider}' (expected: {VALID_PROVIDERS})",
        ))

    # If OpenAI is configured as fallback, key should be present
    if fallback_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        issues.append(ConfigIssue(
            field="OPENAI_API_KEY", severity="info",
            message="Fallback provider is 'openai' but no API key set — deterministic fallback will be used",
        ))

    return issues


# ---------------------------------------------------------------------------
# Budget / cost validation
# ---------------------------------------------------------------------------

def validate_budget() -> List[ConfigIssue]:
    """Validate budget/cost configuration."""
    issues: List[ConfigIssue] = []

    max_cost_str = os.environ.get("VASTAI_MAX_HOURLY_COST", "0.50")
    try:
        max_cost = float(max_cost_str)
        if max_cost <= 0:
            issues.append(ConfigIssue(
                field="VASTAI_MAX_HOURLY_COST", severity="warning",
                message="Max hourly cost is <= 0 — Vast.ai provisioning will always be rejected",
            ))
        elif max_cost > 5.0:
            issues.append(ConfigIssue(
                field="VASTAI_MAX_HOURLY_COST", severity="warning",
                message=f"Max hourly cost is ${max_cost:.2f} — consider a lower budget limit",
            ))
    except ValueError:
        issues.append(ConfigIssue(
            field="VASTAI_MAX_HOURLY_COST", severity="error",
            message=f"Invalid cost value: '{max_cost_str}' — must be a number",
            fix_suggestion="Set VASTAI_MAX_HOURLY_COST to a decimal like 0.50",
        ))

    return issues


# ---------------------------------------------------------------------------
# Safety policy validation
# ---------------------------------------------------------------------------

def validate_safety_policy() -> List[ConfigIssue]:
    """Validate safety policy settings."""
    issues: List[ConfigIssue] = []

    auto_commit = os.environ.get("AUTO_COMMIT", "false").lower()
    auto_push = os.environ.get("AUTO_PUSH", "false").lower()

    if auto_commit == "true":
        issues.append(ConfigIssue(
            field="AUTO_COMMIT", severity="warning",
            message="AUTO_COMMIT is enabled — commits will happen without manual review",
        ))

    if auto_push == "true":
        issues.append(ConfigIssue(
            field="AUTO_PUSH", severity="error",
            message="AUTO_PUSH is enabled — this violates safety policy (no auto-push to remotes)",
            fix_suggestion="Set AUTO_PUSH=false in .env",
        ))

    vastai_auto = os.environ.get("VASTAI_AUTO_PROVISION", "false").lower()
    if vastai_auto == "true":
        issues.append(ConfigIssue(
            field="VASTAI_AUTO_PROVISION", severity="warning",
            message="VASTAI_AUTO_PROVISION is enabled — GPU instances may be created automatically",
            fix_suggestion="Set VASTAI_AUTO_PROVISION=false for safety",
        ))

    vastai_approval = os.environ.get("VASTAI_REQUIRE_APPROVAL", "true").lower()
    if vastai_approval == "false":
        issues.append(ConfigIssue(
            field="VASTAI_REQUIRE_APPROVAL", severity="warning",
            message="VASTAI_REQUIRE_APPROVAL is disabled — Vast.ai actions can execute without approval",
        ))

    return issues


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

def validate_all(project_root: Optional[str] = None) -> ConfigValidationResult:
    """Run all configuration validations."""
    result = ConfigValidationResult()

    sections = [
        ("env", validate_env, {"project_root": project_root}),
        ("config_json", validate_config_json, {"project_root": project_root}),
        ("provider", validate_provider, {}),
        ("budget", validate_budget, {}),
        ("safety_policy", validate_safety_policy, {}),
    ]

    for name, func, kwargs in sections:
        try:
            issues = func(**kwargs)
            for issue in issues:
                result.add_issue(issue)
            result.validated_sections.append(name)
        except Exception as exc:
            result.add_issue(ConfigIssue(
                field=name, severity="error",
                message=f"Validation failed with error: {exc}",
            ))

    return result
