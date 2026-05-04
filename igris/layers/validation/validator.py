"""
Task completion validation for IGRIS_GPT.

Ensures tasks are not marked completed simply because a command returned 0.
Tasks must satisfy their success_criteria to be considered truly complete.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a task against its success criteria."""
    valid: bool = False
    task_id: Optional[int] = None
    criteria_results: List[Dict[str, Any]] = field(default_factory=list)
    overall_status: str = "needs_review"  # completed | needs_review | blocked
    reason: str = ""
    manual_completion_reason: str = ""
    validated_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    validation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reason"] = redact_secrets(d.get("reason", ""))
        d["manual_completion_reason"] = redact_secrets(
            d.get("manual_completion_reason", "")
        )
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        return cls(
            valid=data.get("valid", False),
            task_id=data.get("task_id"),
            criteria_results=data.get("criteria_results", []),
            overall_status=data.get("overall_status", "needs_review"),
            reason=data.get("reason", ""),
            manual_completion_reason=data.get("manual_completion_reason", ""),
            validated_at=data.get("validated_at", ""),
            validation_id=data.get("validation_id", uuid.uuid4().hex[:12]),
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _validations_dir(project_root: Optional[str] = None) -> Path:
    if project_root:
        d = Path(project_root) / ".igris" / "validations"
    else:
        from igris.models.config import CONFIG
        d = CONFIG.project_root / ".igris" / "validations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_validation(result: ValidationResult, project_root: Optional[str] = None) -> None:
    d = _validations_dir(project_root)
    fp = d / f"{result.validation_id}.json"
    fp.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def get_validation(validation_id: str, project_root: Optional[str] = None) -> Optional[ValidationResult]:
    fp = _validations_dir(project_root) / f"{validation_id}.json"
    if not fp.exists():
        return None
    data = json.loads(fp.read_text(encoding="utf-8"))
    return ValidationResult.from_dict(data)


def get_validations_for_task(task_id: int, project_root: Optional[str] = None) -> List[ValidationResult]:
    results = []
    d = _validations_dir(project_root)
    for fp in sorted(d.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if data.get("task_id") == task_id:
                results.append(ValidationResult.from_dict(data))
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Criterion checking
# ---------------------------------------------------------------------------

def _check_criterion_tests_pass(
    criterion: str, reports: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Check if recent test reports show passing tests."""
    for r in reversed(reports):
        if r.get("command_id") in ("run_tests",) or "test" in r.get("command_id", ""):
            if r.get("success"):
                return {"criterion": criterion, "met": True, "evidence": "Recent test report shows success"}
            else:
                return {"criterion": criterion, "met": False, "evidence": f"Test report shows failure: {redact_secrets(r.get('stderr_truncated', '')[:100])}"}
    return {"criterion": criterion, "met": False, "evidence": "No recent test reports found"}


def _check_criterion_file_exists(
    criterion: str, files_changed: List[str], project_root: Optional[str] = None
) -> Dict[str, Any]:
    """Check if a file mentioned in the criterion exists."""
    import re
    # Try to extract a file path from criterion
    path_match = re.search(r'[\w./]+\.\w+', criterion)
    if path_match:
        target = path_match.group()
        # Check in files_changed
        if any(target in f for f in files_changed):
            return {"criterion": criterion, "met": True, "evidence": f"File {target} found in changed files"}
        # Check on disk
        if project_root:
            full = Path(project_root) / target
            if full.exists():
                return {"criterion": criterion, "met": True, "evidence": f"File {target} exists on disk"}
        return {"criterion": criterion, "met": False, "evidence": f"Expected file {target} not found"}
    return {"criterion": criterion, "met": None, "evidence": "Could not extract file path from criterion"}


def _check_criterion_generic(criterion: str) -> Dict[str, Any]:
    """Generic criterion that requires manual verification."""
    return {"criterion": criterion, "met": None, "evidence": "Requires manual verification"}


def validate_success_criteria(
    criteria: List[str],
    reports: List[Dict[str, Any]],
    files_changed: List[str],
    project_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Validate each success criterion and return results."""
    results = []
    for criterion in criteria:
        cl = criterion.lower()
        if any(kw in cl for kw in ("test", "pytest", "pass", "green")):
            results.append(_check_criterion_tests_pass(criterion, reports))
        elif any(kw in cl for kw in ("file", "create", "exist", "add")):
            results.append(_check_criterion_file_exists(criterion, files_changed, project_root))
        else:
            results.append(_check_criterion_generic(criterion))
    return results


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_task_completion(
    task: Any,
    reports: Optional[List[Dict[str, Any]]] = None,
    files_changed: Optional[List[str]] = None,
    manual_completion_reason: str = "",
    project_root: Optional[str] = None,
) -> ValidationResult:
    """Validate whether a task is truly complete.

    Rules:
    - Tasks with success_criteria are validated against them
    - Tasks without criteria cannot be auto-completed (needs_review)
    - Manual completion requires a reason
    - Failed validation leaves task in needs_review or blocked
    """
    result = ValidationResult(task_id=task.id if hasattr(task, 'id') else None)
    reports = reports or []
    files_changed = files_changed or []

    # Check if task has success criteria
    criteria = getattr(task, 'success_criteria', []) or []
    if not criteria:
        if manual_completion_reason:
            result.valid = True
            result.overall_status = "completed"
            result.manual_completion_reason = manual_completion_reason
            result.reason = "Manually completed with reason provided"
        else:
            result.valid = False
            result.overall_status = "needs_review"
            result.reason = "Task has no success criteria — cannot auto-validate"
        _save_validation(result, project_root)
        return result

    # Validate each criterion
    criteria_results = validate_success_criteria(
        criteria, reports, files_changed, project_root
    )
    result.criteria_results = criteria_results

    # Determine overall result
    met_count = sum(1 for c in criteria_results if c.get("met") is True)
    failed_count = sum(1 for c in criteria_results if c.get("met") is False)
    unknown_count = sum(1 for c in criteria_results if c.get("met") is None)

    if failed_count > 0:
        result.valid = False
        result.overall_status = "blocked"
        failed_criteria = [c["criterion"] for c in criteria_results if c.get("met") is False]
        result.reason = f"{failed_count} criteria failed: {'; '.join(failed_criteria[:3])}"
    elif met_count == len(criteria_results):
        result.valid = True
        result.overall_status = "completed"
        result.reason = f"All {met_count} criteria met"
    elif met_count > 0 and unknown_count > 0:
        result.valid = False
        result.overall_status = "needs_review"
        result.reason = f"{met_count} met, {unknown_count} need manual verification"
    else:
        result.valid = False
        result.overall_status = "needs_review"
        result.reason = f"All {unknown_count} criteria need manual verification"

    # Allow manual override
    if not result.valid and manual_completion_reason:
        result.valid = True
        result.overall_status = "completed"
        result.manual_completion_reason = manual_completion_reason
        result.reason += f" — manually overridden: {manual_completion_reason}"

    _save_validation(result, project_root)
    return result


def map_validation_to_status(validation: ValidationResult) -> str:
    """Map validation result to task status string."""
    status_map = {
        "completed": "completed",
        "needs_review": "pending",
        "blocked": "blocked",
    }
    return status_map.get(validation.overall_status, "pending")
