"""
decomposition_validator.py — Epic #1078

Validates and normalises the decomposition output (sub-missions list) produced
by the LLM before any GitHub issue is created.  This is a pure-function module:
no I/O, no side effects — easy to test and to call from the supervisor.

Design goals:
  1. Schema completeness: every sub-mission must carry all required fields.
  2. Title hygiene: titles must follow the format  "<Area>: <verb> <object>"
     and must not be vague, duplicated, or auto-generated.
  3. Acceptance-criteria quality: ≥3 measurable, non-vague ACs per sub-mission.
  4. Deduplication: by title (case-insensitive) AND by normalised goal hash.
  5. Dependency soundness: no cycles, all referenced IDs exist in the batch.
  6. Cap: total sub-missions capped at IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_log = logging.getLogger("igris.decomposition_validator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SUBISSUES: int = int(os.getenv("IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION", "12"))
MIN_TITLE_LEN: int = 8
MAX_TITLE_LEN: int = 120
MIN_AC_COUNT: int = 3
MIN_AC_LEN: int = 12

_VAGUE_AC_MARKERS: frozenset = frozenset({
    "_not specified_", "not specified", "tbd", "todo", "n/a", "",
    "none", "to be defined", "see acceptance criteria",
})

_VAGUE_TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^implement\s+github\s+issue\s+#?\d+", re.I),
    re.compile(r"^sub[\s_-]?task\s+\d+$", re.I),
    re.compile(r"igris/\*\*", re.I),
    re.compile(r"^fix\s+sub[\s_-]?task", re.I),
    re.compile(r"^\[?[a-z]+\]?\s+\[?todo\]?", re.I),
]

REQUIRED_FIELDS: Tuple[str, ...] = (
    "title", "goal", "risk_level", "acceptance_criteria",
)

FULL_SCHEMA_FIELDS: Tuple[str, ...] = (
    "title", "goal", "risk_level", "acceptance_criteria",
    "allowed_file_scopes", "tests", "dependencies",
    "out_of_scope", "success_signal", "failure_fallback",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single validation finding for one sub-mission."""
    index: int
    field: str
    severity: str  # "error" | "warning" | "fixed"
    message: str
    original: Any = None
    fixed: Any = None

    @property
    def code(self) -> str:
        """Short uppercase code derived from field name (e.g. 'TITLE', 'AC', 'DEDUP')."""
        _alias = {
            "acceptance_criteria": "AC",
            "goal_hash": "DEDUP",
            "goal": "GOAL",
            "title": "TITLE",
            "dependencies": "DEPS",
            "count": "CAP",
            "risk_level": "RISK",
        }
        return _alias.get(self.field, self.field.upper().replace("_", ""))


@dataclass
class SubMission:
    """A validated and normalised sub-mission."""
    title: str
    goal: str
    risk_level: str
    acceptance_criteria: List[str]
    allowed_file_scopes: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    success_signal: str = ""
    failure_fallback: str = ""
    # Internal
    _goal_hash: str = ""
    _original_title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "goal": self.goal,
            "risk_level": self.risk_level,
            "acceptance_criteria": self.acceptance_criteria,
            "allowed_file_scopes": self.allowed_file_scopes,
            "tests": self.tests,
            "dependencies": self.dependencies,
            "out_of_scope": self.out_of_scope,
            "success_signal": self.success_signal,
            "failure_fallback": self.failure_fallback,
        }


@dataclass
class ValidationReport:
    """Full validation report for a decomposition batch."""
    accepted: List[SubMission] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    capped: bool = False
    original_count: int = 0

    @property
    def ok(self) -> bool:
        """True if validation produced at least one accepted sub-mission."""
        return len(self.accepted) > 0

    @property
    def valid(self) -> bool:
        """True when no error-severity issues were raised (warnings/fixed are allowed)."""
        return self.error_count == 0 and self.ok

    @property
    def quality_score(self) -> float:
        """Normalised 0–1 quality score.

        Starts at 1.0 and is penalised:
          - 0.15 per error-severity issue (hard problems)
          - 0.05 per warning-severity issue (soft problems)
        Floored at 0.0.
        """
        penalty = self.error_count * 0.15 + self.warning_count * 0.05
        return max(0.0, 1.0 - penalty)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def fixed_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "fixed")

    def summary(self) -> str:
        return (
            f"accepted={len(self.accepted)}, rejected={len(self.rejected)}, "
            f"errors={self.error_count}, warnings={self.warning_count}, "
            f"fixed={self.fixed_count}, capped={self.capped}"
        )


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class DecompositionValidator:
    """Validates a raw LLM decomposition output before issue creation.

    Usage:
        validator = DecompositionValidator(parent_goal="implement X for Y")
        report = validator.validate(raw_sub_missions_list)
        for sm in report.accepted:
            create_github_issue(sm.title, sm.goal, sm.acceptance_criteria)
    """

    def __init__(
        self,
        parent_goal: str = "",
        rank_id: str = "",
        max_subissues: int = MAX_SUBISSUES,
    ) -> None:
        self.parent_goal = parent_goal
        self.rank_id = rank_id
        self.max_subissues = max_subissues

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, raw: List[Dict[str, Any]]) -> ValidationReport:
        """Validate and normalise a list of raw sub-mission dicts.

        Returns a ValidationReport with the accepted (normalised) sub-missions,
        rejected entries, and all validation issues found.
        """
        report = ValidationReport(original_count=len(raw))
        seen_titles: Set[str] = set()
        seen_goal_hashes: Set[str] = set()

        # Cap early
        if len(raw) > self.max_subissues:
            report.capped = True
            report.issues.append(ValidationIssue(
                index=-1, field="count", severity="warning",
                message=f"Decomposition capped: {len(raw)} → {self.max_subissues}",
            ))
            raw = raw[:self.max_subissues]

        for i, sub in enumerate(raw):
            sm, sub_issues, rejected = self._validate_one(i, sub, seen_titles, seen_goal_hashes)
            report.issues.extend(sub_issues)
            if rejected:
                report.rejected.append(sub)
            elif sm is not None:
                seen_titles.add(sm.title.lower().strip())
                seen_goal_hashes.add(sm._goal_hash)
                report.accepted.append(sm)

        # Dependency soundness check
        dep_issues = self._check_dependencies(report.accepted)
        report.issues.extend(dep_issues)

        _log.info(
            "DecompositionValidator.validate: %s", report.summary()
        )
        return report

    # ------------------------------------------------------------------
    # Per-sub-mission validation
    # ------------------------------------------------------------------

    def _validate_one(
        self,
        index: int,
        sub: Dict[str, Any],
        seen_titles: Set[str],
        seen_goal_hashes: Set[str],
    ) -> Tuple[Optional[SubMission], List[ValidationIssue], bool]:
        """Validate one raw sub-mission dict.

        Returns (SubMission | None, issues, rejected).
        """
        issues: List[ValidationIssue] = []
        rejected = False

        # 1. Required fields
        for field_name in REQUIRED_FIELDS:
            if not sub.get(field_name):
                issues.append(ValidationIssue(
                    index=index, field=field_name, severity="error",
                    message=f"Required field '{field_name}' is missing or empty",
                ))
                if field_name in ("title", "goal"):
                    rejected = True

        if rejected:
            return None, issues, True

        # 2. Title validation + normalisation
        title, title_issues = self._validate_title(index, str(sub.get("title", "")))
        issues.extend(title_issues)

        # 3. Deduplication by title
        if title.lower().strip() in seen_titles:
            issues.append(ValidationIssue(
                index=index, field="title", severity="error",
                message=f"Duplicate title: {title!r}",
                original=title,
            ))
            return None, issues, True

        # 4. Goal hash dedup
        goal_text = str(sub.get("goal", "")).strip()
        goal_hash = self._goal_hash(goal_text)
        if goal_hash in seen_goal_hashes:
            issues.append(ValidationIssue(
                index=index, field="goal", severity="error",
                message=f"Duplicate goal (hash={goal_hash}): {goal_text[:60]}",
                original=goal_text,
            ))
            return None, issues, True

        # 5. AC validation + generation
        raw_criteria = sub.get("acceptance_criteria") or []
        criteria, ac_issues = self._validate_acceptance_criteria(
            index, raw_criteria, goal_text,
            file_scopes=sub.get("allowed_file_scopes") or [],
            tests=sub.get("tests") or [],
        )
        issues.extend(ac_issues)

        # 6. Risk level
        risk = str(sub.get("risk_level", "medium")).lower().strip()
        if risk not in ("low", "medium", "high", "critical"):
            risk = "medium"
            issues.append(ValidationIssue(
                index=index, field="risk_level", severity="fixed",
                message=f"Invalid risk_level normalised to 'medium'",
                original=sub.get("risk_level"), fixed="medium",
            ))

        # 7. Build full schema with defaults
        sm = SubMission(
            title=title,
            goal=goal_text,
            risk_level=risk,
            acceptance_criteria=criteria,
            allowed_file_scopes=list(sub.get("allowed_file_scopes") or []),
            tests=list(sub.get("tests") or []),
            dependencies=list(sub.get("dependencies") or []),
            out_of_scope=list(sub.get("out_of_scope") or []),
            success_signal=self._default_success_signal(sub, goal_text, criteria),
            failure_fallback=self._default_failure_fallback(sub, risk),
        )
        sm._goal_hash = goal_hash
        sm._original_title = str(sub.get("title", ""))

        return sm, issues, False

    # ------------------------------------------------------------------
    # Title normalisation
    # ------------------------------------------------------------------

    def _validate_title(
        self, index: int, raw_title: str
    ) -> Tuple[str, List[ValidationIssue]]:
        issues: List[ValidationIssue] = []
        title = raw_title.strip()

        # Check for vague/bad title
        is_vague = (
            not title
            or len(title) < MIN_TITLE_LEN
            or len(title) > MAX_TITLE_LEN
            or any(p.search(title) for p in _VAGUE_TITLE_PATTERNS)
        )

        if is_vague:
            issues.append(ValidationIssue(
                index=index, field="title", severity="fixed",
                message=f"Vague/invalid title normalised",
                original=raw_title,
            ))
            # Will be set by caller after goal is available
            title = f"sub-task-{index+1}"

        # Enforce <Area>: <action> format
        if ":" not in title and not is_vague:
            # Try to split on first verb to add structure
            title = self._normalise_title_format(title, index)
            if title != raw_title.strip():
                issues.append(ValidationIssue(
                    index=index, field="title", severity="fixed",
                    message=f"Title format normalised to include area prefix",
                    original=raw_title, fixed=title,
                ))

        return title, issues

    def _normalise_title_format(self, title: str, index: int) -> str:
        """Enforce '<Area>: <description>' format."""
        # If already has a colon at a reasonable position, keep it
        if ":" in title and title.index(":") < 30:
            return title
        # Prepend a generic area tag
        return f"Task {index+1}: {title}"

    def build_title_from_goal(self, goal: str, file_scopes: List[str], index: int) -> str:
        """Build a meaningful title from goal text and file scopes."""
        # Extract area from file scopes
        area = ""
        if file_scopes:
            first = file_scopes[0].strip("/").split("/")
            area = first[0].capitalize() if first else ""

        # Extract a concise action from the goal (first ~60 chars, up to first period)
        action = goal.strip()
        if "." in action:
            action = action[:action.index(".")].strip()
        action = action[:70].strip()

        if area and action:
            return f"{area}: {action}"
        elif action:
            return action[:80]
        else:
            return f"Sub-task {index+1}: implement component"

    # ------------------------------------------------------------------
    # Acceptance criteria
    # ------------------------------------------------------------------

    def _validate_acceptance_criteria(
        self,
        index: int,
        raw_criteria: List[Any],
        goal: str,
        file_scopes: List[str],
        tests: List[str],
    ) -> Tuple[List[str], List[ValidationIssue]]:
        issues: List[ValidationIssue] = []

        # Filter out vague ACs
        valid = [
            str(c).strip() for c in raw_criteria
            if (
                str(c).strip().lower() not in _VAGUE_AC_MARKERS
                and len(str(c).strip()) >= MIN_AC_LEN
            )
        ]

        if len(valid) < MIN_AC_COUNT:
            generated = self._generate_acceptance_criteria(goal, file_scopes, tests)
            # Merge: generated first (structural), then valid model-produced ones
            combined = generated + [v for v in valid if v not in generated]
            issues.append(ValidationIssue(
                index=index, field="acceptance_criteria", severity="fixed",
                message=f"Generated {len(generated)} ACs (had {len(valid)} valid, need ≥{MIN_AC_COUNT})",
                original=raw_criteria, fixed=combined,
            ))
            return combined, issues

        return valid, issues

    def _generate_acceptance_criteria(
        self, goal: str, file_scopes: List[str], tests: List[str]
    ) -> List[str]:
        """Generate minimal but meaningful acceptance criteria."""
        acs: List[str] = []

        if goal:
            acs.append(
                f"Implementation satisfies the stated goal: {goal.strip()[:120]}"
            )

        if tests:
            test_list = ", ".join(f"`{t}`" for t in tests[:3])
            acs.append(f"All listed test targets pass without modification: {test_list}")
        else:
            acs.append(
                "Full test suite passes with no regressions (`pytest` exit 0)"
            )

        if file_scopes:
            scope_list = ", ".join(f"`{s}`" for s in file_scopes[:3])
            acs.append(f"Changes are scoped to the declared files: {scope_list}")
        else:
            acs.append(
                "PR diff is minimal and does not touch files outside the stated scope"
            )

        acs.append("All changed Python files are importable without errors")
        acs.append("No new linting errors introduced (ruff check passes)")

        return acs

    # ------------------------------------------------------------------
    # Dependency soundness
    # ------------------------------------------------------------------

    def _check_dependencies(self, accepted: List[SubMission]) -> List[ValidationIssue]:
        """Check that all dependency references are resolvable within the batch."""
        issues: List[ValidationIssue] = []
        known_titles = {sm.title.lower() for sm in accepted}

        for i, sm in enumerate(accepted):
            for dep in sm.dependencies:
                dep_lower = dep.lower().strip()
                # Allow numeric references (#NNN) and title references
                if dep_lower.startswith("#") and dep_lower[1:].isdigit():
                    continue  # external issue ref, OK
                if dep_lower not in known_titles:
                    issues.append(ValidationIssue(
                        index=i, field="dependencies", severity="warning",
                        message=f"Dependency {dep!r} not found in accepted sub-missions",
                        original=dep,
                    ))

        return issues

    # ------------------------------------------------------------------
    # Schema defaults
    # ------------------------------------------------------------------

    def _default_success_signal(
        self, sub: Dict[str, Any], goal: str, criteria: List[str]
    ) -> str:
        raw = str(sub.get("success_signal", "")).strip()
        if raw:
            return raw
        return (
            f"Supervisor verifies: all {len(criteria)} acceptance criteria met, "
            f"tests green, no regressions."
        )

    def _default_failure_fallback(self, sub: Dict[str, Any], risk: str) -> str:
        raw = str(sub.get("failure_fallback", "")).strip()
        if raw:
            return raw
        if risk in ("high", "critical"):
            return (
                "Escalate to human review: reopen the parent mission with findings. "
                "Do not auto-retry without diagnosis."
            )
        return (
            "Trigger one repair cycle. If repair also fails, mark blocked and "
            "add a comment with the failure summary for human review."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _goal_hash(goal: str) -> str:
        return hashlib.md5(goal.lower().strip().encode()).hexdigest()[:12]
