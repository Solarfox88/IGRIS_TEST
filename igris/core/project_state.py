"""Project state tracking with family saturation cooldown.

Tracks per-family execution metrics, saturation cooldowns, recovery
escalation patterns, and recent task fingerprints. Integrates with
decision_memory without replacing it — adds cooldown_until, family
metrics, and recovery state on top.

Inspired by IGRIS_DEVIN ProjectState/FamilySaturationState/RecoveryPattern.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from igris.core import decision_memory
from igris.core.safety import redact_secrets
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class FamilyMetrics:
    """Per-family execution metrics."""
    family: str
    total_attempts: int = 0
    successes: int = 0
    failures: int = 0
    last_attempt_ts: float = 0.0
    last_success_ts: float = 0.0
    last_failure_ts: float = 0.0
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    recovery_level: int = 0  # 0=normal, 1=caution, 2=elevated, 3=critical

    @property
    def failure_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.failures / self.total_attempts

    @property
    def is_cooling_down(self) -> bool:
        return self.cooldown_until > time.time()

    @property
    def cooldown_remaining(self) -> float:
        remaining = self.cooldown_until - time.time()
        return max(0.0, remaining)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "total_attempts": self.total_attempts,
            "successes": self.successes,
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 3),
            "last_attempt_ts": self.last_attempt_ts,
            "last_success_ts": self.last_success_ts,
            "last_failure_ts": self.last_failure_ts,
            "cooldown_until": self.cooldown_until,
            "is_cooling_down": self.is_cooling_down,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
            "consecutive_failures": self.consecutive_failures,
            "recovery_level": self.recovery_level,
            "recovery_label": _recovery_label(self.recovery_level),
        }


@dataclass
class ProjectState:
    """Aggregate project execution state."""
    family_metrics: Dict[str, FamilyMetrics] = field(default_factory=dict)
    recent_fingerprints: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "families": {k: v.to_dict() for k, v in self.family_metrics.items()},
            "recent_fingerprints": self.recent_fingerprints[-20:],
            "last_updated": self.last_updated,
            "cooling_down": [k for k, v in self.family_metrics.items() if v.is_cooling_down],
            "critical_families": [k for k, v in self.family_metrics.items() if v.recovery_level >= 3],
            "elevated_families": [k for k, v in self.family_metrics.items() if v.recovery_level >= 2],
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cooldown durations per recovery level (seconds)
COOLDOWN_DURATIONS = {
    0: 0,        # normal: no cooldown
    1: 60,       # caution: 1 minute
    2: 300,      # elevated: 5 minutes
    3: 900,      # critical: 15 minutes
}

MAX_FINGERPRINTS = 50
RECOVERY_THRESHOLD_CAUTION = 2      # consecutive failures
RECOVERY_THRESHOLD_ELEVATED = 4
RECOVERY_THRESHOLD_CRITICAL = 6


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _state_path(project_root: Optional[str] = None) -> Path:
    root = Path(project_root) if project_root else CONFIG.project_root
    d = root / ".igris" / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d / "project_state.json"


def _load_state(project_root: Optional[str] = None) -> ProjectState:
    path = _state_path(project_root)
    if not path.exists():
        return ProjectState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = ProjectState()
        state.last_updated = data.get("last_updated", time.time())
        state.recent_fingerprints = data.get("recent_fingerprints", [])[-MAX_FINGERPRINTS:]
        for fam, m in data.get("families", {}).items():
            state.family_metrics[fam] = FamilyMetrics(
                family=fam,
                total_attempts=m.get("total_attempts", 0),
                successes=m.get("successes", 0),
                failures=m.get("failures", 0),
                last_attempt_ts=m.get("last_attempt_ts", 0),
                last_success_ts=m.get("last_success_ts", 0),
                last_failure_ts=m.get("last_failure_ts", 0),
                cooldown_until=m.get("cooldown_until", 0),
                consecutive_failures=m.get("consecutive_failures", 0),
                recovery_level=m.get("recovery_level", 0),
            )
        return state
    except (json.JSONDecodeError, KeyError, TypeError):
        return ProjectState()


def _save_state(state: ProjectState, project_root: Optional[str] = None) -> None:
    path = _state_path(project_root)
    data = {
        "families": {},
        "recent_fingerprints": state.recent_fingerprints[-MAX_FINGERPRINTS:],
        "last_updated": state.last_updated,
    }
    for fam, m in state.family_metrics.items():
        data["families"][fam] = {
            "total_attempts": m.total_attempts,
            "successes": m.successes,
            "failures": m.failures,
            "last_attempt_ts": m.last_attempt_ts,
            "last_success_ts": m.last_success_ts,
            "last_failure_ts": m.last_failure_ts,
            "cooldown_until": m.cooldown_until,
            "consecutive_failures": m.consecutive_failures,
            "recovery_level": m.recovery_level,
        }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_attempt(
    family: str,
    success: bool,
    fingerprint: Optional[str] = None,
    project_root: Optional[str] = None,
) -> FamilyMetrics:
    """Record a task attempt and update family metrics + cooldowns."""
    state = _load_state(project_root)
    now = time.time()

    if family not in state.family_metrics:
        state.family_metrics[family] = FamilyMetrics(family=family)

    m = state.family_metrics[family]
    m.total_attempts += 1
    m.last_attempt_ts = now

    if success:
        m.successes += 1
        m.last_success_ts = now
        m.consecutive_failures = max(0, m.consecutive_failures - 1)
        # Recovery: success lowers recovery level
        if m.recovery_level > 0:
            m.recovery_level = max(0, m.recovery_level - 1)
    else:
        m.failures += 1
        m.last_failure_ts = now
        m.consecutive_failures += 1
        # Escalate recovery level
        if m.consecutive_failures >= RECOVERY_THRESHOLD_CRITICAL:
            m.recovery_level = 3
        elif m.consecutive_failures >= RECOVERY_THRESHOLD_ELEVATED:
            m.recovery_level = 2
        elif m.consecutive_failures >= RECOVERY_THRESHOLD_CAUTION:
            m.recovery_level = 1
        # Apply cooldown
        cooldown = COOLDOWN_DURATIONS.get(m.recovery_level, 0)
        if cooldown > 0:
            m.cooldown_until = now + cooldown

    # Track fingerprint
    if fingerprint:
        state.recent_fingerprints.append(fingerprint)
        state.recent_fingerprints = state.recent_fingerprints[-MAX_FINGERPRINTS:]

    state.last_updated = now
    _save_state(state, project_root)
    return m


def get_family_metrics(
    family: str,
    project_root: Optional[str] = None,
) -> Optional[FamilyMetrics]:
    """Get metrics for a specific family."""
    state = _load_state(project_root)
    return state.family_metrics.get(family)


def get_project_state(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Get full project state as dict."""
    state = _load_state(project_root)
    return state.to_dict()


def get_cooling_down_families(project_root: Optional[str] = None) -> List[str]:
    """Return list of families currently in cooldown."""
    state = _load_state(project_root)
    return [fam for fam, m in state.family_metrics.items() if m.is_cooling_down]


def is_family_available(family: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Check if a family is available (not cooling down, not saturated)."""
    state = _load_state(project_root)
    m = state.family_metrics.get(family)
    saturated = decision_memory.get_saturated_families(project_root=project_root)

    if family in saturated:
        return {
            "available": False,
            "family": family,
            "reason": "Family is saturated in decision memory",
        }

    if m and m.is_cooling_down:
        return {
            "available": False,
            "family": family,
            "reason": f"Family is cooling down ({round(m.cooldown_remaining)}s remaining, recovery level {m.recovery_level})",
            "cooldown_remaining": round(m.cooldown_remaining, 1),
            "recovery_level": m.recovery_level,
        }

    return {
        "available": True,
        "family": family,
        "reason": "Family is available",
        "metrics": m.to_dict() if m else None,
    }


def reset_family_cooldown(family: str, project_root: Optional[str] = None) -> bool:
    """Manually reset cooldown for a family."""
    state = _load_state(project_root)
    m = state.family_metrics.get(family)
    if not m:
        return False
    m.cooldown_until = 0
    m.recovery_level = max(0, m.recovery_level - 1)
    state.last_updated = time.time()
    _save_state(state, project_root)
    return True


def get_recent_fingerprints(limit: int = 20, project_root: Optional[str] = None) -> List[str]:
    """Return recent task fingerprints."""
    state = _load_state(project_root)
    return state.recent_fingerprints[-limit:]


def has_recent_fingerprint(fingerprint: str, project_root: Optional[str] = None) -> bool:
    """Check if a fingerprint was recently seen."""
    state = _load_state(project_root)
    return fingerprint in state.recent_fingerprints


def get_recovery_summary(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Get recovery escalation summary integrated with decision memory."""
    state = _load_state(project_root)
    saturated = decision_memory.get_saturated_families(project_root=project_root)
    constraints = decision_memory.explain_memory_constraints(project_root=project_root)

    families_status: Dict[str, Dict[str, Any]] = {}
    for fam, m in state.family_metrics.items():
        families_status[fam] = {
            "recovery_level": m.recovery_level,
            "recovery_label": _recovery_label(m.recovery_level),
            "cooling_down": m.is_cooling_down,
            "cooldown_remaining": round(m.cooldown_remaining, 1),
            "consecutive_failures": m.consecutive_failures,
            "failure_rate": round(m.failure_rate, 3),
            "saturated": fam in saturated,
        }

    return {
        "families": families_status,
        "cooling_down": [fam for fam, m in state.family_metrics.items() if m.is_cooling_down],
        "critical": [fam for fam, m in state.family_metrics.items() if m.recovery_level >= 3],
        "elevated": [fam for fam, m in state.family_metrics.items() if m.recovery_level >= 2],
        "saturated_from_memory": saturated,
        "avoid_families": constraints["avoid_families"],
        "memory_constraints": constraints,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recovery_label(level: int) -> str:
    labels = {0: "normal", 1: "caution", 2: "elevated", 3: "critical"}
    return labels.get(level, "unknown")
