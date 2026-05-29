"""ContextSectionWeighter — feedback loop on prompt section utility.

Part of GitHub issue #524: feat(context): Context section weighting.
Fase 2bis — Gap 7.

Records which context sections are "used" (cited or referenced) by the model
in successful steps, then adjusts budget allocation so useful sections get
more tokens and ignored sections get less.

Usage::

    from igris.core.context_section_weighter import ContextSectionWeighter

    weighter = ContextSectionWeighter(project_root)

    # After each reasoning step:
    weighter.record_step(
        step_id="abc123",
        sections_present=["memory_context", "state_context", "recent_actions"],
        model_response="Based on recent_actions I can see the fix is...",
        success=True,
    )

    # Get budget multipliers for next context build:
    multipliers = weighter.get_budget_multipliers()
    # => {"memory_context": 1.2, "state_context": 0.8, ...}
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USAGE_FILE = ".igris/section_usage.json"
_WEIGHT_FILE = ".igris/section_weights.json"
_MIN_SAMPLES = 10          # minimum data points before adjusting weights
_MAX_RECORDS = 500         # keep only the last N records (rolling window)
_MULTIPLIER_MIN = 0.5      # never reduce below 50% of base budget
_MULTIPLIER_MAX = 2.0      # never exceed 200% of base budget

# Sections tracked (must match ContextPacket field names)
TRACKED_SECTIONS = [
    "mission_context",
    "error_context",
    "recent_actions",
    "state_context",
    "memory_context",
    "file_context",
]

# Keywords the model uses when citing each section.
# Proxy: if any keyword appears in the model response, section is "cited".
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "mission_context": ["goal", "mission", "task", "objective"],
    "error_context": ["error", "traceback", "exception", "test fail", "pytest", "AssertionError"],
    "recent_actions": ["recent_action", "previous step", "last action", "already did", "previously"],
    "state_context": ["world_state", "service", "health", "running", "state"],
    "memory_context": ["memory", "lesson", "recall", "according to", "previously learned"],
    "file_context": ["file", "function", "class", "import", "def ", "line "],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepUsageRecord:
    step_id: str
    sections_present: List[str]
    sections_cited: List[str]    # sections referenced in model response
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class SectionStats:
    section: str
    total_steps: int
    cited_in_success: int        # cited AND step was successful
    cited_count: int             # total cited (success or not)
    present_count: int           # section was present in context
    weight: float = 1.0          # computed weight (budget multiplier)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _usage_path(project_root: str) -> Path:
    return Path(project_root) / _USAGE_FILE


def _weight_path(project_root: str) -> Path:
    return Path(project_root) / _WEIGHT_FILE


def load_usage_records(project_root: str) -> List[StepUsageRecord]:
    path = _usage_path(project_root)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            StepUsageRecord(
                step_id=str(r.get("step_id", "")),
                sections_present=list(r.get("sections_present", [])),
                sections_cited=list(r.get("sections_cited", [])),
                success=bool(r.get("success", False)),
                timestamp=float(r.get("timestamp", 0.0)),
            )
            for r in raw
        ]
    except Exception:
        return []


def save_usage_records(project_root: str, records: List[StepUsageRecord]) -> None:
    path = _usage_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "step_id": r.step_id,
            "sections_present": r.sections_present,
            "sections_cited": r.sections_cited,
            "success": r.success,
            "timestamp": r.timestamp,
        }
        for r in records[-_MAX_RECORDS:]
    ]
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, str(path))


def load_section_weights(project_root: str) -> Dict[str, float]:
    path = _weight_path(project_root)
    if not path.exists():
        return {}
    try:
        return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def save_section_weights(project_root: str, weights: Dict[str, float]) -> None:
    path = _weight_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(weights, indent=2), encoding="utf-8")
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Citation detection
# ---------------------------------------------------------------------------

def detect_cited_sections(response_text: str, sections_present: List[str]) -> List[str]:
    """Return which sections are cited/referenced in the model response."""
    response_lower = response_text.lower()
    cited = []
    for section in sections_present:
        keywords = _SECTION_KEYWORDS.get(section, [])
        if any(kw.lower() in response_lower for kw in keywords):
            cited.append(section)
    return cited


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_section_stats(records: List[StepUsageRecord]) -> Dict[str, SectionStats]:
    """Compute per-section statistics from historical records."""
    stats: Dict[str, SectionStats] = {
        s: SectionStats(section=s, total_steps=0, cited_in_success=0,
                        cited_count=0, present_count=0)
        for s in TRACKED_SECTIONS
    }
    for rec in records:
        for s in TRACKED_SECTIONS:
            if s in rec.sections_present:
                stats[s].total_steps += 1
                stats[s].present_count += 1
                if s in rec.sections_cited:
                    stats[s].cited_count += 1
                    if rec.success:
                        stats[s].cited_in_success += 1
    return stats


def compute_weights(
    records: List[StepUsageRecord],
    min_samples: int = _MIN_SAMPLES,
) -> Dict[str, float]:
    """Compute budget multipliers for each section based on correlation with success.

    Algorithm:
        P(success | cited) = cited_in_success / cited_count
        P(success_baseline) = total_successes / total_steps
        weight = P(success | cited) / P(success_baseline)
        Clamped to [MULTIPLIER_MIN, MULTIPLIER_MAX]

    Returns 1.0 for sections with insufficient data.
    """
    if len(records) < min_samples:
        return {s: 1.0 for s in TRACKED_SECTIONS}

    total_success = sum(1 for r in records if r.success)
    total = len(records)
    baseline = total_success / total if total else 0.0

    stats = compute_section_stats(records)
    weights: Dict[str, float] = {}

    for section, st in stats.items():
        if st.cited_count < max(3, min_samples // 3):
            # Not enough data for this section
            weights[section] = 1.0
            continue
        p_success_given_cited = st.cited_in_success / st.cited_count
        if baseline > 0:
            raw = p_success_given_cited / baseline
        else:
            raw = 1.0
        weights[section] = max(_MULTIPLIER_MIN, min(_MULTIPLIER_MAX, raw))

    return weights


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContextSectionWeighter:
    """Records section usage and exposes budget multipliers for ContextManager.

    Thread-safe reads; file writes are atomic (tmp+replace).
    """

    def __init__(self, project_root: str, min_samples: int = _MIN_SAMPLES) -> None:
        self._root = project_root
        self._min_samples = min_samples

    def record_step(
        self,
        step_id: str,
        sections_present: List[str],
        model_response: str,
        success: bool,
    ) -> StepUsageRecord:
        """Record section citation for a completed step and persist to disk."""
        cited = detect_cited_sections(model_response, sections_present)
        record = StepUsageRecord(
            step_id=step_id,
            sections_present=list(sections_present),
            sections_cited=cited,
            success=success,
        )
        records = load_usage_records(self._root)
        records.append(record)
        save_usage_records(self._root, records)
        # Recompute and persist weights every 10 records
        if len(records) % 10 == 0:
            self._recompute_and_save(records)
        return record

    def get_budget_multipliers(self) -> Dict[str, float]:
        """Return current budget multipliers for all tracked sections.

        Falls back to 1.0 for all sections if weights file is missing or stale.
        """
        weights = load_section_weights(self._root)
        if not weights:
            records = load_usage_records(self._root)
            if len(records) >= self._min_samples:
                weights = compute_weights(records, self._min_samples)
                save_section_weights(self._root, weights)
        return {s: weights.get(s, 1.0) for s in TRACKED_SECTIONS}

    def _recompute_and_save(self, records: List[StepUsageRecord]) -> None:
        try:
            weights = compute_weights(records, self._min_samples)
            save_section_weights(self._root, weights)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Return raw statistics for diagnostics."""
        records = load_usage_records(self._root)
        stats = compute_section_stats(records)
        return {
            "total_records": len(records),
            "sections": {
                s: {
                    "present_count": st.present_count,
                    "cited_count": st.cited_count,
                    "cited_in_success": st.cited_in_success,
                    "weight": self.get_budget_multipliers().get(s, 1.0),
                }
                for s, st in stats.items()
            },
        }
