"""Failure memory for IGRIS SelfRepairSupervisor.

Records structured failure patterns from blocked/failed runs and provides
similarity-based risk scoring for new missions.  All data is persisted in
.igris/failure_patterns.json.  The module is intentionally simple: keyword
overlap is enough to surface relevant history without requiring embeddings.

Public API
----------
FailureMemory.record(goal, failure_class, capability_signals, repair_cycles)
    Persist a new failure entry (call after any blocked/failed run).

FailureMemory.check(goal) -> FailureRisk
    Return a FailureRisk summary for a new mission goal.

FailureRisk
    .risk_level: "low" | "medium" | "high"
    .similar_count: int
    .dominant_failure: str
    .notes: list[str]
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_STORE = Path(".igris/failure_patterns.json")

# Minimum keyword overlap (intersection / union) to call two goals "similar".
_SIMILARITY_THRESHOLD = 0.20

# Words filtered from keyword extraction (too common to be meaningful).
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "with",
    "add", "new", "fix", "get", "set", "run", "use", "on", "at",
    "is", "it", "be", "as", "by", "from", "that", "this", "so",
    "we", "do", "no", "if", "not", "but",
})

_MAX_PATTERNS = 200  # rolling cap to keep the file small


def _keywords(text: str) -> frozenset:
    tokens = re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass
class FailureRisk:
    risk_level: str = "low"        # "low" | "medium" | "high"
    similar_count: int = 0
    dominant_failure: str = ""
    notes: List[str] = field(default_factory=list)


class FailureMemory:
    """Persistent failure pattern store."""

    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        self._path = store_path
        self._patterns: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        goal: str,
        failure_class: str,
        capability_signals: Optional[Dict[str, int]] = None,
        repair_cycles: int = 0,
    ) -> None:
        """Persist a failure pattern from a blocked/failed run."""
        entry: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "goal": goal[:500],
            "keywords": sorted(_keywords(goal)),
            "failure_class": failure_class,
            "capability_signals": dict(capability_signals or {}),
            "repair_cycles": repair_cycles,
        }
        self._patterns.append(entry)
        # Keep rolling window
        if len(self._patterns) > _MAX_PATTERNS:
            self._patterns = self._patterns[-_MAX_PATTERNS:]
        self._save()

    def check(self, goal: str) -> FailureRisk:
        """Return a risk assessment based on past failures similar to goal."""
        goal_kw = _keywords(goal)
        matches: List[Dict[str, Any]] = []
        for p in self._patterns:
            past_kw = frozenset(p.get("keywords") or [])
            if _jaccard(goal_kw, past_kw) >= _SIMILARITY_THRESHOLD:
                matches.append(p)

        if not matches:
            return FailureRisk(risk_level="low")

        # Count failure classes among matches
        class_counts: Dict[str, int] = {}
        for m in matches:
            fc = m.get("failure_class", "unknown")
            class_counts[fc] = class_counts.get(fc, 0) + 1
        dominant = max(class_counts, key=class_counts.__getitem__)
        count = len(matches)

        if count >= 3:
            risk_level = "high"
        elif count >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"

        notes: List[str] = [
            f"Found {count} similar past failure(s) (dominant: {dominant}).",
        ]
        # Surface capability signals if present
        all_signals: Dict[str, int] = {}
        for m in matches:
            for sig, cnt in (m.get("capability_signals") or {}).items():
                all_signals[sig] = all_signals.get(sig, 0) + cnt
        if all_signals:
            notes.append(f"Accumulated capability signals from history: {all_signals}")

        return FailureRisk(
            risk_level=risk_level,
            similar_count=count,
            dominant_failure=dominant,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._patterns = list(data.get("patterns", []))
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            self._patterns = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"patterns": self._patterns}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError:
            pass  # non-fatal: memory is advisory, never blocks a run
