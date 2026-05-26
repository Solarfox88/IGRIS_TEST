"""ToolTracker — per-tool effectiveness stats with persistence.

Inspired by src/openhuman/learning/tool_tracker.rs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolStats:
    """Statistics for a single tool."""

    tool_name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    avg_duration_ms: float = 0.0
    common_error_patterns: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class ToolTracker:
    """Tracks per-tool execution effectiveness.

    Persists stats to .igris/tool_stats.json.
    """

    DEFAULT_MAX_ERROR_PATTERNS = 5

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        self.storage_dir = os.path.join(project_root, ".igris")
        self.storage_path = os.path.join(self.storage_dir, "tool_stats.json")
        self.max_error_patterns = self.DEFAULT_MAX_ERROR_PATTERNS
        self._stats: dict[str, ToolStats] = {}
        self._load()

    # ------------------------------------------------------------------ public

    def record(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        error_snippet: str | None = None,
    ) -> None:
        """Record one tool invocation."""
        stat = self._get_or_create(tool_name)
        stat.total_calls += 1
        if success:
            stat.successes += 1
        else:
            stat.failures += 1
            if error_snippet:
                snippet = error_snippet.strip()[:200]
                if snippet not in stat.common_error_patterns:
                    stat.common_error_patterns.append(snippet)
                    if len(stat.common_error_patterns) > self.max_error_patterns:
                        stat.common_error_patterns = stat.common_error_patterns[-self.max_error_patterns:]

        # running average
        if stat.total_calls == 1:
            stat.avg_duration_ms = duration_ms
        else:
            stat.avg_duration_ms = (
                (stat.avg_duration_ms * (stat.total_calls - 1) + duration_ms)
                / stat.total_calls
            )
        stat.last_updated = time.time()
        self._save()

    def get_stats(self, tool_name: str) -> ToolStats | None:
        """Return stats for *tool_name* or None."""
        return self._stats.get(tool_name)

    def get_all_stats(self) -> dict[str, ToolStats]:
        """Return a shallow copy of all stats."""
        return dict(self._stats)

    def get_unreliable_tools(
        self, min_calls: int = 5, max_success_rate: float = 0.6
    ) -> list[str]:
        """Return tool names with success rate < max_success_rate after ≥ min_calls."""
        unreliable: list[str] = []
        for name, s in self._stats.items():
            if s.total_calls < min_calls:
                continue
            if s.total_calls == 0:
                continue
            rate = s.successes / s.total_calls
            if rate < max_success_rate:
                unreliable.append(name)
        return unreliable

    # ----------------------------------------------------------------- private

    def _get_or_create(self, tool_name: str) -> ToolStats:
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolStats(tool_name=tool_name)
        return self._stats[tool_name]

    def _load(self) -> None:
        """Load stats from disk if present."""
        if not os.path.isfile(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        for name, d in data.items():
            ts = ToolStats(
                tool_name=name,
                total_calls=d.get("total_calls", 0),
                successes=d.get("successes", 0),
                failures=d.get("failures", 0),
                avg_duration_ms=d.get("avg_duration_ms", 0.0),
                common_error_patterns=d.get("common_error_patterns", []),
                last_updated=d.get("last_updated", 0.0),
            )
            self._stats[name] = ts

    def _save(self) -> None:
        """Atomically persist stats to disk."""
        os.makedirs(self.storage_dir, exist_ok=True)
        tmp_path = self.storage_path + ".tmp"
        data = {
            name: {
                "tool_name": s.tool_name,
                "total_calls": s.total_calls,
                "successes": s.successes,
                "failures": s.failures,
                "avg_duration_ms": s.avg_duration_ms,
                "common_error_patterns": s.common_error_patterns,
                "last_updated": s.last_updated,
            }
            for name, s in self._stats.items()
        }
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, self.storage_path)
