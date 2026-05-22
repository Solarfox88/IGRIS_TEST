"""Background memory quality control: decay, stale detection, contradiction marking."""

from __future__ import annotations

import logging
from igris.core.memory_graph import MemoryGraph

logger = logging.getLogger(__name__)


class MemoryValidator:
    def __init__(self, project_root: str) -> None:
        self.project_root = project_root

    def run(self, *, half_life_days: float = 14.0, max_age_days: float = 30.0) -> dict:
        """Run all quality control passes. Returns summary dict."""
        mg = MemoryGraph(self.project_root)
        decayed = mg.decay_confidence(max_age_days=max_age_days, half_life_days=half_life_days)
        stale = mg.deprecate_stale_lessons(self.project_root)
        contradictions = mg.detect_and_mark_contradictions()
        logger.info(
            "MemoryValidator: decayed=%d stale=%d contradictions=%d",
            decayed,
            stale,
            contradictions,
        )
        return {"decayed": decayed, "stale": stale, "contradictions": contradictions}
