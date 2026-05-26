"""
ToolOutputCompactor: rule-driven stdout compaction before LLM injection.
Implements TokenJuice concept: 80% token reduction via configurable rules.
"""

import re
import json
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class CompactorConfig:
    """Configuration for the compactor, can be loaded from .igris/compactor_rules.json."""
    max_chars: int = 8000
    tail_lines: int = 50
    stack_head: int = 10
    stack_tail: int = 5
    collapse_patterns: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.collapse_patterns is None:
            self.collapse_patterns = [
                {"pattern": r"(\\.{3,})", "name": "."},
                {"pattern": r"(\={3,})", "name": "="},
                {"pattern": r"(\-{3,})", "name": "-"},
            ]

    @classmethod
    def from_file(cls, path: str) -> "CompactorConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            max_chars=data.get("max_chars", 8000),
            tail_lines=data.get("tail_lines", 50),
            stack_head=data.get("stack_head", 10),
            stack_tail=data.get("stack_tail", 5),
            collapse_patterns=data.get("collapse_patterns", None),
        )

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)


class ToolOutputCompactor:
    """Compacts tool output using configurable rules."""

    ANSI_ESCAPE = re.compile(r"(\x1b\[[0-9;]*[a-zA-Z]|", re.UNICODE)

    def __init__(self, config: Optional[CompactorConfig] = None):
        self.config = config or CompactorConfig()

    def compress(self, text: str, source_type: str = "generic") -> str:
        """Apply all compaction rules in order. source_type can be 'test_runner' to apply tail-first."""
        if not text:
            return text

        # 1. Strip ANSI escape codes
        text = self._strip_ansi(text)

        # 2. Dedup consecutive identical lines
        text = self._dedup_lines(text)

        # 3. Collapse repeated patterns (like ........ to [Nx '.'])
        text = self._collapse_repeated_patterns(text)

        # 4. Truncate long stack traces
        text = self._truncate_stack_traces(text)

        # 5. Tail-first for test runners
        if source_type == "test_runner":
            text = self._tail_first_test_runner(text)

        # 6. Max length cap (final)
        text = self._hard_truncate(text)

        return text

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes."""
        return self.ANSI_ESCAPE.sub("", text)

    def _dedup_lines(self, text: str) -> str:
        """Dedup consecutive identical lines, but keep first and count."""
        lines = text.splitlines(keepends=True)
        if not lines:
            return text

        compressed = []
        prev = None
        count = 0
        for line in lines:
            stripped = line.rstrip('\n\r')
            if stripped == prev:
                count += 1
            else:
                if count > 1:
                    # Append the dedup marker for the previous run
                    compressed.append(f"[repeated {count} times] {prev}\n")
                    count = 0
                elif count == 1:
                    # Only one occurrence, just append the line as is
                    compressed.append(line)
                    count = 0
                # Reset for new line
                prev = stripped
                count = 1
        # Handle trailing run
        if count > 1:
            compressed.append(f"[repeated {count} times] {prev}\n")
        elif count == 1:
            compressed.append(line)

        return "".join(compressed)

    def _collapse_repeated_patterns(self, text: str) -> str:
        """Collapse repeated patterns like ...... to [Nx '.']"""
        for pattern_def in self.config.collapse_patterns:
            pattern = pattern_def["pattern"]
            name = pattern_def.get("name", pattern_def["pattern"])
            # We need to replace each run of the same character with a marker
            # Example: "........." -> "[9x .]"
            text = re.sub(pattern, lambda m: self._replace_run(m, name), text)
        return text

    def _replace_run(self, match: re.Match, char: str) -> str:
        """Replace a run of identical characters with [Nx char] marker."""
        run = match.group(1)
        if len(run) >= 3:  # only collapse if 3 or more
            return f"[{len(run)}× '{char}']"
        return run

    def _truncate_stack_traces(self, text: str) -> str:
        """Truncate long stack traces: keep first 10 + last 5 lines, elide middle if more than 15 lines total."""
        lines = text.splitlines(keepends=True)
        # Heuristic: find multi-line "Traceback" or "stack trace" sections
        # Simpler: if entire text has many lines and looks like a stack trace (starting with "Traceback" or containing "File "), compress.
        # But better: compress sections delimited by blank lines? For initial implementation, if total lines > 30 and contains "Traceback", apply.
        if len(lines) < 15:
            return text

        # Find traceback sections and compress each.
        segments = re.split(r"(\n\n)", text)
        compressed_segments = []
        for seg in segments:
            seg_lines = seg.splitlines(keepends=True)
            if len(seg_lines) >= 15 and ("Traceback" in seg or "stack trace" in seg.lower()):
                head = seg_lines[: self.config.stack_head]
                tail = seg_lines[-self.config.stack_tail :]
                middle_elision = f"... [{len(seg_lines) - self.config.stack_head - self.config.stack_tail} lines elided] ...\n"
                compressed_segments.append("".join(head) + middle_elision + "".join(tail))
            else:
                compressed_segments.append(seg)
        return "".join(compressed_segments)

    def _tail_first_test_runner(self, text: str) -> str:
        """For test runner output, keep FAILED summary + last N lines."""
        lines = text.splitlines()
        if not lines:
            return text

        # Extract summary section (if any) starting with "FAILURES" or "=== short test summary info ===" etc.
        summary_start = -1
        summary_end = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("FAILURES") or line.strip().startswith("short test summary info"):
                summary_start = i
                # Find end of summary (next blank line or end)
                for j in range(i, len(lines)):
                    if lines[j].strip() == "":
                        summary_end = j
                        break
                if summary_end == -1:
                    summary_end = len(lines) - 1
                break

        # If summary found, keep summary + last tail_lines lines
        tail_len = self.config.tail_lines
        if summary_start != -1:
            kept = (
                lines[summary_start : summary_end + 1]
                + ["", "[last {} lines of output]".format(tail_len)]
                + lines[-tail_len:]
            )
        else:
            kept = ["[no summary found, last {} lines of output]".format(tail_len)] + lines[-tail_len:]

        return "\n".join(kept)

    def _hard_truncate(self, text: str) -> str:
        """Hard truncate at max_chars, appending marker."""
        if len(text) > self.config.max_chars:
            truncated = text[: self.config.max_chars]
            truncated += f"\n[truncated {len(text) - self.config.max_chars} chars]"
            return truncated
        return text
