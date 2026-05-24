"""ToolOutputCompactor - Rule-driven stdout compaction before LLM injection.

Implements the TokenJuice concept from tinyhumansai/openhuman to reduce token
usage by compressing tool outputs before they reach the reasoning loop.
"""

import re
from typing import Optional, Dict, List, Any


class ToolOutputCompactor:
    """Compacts tool output using configurable rules.

    Rules:
        1. Dedup consecutive identical lines
        2. Collapse repeated patterns (e.g., ``..........`` → ``[10× '.']``)
        3. Strip ANSI escape codes
        4. Truncate long stack traces (keep first 10, last 5)
        5. Tail-first for test runners (pytest/jest) — keep head + tail

    Args:
        config: dict with keys:
            - dedup_enabled (bool): default True
            - collapse_repeated_enabled (bool): default True
            - strip_ansi_enabled (bool): default True
            - truncate_stacktrace_enabled (bool): default True
            - tail_first_enabled (bool): default True
            - tail_first_head_lines (int): lines to keep from beginning
            - tail_first_tail_lines (int): lines to keep from end
            - tail_first_max_lines (int): threshold to trigger tail-first
            - truncate_stacktrace_head (int): lines to keep at top
            - truncate_stacktrace_tail (int): lines to keep at bottom
            - collapse_min_repeats (int): minimum char repeats to collapse
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "dedup_enabled": True,
        "collapse_repeated_enabled": True,
        "strip_ansi_enabled": True,
        "truncate_stacktrace_enabled": True,
        "tail_first_enabled": True,
        "tail_first_head_lines": 50,
        "tail_first_tail_lines": 200,
        "tail_first_max_lines": 500,
        "truncate_stacktrace_head": 10,
        "truncate_stacktrace_tail": 5,
        "collapse_min_repeats": 10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def compact(self, text: str, source_type: str = "generic") -> str:
        """Apply all enabled compaction rules.

        Args:
            text: raw tool output
            source_type: hint ("generic", "test_runner", "pytest", "jest")

        Returns:
            compacted text
        """
        if not text:
            return text

        result = text

        # Rule 3: Strip ANSI early to avoid interfering with other rules
        if self.config["strip_ansi_enabled"]:
            result = self._strip_ansi(result)

        # Rule 1: Dedup consecutive identical lines
        if self.config["dedup_enabled"]:
            result = self._dedup_consecutive(result)

        # Rule 2: Collapse repeated patterns
        if self.config["collapse_repeated_enabled"]:
            result = self._collapse_repeated_patterns(result)

        # Rule 4: Truncate long stack traces (only if generic or unknown; test runners have their own handling)
        if self.config["truncate_stacktrace_enabled"] and source_type == "generic":
            result = self._truncate_stacktrace(result)

        # Rule 5: Tail-first for test runners
        if self.config["tail_first_enabled"] and source_type in ("test_runner", "pytest", "jest"):
            result = self._tail_first(result, source_type)

        return result

    # ------------------------------------------------------------------
    # Private rule implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_consecutive(text: str) -> str:
        """Remove consecutive duplicate lines."""
        lines = text.splitlines()
        deduped = []
        prev = None
        for line in lines:
            if line != prev:
                deduped.append(line)
                prev = line
        return "\n".join(deduped)

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Strip ANSI escape sequences."""
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
        return ansi_escape.sub('', text)

    def _collapse_repeated_patterns(self, text: str) -> str:
        """Collapse lines consisting of a single repeated character.

        Example: ``..........`` → ``[10× '.']``
        """
        min_repeats = self.config["collapse_min_repeats"]
        lines = text.splitlines()
        collapsed = []
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) >= min_repeats and len(set(stripped)) == 1:
                char = stripped[0]
                count = len(stripped)
                # Replace with collapsed notation, preserving leading whitespace
                leading_ws = line[:len(line) - len(line.lstrip())]
                collapsed.append(f"{leading_ws}[{count}× '{char}']")
            else:
                collapsed.append(line)
        return "\n".join(collapsed)

    def _truncate_stacktrace(self, text: str) -> str:
        """Truncate long Python tracebacks: keep first N and last M lines.

        Recognises typical Python stack frames (lines starting with ``  File "...", line ...``).
        """
        head_lines = self.config["truncate_stacktrace_head"]
        tail_lines = self.config["truncate_stacktrace_tail"]
        lines = text.splitlines()

        # Detect stack frames: lines that match '  File '
        frame_pattern = re.compile(r'^  File ".+", line \d+')
        frame_indices = [i for i, line in enumerate(lines) if frame_pattern.match(line)]

        if len(frame_indices) <= head_lines + tail_lines:
            return text  # Not long enough to truncate

        # Split into blocks of consecutive frames
        # A simple heuristic: if there are many frames, truncate the middle.
        # We'll keep the first `head_lines` frames and the last `tail_lines` frames.
        # Find the corresponding line indices.
        keep_first_end = frame_indices[head_lines - 1] if head_lines > 0 else -1
        keep_last_start = frame_indices[-tail_lines] if tail_lines > 0 else len(lines)

        if keep_first_end >= keep_last_start:
            return text  # overlap, nothing to truncate

        # Build result: lines before first frame? Keep everything before first frame?
        # We'll keep the entire preamble (lines before any frame), then first N frames,
        # then insert a truncation marker, then the last M frames, then the final error message.
        # The final error line is typically after the last frame.
        if not frame_indices:
            return text

        preamble = lines[:frame_indices[0]]
        first_block = lines[frame_indices[0]: keep_first_end + 1]
        # Find the end of the last frame we keep, plus any following lines until the start of the
        # last frames block. Better approach: keep lines up to the end of the head_frame block,
        # then insert marker, then include lines from the start of the tail block onward.
        # If tail_lines == 0, we just truncate after head.

        if tail_lines == 0:
            result = preamble + first_block + ["[... truncated ...]"]
            return "\n".join(result)

        tail_start_index = frame_indices[-tail_lines]
        tail_block = lines[tail_start_index:]
        result = preamble + first_block + ["[... truncated ...]"] + tail_block
        return "\n".join(result)

    def _tail_first(self, text: str, source_type: str) -> str:
        """Keep only the head and tail of the output for test runners."""
        head_lines = self.config["tail_first_head_lines"]
        tail_lines = self.config["tail_first_tail_lines"]
        max_lines = self.config["tail_first_max_lines"]
        lines = text.splitlines()
        total = len(lines)

        if total <= max_lines:
            return text

        # Keep first `head_lines` and last `tail_lines`
        head = lines[:head_lines]
        tail = lines[-tail_lines:]

        # Insert a note about how many lines were omitted
        skipped = total - head_lines - tail_lines
        if skipped <= 0:
            return text

        result = head + [f"[... {skipped} lines skipped ...]"] + tail
        return "\n".join(result)
