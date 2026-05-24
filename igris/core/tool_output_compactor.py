import re
from typing import List, Optional


class ToolOutputCompactor:
    """Rule-driven compactor for tool stdout before LLM injection.

    Applies configurable rules to reduce token count while preserving
    semantic content. Inspired by the TokenJuice concept.
    """

    def __init__(
        self,
        dedup_consecutive: bool = True,
        truncate_stack_traces: bool = True,
        stack_first_lines: int = 10,
        stack_last_lines: int = 5,
        collapse_repeated_patterns: bool = True,
        strip_ansi: bool = True,
        tail_first_test_runners: bool = True,
    ):
        self.dedup_consecutive = dedup_consecutive
        self.truncate_stack_traces = truncate_stack_traces
        self.stack_first_lines = stack_first_lines
        self.stack_last_lines = stack_last_lines
        self.collapse_repeated_patterns = collapse_repeated_patterns
        self.strip_ansi = strip_ansi
        self.tail_first_test_runners = tail_first_test_runners

    def compact(self, text: str) -> str:
        if not text:
            return text
        result = text
        if self.strip_ansi:
            result = self._strip_ansi(result)
        if self.dedup_consecutive:
            result = self._dedup_consecutive_lines(result)
        if self.collapse_repeated_patterns:
            result = self._collapse_repeated_patterns(result)
        if self.truncate_stack_traces:
            result = self._truncate_stack_traces(result)
        if self.tail_first_test_runners:
            result = self._tail_first_for_test_runners(result)
        return result

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences."""
        ansi_escape = re.compile(r'\033(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    @staticmethod
    def _dedup_consecutive_lines(text: str) -> str:
        """Remove consecutive duplicate lines."""
        lines = text.split('\n')
        if not lines:
            return text
        result = [lines[0]]
        for line in lines[1:]:
            if line != result[-1]:
                result.append(line)
        return '\n'.join(result)

    def _collapse_repeated_patterns(self, text: str) -> str:
        """Collapse patterns like '..........' -> '[10× "."]'."""
        pattern = re.compile(r'((.)\2{3,})', re.MULTILINE)
        def replacer(m):
            full = m.group(1)
            char = m.group(2)
            return f'[{len(full)}× {repr(char)}]'
        return pattern.sub(replacer, text)

    def _truncate_stack_traces(self, text: str) -> str:
        """Keep first N and last M lines of tracebacks, truncating middle."""
        lines = text.split('\n')
        # Detect traceback: lines containing 'Traceback' or 'File "..."'
        traceback_indices = []
        in_traceback = False
        for i, line in enumerate(lines):
            if 'Traceback' in line or re.match(r'^\s*File "', line):
                if not in_traceback:
                    traceback_indices.append(i)
                in_traceback = True
            else:
                in_traceback = False
        if not traceback_indices:
            return text
        # For simplicity, truncate all tracebacks found
        result = []
        tb_start = traceback_indices[0]
        # Find end of traceback (last line of stack before non-indented line)
        tb_end = tb_start
        for i in range(tb_start, len(lines)):
            if lines[i].startswith(' ') or 'File "' in lines[i] or 'Traceback' in lines[i]:
                tb_end = i
            else:
                break
        tb_lines = lines[tb_start:tb_end+1]
        first = tb_lines[:self.stack_first_lines]
        last = tb_lines[-self.stack_last_lines:]
        truncated = first + [f'    ... truncated {len(tb_lines) - self.stack_first_lines - self.stack_last_lines} lines'] + last
        result = lines[:tb_start] + truncated + lines[tb_end+1:]
        return '\n'.join(result)

    def _tail_first_for_test_runners(self, text: str) -> str:
        """Reorder lines so that test results/passes/fails appear first.

        For pytest/jest output, 'PASS', 'FAIL', 'failed', 'passed' lines are moved up.
        """
        lines = text.split('\n')
        important = []
        rest = []
        for line in lines:
            if re.search(r'\b(PASS|FAIL|passed|failed|ok|ERROR|error)\b', line):
                important.append(line)
            else:
                rest.append(line)
        return '\n'.join(important + rest)
