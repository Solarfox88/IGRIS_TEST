"""
Validation utilities for tests and reports.

In the future this module will include functions to parse test results and
generate structured reports.  For now it defines a simple schema for a test
report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TestReport:
    """Represents the outcome of running the test suite."""

    success: bool
    output: str
    errors: str

    def summary(self) -> str:
        return "success" if self.success else "failure"