"""
Definition of safe commands for the terminal MVP.

Each entry defines a human‑friendly name mapped to a list representing the
command and its arguments.  Only commands listed here may be executed by the
safe runner.  The user interface should present these commands as options
rather than allowing free‑form input.
"""

from __future__ import annotations

from typing import Dict, List


ALLOWED_COMMANDS: Dict[str, List[str]] = {
    "git_status": ["git", "status", "--short"],
    "git_log": ["git", "log", "--oneline", "-10"],
    "run_tests": ["python", "-m", "pytest", "-q"],
    "list_files": ["ls", "-1"],
}