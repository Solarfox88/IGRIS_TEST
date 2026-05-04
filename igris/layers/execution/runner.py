"""
Safe command runner.

This module exposes functions to run whitelisted commands defined in
`safe_commands.py`.  Each command is executed with a timeout and limited
working directory to prevent arbitrary code execution.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
from igris.models.config import CONFIG


class CommandError(Exception):
    """Raised when an invalid or unsafe command is requested."""

    pass


def run_safe_command(command_name: str, timeout: int = 30) -> dict:
    """Execute a whitelisted command and return its output and status.

    :param command_name: Key in ALLOWED_COMMANDS.
    :param timeout: Maximum number of seconds to allow the command to run.
    :return: A dict with stdout, stderr and returncode.
    :raises CommandError: If the command is not in the allowlist.
    """
    if command_name not in ALLOWED_COMMANDS:
        raise CommandError(f"Command '{command_name}' is not allowed")
    cmd = ALLOWED_COMMANDS[command_name]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CONFIG.project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out", "returncode": -1}
    return {
        "stdout": result.stdout[:10_000],
        "stderr": result.stderr[:10_000],
        "returncode": result.returncode,
    }


def run_tests(timeout: int = 300) -> dict:
    """Run the test suite using the safe test command.

    This is a convenience wrapper that calls the `run_tests` command defined
    in `ALLOWED_COMMANDS`.  The timeout may be larger for tests.
    """
    return run_safe_command("run_tests", timeout=timeout)