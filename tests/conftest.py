"""Project-wide pytest configuration.

Session fixture: clear accumulated IGRIS state from the shared /tmp/project
test root before each session.  Many supervisor tests use /tmp/project as their
project_root and write events to .igris/supervisor_runs.json.  Without a cleanup,
that file grows across sessions (12 MB+ observed) and makes every read/write
slow enough to exceed the supervisor's idle-timeout check during IGRIS baseline
runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


_SHARED_IGRIS_DIR = Path("/tmp/project/.igris")


@pytest.fixture(scope="session", autouse=True)
def _reset_shared_igris_state() -> None:
    """Delete and recreate /tmp/project/.igris at the start of each test session.

    This prevents supervisor_runs.json and failure_patterns.json from growing
    unboundedly across sessions and causing I/O-induced idle timeouts.
    """
    if _SHARED_IGRIS_DIR.exists():
        shutil.rmtree(_SHARED_IGRIS_DIR, ignore_errors=True)
    _SHARED_IGRIS_DIR.mkdir(parents=True, exist_ok=True)
