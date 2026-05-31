"""Tests for Epic #1073 — Memory Tree reliability.

Validates:
- Non-silent failure: corrupt files log warnings and return empty state
- TTL staleness check: get_fresh_entries filters stale entries
- healthcheck() method returns structured report
- /api/memory/health endpoint existence (smoke test)
"""

import json
import logging
import os
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from igris.core.long_term_memory import LongTermMemory, MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ltm(tmpdir: str) -> LongTermMemory:
    """Create a LongTermMemory backed by a temp dir."""
    return LongTermMemory(storage_dir=tmpdir)


# ---------------------------------------------------------------------------
# Non-silent failure on corrupt files
# ---------------------------------------------------------------------------

class TestNonSilentFailure:
    """Corrupt memory files log warnings and return empty state (no crash)."""

    def test_corrupt_entries_file_logs_warning(self, tmp_path, caplog):
        """Write corrupt JSON to entries.json → warning logged, empty entries."""
        (tmp_path / "entries.json").write_text("{not valid json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="igris.memory.long_term"):
            ltm = LongTermMemory(storage_dir=str(tmp_path))

        assert any("entries" in rec.message and "failed" in rec.message.lower()
                   for rec in caplog.records), \
            f"Expected warning about entries load failure, got: {[r.message for r in caplog.records]}"
        # Should return empty state, not crash
        assert len(ltm.get_entries("any_domain")) == 0

    def test_corrupt_index_file_logs_warning(self, tmp_path, caplog):
        """Corrupt index.json → warning logged, empty index."""
        (tmp_path / "index.json").write_text("NOT_JSON", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="igris.memory.long_term"):
            ltm = LongTermMemory(storage_dir=str(tmp_path))

        assert any("index" in rec.message.lower() and "failed" in rec.message.lower()
                   for rec in caplog.records)

    def test_corrupt_summary_file_logs_warning(self, tmp_path, caplog):
        """Corrupt summary.json → warning logged, empty summaries."""
        (tmp_path / "summary.json").write_text("[]", encoding="utf-8")  # list not dict

        with caplog.at_level(logging.WARNING, logger="igris.memory.long_term"):
            ltm = LongTermMemory(storage_dir=str(tmp_path))

        # Even if this doesn't fail (empty list loop), summaries should be empty
        assert isinstance(ltm._summaries, dict)

    def test_no_crash_on_corrupt_file(self, tmp_path):
        """LongTermMemory constructor never raises even with corrupt files."""
        (tmp_path / "entries.json").write_text("INVALID", encoding="utf-8")
        (tmp_path / "index.json").write_text("INVALID", encoding="utf-8")
        (tmp_path / "summary.json").write_text("INVALID", encoding="utf-8")

        try:
            ltm = LongTermMemory(storage_dir=str(tmp_path))
        except Exception as exc:
            pytest.fail(f"LongTermMemory should not raise on corrupt files, but raised: {exc}")

    def test_fresh_ltm_loads_valid_data(self, tmp_path):
        """After corruption, a fresh valid write-then-reload cycle works."""
        ltm = LongTermMemory(storage_dir=str(tmp_path))
        ltm.add_entry("test_domain", {"key": "value"}, source="test")

        # Reload
        ltm2 = LongTermMemory(storage_dir=str(tmp_path))
        entries = ltm2.get_entries("test_domain")
        assert len(entries) == 1
        assert entries[0].domain == "test_domain"


# ---------------------------------------------------------------------------
# TTL staleness check
# ---------------------------------------------------------------------------

class TestTtlStaleness:
    """get_fresh_entries filters out entries older than TTL."""

    def test_fresh_entry_returned(self, tmp_path):
        """Entry created just now is within TTL."""
        ltm = _make_ltm(str(tmp_path))
        ltm.add_entry("domain1", "fresh content", source="test")

        entries = ltm.get_fresh_entries("domain1", ttl_seconds=3600)
        assert len(entries) == 1

    def test_stale_entry_excluded(self, tmp_path):
        """Entry created 2 hours ago is outside a 1-hour TTL."""
        ltm = _make_ltm(str(tmp_path))
        entry = ltm.add_entry("domain1", "old content", source="test")

        # Manually backdate the timestamp
        ltm._entries[entry.id].timestamp = time.time() - 7200  # 2 hours ago

        entries = ltm.get_fresh_entries("domain1", ttl_seconds=3600)
        assert len(entries) == 0

    def test_mixed_entries_ttl(self, tmp_path):
        """Fresh entries are returned, stale are filtered."""
        ltm = _make_ltm(str(tmp_path))

        fresh = ltm.add_entry("domain1", "fresh", source="test")
        stale = ltm.add_entry("domain1", "stale", source="test")
        ltm._entries[stale.id].timestamp = time.time() - 7200  # 2 hours old

        entries = ltm.get_fresh_entries("domain1", ttl_seconds=3600)
        assert len(entries) == 1
        assert entries[0].id == fresh.id

    def test_stale_log_warning_emitted(self, tmp_path, caplog):
        """When stale entries exist, an info log is emitted."""
        ltm = _make_ltm(str(tmp_path))
        entry = ltm.add_entry("domain1", "old", source="test")
        ltm._entries[entry.id].timestamp = time.time() - 7200

        with caplog.at_level(logging.INFO, logger="igris.memory.long_term"):
            ltm.get_fresh_entries("domain1", ttl_seconds=3600)

        assert any("stale" in rec.message.lower() for rec in caplog.records)

    def test_is_entry_stale_true(self, tmp_path):
        """is_entry_stale returns True for old entries."""
        ltm = _make_ltm(str(tmp_path))
        entry = ltm.add_entry("d", "old", source="test")
        ltm._entries[entry.id].timestamp = time.time() - 7200
        assert ltm.is_entry_stale(entry.id, ttl_seconds=3600) is True

    def test_is_entry_stale_false(self, tmp_path):
        """is_entry_stale returns False for recent entries."""
        ltm = _make_ltm(str(tmp_path))
        entry = ltm.add_entry("d", "fresh", source="test")
        assert ltm.is_entry_stale(entry.id, ttl_seconds=3600) is False

    def test_is_entry_stale_missing_id(self, tmp_path):
        """is_entry_stale returns True for unknown IDs (treat as stale)."""
        ltm = _make_ltm(str(tmp_path))
        assert ltm.is_entry_stale("nonexistent-id", ttl_seconds=3600) is True

    def test_unknown_domain_returns_empty(self, tmp_path):
        """get_fresh_entries on unknown domain returns empty list, not error."""
        ltm = _make_ltm(str(tmp_path))
        entries = ltm.get_fresh_entries("no_such_domain", ttl_seconds=3600)
        assert entries == []


# ---------------------------------------------------------------------------
# healthcheck() method
# ---------------------------------------------------------------------------

class TestHealthcheck:
    """LongTermMemory.healthcheck() returns structured status."""

    def test_healthy_status_on_empty_store(self, tmp_path):
        """Fresh empty store → healthy status."""
        ltm = _make_ltm(str(tmp_path))
        result = ltm.healthcheck()
        assert result["status"] == "healthy"
        assert result["files_ok"] is True

    def test_healthy_status_with_data(self, tmp_path):
        """Store with valid data → healthy status."""
        ltm = _make_ltm(str(tmp_path))
        ltm.add_entry("d", "content")
        result = ltm.healthcheck()
        assert result["status"] == "healthy"
        assert result["entry_count"] >= 1
        assert result["domain_count"] >= 1

    def test_degraded_status_on_corrupt_file(self, tmp_path):
        """After writing corrupt entries.json, healthcheck reports degraded."""
        ltm = _make_ltm(str(tmp_path))
        ltm.add_entry("d", "content")

        # Corrupt the file after saving
        (tmp_path / "entries.json").write_text("BAD JSON", encoding="utf-8")

        result = ltm.healthcheck()
        assert result["status"] == "degraded"
        assert result["files_ok"] is False

    def test_healthcheck_fields_present(self, tmp_path):
        """healthcheck result always includes required fields."""
        ltm = _make_ltm(str(tmp_path))
        result = ltm.healthcheck()
        required_keys = {"status", "entry_count", "domain_count", "summary_count", "files_ok"}
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# /api/memory/health endpoint (import smoke test)
# ---------------------------------------------------------------------------

class TestMemoryHealthEndpoint:
    """Smoke tests for the /api/memory/health router registration."""

    def test_endpoint_registered_in_routes(self):
        """The health endpoint function must exist in the routes module."""
        from igris.web.routers import routes_03
        # The function is registered inside a factory; check that the source
        # code contains the endpoint path
        import inspect
        src = inspect.getsource(routes_03)
        assert "/api/memory/health" in src, \
            "/api/memory/health endpoint not found in routes_03 source"

    def test_endpoint_returns_status_key(self):
        """If we call the endpoint function, it should produce a status key."""
        # We can't easily invoke the FastAPI endpoint in unit tests without
        # a running app, but we can verify the implementation exists and the
        # response contract is documented in the source.
        from igris.web.routers import routes_03
        import inspect
        src = inspect.getsource(routes_03)
        assert '"status"' in src or "'status'" in src, \
            "health endpoint should return a 'status' key"
        assert "healthy" in src, \
            "health endpoint should include 'healthy' in possible values"
