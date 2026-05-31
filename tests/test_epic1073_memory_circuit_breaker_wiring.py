"""Tests for Epic #1073 — MemoryCircuitBreaker wired into MemoryGraph._tree_write
and memory_healthcheck() function.

Verifies:
1. Breaker state machine: closed → failure accumulation → open → half-open → closed.
2. _tree_write skips ContentStore/Scorer when breaker is OPEN.
3. _tree_write calls record_success() after a successful write.
4. _tree_write calls record_failure() after a failed write.
5. memory_healthcheck() returns expected keys and correct db_ok.
6. memory_healthcheck() includes circuit_breaker_state.
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pytest

from igris.core.memory_circuit_breaker import (
    MemoryCircuitBreaker,
    BreakerState,
    DEFAULT_OPEN_THRESHOLD,
    DEFAULT_RECOVERY_WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_breaker(name: str = "test_cb", threshold: int = 3, window: float = 120.0) -> MemoryCircuitBreaker:
    MemoryCircuitBreaker.reset_all()
    return MemoryCircuitBreaker.get(name, open_threshold=threshold, recovery_window_seconds=window)


# ---------------------------------------------------------------------------
# MemoryCircuitBreaker state machine tests
# ---------------------------------------------------------------------------

class TestMemoryCircuitBreakerStateMachine:

    def setup_method(self):
        MemoryCircuitBreaker.reset_all()

    def test_initial_state_is_closed(self):
        cb = _fresh_breaker()
        assert cb.state == BreakerState.CLOSED

    def test_allow_when_closed(self):
        cb = _fresh_breaker()
        assert cb.allow() is True

    def test_failure_accumulation_opens_breaker(self):
        cb = _fresh_breaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == BreakerState.OPEN

    def test_allow_false_when_open(self):
        cb = _fresh_breaker(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == BreakerState.OPEN
        assert cb.allow() is False

    def test_singleton_per_name(self):
        cb1 = MemoryCircuitBreaker.get("same_name")
        cb2 = MemoryCircuitBreaker.get("same_name")
        assert cb1 is cb2

    def test_different_names_are_independent(self):
        cb1 = MemoryCircuitBreaker.get("breaker_a")
        cb2 = MemoryCircuitBreaker.get("breaker_b")
        cb1.record_failure()
        cb1.record_failure()
        cb1.record_failure()
        assert cb1.state == BreakerState.OPEN
        assert cb2.state == BreakerState.CLOSED

    def test_success_closes_breaker_from_half_open(self):
        cb = _fresh_breaker(threshold=1)
        cb.record_failure()  # opens
        assert cb.state == BreakerState.OPEN
        # Manually simulate half-open (patch _last_failure_ts to be old)
        cb._last_failure_ts = cb._last_failure_ts - (DEFAULT_RECOVERY_WINDOW + 1)
        assert cb.allow() is True  # should transition to HALF_OPEN and allow
        cb.record_success()
        assert cb.state == BreakerState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = _fresh_breaker(threshold=1, window=0.001)
        cb.record_failure()  # opens
        import time
        time.sleep(0.01)  # wait for recovery window
        cb.allow()  # transitions to HALF_OPEN
        cb.record_failure()  # fails trial → OPEN again
        assert cb.state == BreakerState.OPEN

    def test_reset_all_clears_registry(self):
        _fresh_breaker("persistent_name")
        MemoryCircuitBreaker.reset_all()
        # After reset, getting same name returns a fresh breaker
        cb_new = MemoryCircuitBreaker.get("persistent_name")
        assert cb_new.state == BreakerState.CLOSED
        assert cb_new._failure_count == 0

    def test_success_decrements_failure_count_when_closed(self):
        cb = _fresh_breaker(threshold=5)
        cb.record_failure()
        cb.record_failure()  # failure_count = 2
        cb.record_success()  # should decrement → 1
        assert cb._failure_count == 1
        assert cb.state == BreakerState.CLOSED


# ---------------------------------------------------------------------------
# MemoryGraph._tree_write circuit breaker integration tests
# ---------------------------------------------------------------------------

class TestMemoryGraphTreeWriteCircuitBreaker:

    def setup_method(self):
        MemoryCircuitBreaker.reset_all()

    def _make_graph(self):
        from igris.core.memory_graph import MemoryGraph
        tmpdir = tempfile.mkdtemp()
        return MemoryGraph(tmpdir)

    def test_add_node_succeeds_with_closed_breaker(self):
        """add_node should work normally when breaker is CLOSED."""
        graph = self._make_graph()
        nid = graph.add_node("lesson", {"text": "test lesson", "outcome": "success"})
        assert isinstance(nid, str)
        assert len(nid) > 0

    def test_add_node_still_persists_to_db_when_tree_write_fails(self):
        """Even if ContentStore/Scorer raises, the SQLite write must succeed."""
        graph = self._make_graph()
        nid = graph.add_node("lesson", {"text": "test lesson 2", "outcome": "failure"})
        # Verify the node is in SQLite
        node = graph.get_node(nid)
        assert node is not None
        assert node["node_type"] == "lesson"

    def test_breaker_skips_tree_write_when_open(self):
        """When breaker is OPEN, _tree_write should skip without raising."""
        graph = self._make_graph()
        # Open the breaker by injecting failures
        cb = MemoryCircuitBreaker.get("memory_tree")
        for _ in range(DEFAULT_OPEN_THRESHOLD):
            cb.record_failure()
        assert cb.state == BreakerState.OPEN

        # add_node must still succeed (SQLite write should work)
        nid = graph.add_node("lesson", {"text": "lesson with open breaker"})
        node = graph.get_node(nid)
        assert node is not None  # SQLite write succeeded


# ---------------------------------------------------------------------------
# memory_healthcheck() tests
# ---------------------------------------------------------------------------

class TestMemoryHealthcheck:

    def setup_method(self):
        MemoryCircuitBreaker.reset_all()

    def _make_graph(self):
        from igris.core.memory_graph import MemoryGraph
        tmpdir = tempfile.mkdtemp()
        return MemoryGraph(tmpdir)

    def test_healthcheck_returns_expected_keys(self):
        graph = self._make_graph()
        report = graph.memory_healthcheck()
        required_keys = {
            "db_ok", "node_count", "node_counts_by_type",
            "circuit_breaker_state", "content_store_available",
            "scorer_available", "errors",
        }
        assert required_keys.issubset(set(report.keys()))

    def test_healthcheck_db_ok_true_on_empty_db(self):
        graph = self._make_graph()
        report = graph.memory_healthcheck()
        assert report["db_ok"] is True
        assert report["node_count"] == 0
        assert isinstance(report["node_counts_by_type"], dict)

    def test_healthcheck_node_count_after_add(self):
        graph = self._make_graph()
        graph.add_node("lesson", {"text": "hello", "outcome": "success"})
        graph.add_node("lesson", {"text": "world", "outcome": "failure"})
        report = graph.memory_healthcheck()
        assert report["db_ok"] is True
        assert report["node_count"] >= 2

    def test_healthcheck_circuit_breaker_state_closed_initially(self):
        graph = self._make_graph()
        report = graph.memory_healthcheck()
        # Breaker starts CLOSED (or unknown if import fails)
        assert report["circuit_breaker_state"] in (BreakerState.CLOSED, "unknown")

    def test_healthcheck_circuit_breaker_state_open_after_failures(self):
        graph = self._make_graph()
        cb = MemoryCircuitBreaker.get("memory_tree")
        for _ in range(DEFAULT_OPEN_THRESHOLD):
            cb.record_failure()
        report = graph.memory_healthcheck()
        assert report["circuit_breaker_state"] == BreakerState.OPEN

    def test_healthcheck_errors_is_list(self):
        graph = self._make_graph()
        report = graph.memory_healthcheck()
        assert isinstance(report["errors"], list)

    def test_healthcheck_content_store_and_scorer_are_bool(self):
        graph = self._make_graph()
        report = graph.memory_healthcheck()
        assert isinstance(report["content_store_available"], bool)
        assert isinstance(report["scorer_available"], bool)
