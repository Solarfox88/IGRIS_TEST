"""Tests for Real Operational Benchmark /api/ping — Epic #64.

Validates the benchmark harness, each phase independently, and the
full deterministic benchmark execution.
"""

import pytest
import os
from pathlib import Path

from igris.core.benchmark_ping import (
    BenchmarkRunner,
    BenchmarkResult,
    BENCHMARK_GOAL,
    BENCHMARK_PHASES,
)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_default(self):
        r = BenchmarkResult()
        assert r.mode == "deterministic"
        assert r.status == "pending"
        assert r.total_phases == 0

    def test_to_dict(self):
        r = BenchmarkResult(
            status="passed",
            total_phases=8,
            code_navigation_ok=True,
        )
        d = r.to_dict()
        assert d["status"] == "passed"
        assert d["code_navigation_ok"] is True

    def test_to_dict_redacts(self):
        fake = "sk-" + "a" * 30
        r = BenchmarkResult(final_report=f"key={fake}")
        d = r.to_dict()
        assert fake not in d["final_report"]

    def test_benchmark_id_format(self):
        r = BenchmarkResult()
        assert r.benchmark_id.startswith("bench-")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_goal(self):
        assert "ping" in BENCHMARK_GOAL
        assert "pong" in BENCHMARK_GOAL

    def test_phases(self):
        assert len(BENCHMARK_PHASES) == 8
        assert "code_navigation" in BENCHMARK_PHASES
        assert "context_manager" in BENCHMARK_PHASES
        assert "reasoning_loop" in BENCHMARK_PHASES
        assert "tool_runtime" in BENCHMARK_PHASES
        assert "risk_engine" in BENCHMARK_PHASES
        assert "test_execution" in BENCHMARK_PHASES
        assert "memory" in BENCHMARK_PHASES
        assert "governor" in BENCHMARK_PHASES


# ---------------------------------------------------------------------------
# Phase tests (individual)
# ---------------------------------------------------------------------------

class TestPhaseCodeNavigation:
    """Test code navigation phase."""

    def test_finds_server(self, tmp_path):
        """Navigation should find server.py in real project."""
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = BenchmarkResult()
        runner._phase_code_navigation(result)
        assert result.code_navigation_ok is True
        assert "code_navigation" in result.phases_completed


class TestPhaseContextManager:
    """Test context manager phase."""

    def test_builds_context(self):
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = BenchmarkResult()
        runner._phase_context_manager(result)
        assert result.context_manager_ok is True
        assert "context_manager" in result.phases_completed


@pytest.mark.slow
class TestPhaseReasoningLoop:
    """Test reasoning loop phase — marked slow: calls _phase_reasoning_loop which uses real LLM."""

    def test_loop_runs(self, tmp_path):
        runner = BenchmarkRunner(project_root=str(tmp_path))
        result = BenchmarkResult()
        runner._phase_reasoning_loop(result)
        # Without LLM, loop still completes (degraded mode)
        assert result.reasoning_loop_ok is True
        assert "reasoning_loop" in result.phases_completed


class TestPhaseToolRuntime:
    """Test tool runtime phase."""

    def test_git_status(self):
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = BenchmarkResult()
        runner._phase_tool_runtime(result)
        assert result.tool_runtime_ok is True
        assert "tool_runtime" in result.phases_completed
        assert "git status" in result.commands_executed


class TestPhaseRiskEngine:
    """Test risk engine phase."""

    def test_classifies_correctly(self, tmp_path):
        runner = BenchmarkRunner(project_root=str(tmp_path))
        result = BenchmarkResult()
        runner._phase_risk_engine(result)
        assert result.risk_engine_ok is True
        assert "risk_engine" in result.phases_completed


@pytest.mark.slow
class TestPhaseTestExecution:
    """Test test execution phase — marked slow: spawns a pytest subprocess."""

    def test_runs_tests(self):
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = BenchmarkResult()
        runner._phase_test_execution(result)
        assert result.test_execution_ok is True
        assert "test_execution" in result.phases_completed


class TestPhaseMemory:
    """Test memory phase."""

    def test_records_decision(self, tmp_path):
        runner = BenchmarkRunner(project_root=str(tmp_path))
        result = BenchmarkResult()
        runner._phase_memory(result)
        assert result.memory_ok is True
        assert "memory" in result.phases_completed


class TestPhaseGovernor:
    """Test governor phase."""

    def test_evaluates_task(self, tmp_path):
        runner = BenchmarkRunner(project_root=str(tmp_path))
        result = BenchmarkResult()
        runner._phase_governor(result)
        assert result.governor_ok is True
        assert "governor" in result.phases_completed


# ---------------------------------------------------------------------------
# Full deterministic benchmark
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDeterministicBenchmark:
    """Test full deterministic benchmark execution.

    Marked slow: run_deterministic() spawns a pytest subprocess that collects
    the full test suite, causing >300s of silence in a parent pytest process.
    Excluded from supervised baseline runs (which use -m 'not slow').
    """

    def test_runs_all_phases(self):
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = runner.run_deterministic()
        assert result.mode == "deterministic"
        assert result.total_phases == 8
        assert result.status in ("passed", "partial")
        assert result.total_duration_ms >= 0
        assert result.final_report != ""
        assert "IGRIS Benchmark" in result.final_report
        assert isinstance(result.to_dict(), dict)

    def test_report_format(self):
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = runner.run_deterministic()
        report = result.final_report
        assert "Status:" in report
        assert "Phases" in report
        assert "code_navigation" in report

    def test_phases_tracked(self):
        runner = BenchmarkRunner(project_root=str(Path(__file__).parent.parent))
        result = runner.run_deterministic()
        # At minimum, these should pass in test environment
        assert "code_navigation" in result.phases_completed
        assert "context_manager" in result.phases_completed
        assert "risk_engine" in result.phases_completed
        assert "memory" in result.phases_completed
        assert "governor" in result.phases_completed


# ---------------------------------------------------------------------------
# Integration benchmark (degraded — no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIntegrationBenchmark:
    """Test integration benchmark — marked slow: runs the full LLM integration pipeline."""

    def test_runs_without_llm(self, tmp_path):
        runner = BenchmarkRunner(project_root=str(tmp_path))
        result = runner.run_integration(max_steps=2)
        assert result.mode == "integration"
        assert result.status in ("passed", "partial")
        assert isinstance(result.to_dict(), dict)
