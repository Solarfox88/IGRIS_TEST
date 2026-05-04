#!/usr/bin/env bash
set -euo pipefail

# Run IGRIS_GPT operational benchmark suite
# Usage: bash scripts/run_operational_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== IGRIS_GPT Operational Benchmark Suite ==="
echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Project: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

echo "--- Benchmark 1: Docs-Only Task ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmark1DocsOnlyTask -v --tb=short 2>&1
echo ""

echo "--- Benchmark 2: Bugfix Small ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmark2BugfixSmall -v --tb=short 2>&1
echo ""

echo "--- Benchmark 3: Test Failure Recovery ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmark3TestFailureRecovery -v --tb=short 2>&1
echo ""

echo "--- Benchmark 4: Multi-File Safe Patch ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmark4MultiFilePatch -v --tb=short 2>&1
echo ""

echo "--- Benchmark 5: Full Loop Smoke ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmark5FullLoopSmoke -v --tb=short 2>&1
echo ""

echo "--- Safety Cross-Checks ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmarkSafety -v --tb=short 2>&1
echo ""

echo "--- Record Structure ---"
python -m pytest tests/test_operational_benchmark.py::TestBenchmarkRecordStructure -v --tb=short 2>&1
echo ""

echo "=== All Benchmarks Complete ==="
echo "Total: $(python -m pytest tests/test_operational_benchmark.py --co -q 2>&1 | tail -1)"
