# Real Task Benchmarks

Operational benchmark suite proving IGRIS_GPT workflows end-to-end.

All benchmarks run in deterministic/fallback mode — no LLM required, no costs, no external calls.

## Benchmarks

### 1. Docs-Only Task

Full workflow for a safe documentation update:

| Step | Action | Result |
|------|--------|--------|
| Mission | Create "Update docs" mission | Created |
| Plan | Generate deterministic plan | Steps generated |
| Materialize | Create persistent tasks | Tasks in TaskEngine |
| Patch | Create CHANGELOG.md patch proposal | Diff generated |
| Validate | Safety validation | Valid, no violations |
| Apply | Apply patch | File written |
| Report | Decision report | Persisted with outcome |

**Proves:** mission → plan → materialize → patch → validate → apply → decision report.

### 2. Bugfix Small

Fix a simulated bug with patch workflow:

| Step | Action | Result |
|------|--------|--------|
| Setup | Create buggy `utils.py` (a - b instead of a + b) | File written |
| Mission | Create bugfix mission | Created |
| Plan | Generate fix plan | Steps generated |
| Patch | Create fix patch (modify action) | Diff shows change |
| Validate | Safety check | Valid |
| Apply | Apply fix | File corrected |
| Verify | Read file content | `a + b` present, `a - b` gone |

**Proves:** bugfix workflow from mission to verified fix.

### 3. Test Failure Recovery

Simulated test failure with recovery cycle:

| Step | Action | Result |
|------|--------|--------|
| Task | Create "Run tests" task | Created |
| Simulate | Fake failed outcome report | Report generated |
| Route | Outcome router processes failure | Recommendation returned |
| Memory | Record failure event | Persisted in decision_memory |
| State | Record in project_state | Family metrics updated |
| Verify | Check failure recorded | failures >= 1 |
| Recover | Record success attempt | successes >= 1 |
| Teacher | Propose remediation | Task description returned |

**Proves:** failure → routing → memory → state → recovery → remediation.

### 4. Multi-File Safe Patch

Patch two files in one proposal:

| Step | Action | Result |
|------|--------|--------|
| Patch | Create proposal with README.md + NOTES.md changes | 2 files, both with diffs |
| Validate | Safety check on both files | Valid |
| Apply | Apply atomically | Both files written |
| Verify | Read both files | New content present |

Also tests mixed create + modify operations in a single proposal.

**Proves:** multi-file patch → validate → apply workflow.

### 5. Full Loop Smoke

Complete autonomous loop cycle:

| Step | Action | Result |
|------|--------|--------|
| Mission | Create test execution mission | Created |
| Plan | Generate plan | Steps generated |
| Materialize | Create tasks in engine | task_ids populated |
| Loop Step | execute_step() | Task selected and processed |
| Report | Decision report | Persisted with selection data |
| Memory | Verify memory events | Decisions recorded |
| Constraints | Check memory constraints | avoid/saturated families returned |

Additional coverage:
- Loop stops on no tasks (graceful exit)
- run_loop respects max_steps
- Timeline events recorded

**Proves:** mission → plan → materialize → select → loop step → report → memory → decision report.

## Safety Cross-Checks

All benchmarks verify safety gates:

| Check | Result |
|-------|--------|
| `.env` file creation blocked | Validation fails |
| `.git/` path blocked | Validation fails |
| Path traversal (`../../../etc/passwd`) blocked | Validation fails |
| Secret content (API keys) blocked | Validation fails |
| Delete action blocked | Validation fails |
| No `/api/git/push` endpoint | 404 |
| No `/api/git/merge` endpoint | 404 |
| `redact_secrets()` removes sk-*/ghp_* | Confirmed |

## Running Benchmarks

```bash
# Run all operational benchmarks
python -m pytest tests/test_operational_benchmark.py -v

# Run specific benchmark
python -m pytest tests/test_operational_benchmark.py::TestBenchmark1DocsOnlyTask -v

# Run with the benchmark script
bash scripts/run_operational_benchmark.sh
```

## Record Fields

Each benchmark produces a record with:

| Field | Description |
|-------|-------------|
| benchmark | Benchmark name |
| mission_input | Mission title |
| plan | Plan summary |
| tasks_created | Number of tasks materialized |
| patch_proposal | Proposal ID |
| validation_result | valid/invalid |
| outcome | success/failure |
| decision_report_id | Report ID (if generated) |
| manual_intervention_needed | Whether human action was required |

## Key Observations

1. **All workflows run without LLM** — deterministic planning and fallback mode work end-to-end.
2. **Safety gates catch all dangerous operations** — path traversal, secrets, .env, .git, delete all blocked.
3. **Memory and state track correctly** — failures, recoveries, and constraints propagate through the system.
4. **Patch proposals are atomic** — multi-file patches validate and apply together.
5. **Loop respects boundaries** — max_steps honored, no tasks = graceful stop.
