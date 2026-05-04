# Prepared but Not Fully Implemented Capabilities

This document tracks the IGRIS_GPT capabilities that are intentionally prepared, scaffolded, documented, or partially integrated, but are **not yet fully operational production features**.

The goal is to make future work easy to rediscover and to avoid confusing an installable/safe baseline with a fully autonomous, cost-incurring, production-grade Devin replacement.

Last updated: 2026-05-04

---

## Current Baseline

IGRIS_GPT v0.2 already provides an installable, safety-first, Devin-like engineering loop MVP:

- Ubuntu install scripts and server lifecycle scripts
- FastAPI backend
- Web console with many operational tabs
- local-first chat with fallback behavior
- mission planner and task graph
- persistent task engine
- patch proposal, diff preview, validation and safe apply
- controlled Git workflow without push automation
- decision/failure memory
- autonomous loop MVP with bounded steps
- validation/definition-of-done layer
- A2A task/artifact store
- cost router and provider availability checks
- timeline/reports/safety/cost visibility

This is functional and usable as a controlled local engineering assistant.

The sections below list the parts that are intentionally **not yet complete**.

---

## 1. Vast.ai Real Runtime / DeepSeek GPU Provider

### Current state

Vast.ai is currently prepared mainly as a routing/cost provider:

- configuration hooks exist or are planned;
- availability can report whether a Vast.ai key is present;
- route estimation can mention Vast.ai;
- budget/cost concepts exist;
- no automatic cost-incurring action should happen by default.

### Not fully implemented yet

The following are not considered production-ready in IGRIS_GPT yet:

- real instance provisioning from IGRIS_GPT;
- GPU offer search against live Vast.ai in production mode;
- creating Vast.ai instances;
- installing/starting Ollama/vLLM remotely;
- pulling/running DeepSeek on the remote GPU;
- querying the remote DeepSeek model from the router;
- automatic instance shutdown/destroy lifecycle;
- robust real cost accounting from Vast.ai;
- persistent/on-demand mode with full safeguards;
- production UI controls for provisioning/destroy.

### Useful prior work to reference

Known prior implementations/prototypes exist in older repositories:

- `Solarfox88/IGRIS_OH`, branch `openhands-test-clean`
  - `igris/layers/advisory/vastai_manager.py`
  - includes a more mature VastAIManager flow and DeepSeek-R1 32B target.
- `Solarfox88/IGRIS_DEVIN`, branch `init-branch`
  - `igris/layers/advisory/vastai_manager.py`
  - older Qwen coder based implementation.
  - `igris/web/server.py`
  - contained experimental/debug Vast.ai endpoints.

### Recommended future implementation

Implement as a dedicated gated sprint:

`Vast.ai Gated DeepSeek Runtime`

Minimum requirements:

- `VASTAI_MODEL=deepseek-r1:32b`
- `VASTAI_FALLBACK_MODEL=qwen2.5-coder:7b`
- `VASTAI_REQUIRE_APPROVAL=true`
- `VASTAI_AUTO_PROVISION=false` by default
- `VASTAI_MAX_HOURLY_COST` budget gate
- explicit approval string required before provisioning, for example:
  - `I_APPROVE_VASTAI_COSTS`
- mock all Vast.ai HTTP calls in CI tests;
- never expose API keys;
- never create instances during CI;
- no automatic cost-incurring action from the autonomous loop.

Suggested endpoints:

- `GET /api/vastai/config`
- `GET /api/vastai/status`
- `POST /api/vastai/offers/search`
- `POST /api/vastai/estimate`
- `POST /api/vastai/provision` gated by explicit approval
- `POST /api/vastai/destroy` gated and audited
- `POST /api/vastai/set-mode` gated and audited

Definition of done:

- dry-run works without cost;
- provision refuses without approval;
- provision with approval works against mocked Vast.ai API;
- real mode is documented as cost-incurring;
- no duplicate instance creation;
- destroy is safe and state-aware;
- router can use Vast.ai only when explicitly configured and ready.

---

## 2. LLM-Based Mission Planning

### Current state

The mission planner is deterministic/rule-based. It can create missions, generate a structured plan, materialize tasks, and expose a task graph.

### Not fully implemented yet

- LLM-generated multi-step plans;
- JSON-schema validated LLM planning output;
- plan critic/reviewer;
- automatic re-planning after failure;
- model-specific planning strategy;
- explicit comparison between deterministic and LLM-generated plan.

### Recommended future implementation

Add an LLM planner behind a safe fallback:

1. Try deterministic planner first or as fallback.
2. Ask LLM for structured JSON plan only.
3. Validate against a strict schema.
4. Redact secrets.
5. Reject invalid/unsafe plan steps.
6. Require success criteria on every step.
7. Do not auto-execute generated tasks.

Definition of done:

- deterministic fallback always works;
- invalid LLM plan never breaks mission creation;
- LLM plan includes success criteria, risk and safe capabilities;
- tests cover malformed LLM output and fallback behavior.

---

## 3. LLM-Based Memory Analysis

### Current state

Decision/failure memory records events and deterministic constraints. The teacher and task selection can use memory constraints such as repeated failures or saturated families.

### Not fully implemented yet

- LLM analysis of recurring failure patterns;
- periodic memory summarization;
- root-cause analysis from failures;
- long-term learning from task/report history;
- automatic archival/compaction strategy;
- semantic clustering of failures/decisions.

### Recommended future implementation

Add a memory analyzer that produces safe summaries, not direct actions:

- `GET /api/memory/analysis`
- `POST /api/memory/analyze`
- `GET /api/memory/lessons`

Rules:

- output is advisory only;
- never execute from memory analysis;
- redact secrets;
- keep deterministic fallback.

Definition of done:

- analysis explains repeated failures;
- teacher payload can include summarized lessons;
- task selection can prefer safer families based on memory;
- tests cover secret redaction and bounded output.

---

## 4. Intelligent Patch Generation

### Current state

IGRIS_GPT supports patch proposals, diff preview, validation and safe apply. The workflow is controlled and safe.

### Not fully implemented yet

- robust LLM-generated patches from arbitrary natural language goals;
- multi-file patch planning;
- patch self-review;
- automatic repair after failing tests;
- rollback strategy;
- semantic diff explanation;
- confidence scoring.

### Recommended future implementation

Keep the current safe workflow, but add an LLM proposal generator:

`mission/task -> LLM patch draft -> patch proposal -> validation -> diff review -> manual/app gated apply`

Do not bypass the existing validation/apply gates.

Definition of done:

- generated patch is always stored as a proposal;
- apply still requires validation;
- secrets/path traversal/binary file rules still apply;
- failing tests create remediation task, not uncontrolled retries.

---

## 5. Full GitHub PR Workflow

### Current state

IGRIS_GPT supports controlled Git visibility and proposal logic:

- status;
- diff;
- diffstat;
- branch listing/creation;
- safety checks;
- commit proposal;
- PR summary;
- no push endpoint by design.

### Not fully implemented yet

- real commit creation from the UI/API;
- pushing branches to GitHub;
- opening pull requests from IGRIS_GPT;
- updating PR descriptions;
- review checklist enforcement;
- rollback branch creation;
- conflict/rebase workflow;
- merge automation.

### Recommended future implementation

Implement as a gated workflow, not as free Git automation:

1. pre-commit safety check;
2. tests must be green or explicit override required;
3. commit message generated and reviewed;
4. commit creation requires approval;
5. push requires approval and branch allowlist;
6. PR creation requires approval and summary review.

Definition of done:

- no push to `main`;
- no force push;
- no secret/runtime artifacts;
- no automatic merge;
- every write operation is audited.

---

## 6. Chat Streaming and Session Tier Selector

### Current state

The chat works, with local-first/fallback behavior depending on current implementation state. UI provider/fallback visibility exists or is being hardened.

### Not fully implemented yet

- robust SSE/token streaming in the current IGRIS_GPT branch;
- per-session tier selector fully integrated in UI and backend;
- tier choices such as `auto`, `local`, `fallback` persisted per session;
- streaming metadata at end of response;
- cancellation of streaming generation.

### Useful prior work to reference

`IGRIS_DEVIN` had an SSE endpoint and per-session tier concept:

- `/api/sessions/{session_id}/messages/stream`
- tier values such as `auto`, `local`, `api`, `vastai`

Do not port the unsafe auto-execution behavior from IGRIS_DEVIN.

### Recommended future implementation

Implement safe streaming only:

- text streaming;
- provider/model/fallback metadata;
- no command execution;
- no `[CMD]` execution;
- no `[WRITE_FILE]` execution;
- chat may create proposals/tasks only through safe endpoints.

---

## 7. Context-Enriched Chat

### Current state

IGRIS_GPT has many context sources: missions, tasks, reports, memory, loop status, Git status, patches, cost/routing. Chat does not yet fully use all of them as a coherent context package.

### Not fully implemented yet

- chat prompt enriched with current mission;
- selected/active tasks;
- recent reports;
- validation state;
- patch proposal status;
- Git dirty state;
- decision memory constraints;
- cost/routing state;
- loop stop reason.

### Recommended future implementation

Create a `ChatContextBuilder` that composes safe, bounded context:

- current mission summary;
- active tasks;
- recent failures;
- memory constraints;
- last loop step;
- git status summary;
- provider/cost state.

Rules:

- bounded token size;
- redacted secrets;
- no raw `.env` or secret files;
- chat proposes actions, it does not directly execute them.

---

## 8. Advanced Operational Diagnostics

### Current state

Diagnostics are planned from prior repositories and may be under active development in post-v0.2 work.

### Not fully implemented until verified

- task starvation diagnostics;
- observation-loop diagnostics;
- blocked-task accumulation warnings;
- family failure health;
- excessive recovery escalation;
- UI surfacing of diagnostics;
- diagnostics impact on loop stop/recovery.

### Useful prior work to reference

`IGRIS_DEVIN` included diagnostics for starvation, observation loops, blocked accumulation, family health and recovery escalation.

### Recommended future implementation

Expose:

- `GET /api/diagnostics`
- `GET /api/diagnostics/summary`

Integrate with:

- Safety tab;
- Loop tab;
- Agent timeline;
- teacher remediation.

---

## 9. Strict Safety Policy as Second-Layer Guard

### Current state

IGRIS_GPT uses safe command IDs and avoids free shell execution.

### Not fully implemented until verified

- a reusable strict command safety policy applied after command ID resolution;
- explicit destructive-pattern detector as a universal second layer;
- consistent `SafetyDecision` object across terminal, loop, tests and future Git operations.

### Useful prior work to reference

`IGRIS_DECO` had a `SafeCommandPolicy` with strict mode, allowlist checks and destructive command/pattern blocks.

### Recommended future implementation

Keep command IDs. Add strict validation after command resolution:

`command_id -> resolved argv -> strict safety policy -> execute`

Never replace safe command IDs with free shell strings.

---

## 10. Explainable Task Selection

### Current state

Task selection exists and respects memory/anti-loop constraints.

### Not fully implemented until verified

- complete selection decision object;
- rejected candidate reasons;
- task scoring explanation;
- surfaced selection explanation in loop response/UI;
- endpoint dedicated to selection explanation.

### Useful prior work to reference

`IGRIS_DECO` had a task selector that returned selected task, score, why, rejected reasons, saturated families, blocked families and recent counts.

### Recommended future implementation

Add:

- `GET /api/tasks/selection/explain`
- `selection_decision` field in `/api/loop/step` and `/api/loop/recent`

Definition of done:

- every skipped task has a reason;
- selected task has a score and reason;
- UI can show why the agent chose or refused work.

---

## 11. Project State and Saturation Cooldown

### Current state

IGRIS_GPT has decision memory and saturation concepts.

### Not fully implemented until verified

- unified `ProjectState` persisted snapshot;
- family metrics with completed/failed/blocked counts;
- `cooldown_until` per family;
- recent task fingerprints;
- recovery escalation count;
- state-driven diagnostics;
- state-driven teacher recovery.

### Useful prior work to reference

`IGRIS_DEVIN` and `IGRIS_DECO` had `ProjectState`, `FamilySaturationState`, `FamilyMetrics` and `RecoveryPattern` models.

### Recommended future implementation

Do not replace existing memory. Integrate state as an additional operational snapshot:

- `.igris/state/project_state.json`
- update state from loop/task outcomes;
- expose state safely via API;
- use state in diagnostics and teacher payload.

---

## 12. Decision Reports per Loop Cycle

### Current state

IGRIS_GPT has reports, timeline and loop step results.

### Not fully implemented until verified

- a dedicated JSON decision report for every loop cycle;
- complete reasoning trace of selected/rejected tasks;
- safety decisions;
- action chosen;
- outcome;
- memory constraints;
- teacher recommendations;
- next-step recommendation.

### Recommended future implementation

Persist reports under:

`.igris/decision_reports/`

Expose:

- `GET /api/decision-reports`
- `GET /api/decision-reports/{id}`

Definition of done:

- every loop step produces one decision report;
- reports are redacted;
- reports are linked from timeline;
- UI can inspect them.

---

## 13. Real-Task Benchmark Hardening

### Current state

The internal test suite is large and green, but passing unit/API/E2E tests is not the same as proving real-world autonomy.

### Not fully implemented yet

- benchmark suite on real repositories;
- repeated bugfix/feature/refactor tasks;
- scoring of patch quality;
- recovery after failed tests;
- regression tracking;
- comparison between local/fallback/Vast models.

### Recommended future implementation

Create an `ops-benchmarks/` or `docs/REAL_TASK_BENCHMARKS.md` workflow:

- 3 to 5 real tasks first;
- then 10+ repeatable benchmark tasks;
- record mission, plan, patch, tests, report, outcome;
- measure manual intervention needed.

---

## Why These Are Not Enabled by Default

Some capabilities are deliberately left prepared rather than fully active because they can create risks if enabled too early:

- financial cost risk: Vast.ai provisioning, remote GPU runtimes;
- data/security risk: secrets in prompts, logs or remote instances;
- repo integrity risk: commits, pushes, force pushes, PR automation;
- execution risk: free shell commands or LLM-generated commands;
- reliability risk: LLM planning/patching without validation;
- trust risk: autonomous loops without explainability and stop conditions.

A feature can be considered ready only when it has:

1. safe defaults;
2. explicit approval gates for destructive/costly actions;
3. tests and E2E coverage;
4. documentation;
5. no secret leakage;
6. no runtime artifacts committed;
7. rollback or stop behavior;
8. clear UI/API state.

---

## Operational Interpretation

A feature being listed here does **not** mean IGRIS_GPT is broken. It means the feature is not required for the current local safety-first baseline, or it would be unsafe/costly to enable automatically.

IGRIS_GPT should always remain operational without these advanced capabilities:

- if Vast.ai is not implemented, use local/fallback providers;
- if LLM planning fails, use deterministic planning;
- if streaming is unavailable, use non-streaming chat;
- if GitHub PR automation is unavailable, use commit/PR proposals;
- if memory analysis is unavailable, use deterministic memory constraints;
- if DeepSeek GPU runtime is unavailable, continue with `phi4-mini`/fallback.

The target is not to hide unfinished work. The target is to keep the system usable, honest, safe and incrementally improvable.
