# Prepared but Not Fully Implemented Capabilities

This document tracks the IGRIS_GPT capabilities that are intentionally prepared, scaffolded, documented, or partially integrated, but are **not yet fully operational production features**.

The goal is to make future work easy to rediscover and to avoid confusing an installable/safe baseline with a fully autonomous, cost-incurring, production-grade agent.

Last updated: 2026-05-04 (v0.5-real-world-candidate)

---

## Current Baseline

IGRIS_GPT v0.5 provides an installable, safety-first, real-world-validated engineering loop:

- Ubuntu install scripts and server lifecycle scripts
- FastAPI backend with 80+ API endpoints
- Web console with 14+ operational tabs
- Local-first chat with phi4-mini, streaming, tier selector, context enrichment
- Mission planner with deterministic + LLM-based planning (safe schema)
- Persistent task engine with explainable selection
- Patch proposal, diff preview, validation and safe apply
- **LLM patch generation (proposal-only, never auto-applies)**
- Controlled Git workflow + gated GitHub PR workflow
- **GitHub PR dry-run benchmark (full workflow without side effects)**
- Decision/failure memory with LLM analysis
- Autonomous loop MVP with bounded steps, diagnostics, decision reports
- Validation/definition-of-done layer
- A2A task/artifact store
- Cost router and provider availability checks
- Vast.ai gated/mock-safe GPU management
- Operational diagnostics (starvation, blocked, family health)
- ProjectState + saturation cooldown
- Strict safety policy + safe command policy
- Timeline/reports/safety/cost visibility
- **Human acceptance verification (21-step checklist + automated script)**
- **External repo sandbox benchmarks (5 scenarios on sandbox project)**
- Operational benchmarks (5 workflow scenarios documented)
- **928 tests passing**

The sections below list the parts that are intentionally **not yet complete**.

---

## 1. Vast.ai Real API Integration

### Implemented in v0.4 (Sprint 22)

- Full gated manager with 7 endpoints
- Config: deepseek-r1:32b default, qwen2.5-coder:7b fallback
- Approval token: `I_APPROVE_VASTAI_COSTS`
- Budget gate, anti-duplicate guard, state-aware destroy
- Mode management: on_demand | always_on | disabled
- 48 tests, all mock/dry-run

### Still not implemented

- Real HTTP calls to Vast.ai API
- Creating actual GPU instances
- Installing/starting Ollama/vLLM remotely on GPU instances
- Pulling/running DeepSeek on remote GPU
- Querying remote DeepSeek model from router
- Automatic instance shutdown/destroy lifecycle
- Real cost accounting from Vast.ai billing
- Production UI controls for provisioning/destroy

### When to implement

When ready to incur real GPU costs. The gated framework is ready — only the HTTP transport layer needs to be connected.

---

## 2. Intelligent Patch Generation (LLM-based)

### Implemented in v0.5 (Sprint 28)

- LLM patch generation module (`igris/core/llm_patch_generator.py`)
- Endpoints: `POST /api/patches/generate`, `POST /api/tasks/{id}/generate-patch`
- Schema-validated JSON output
- Path/content/secret validation
- Deterministic fallback when LLM unavailable
- `proposal_only: true` flag — never auto-applies
- 44 tests

### Not fully implemented yet

- Robust multi-file patch planning from complex goals
- Patch self-review / quality scoring
- Automatic repair after failing tests
- Rollback strategy
- Semantic diff explanation
- Confidence scoring

---

## 3. Real-Task Benchmark Hardening

### Implemented in v0.5 (Sprints 20, 27, 29)

- 5 operational benchmarks (docs-only, bugfix, test failure, multi-file, full loop) — Sprint 20
- External repo sandbox benchmarks (5 scenarios: bugfix, test repair, docs, refactor, multi-file) — Sprint 27
- GitHub PR dry-run benchmark (full workflow without side effects) — Sprint 29
- Human acceptance verification (21-step checklist) — Sprint 26
- All deterministic/mock-based — no LLM fragility

### Not fully implemented yet

- Benchmarks on real public repositories (network dependencies)
- Scoring of patch quality across versions
- Regression tracking
- Comparison between local/fallback/Vast models
- Automated benchmark runner with reporting dashboard

---

## 4. WebSocket Live Updates

### Not implemented

- UI currently uses polling (15s auto-refresh)
- WebSocket for real-time task progress, timeline, loop state updates

---

## 5. Vector Search Memory

### Not implemented

- Memory is currently simple JSON append with file-based persistence
- Semantic vector search for similar failures/decisions
- Embedding-based memory clustering

---

## 6. Multi-Repo Management

### Not implemented

- Single project root assumed
- Managing tasks/patches/missions across multiple repositories

---

## Why These Are Not Enabled by Default

Some capabilities are deliberately left prepared rather than fully active because they can create risks if enabled too early:

- financial cost risk: Vast.ai real provisioning, remote GPU runtimes
- data/security risk: secrets in prompts, logs or remote instances
- repo integrity risk: free shell commands or LLM-generated commands
- reliability risk: LLM patching without validation
- trust risk: autonomous loops without explainability and stop conditions

A feature can be considered ready only when it has:

1. safe defaults
2. explicit approval gates for destructive/costly actions
3. tests and E2E coverage
4. documentation
5. no secret leakage
6. no runtime artifacts committed
7. rollback or stop behavior
8. clear UI/API state

---

## Operational Interpretation

A feature being listed here does **not** mean IGRIS_GPT is broken. It means the feature is not required for the current real-world-validated baseline, or it would be unsafe/costly to enable automatically.

IGRIS_GPT always remains operational without these advanced capabilities:

- if Vast.ai real API is not connected, use local/fallback providers (gated mock works)
- if LLM planning fails, deterministic planning takes over
- if LLM patch generation fails, deterministic fallback returns safe placeholder
- if streaming is unavailable, use non-streaming chat
- if GitHub PR workflow is not approved, use commit/PR proposals
- if memory analysis LLM is unavailable, deterministic analysis works
- if DeepSeek GPU runtime is unavailable, continue with phi4-mini/fallback

The target is not to hide unfinished work. The target is to keep the system usable, honest, safe and incrementally improvable.
