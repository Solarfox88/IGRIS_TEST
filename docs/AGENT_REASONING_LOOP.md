# Agent Reasoning Loop — Epic #61

## Overview

The Agent Reasoning Loop is the cognitive core of IGRIS. It implements
the observe-reason-act-observe cycle that makes IGRIS an autonomous
reasoning agent rather than a static task runner.

## Architecture

```
Goal + Mission ID
        |
        v
+------------------+
| build_context()  |  <-- Context Manager (Epic #60)
+------------------+
        |
        v
+------------------+
| Model Orchestrator|  <-- decide next action (Epic #58)
| .complete()      |
+------------------+
        |
        v
+------------------+
| validate_action()|  <-- Agent Action Schema (Epic #58)
+------------------+
        |
        v
+------------------+
| route + execute  |  <-- CodeNavigator / ToolRuntime / Risk Engine
+------------------+
        |
        v
+------------------+
| observe result   |
| update state     |
| check governor   |
+------------------+
        |
        v
   next step or stop
```

## Action Routes

| Route | Handler | Side Effects |
|---|---|---|
| code_navigation | CodeNavigator | None (read-only) |
| tool_runtime | ToolRuntime | Governed |
| command_risk_engine | Blocked (Epic #63) | None |
| mission_controller | Plan update | State only |
| memory | Record decision | Memory only |
| human_gate | Ask user | Stop condition |
| terminal | Finish/Blocked | Stop condition |

## Stop Conditions

| Reason | Trigger |
|---|---|
| finish | LLM declares mission complete |
| blocked | LLM cannot proceed |
| ask_user | LLM needs human input |
| budget_exceeded | Too many consecutive errors |
| max_steps | Step limit reached |
| governor_stop | Anti-loop governor triggered |
| llm_unavailable | No suitable LLM provider |
| risk_blocked | High-risk action blocked |

## API Endpoints

### POST /api/loop/run
Run the full reasoning loop for a goal.

### POST /api/loop/step
Execute a single step (testing/debugging).

### GET /api/loop/stop-reasons
List all possible stop reasons.

## Degraded Mode

When no LLM provider is available, the loop returns a `blocked`
action with reason "LLM unavailable". This ensures graceful
degradation without crashes.

## Key Properties

- Every action validated against Agent Action Schema
- Raw shell proposals blocked until Command Risk Engine (Epic #63)
- All output through `redact_secrets()`
- File modifications tracked in result
- Memory items accumulated across steps
- Full execution trace in LoopResult
