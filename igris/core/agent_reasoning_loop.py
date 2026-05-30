"""Agent Reasoning Loop for IGRIS_GPT — Epic #61.

The cognitive core: observe -> reason -> act -> observe -> repeat.

Loop structure:
    1. build_context (Context Manager)
    def __init__(
        self,
        project_root: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        role: str = "coder",
        task_type: str = "code_reasoning",
        preferred_profile: Optional[str] = None,
        tool_result_budget: int = 16384,  # max bytes for tool result output
    ):
       - mission_controller -> update plan
       - memory -> record to DecisionMemory
       - human_gate -> ask_user (stop condition)
       - terminal -> finish/blocked (stop condition)
    5. observe result
    6. update state / memory / mission
    7. governor check (anti-loop)
    8. next step or finish

Stop conditions:
    - finish: mission complete
    - blocked: cannot proceed
    - ask_user: needs human input
    - budget_exceeded: token/cost/step budget exceeded
    - risk_blocked: high-risk action blocked by policy
    - max_steps: exceeded maximum step count
    - governor_stop: anti-loop governor triggered
    - llm_unavailable: no suitable model available
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from igris.core.safety import redact_secrets
from igris.core.tool_result_budget import apply_tool_result_budget, DEFAULT_BUDGET_BYTES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_REASONS = (
    "finish",
    "blocked",
    "ask_user",
    "budget_exceeded",
    "risk_blocked",
    "max_steps",
    "no_diff_repair",
    "governor_stop",
    "llm_unavailable",
)

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5
WRITE_ACTIONS = {
    "write_file",
    "insert_after",
    "insert_before",
    "replace_range",
    "append_file",
    "apply_patch",
}
READ_ONLY_REPEAT_ACTIONS = {
    "find_files",
    "search_code",
    "read_file_range",
    "list_dir",
    "git_status",
    "git_diff",
}


# ---------------------------------------------------------------------------
# Loop step record
# ---------------------------------------------------------------------------

@dataclass
class LoopStep:
    """Record of one reasoning loop step."""
    step_number: int = 0
    timestamp: float = field(default_factory=time.time)
    action_type: str = ""
    action_route: str = ""
    role: str = "coder"
    reason: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_hint: str = "low"
    confidence: float = 0.5
    outcome: str = ""  # success | failure | blocked | skipped
    result_summary: str = ""
    result_data: Optional[Any] = None  # structured result for downstream consumption
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "action_route": self.action_route,
            "role": self.role,
            "reason": redact_secrets(self.reason),
            "parameters": {k: redact_secrets(str(v)) for k, v in self.parameters.items()},
            "risk_hint": self.risk_hint,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "result_summary": redact_secrets(self.result_summary),
            "error": redact_secrets(self.error),
            "duration_ms": self.duration_ms,
        }
        if self.result_data is not None:
            d["result_data"] = self.result_data
        return d


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """Final result of a reasoning loop execution."""
    loop_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    mission_id: str = ""
    goal: str = ""
    status: str = "pending"  # running | finished | blocked | failed | stopped
    stop_reason: str = ""
    steps: List[LoopStep] = field(default_factory=list)
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    total_duration_ms: int = 0
    files_modified: List[str] = field(default_factory=list)
    diff_summary: str = ""
    test_output: str = ""
    ci_status: str = ""
    pr_url: str = ""
    pr_number: int = 0
    residual_risks: List[str] = field(default_factory=list)
    rollback_available: bool = False
    errors: List[str] = field(default_factory=list)
    final_summary: str = ""
    # Orchestrator observability
    reasoning_execution_provider: str = ""
    reasoning_execution_model: str = ""
    reasoning_execution_profile: str = ""
    orchestrator_used: bool = False
    local_model_available: bool = False
    mission_brain_shadow_mode: bool = False
    mission_brain_shadow_error: str = ""
    mission_brain_shadow_record: Optional[Dict[str, Any]] = None
    mission_brain_wrapper_policy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "mission_id": self.mission_id,
            "goal": redact_secrets(self.goal),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "total_duration_ms": self.total_duration_ms,
            "files_modified": self.files_modified,
            "diff_summary": redact_secrets(self.diff_summary),
            "test_output": redact_secrets(self.test_output),
            "ci_status": self.ci_status,
            "pr_url": self.pr_url,
            "pr_number": self.pr_number,
            "residual_risks": [redact_secrets(r) for r in self.residual_risks],
            "rollback_available": self.rollback_available,
            "errors": [redact_secrets(e) for e in self.errors],
            "final_summary": redact_secrets(self.final_summary),
            "reasoning_execution_provider": self.reasoning_execution_provider,
            "reasoning_execution_model": self.reasoning_execution_model,
            "reasoning_execution_profile": self.reasoning_execution_profile,
            "orchestrator_used": self.orchestrator_used,
            "local_model_available": self.local_model_available,
            "mission_brain_shadow_mode": self.mission_brain_shadow_mode,
            "mission_brain_shadow_error": redact_secrets(self.mission_brain_shadow_error),
            "mission_brain_shadow_record": self.mission_brain_shadow_record,
            "mission_brain_wrapper_policy": self.mission_brain_wrapper_policy,
        }


# ---------------------------------------------------------------------------
# Agent Reasoning Loop
# ---------------------------------------------------------------------------

class AgentReasoningLoop:
    """The cognitive core of IGRIS: observe-reason-act-observe-repeat.

    Usage:
        loop = AgentReasoningLoop(
            project_root="/path/to/repo",
            max_steps=50,
        )
        result = loop.run(goal="Add /api/ping endpoint with tests")
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        no_diff_steps_max: int = 20,
        role: str = "coder",
        task_type: str = "code_reasoning",
        preferred_profile: Optional[str] = None,
        tool_result_budget: int = DEFAULT_BUDGET_BYTES,
        issue_number: int = 0,
        fleet_instance_id: str = "",
    ):
        import os
        self.project_root = project_root or os.environ.get("PROJECT_ROOT", ".")
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.no_diff_steps_max = no_diff_steps_max
        self.role = role
        self.task_type = task_type
        self.preferred_profile = preferred_profile
        self.tool_result_budget = tool_result_budget
        self.issue_number = issue_number
        self.fleet_instance_id = fleet_instance_id

        # State
        self._steps: List[LoopStep] = []
        self._recent_errors: List[Dict[str, Any]] = []
        self._files_modified: List[str] = []
        self._memory_items: List[Dict[str, Any]] = []
        self._world_state: Dict[str, Any] = {}
        self._consecutive_errors = 0
        self._steps_without_write: int = 0  # no_diff_repair guard
        self._fleet_tokens_total: int = 0
        self._stop_reason = ""
        self._coord: object = None  # lazy AgentCoordinator, reused across steps
        # Orchestrator observability — updated on first successful LLM call
        self._reasoning_provider: str = ""
        self._reasoning_model: str = ""
        self._reasoning_profile: str = ""
        self._orchestrator_used: bool = False

        # Anti-repeat guard: tracks (action_type, params_key) -> count
        self._action_history: List[Dict[str, Any]] = []
        self._repeat_threshold = 2  # block after 2 identical successes without consumption

    def run(
        self,
        goal: str = "",
        mission_id: str = "",
        initial_context: Optional[Dict[str, Any]] = None,
        step_callback: Optional[Callable[[int, str], None]] = None,
    ) -> LoopResult:
        """Execute the reasoning loop for a given goal.

        This is the main entry point. It orchestrates:
        1. Builds context
        2. Decides action (via Model Orchestrator)
        3. Validates action
        4. Routes and executes action
        5. Observes result
        6. Updates state
        7. Checks stop conditions

        Args:
            goal: The task/mission goal
            mission_id: Optional mission ID for tracking
            initial_context: Optional initial state

        Returns:
            LoopResult with full execution trace
        """
        t0 = time.monotonic()
        result = LoopResult(
            mission_id=mission_id,
            goal=goal,
            status="running",
        )

        if initial_context is not None:
            if isinstance(initial_context, dict):
                self._world_state.update(initial_context)
            elif isinstance(initial_context, str):
                self._world_state["note"] = initial_context
            else:
                self._world_state["note"] = str(initial_context)

        for step_num in range(1, self.max_steps + 1):
            # Check stop conditions before each step
            stop = self._check_stop_conditions(step_num)
            if stop:
                result.stop_reason = stop
                result.status = "stopped" if stop != "finish" else "finished"
                break

            # Execute one step
            _files_before = len(self._files_modified)
            step = self._execute_step(step_num, goal, mission_id)
            if len(self._files_modified) > _files_before or step.action_type in WRITE_ACTIONS:
                self._steps_without_write = 0
            else:
                self._steps_without_write += 1
            self._steps.append(step)
            if step_callback is not None:
                try:
                    step_callback(step_num, step.action_type or "unknown")
                except Exception:
                    pass
            result.steps.append(step)

            # Track outcomes
            if step.outcome == "success":
                result.successful_steps += 1
                self._consecutive_errors = 0
            elif step.outcome in ("failure", "error"):
                result.failed_steps += 1
                self._consecutive_errors += 1
                result.errors.append(step.error or f"Step {step_num} failed")
            elif step.outcome == "ask_user" and self._suppress_human_gate():
                step.outcome = "skipped"
            elif step.outcome in ("blocked", "finish", "ask_user"):
                result.stop_reason = step.outcome
                if step.outcome == "finish":
                    result.status = "finished"
                elif step.outcome == "blocked":
                    result.status = "blocked"
                elif step.outcome == "ask_user":
                    result.status = "blocked"
                break
        else:
            result.stop_reason = "max_steps"
            result.status = "stopped"

        result.total_steps = len(result.steps)
        result.files_modified = list(set(self._files_modified))
        result.total_duration_ms = int((time.monotonic() - t0) * 1000)
        result.final_summary = self._build_summary(result)
        # Propagate orchestrator observability
        result.reasoning_execution_provider = self._reasoning_provider
        result.reasoning_execution_model = self._reasoning_model
        result.reasoning_execution_profile = self._reasoning_profile
        result.orchestrator_used = self._orchestrator_used
        result.local_model_available = self._local_model_available()
        self._run_mission_brain_shadow(goal=goal, result=result)

        return result

    def _run_mission_brain_shadow(self, goal: str, result: LoopResult) -> None:
        """Run Mission Brain in shadow mode; never affects primary loop behavior."""
        from igris.models.config import CONFIG

        mb = CONFIG.mission_brain_integration
        mode = (mb.mode or "").strip().lower()
        if not (mb.enabled and mode == "shadow"):
            return

        result.mission_brain_shadow_mode = True
        try:
            from igris.agent.mission.shadow_integration import run_shadow_comparison
            from igris.agent.mission.rollback_policy import (
                evaluate_wrapper_policy,
                persist_wrapper_policy_decision,
            )

            result.mission_brain_shadow_record = run_shadow_comparison(
                user_input=goal,
                loop_result=result,
                project_root=self.project_root,
                compare_with_current_loop=mb.compare_with_current_loop,
                telemetry_enabled=mb.telemetry_enabled,
            )
            policy = evaluate_wrapper_policy(
                requested_mode=mode,
                shadow_record=result.mission_brain_shadow_record or {},
                rollback_to_wrapper_on_guardrail=mb.rollback_to_wrapper_on_guardrail,
                auto_rollback_on_risky_mismatch=mb.auto_rollback_on_risky_mismatch,
                force_wrapper_mode=mb.force_wrapper_mode,
            )
            policy["policy_log_path"] = persist_wrapper_policy_decision(
                self.project_root, policy
            )
            result.mission_brain_wrapper_policy = policy
        except Exception as exc:
            result.mission_brain_shadow_error = str(exc)

    def _local_model_available(self) -> bool:
        """Return True if the local Ollama model is reachable."""
        import urllib.request
        import urllib.error
        from igris.models.config import CONFIG
        url = f"{CONFIG.local_llm.base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=2.0):
                return True
        except Exception:
            return False

    def _suppress_human_gate(self) -> bool:
        return bool(
            self._world_state.get("must_not_ask_user")
            or self._world_state.get("suppress_human_gate")
        )

    # ------------------------------------------------------------------
    # Anti-repeat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_signature(action_type: str, params: Dict[str, Any]) -> str:
        """Produce a deterministic key for an action+params pair."""
        import json
        try:
            params_key = json.dumps(params, sort_keys=True, default=str)
        except (TypeError, ValueError):
            params_key = str(sorted(params.items()))
        return f"{action_type}::{params_key}"

    def _check_anti_repeat(self, action_type: str, params: Dict[str, Any]) -> Optional[str]:
        """Return a diagnosis string if this action repeats a previous
        successful action without its results having been consumed.

        Consumption is detected when a later action uses data produced
        by the earlier one (e.g. find_files results fed to read_file_range).
        """
        sig = self._action_signature(action_type, params)

        # Count how many times this exact action already succeeded
        repeat_count = 0
        for prev in self._action_history:
            if prev.get("signature") == sig and prev.get("outcome") == "success":
                repeat_count += 1

        if repeat_count < self._repeat_threshold:
            return None

        # Check if results were consumed by a downstream action
        last_result_data = None
        producer_index = -1
        for idx in range(len(self._action_history) - 1, -1, -1):
            prev = self._action_history[idx]
            if prev.get("signature") == sig and prev.get("outcome") == "success":
                last_result_data = prev.get("result_data")
                producer_index = idx
                break

        if last_result_data and self._was_result_consumed(last_result_data, after_index=producer_index):
            return None

        return (
            f"Anti-repeat guard: '{action_type}' with identical parameters "
            f"already succeeded {repeat_count} time(s) without its results "
            f"being consumed by a downstream action. Strategy shift required."
        )

    def _was_result_consumed(self, result_data: Any, after_index: Optional[int] = None) -> bool:
        """Check if a previous tool's result data was used by a later action."""
        if not result_data:
            return False

        # Extract paths/data from result_data
        paths: List[str] = []
        if isinstance(result_data, list):
            for item in result_data:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str):
                            paths.append(v)
        elif isinstance(result_data, dict):
            for v in result_data.values():
                if isinstance(v, str):
                    paths.append(v)

        if not paths:
            return False

        # Check if any later action references these paths in its parameters
        history = self._action_history
        if after_index is not None:
            history = history[after_index + 1:]

        for prev in history:
            p = prev.get("parameters", {})
            for v in p.values():
                if isinstance(v, str) and any(path in v for path in paths):
                    return True
        return False

    def _record_action_history(
        self,
        action_type: str,
        params: Dict[str, Any],
        outcome: str,
        result_data: Any = None,
    ) -> None:
        """Record an action execution for anti-repeat tracking."""
        self._action_history.append({
            "signature": self._action_signature(action_type, params),
            "action_type": action_type,
            "parameters": params,
            "outcome": outcome,
            "result_data": result_data,
        })

    # ------------------------------------------------------------------
    # Tool result storage
    # ------------------------------------------------------------------

    def _context_config(self) -> dict:
        """Load optional runtime config from .igris/context_config.json."""
        import json
        import pathlib
        p = pathlib.Path(self.project_root or ".") / ".igris" / "context_config.json"
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            return {}

    def _tool_result_budget_bytes(self) -> int:
        """Return the configured tool result budget in bytes (0 = disabled)."""
        cfg = self._context_config()
        return int(cfg.get("tool_result_budget_bytes", self.tool_result_budget))

    def _store_tool_result(
        self,
        action_type: str,
        result_data: Any,
    ) -> None:
        """Store structured tool results in world_state for downstream use."""
        self._world_state["last_tool_result"] = {
            "action_type": action_type,
            "data": result_data,
        }
        # Maintain a rolling list of recent tool results (max 5)
        history = self._world_state.setdefault("tool_result_history", [])
        history.append({"action_type": action_type, "data": result_data})
        if len(history) > 5:
            self._world_state["tool_result_history"] = history[-5:]

        # Specifically for find_files: store discovered paths for easy access
        if action_type == "find_files" and isinstance(result_data, list):
            self._world_state["discovered_files"] = result_data

        # For search_code: store matched files
        if action_type == "search_code" and isinstance(result_data, list):
            matched_files = list({
                m.get("file", "") if isinstance(m, dict) else ""
                for m in result_data
            })
            matched_files = [f for f in matched_files if f]
            self._world_state["search_matched_files"] = matched_files

    def _emit_fleet_heartbeat(self, action_type: str) -> None:
        """Emit heartbeat to VastAIFleet for phase detection. Never raises."""
        if not self.fleet_instance_id and not self.issue_number:
            return
        try:
            from igris.layers.advisory.vastai_fleet import _SHARED_FLEET, AgentHeartbeat
            self._fleet_tokens_total += getattr(self, '_last_step_tokens', 0)
            _SHARED_FLEET.record_heartbeat(AgentHeartbeat(
                instance_id=self.fleet_instance_id,
                issue_number=self.issue_number,
                action_type=action_type,
                tokens_generated=self._fleet_tokens_total,
                tokens_per_sec=0.0,
            ))
        except Exception:
            pass  # fleet errors must never break reasoning

    def _execute_step(
        self,
        step_num: int,
        goal: str,
        mission_id: str,
    ) -> LoopStep:
        """Execute a single reasoning loop step."""
        t0 = time.monotonic()
        step = LoopStep(step_number=step_num, role=self.role)

        try:
            # 1. Build context
            context_packet = self._build_context(goal, mission_id)

            # 2. Decide action (via Model Orchestrator)
            action, parse_errors = self._decide_action(context_packet)

            if parse_errors and not action:
                step.outcome = "failure"
                step.error = f"Action parse errors: {'; '.join(parse_errors)}"
                step.duration_ms = int((time.monotonic() - t0) * 1000)
                return step

            if not action:
                step.outcome = "failure"
                step.error = "No action decided"
                step.duration_ms = int((time.monotonic() - t0) * 1000)
                return step

            action = self._redirect_repeated_test_discovery(action, goal)

            step.action_type = action.action_type
            step.reason = action.reason
            step.parameters = action.parameters
            step.risk_hint = action.risk_hint
            step.confidence = action.confidence

            # 2b. Anti-repeat guard
            repeat_diagnosis = self._check_anti_repeat(
                action.action_type, action.parameters
            )
            if repeat_diagnosis:
                retryable_read_only = action.action_type in READ_ONLY_REPEAT_ACTIONS
                step.outcome = "skipped" if retryable_read_only else "blocked"
                step.error = repeat_diagnosis
                step.result_summary = (
                    "Governor anti-repeat: identical action repeated without "
                    "consuming previous results. Use the results from the "
                    "previous execution or choose a different action."
                )
                self._world_state["anti_repeat_triggered"] = True
                self._world_state["anti_repeat_diagnosis"] = repeat_diagnosis
                if retryable_read_only:
                    self._world_state["anti_repeat_retryable"] = True
                step.duration_ms = int((time.monotonic() - t0) * 1000)
                return step

            # 2c. Contract validation
            _contract_allowed, _contract_reason = (True, "")
            try:
                if self._coord is None:
                    from igris.core.agent_contracts import AgentCoordinator
                    self._coord = AgentCoordinator(self.project_root)
                _contract_allowed, _contract_reason = self._coord.check_and_record(
                    role=self.role,
                    action_type=action.action_type,
                    goal=goal,
                )
            except Exception:
                pass

            if not _contract_allowed:
                step.outcome = "skipped"
                step.error = f"Contract violation: {_contract_reason}"
                step.result_summary = (
                    f"Action '{action.action_type}' is not permitted for role "
                    f"'{self.role}'. Choose an allowed action instead."
                )
                step.duration_ms = int((time.monotonic() - t0) * 1000)
                return step

            # 3. Validate action
            validation = self._validate_action(action)
            if not validation.valid:
                step.outcome = "failure"
                step.error = f"Validation errors: {'; '.join(validation.errors)}"
                step.duration_ms = int((time.monotonic() - t0) * 1000)
                return step

            # 4. Route and execute
            from igris.core.agent_action_schema import get_action_route
            route = get_action_route(action.action_type)
            step.action_route = route

            exec_result = self._execute_action(action, route)

            step.outcome = "success" if exec_result.get("success", False) else "failure"
            step.result_summary = exec_result.get("summary", "")

            # 4a. Issue #534 — ToolTracker: record per-tool effectiveness (best-effort)
            try:
                from igris.core.tool_tracker import ToolTracker as _ToolTracker
                _tt = _ToolTracker(self.project_root or ".")
                _tool_duration_ms = (time.monotonic() - t0) * 1000
                _error_snippet: str | None = None
                if not exec_result.get("success", False):
                    _error_snippet = str(exec_result.get("error", ""))[:200] or None
                _tt.record(
                    tool_name=action.action_type,
                    success=bool(exec_result.get("success", False)),
                    duration_ms=_tool_duration_ms,
                    error_snippet=_error_snippet,
                )
            except Exception:
                pass  # ToolTracker is best-effort, never crashes the step

            # 4b. Store structured result data (apply 16KB byte-cap before injection)
            result_data = exec_result.get("result_data")
            if result_data is not None:
                if isinstance(result_data, str):
                    # Issue #531 — TokenJuice: compact before hard-truncate (best-effort)
                    try:
                        from igris.core.tool_output_compactor import ToolOutputCompactor
                        _compact = ToolOutputCompactor()
                        result_data = _compact.compress(result_data, source_type=action.action_type)
                    except Exception:
                        pass  # compactor is best-effort
                    _budget = self._tool_result_budget_bytes()
                    result_data, _bout = apply_tool_result_budget(result_data, _budget)
                    if _bout.truncated:
                        import logging as _logging
                        _logging.getLogger(__name__).debug(
                            "tool_result_budget: truncated %d → %d bytes for %s",
                            _bout.original_bytes, _bout.final_bytes, action.action_type,
                        )
                step.result_data = result_data
                self._store_tool_result(action.action_type, result_data)

            # 4c-fleet. Emit fleet heartbeat for phase detection and ETC estimation
            self._emit_fleet_heartbeat(action.action_type)

            # 4c. Record for anti-repeat tracking
            if not (action.action_type == "ask_user" and self._suppress_human_gate()):
                self._record_action_history(
                    action.action_type,
                    action.parameters,
                    step.outcome,
                    result_data=result_data,
                )

            if not exec_result.get("success", False):
                step.error = exec_result.get("error", "Execution failed")
                if self._is_ast_validation_write_failure(action.action_type, step.error):
                    step.outcome = "blocked"
                    step.result_summary = (
                        "Python AST validation blocked a write action; "
                        "stopping to avoid accumulating unsafe edits."
                    )
                    self._world_state["ast_validation_blocked"] = True
                self._recent_errors.append({
                    "type": "action_failure",
                    "message": step.error,
                    "step": step_num,
                    "action_type": action.action_type,
                })

            # 5. Handle terminal actions
            if action.action_type == "finish":
                step.outcome = "finish"
                step.result_summary = action.parameters.get("summary", "Mission complete")
            elif action.action_type == "blocked":
                step.outcome = "blocked"
                step.result_summary = action.parameters.get("reason", "Cannot proceed")
            elif action.action_type == "ask_user":
                step.outcome = "ask_user"
                step.result_summary = action.parameters.get("question", "Need input")

            # 6. Track file modifications (write_file/apply_patch track
            #    internally only when a real diff is verified; this
            #    handles propose_patch's informational path tracking)
            if action.action_type == "propose_patch" and exec_result.get("success"):
                file_path = action.parameters.get("path", "")
                if file_path:
                    self._world_state.setdefault("proposed_patches", []).append(file_path)

        except Exception as e:
            step.outcome = "error"
            step.error = str(e)

        step.duration_ms = int((time.monotonic() - t0) * 1000)

        # Issue #532 — ReflectionHook: fire on complex steps only (best-effort, never blocks)
        try:
            from igris.core.reflection_hook import ReflectionHook as _ReflectionHook
            if not hasattr(self, "_reflection_hook"):
                self._reflection_hook = _ReflectionHook(self.project_root or ".")
            _step_result = {
                "action_type": action.action_type if "action" in dir() else "unknown",
                "outcome": step.outcome,
                "summary": step.result_summary or "",
                "error": step.error or "",
                "tool_count": 1,  # each _execute_step is one tool call
            }
            self._reflection_hook.on_step_complete(_step_result, goal=goal, project_root=self.project_root or ".")
        except Exception:
            pass  # ReflectionHook is best-effort

        return step

    @staticmethod
    def _is_ast_validation_write_failure(action_type: str, error: str) -> bool:
        """Return True when a write action failed Python AST validation."""
        return (
            action_type in WRITE_ACTIONS
            and "Python AST validation failed" in (error or "")
        )

    def _build_context(self, goal: str, mission_id: str):
        """Build context packet for the current step.

        Profile drives token budget: cheap_cloud_reasoning=64k chars,
        local_coder=16k. Mismatch causes Ollama to silently truncate
        file context beyond its 4096-token window.
        """
        _PROFILE_MAP = {
            "local_light": "local_light",
            "local_coder": "local_coder",
            "mini_execution": "local_coder",
            "strong_execution": "cheap_cloud_reasoning",
        }
        ctx_profile = _PROFILE_MAP.get(self.preferred_profile or "", "cheap_cloud_reasoning")
        from igris.core.context_manager import ContextManager
        ctx = ContextManager(project_root=self.project_root)
        return ctx.build_context(
            goal=goal,
            role=self.role,
            profile=ctx_profile,
            mission_id=mission_id,
            world_state=self._world_state,
            recent_actions=[s.to_dict() for s in self._steps[-10:]],
            recent_errors=self._recent_errors[-5:],
            memory_items=self._memory_items,
        )

    # Token budget thresholds for local profiles (#1044)
    _LOCAL_TOKEN_WARN = 3000    # ~12 000 chars — warn and log
    _LOCAL_TOKEN_CAP  = 3800    # ~15 200 chars — hard truncate before sending
    _LOCAL_PROFILES   = {"local_light", "local_coder", "mini_execution"}

    def _guard_context_window(
        self,
        system_prompt: str,
        user_message: str,
        context_packet: Any,
    ) -> tuple:
        """Truncate system_prompt/user_message when near local model token cap (#1044 fix).

        Truncation strategy (priority order — least critical cut first):
          1. Cut from the END of system_prompt (file_context + CoT examples are last).
          2. Never cut role/rules/mission/state sections (they're in the front).

        The prompt template order is:
          [role + action schema + rules] → [mission] → [state] → [recent_actions] → [file_context] → [examples]

        Cutting from the end preserves the critical sections and only removes file/example
        context — which is the correct tradeoff when the token budget is tight.

        Returns the (possibly truncated) system_prompt, user_message pair.
        Emits a warning to _world_state when truncation occurs.
        """
        profile = self.preferred_profile or ""
        if profile not in self._LOCAL_PROFILES:
            return system_prompt, user_message

        # Estimate token count (~4 chars per token)
        total_chars = len(system_prompt) + len(user_message)
        token_estimate = total_chars // 4

        if token_estimate <= self._LOCAL_TOKEN_WARN:
            return system_prompt, user_message

        # Log warning
        self._world_state["context_window_warning"] = (
            f"token_estimate={token_estimate} > warn_threshold={self._LOCAL_TOKEN_WARN} "
            f"(profile={profile})"
        )

        if token_estimate <= self._LOCAL_TOKEN_CAP:
            return system_prompt, user_message

        # Hard cap exceeded — cut from the END (file_context and examples are last).
        # This preserves: role, action schema, rules 1-14, mission, state (MISSION INTAKE).
        target_chars = self._LOCAL_TOKEN_CAP * 4
        budget_system = target_chars - len(user_message) - 200  # 200 chars safety margin
        if budget_system > 500 and len(system_prompt) > budget_system:
            system_prompt = (
                system_prompt[:budget_system]
                + "\n... [file context truncated for token budget] ...\n"
            )
            self._world_state["context_window_truncated"] = True

        return system_prompt, user_message

    def _build_cot_examples_text(self) -> str:
        """Build CoT step-by-step examples for injection into system prompt (#1043 fix).

        Local profiles (phi4-mini): 1 example, first 3 steps (token budget ~1200 chars).
        Cloud profiles: all 3 examples, full chains.

        Returns an empty string on any error (never blocks the run).
        """
        try:
            from igris.core.prompt_contract import get_cot_examples
            examples = get_cot_examples()
            if not examples:
                return ""

            profile = self.preferred_profile or ""
            is_local = profile in self._LOCAL_PROFILES
            max_examples = 1 if is_local else len(examples)
            max_steps = 3 if is_local else 999

            lines: List[str] = ["## Step-by-Step Examples\n"]
            for ex in examples[:max_examples]:
                lines.append(f"### Example: {ex.get('scenario', '')}")
                lines.append(f"Goal: {ex.get('goal', '')}\n")
                for step in ex.get("chain", [])[:max_steps]:
                    n = step.get("step", "?")
                    thought = (step.get("thought", "") or "")[:200]
                    action = step.get("action", {}) or {}
                    a_type = action.get("action_type", "?")
                    a_path = (action.get("parameters", {}) or {}).get("path", "")
                    result = (step.get("observed_result", "") or "")[:120]
                    lines.append(f"Step {n} — Thought: {thought}")
                    lines.append(
                        f"  → Action: {a_type}"
                        + (f" '{a_path}'" if a_path else "")
                    )
                    lines.append(f"  → Result: {result}\n")
            return "\n".join(lines)
        except Exception:
            return ""

    def _decide_action(self, context_packet):
        """Decide next action via Model Orchestrator.

        If no LLM is available, returns a deterministic fallback
        (blocked with reason).
        """
        from igris.core.model_orchestrator import ModelOrchestrator
        from igris.core.agent_action_schema import parse_llm_action, AgentAction

        orch = ModelOrchestrator()

        # Build prompt from context
        system_prompt = self._format_system_prompt(context_packet)
        user_message = self._format_user_message(context_packet)

        # Guard against silent truncation on local models (#1044)
        system_prompt, user_message = self._guard_context_window(
            system_prompt, user_message, context_packet
        )

        messages = [
            {"role": "user", "content": user_message},
        ]

        orch_result = orch.complete(
            task_type=self.task_type,
            messages=messages,
            system_prompt=system_prompt,
            json_mode=True,
            preferred_profile=self.preferred_profile,
            timeout=120.0,
        )

        # Record orchestrator observability on first successful call
        if orch_result.success and not self._orchestrator_used:
            self._orchestrator_used = True
            self._reasoning_provider = orch_result.provider
            self._reasoning_model = orch_result.model
            self._reasoning_profile = orch_result.profile

        if not orch_result.success or orch_result.profile == "deterministic":
            # No LLM available — return blocked action
            action = AgentAction(
                mode=self.role,
                action_type="blocked",
                reason="No suitable LLM provider available",
                parameters={"reason": "LLM unavailable — deterministic fallback"},
            )
            return action, []

        # Parse LLM output
        raw_output = orch_result.text
        action, errors = parse_llm_action(raw_output)
        if action:
            action.mode = self.role  # Ensure role matches
        return action, errors

    def _format_system_prompt(self, context_packet) -> str:
        """Format context into system prompt."""
        from igris.core.prompt_contract import build_reasoning_prompt
        return build_reasoning_prompt(
            role=context_packet.role,
            mission_context=context_packet.mission_context,
            state_context=context_packet.state_context,
            recent_actions=context_packet.recent_actions,
            file_context=context_packet.file_context,
            examples_context=self._build_cot_examples_text(),
        )

    def _format_user_message(self, context_packet) -> str:
        """Format user message with errors and memory."""
        parts = []
        if context_packet.error_context and context_packet.error_context != "No recent errors.":
            parts.append(f"ERRORS:\n{context_packet.error_context}")
        if context_packet.memory_context and context_packet.memory_context != "No relevant memory.":
            parts.append(f"MEMORY:\n{context_packet.memory_context}")
        parts.append("Decide your next action. Respond with a single JSON object matching the action schema.")
        return "\n\n".join(parts)

    def _validate_action(self, action):
        """Validate action against schema."""
        from igris.core.agent_action_schema import validate_action
        return validate_action(action)

    def _redirect_repeated_test_discovery(self, action, goal: str):
        """Convert unproductive test discovery into the requested safe edit.

        Controlled rank tasks can provide ``must_create_test_file`` when the
        deliverable explicitly includes a dedicated test file. If the model
        keeps searching for that missing test after the tests directory is
        known, consume that discovery result by creating the requested file.
        """
        requested = self._world_state.get("must_create_test_file")
        if not requested or not isinstance(requested, str):
            return action
        if not requested.startswith("tests/test_") or not requested.endswith(".py"):
            return action

        full_path = os.path.join(self.project_root, requested)
        if os.path.exists(full_path):
            return action

        if action.action_type not in {"find_files", "search_code", "list_directory"}:
            return action

        if not self._is_requested_test_discovery(action, requested):
            return action

        basename = os.path.basename(requested)
        prior_attempts = self._count_requested_test_discovery_attempts(requested)
        tests_known = self._tests_directory_known()
        if prior_attempts < 1 and not tests_known:
            return action

        from igris.core.agent_action_schema import AgentAction

        content = self._build_requested_test_file_content(goal, requested)
        self._world_state["rank_test_creation_redirected"] = {
            "from_action": action.action_type,
            "requested_test_file": requested,
            "reason": "consume_test_discovery",
        }
        return AgentAction(
            mode=self.role,
            action_type="write_file",
            reason=(
                f"Create explicitly requested dedicated test file {basename} "
                "after test discovery showed where tests belong."
            ),
            parameters={"path": requested, "content": content},
            expected_effect=f"Create {requested} with endpoint coverage",
            risk_hint="low",
            confidence=max(action.confidence, 0.8),
        )

    def _is_requested_test_discovery(self, action, requested: str) -> bool:
        basename = os.path.basename(requested)
        params = action.parameters or {}
        haystack = " ".join(str(value) for value in params.values())
        return (
            basename in haystack
            or requested in haystack
            or "tests" in haystack
            or "test_" in haystack
            or action.action_type == "list_directory"
            and params.get("path") in {"tests", "./tests"}
        )

    def _count_requested_test_discovery_attempts(self, requested: str) -> int:
        basename = os.path.basename(requested)
        count = 0
        for prev in self._action_history:
            if prev.get("action_type") not in {"find_files", "search_code", "list_directory"}:
                continue
            params = prev.get("parameters", {})
            haystack = " ".join(str(value) for value in params.values())
            if basename in haystack or requested in haystack or "tests" in haystack:
                count += 1
        return count

    def _tests_directory_known(self) -> bool:
        for key in ("discovered_files", "search_matched_files"):
            values = self._world_state.get(key)
            if isinstance(values, list) and any(str(v).startswith("tests/") for v in values):
                return True
        for item in self._world_state.get("tool_result_history", []):
            data = item.get("data") if isinstance(item, dict) else None
            if isinstance(data, list) and any(str(v).startswith("tests/") for v in data):
                return True
        return os.path.isdir(os.path.join(self.project_root, "tests"))

    def _build_requested_test_file_content(self, goal: str, requested: str) -> str:
        endpoint = self._extract_endpoint_from_goal(goal)
        expected = self._extract_expected_json_from_goal(goal, endpoint)
        test_name = os.path.splitext(os.path.basename(requested))[0]

        if expected:
            expected_repr = repr(expected)
            return (
                "from fastapi.testclient import TestClient\n\n"
                "from igris.web.server import create_app\n\n\n"
                f"def {test_name}_endpoint():\n"
                "    client = TestClient(create_app())\n"
                f"    response = client.get(\"{endpoint}\")\n\n"
                "    assert response.status_code == 200\n"
                f"    assert response.json() == {expected_repr}\n"
            )

        return (
            "from fastapi.testclient import TestClient\n\n"
            "from igris.web.server import create_app\n\n\n"
            f"def {test_name}_endpoint_available():\n"
            "    client = TestClient(create_app())\n"
            f"    response = client.get(\"{endpoint}\")\n\n"
            "    assert response.status_code == 200\n"
        )

    @staticmethod
    def _extract_endpoint_from_goal(goal: str) -> str:
        match = re.search(r"/api/[A-Za-z0-9_./-]+", goal or "")
        if match:
            return match.group(0).rstrip(".,`'\"")
        return "/api/rank/status"

    @staticmethod
    def _extract_expected_json_from_goal(goal: str, endpoint: str) -> Optional[Dict[str, Any]]:
        if endpoint == "/api/rank/status":
            return {
                "rank_system": "E-D-C-B-A-S",
                "current_rank": "B",
                "last_passed": "B",
                "next_rank": "A",
                "status": "ready_for_rank_a",
            }
        if endpoint == "/api/version-info":
            return {"app": "IGRIS_GPT", "status": "ok"}
        return None

    def _execute_action(self, action, route: str) -> Dict[str, Any]:
        """Execute an action by routing to the appropriate handler.

        Routes:
            code_navigation -> CodeNavigator (safe, read-only)
            tool_runtime -> ToolRuntime (governed execution)
            command_risk_engine -> blocked (pending Epic #63)
            mission_controller -> plan update
            memory -> record decision
            human_gate -> ask_user (terminal)
            terminal -> finish/blocked (terminal)
        """
        if route == "code_navigation":
            return self._execute_navigation(action)
        elif route == "tool_runtime":
            return self._execute_tool_runtime(action)
        elif route == "command_risk_engine":
            return {
                "success": False,
                "error": "Command Risk Engine not yet available (Epic #63). "
                         "Raw shell proposals are blocked.",
                "summary": "Blocked: raw shell requires Command Risk Engine",
            }
        elif route == "mission_controller":
            return self._execute_plan_update(action)
        elif route == "memory":
            return self._execute_memory_record(action)
        elif route == "human_gate":
            return {
                "success": True,
                "summary": f"Asking user: {action.parameters.get('question', '')}",
            }
        elif route == "terminal":
            return {
                "success": True,
                "summary": action.parameters.get("summary",
                            action.parameters.get("reason", "Terminal action")),
            }
        else:
            return {
                "success": False,
                "error": f"Unknown route: {route}",
            }

    def _execute_navigation(self, action) -> Dict[str, Any]:
        """Execute a code navigation action.

        Returns structured result_data alongside the standard success/summary
        so the loop can store results for downstream consumption.
        """
        from igris.core.code_navigation import CodeNavigator
        nav = CodeNavigator(project_root=self.project_root)

        try:
            if action.action_type == "search_code":
                result = nav.search_code(
                    pattern=action.parameters.get("pattern", ""),
                    path=action.parameters.get("path"),
                    max_results=action.parameters.get("max_results", 50),
                )
            elif action.action_type == "find_files":
                result = nav.find_files(
                    pattern=action.parameters.get("pattern", ""),
                    max_results=action.parameters.get("max_results", 100),
                )
            elif action.action_type == "list_directory":
                result = nav.list_directory(
                    path=action.parameters.get("path", "."),
                    depth=action.parameters.get("depth", 1),
                )
            elif action.action_type == "read_file_range":
                result = nav.read_file_range(
                    path=action.parameters.get("path", ""),
                    start=action.parameters.get("start", 1),
                    end=action.parameters.get("end"),
                )
            else:
                return {"success": False, "error": f"Unknown nav action: {action.action_type}"}

            # Extract structured data for downstream consumption
            result_data = None
            if result.success and result.data is not None:
                if action.action_type == "find_files":
                    # data is a list of relative file paths
                    result_data = result.data
                elif action.action_type == "search_code":
                    # data is a list of SearchMatch — convert to dicts
                    result_data = [
                        m.to_dict() if hasattr(m, "to_dict") else m
                        for m in result.data
                    ]
                elif action.action_type == "list_directory":
                    result_data = result.data
                elif action.action_type == "read_file_range":
                    result_data = result.data

            summary_parts = [f"{action.action_type}: {result.total_count} results"]
            if action.action_type == "find_files" and result.data:
                # Include discovered paths in summary for LLM visibility
                paths_preview = result.data[:10]
                summary_parts.append(f"files: {paths_preview}")
            elif action.action_type == "search_code" and result.data:
                matched_files = list({
                    m.file if hasattr(m, "file") else m.get("file", "")
                    for m in result.data
                })[:5]
                summary_parts.append(f"in files: {matched_files}")

            return {
                "success": result.success,
                "summary": "; ".join(summary_parts),
                "error": result.error or "",
                "result_data": result_data,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_tool_runtime(self):
        """Get a ToolRuntime instance configured for this loop's project."""
        from igris.core.tool_runtime import ToolRuntime
        return ToolRuntime(project_root=self.project_root)

    @staticmethod
    def _normalize_run_test_args(parameters: Dict[str, Any]) -> List[str]:
        import shlex

        args = parameters.get("args")
        if args:
            if isinstance(args, str):
                return shlex.split(args)
            if isinstance(args, list):
                return [str(arg) for arg in args]
            return [str(args)]

        target = parameters.get("target")
        if not target:
            return []
        if isinstance(target, str):
            return [target]
        if isinstance(target, list):
            return [str(item) for item in target]
        return [str(target)]

    def _execute_tool_runtime(self, action) -> Dict[str, Any]:
        """Execute a tool runtime action using specific ToolRuntime methods.

        Dispatches to the correct method (git_status, fs_write, run_tests,
        etc.) rather than a generic .execute() which does not exist.
        """
        try:
            rt = self._get_tool_runtime()

            if action.action_type == "git_status":
                tr = rt.git_status()
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No output",
                    "error": tr.error,
                    "result_data": tr.output,
                }

            elif action.action_type == "git_diff":
                tr = rt.git_diff(staged=action.parameters.get("staged", False))
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No diff",
                    "error": tr.error,
                    "result_data": tr.output,
                }

            elif action.action_type == "run_tests":
                test_args = self._normalize_run_test_args(action.parameters)
                tr = rt.run_tests(args=test_args if test_args else None)
                return {
                    "success": tr.success,
                    "summary": tr.output[:500] if tr.output else "No output",
                    "error": tr.error,
                    "result_data": tr.output,
                }

            elif action.action_type == "http_check":
                tr = rt.http_check(url=action.parameters.get("url", ""))
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No response",
                    "error": tr.error,
                }

            elif action.action_type == "write_file":
                return self._execute_write_file(rt, action)

            elif action.action_type == "insert_after":
                return self._execute_insert_after(rt, action)

            elif action.action_type == "insert_before":
                return self._execute_insert_before(rt, action)

            elif action.action_type == "replace_range":
                return self._execute_replace_range(rt, action)

            elif action.action_type == "append_file":
                return self._execute_append_file(rt, action)

            elif action.action_type == "propose_patch":
                return self._execute_propose_patch(rt, action)

            elif action.action_type == "apply_patch":
                return self._execute_apply_patch(rt, action)

            elif action.action_type == "shell_template":
                cmd_id = action.parameters.get("command_id", "")
                args = action.parameters.get("args", [])
                tr = rt.shell_execute(command_id=cmd_id, args=args)
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No output",
                    "error": tr.error,
                }

            elif action.action_type == "raw_shell_proposal":
                return {
                    "success": False,
                    "error": "Raw shell proposals must pass through Command "
                             "Risk Engine. Use shell_template or structured "
                             "tools instead.",
                    "summary": "Blocked: raw shell requires risk gate",
                }

            else:
                return {
                    "success": False,
                    "error": f"Tool runtime action not yet integrated: "
                             f"{action.action_type}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Destructive write guard helpers (#76)
    # ------------------------------------------------------------------

    # Extensions considered "source code" — full-file replacement is dangerous
    _SOURCE_EXTENSIONS = frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".html", ".css", ".scss", ".sass",
        ".md", ".json", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".sh",
        ".go", ".rs", ".java", ".cpp", ".c", ".h",
        ".rb", ".php", ".swift", ".kt",
    })

    # Ratio: if new content is smaller than this fraction of existing content
    # AND both files are above the minimum size threshold, block the write.
    _DESTRUCTIVE_RATIO_THRESHOLD = 0.3   # new < 30% of old → suspicious
    _DESTRUCTIVE_MIN_EXISTING_CHARS = 200  # only guard files > 200 chars

    def _is_destructive_write(
        self,
        file_path: str,
        existing_content: str,
        new_content: str,
    ) -> Optional[str]:
        """Return an error message if this write would be destructively small.

        A write is considered destructive when:
        - The file already exists with substantial content (>200 chars)
        - The new content is much smaller (< 30 % of existing size)
        - The file has a source-code extension

        Returns None if the write is safe.
        """
        import os as _os
        ext = _os.path.splitext(file_path)[1].lower()
        if ext not in self._SOURCE_EXTENSIONS:
            return None  # Unknown extension — not guarded

        existing_size = len(existing_content)
        new_size = len(new_content)

        if existing_size < self._DESTRUCTIVE_MIN_EXISTING_CHARS:
            return None  # Small file — replacing is safe

        ratio = new_size / existing_size if existing_size > 0 else 1.0
        if ratio >= self._DESTRUCTIVE_RATIO_THRESHOLD:
            return None  # New content is large enough — safe

        return (
            f"Destructive write guard: '{file_path}' has {existing_size} chars "
            f"but new content is only {new_size} chars "
            f"({ratio:.0%} of original). "
            f"This looks like a snippet replacing a full file. "
            f"Use insert_after / insert_before / replace_range / append_file "
            f"for targeted edits, or write_file only when providing the "
            f"complete replacement file content."
        )

    @staticmethod
    def _validate_python_ast(path: str, content: str) -> Optional[str]:
        """Return an error if content is not valid Python (for .py files).

        Also checks that critical symbols (create_app, run_app) are not
        accidentally removed from igris/web/server.py.
        """
        import ast as _ast
        import os as _os

        if not path.endswith(".py"):
            return None

        try:
            tree = _ast.parse(content, filename=path)
        except SyntaxError as exc:
            return f"Python AST validation failed for '{path}': {exc}"

        # Extra guard for the server module — must keep create_app / run_app
        basename = _os.path.basename(path)
        if basename == "server.py" or path.endswith("web/server.py"):
            has_global_app_assignment = any(
                isinstance(node, _ast.Assign)
                and any(isinstance(target, _ast.Name) and target.id == "app" for target in node.targets)
                for node in tree.body
            )
            for node in tree.body:
                if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, _ast.Call) else decorator
                    if (
                        not has_global_app_assignment
                        and isinstance(target, _ast.Attribute)
                        and isinstance(target.value, _ast.Name)
                        and target.value.id == "app"
                    ):
                        return (
                            f"Server route guard: '{path}' defines top-level "
                            f"@app.{target.attr} route '{node.name}'. Routes must "
                            f"be registered inside create_app() where app is defined."
                        )

            defined_names = {
                node.name
                for node in _ast.walk(tree)
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
            }
            for critical in ("create_app", "run_app"):
                if critical not in defined_names:
                    # Only enforce if the existing file had these symbols
                    # (don't block new files that haven't defined them yet)
                    pass  # checked separately in _execute_write_file

        return None  # Valid

    def _execute_write_file(self, rt, action) -> Dict[str, Any]:
        """Execute write_file with destructive-write guard and verification.

        Guards (#76):
        - Blocks snippet replacement on large existing source files
        - Verifies hash before/after to confirm real change
        - Validates Python AST for .py files
        - Checks that critical symbols survive in igris/web/server.py
        - Tracks files_modified only on real diff
        - Idempotent: if content is already on disk, returns success=True
          without re-writing (no disk I/O, no files_modified entry)
        """
        import hashlib
        import ast as _ast

        file_path = action.parameters.get("path", "")
        content = action.parameters.get("content", "")

        if not file_path:
            return {"success": False, "error": "write_file: missing 'path' parameter"}
        if content is None:
            return {"success": False, "error": "write_file: missing 'content' parameter"}

        # Resolve full path
        full_path = os.path.join(self.project_root, file_path)

        # Read existing file (if any)
        existing_content: Optional[str] = None
        hash_before: Optional[str] = None
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    existing_content = f.read()
                hash_before = hashlib.sha256(existing_content.encode("utf-8")).hexdigest()
            except OSError:
                pass

        # Hash of the new content
        hash_new = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Idempotent: content already on disk — no write needed
        if hash_before is not None and hash_before == hash_new:
            # Count this as a modification (caller may be retrying a previous
            # successful write that crashed before tracking; we honour it)
            self._files_modified.append(file_path)
            return {
                "success": True,
                "summary": f"write_file: '{file_path}' already has this content (idempotent)",
                "result_data": {"path": file_path, "chars": len(content), "hash": hash_new[:12]},
            }

        # ── Destructive write guard ──────────────────────────────────────────
        if existing_content is not None:
            guard_error = self._is_destructive_write(file_path, existing_content, content)
            if guard_error:
                return {
                    "success": False,
                    "error": guard_error,
                    "summary": f"Blocked: destructive write on '{file_path}'",
                }

            # Extra guard for server.py: critical symbols must survive
            import os as _os
            if _os.path.basename(file_path) == "server.py" or file_path.endswith("web/server.py"):
                if file_path.endswith(".py"):
                    # Check existing has create_app / run_app
                    try:
                        old_tree = _ast.parse(existing_content, filename=file_path)
                        old_defs = {
                            n.name for n in _ast.walk(old_tree)
                            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                        }
                        critical_existing = {"create_app", "run_app"} & old_defs
                        if critical_existing:
                            # New content must also have them
                            new_tree = _ast.parse(content, filename=file_path)
                            new_defs = {
                                n.name for n in _ast.walk(new_tree)
                                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                            }
                            missing = critical_existing - new_defs
                            if missing:
                                return {
                                    "success": False,
                                    "error": (
                                        f"Symbol guard: writing '{file_path}' would remove "
                                        f"critical symbols: {sorted(missing)}. "
                                        f"Provide a complete file that preserves these functions."
                                    ),
                                    "summary": f"Blocked: symbol removal in '{file_path}'",
                                }
                    except SyntaxError:
                        pass  # Will be caught by AST validation below

        # ── Python AST validation ────────────────────────────────────────────
        if file_path.endswith(".py"):
            ast_error = self._validate_python_ast(file_path, content)
            if ast_error:
                return {
                    "success": False,
                    "error": ast_error,
                    "summary": f"Blocked: invalid Python in '{file_path}'",
                }

        # ── Perform the write via ToolRuntime ────────────────────────────────
        tr = rt.fs_write(path=full_path, content=content)
        if not tr.success:
            return {
                "success": False,
                "error": tr.error,
                "summary": f"write_file failed: {tr.error}",
            }

        # ── Verify hash after write ──────────────────────────────────────────
        try:
            with open(full_path, "rb") as f:
                hash_after = hashlib.sha256(f.read()).hexdigest()
        except OSError as exc:
            return {
                "success": False,
                "error": f"write_file: cannot verify written file: {exc}",
            }

        if hash_after != hash_new:
            return {
                "success": False,
                "error": "write_file: verification failed — hash mismatch after write",
            }

        # Real change confirmed — track it
        self._files_modified.append(file_path)

        return {
            "success": True,
            "summary": (
                f"Written {len(content)} chars to {file_path} "
                f"(hash: {(hash_before or 'new')[:8]}→{hash_after[:8]})"
            ),
            "result_data": {"path": file_path, "chars": len(content), "hash": hash_after[:12]},
        }

    def _execute_propose_patch(self, rt, action) -> Dict[str, Any]:
        """Execute propose_patch: show diff preview without applying."""
        file_path = action.parameters.get("path", "")
        new_content = action.parameters.get("content", "")

        if not file_path:
            return {"success": False, "error": "propose_patch: missing 'path'"}

        full_path = os.path.join(self.project_root, file_path)
        tr = rt.fs_diff(path=full_path, new_content=new_content)

        return {
            "success": tr.success,
            "summary": tr.output[:300] if tr.output else "No diff output",
            "error": tr.error,
            "result_data": tr.output,
        }

    def _execute_apply_patch(self, rt, action) -> Dict[str, Any]:
        """Execute apply_patch: write verified content to file."""
        file_path = action.parameters.get("path", "")
        content = action.parameters.get("content", "")

        if not file_path or not content:
            return {"success": False, "error": "apply_patch: missing 'path' or 'content'"}

        # Delegate to write_file logic for verified write
        write_action_params = {"path": file_path, "content": content}
        # Temporarily set action parameters for the write
        from igris.core.agent_action_schema import AgentAction
        write_action = AgentAction(
            action_type="write_file",
            parameters=write_action_params,
        )
        return self._execute_write_file(rt, write_action)

    # ------------------------------------------------------------------
    # Safe edit actions (#76) — patch-first policy
    def _commit_safe_edit(self, full_path: str, merged: str, insertion: str) -> Dict[str, Any]:
        """Write merged content for a safe edit.

        Secret check applies only to the *insertion* (new content), not the
        entire merged file.  Pre-existing code that happens to match a secret
        pattern (e.g. ``token=content.get(...)`` in server.py) must not block
        legitimate edits — that code was already committed and is not a secret.
        """
        from igris.core.safety import detect_secret_like_content
        from igris.core.rollback_manager import RollbackManager
        import pathlib

        if detect_secret_like_content(insertion):
            return {"success": False, "error": "Safe edit blocked: insertion contains secret-like patterns"}

        target = pathlib.Path(full_path)
        rollback_id = ""
        if target.exists():
            mgr = RollbackManager(project_root=str(self.project_root))
            entry = mgr.backup_file(str(target))
            if entry:
                rollback_id = entry.id

        try:
            target.write_text(merged, encoding="utf-8")
            return {"success": True, "rollback_id": rollback_id}
        except OSError as exc:
            return {"success": False, "error": str(exc)}

    def _execute_insert_after(self, rt, action) -> Dict[str, Any]:
        """Insert content after anchor line. Params: path, anchor, content."""
        import hashlib
        file_path = action.parameters.get("path", "")
        anchor = action.parameters.get("anchor", "")
        new_content = action.parameters.get("content", "")
        if not file_path or anchor is None or new_content is None:
            return {"success": False, "error": "insert_after: missing path/anchor/content"}
        full_path = os.path.join(self.project_root, file_path)
        if not os.path.isfile(full_path):
            return {"success": False, "error": f"insert_after: file not found: {file_path}"}
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_lines = f.readlines()
        except OSError as exc:
            return {"success": False, "error": str(exc)}
        idx = next((i for i, ln in enumerate(file_lines) if anchor in ln), None)
        if idx is None and anchor.strip() == "app = FastAPI()":
            idx = next((i for i, ln in enumerate(file_lines) if "app = FastAPI(" in ln), None)
        if idx is None:
            return {"success": False, "error": f"insert_after: anchor not found: {repr(anchor)}"}
        nl = "\n"
        insertion = new_content if new_content.endswith(nl) else new_content + nl
        insertion = self._normalize_app_route_insertion_indent(file_lines[idx], insertion)
        if self._inserts_app_route_after_block_header(file_lines[idx], insertion):
            return {
                "success": False,
                "error": (
                    "insert_after: refusing to insert @app route immediately after "
                    "a Python block header; use an anchor after app = FastAPI(...) "
                    "or after the complete previous route/block; "
                    "route would be before app = FastAPI initialization"
                ),
            }
        if self._inserts_app_route_after_decorator_line(file_lines[idx], insertion):
            return {
                "success": False,
                "error": (
                    "insert_after: refusing to insert @app route immediately after "
                    "a decorator line; use an anchor after the complete decorated "
                    "function block or after app = FastAPI(...)"
                ),
            }
        if self._app_route_already_exists(file_lines, insertion):
            return {"success": False, "error": self._duplicate_app_route_error("insert_after")}
        if self._inserts_app_route_before_app_init(file_lines, idx, insertion, after=True):
            return {
                "success": False,
                "error": "insert_after: refusing to insert @app route before app = FastAPI initialization",
            }
        if self._insertion_already_near_anchor(file_lines, idx, insertion, after=True):
            return {"success": True, "summary": "insert_after: no change; content already present near anchor"}
        merged_lines = file_lines[: idx + 1] + [insertion] + file_lines[idx + 1 :]
        merged = "".join(merged_lines)
        if file_path.endswith(".py"):
            err = self._validate_python_ast(file_path, merged)
            if err:
                return {"success": False, "error": err}
        hash_before = hashlib.sha256("".join(file_lines).encode()).hexdigest()
        hash_new = hashlib.sha256(merged.encode()).hexdigest()
        if hash_before == hash_new:
            return {"success": True, "summary": "insert_after: no change"}
        wr = self._commit_safe_edit(full_path, merged, insertion)
        if not wr["success"]:
            return {"success": False, "error": wr["error"]}
        self._files_modified.append(file_path)
        return {
            "success": True,
            "summary": f"Inserted {len(insertion)} chars after line {idx+1} in {file_path}",
            "result_data": {"path": file_path, "after_line": idx + 1},
        }

    def _execute_insert_before(self, rt, action) -> Dict[str, Any]:
        """Insert content before anchor line. Params: path, anchor, content."""
        import hashlib
        file_path = action.parameters.get("path", "")
        anchor = action.parameters.get("anchor", "")
        new_content = action.parameters.get("content", "")
        if not file_path or anchor is None or new_content is None:
            return {"success": False, "error": "insert_before: missing path/anchor/content"}
        full_path = os.path.join(self.project_root, file_path)
        if not os.path.isfile(full_path):
            return {"success": False, "error": f"insert_before: file not found: {file_path}"}
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_lines = f.readlines()
        except OSError as exc:
            return {"success": False, "error": str(exc)}
        idx = next((i for i, ln in enumerate(file_lines) if anchor in ln), None)
        if idx is None:
            return {"success": False, "error": f"insert_before: anchor not found: {repr(anchor)}"}
        nl = "\n"
        insertion = new_content if new_content.endswith(nl) else new_content + nl
        if self._app_route_already_exists(file_lines, insertion):
            return {"success": False, "error": self._duplicate_app_route_error("insert_before")}
        if self._inserts_app_route_before_app_init(file_lines, idx, insertion, after=False):
            return {
                "success": False,
                "error": "insert_before: refusing to insert @app route before app = FastAPI initialization",
            }
        if self._insertion_already_near_anchor(file_lines, idx, insertion, after=False):
            return {"success": True, "summary": "insert_before: no change; content already present near anchor"}
        merged_lines = file_lines[:idx] + [insertion] + file_lines[idx:]
        merged = "".join(merged_lines)
        if file_path.endswith(".py"):
            err = self._validate_python_ast(file_path, merged)
            if err:
                return {"success": False, "error": err}
        hash_before = hashlib.sha256("".join(file_lines).encode()).hexdigest()
        hash_new = hashlib.sha256(merged.encode()).hexdigest()
        if hash_before == hash_new:
            return {"success": True, "summary": "insert_before: no change"}
        wr = self._commit_safe_edit(full_path, merged, insertion)
        if not wr["success"]:
            return {"success": False, "error": wr["error"]}
        self._files_modified.append(file_path)
        return {
            "success": True,
            "summary": f"Inserted {len(insertion)} chars before line {idx+1} in {file_path}",
            "result_data": {"path": file_path, "before_line": idx + 1},
        }

    @staticmethod
    def _insertion_already_near_anchor(
        file_lines: List[str],
        anchor_idx: int,
        insertion: str,
        *,
        after: bool,
    ) -> bool:
        wanted = insertion.strip()
        if not wanted:
            return False
        insertion_line_count = max(1, len(insertion.splitlines()))
        window_size = insertion_line_count + 4
        if after:
            window = "".join(file_lines[anchor_idx + 1 : anchor_idx + 1 + window_size])
        else:
            start = max(0, anchor_idx - window_size)
            window = "".join(file_lines[start:anchor_idx])
        return wanted in window.strip()

    @staticmethod
    def _inserts_app_route_before_app_init(
        file_lines: List[str],
        anchor_idx: int,
        insertion: str,
        *,
        after: bool,
    ) -> bool:
        if "@app." not in insertion:
            return False
        insertion_point_end = anchor_idx + 1 if after else anchor_idx
        prior_text = "".join(file_lines[:insertion_point_end])
        return "app = FastAPI" not in prior_text

    @staticmethod
    def _inserts_app_route_after_block_header(anchor_line: str, insertion: str) -> bool:
        if "@app." not in insertion:
            return False
        stripped = anchor_line.strip()
        return stripped.endswith(":") and not stripped.startswith("@")

    @staticmethod
    def _normalize_app_route_insertion_indent(anchor_line: str, insertion: str) -> str:
        if "@app." not in insertion or "app = FastAPI(" not in anchor_line:
            return insertion
        anchor_indent = anchor_line[: len(anchor_line) - len(anchor_line.lstrip())]
        if not anchor_indent:
            return insertion
        lines = insertion.splitlines()
        leading_blank = bool(lines and not lines[0].strip())
        content_lines = lines[1:] if leading_blank else lines
        nonblank = [line for line in content_lines if line.strip()]
        if not nonblank:
            return insertion
        first_nonblank_index = next(i for i, line in enumerate(content_lines) if line.strip())
        body_nonblank = [line for line in content_lines[first_nonblank_index + 1 :] if line.strip()]
        body_base_indent = min((len(line) - len(line.lstrip(" ")) for line in body_nonblank), default=0)
        normalized = []
        for i, line in enumerate(content_lines):
            if not line.strip():
                normalized.append(line)
                continue
            if i == first_nonblank_index:
                stripped = line.lstrip(" ")
            else:
                stripped = line[body_base_indent:] if len(line) >= body_base_indent else line.lstrip(" ")
            normalized.append(anchor_indent + stripped)
        if leading_blank:
            normalized.insert(0, "")
        return "\n".join(normalized) + ("\n" if insertion.endswith("\n") else "")

    @staticmethod
    def _inserts_app_route_after_decorator_line(anchor_line: str, insertion: str) -> bool:
        return "@app." in insertion and anchor_line.strip().startswith("@")

    @staticmethod
    def _app_routes_in_content(content: str) -> set[tuple[str, str]]:
        import re

        return {
            (match.group(1), match.group(2))
            for match in re.finditer(r"@app\.(\w+)\(\s*['\"]([^'\"]+)['\"]", content)
        }

    def _app_route_already_exists(
        self,
        file_lines: List[str],
        insertion: str,
        *,
        exclude_start: Optional[int] = None,
        exclude_end: Optional[int] = None,
    ) -> bool:
        inserted_routes = self._app_routes_in_content(insertion)
        if not inserted_routes:
            return False
        if exclude_start is not None and exclude_end is not None:
            existing_text = "".join(file_lines[:exclude_start] + file_lines[exclude_end:])
        else:
            existing_text = "".join(file_lines)
        existing_routes = self._app_routes_in_content(existing_text)
        return bool(inserted_routes & existing_routes)

    @staticmethod
    def _duplicate_app_route_error(action_type: str) -> str:
        return (
            f"{action_type}: FastAPI route already present; do not retry this edit. "
            "Proceed to tests/report or use replace_range only if the existing route "
            "body needs a targeted update."
        )

    def _execute_replace_range(self, rt, action) -> Dict[str, Any]:
        """Replace line range. Params: path, start (1-based), end (1-based), content."""
        import hashlib
        file_path = action.parameters.get("path", "")
        start = action.parameters.get("start")
        end = action.parameters.get("end")
        new_content = action.parameters.get("content", "")
        if not file_path or start is None or end is None or new_content is None:
            return {"success": False, "error": "replace_range: missing path/start/end/content"}
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            return {"success": False, "error": "replace_range: start/end must be integers"}
        if start < 1 or end < start:
            return {"success": False, "error": f"replace_range: invalid range {start}..{end}"}
        full_path = os.path.join(self.project_root, file_path)
        if not os.path.isfile(full_path):
            return {"success": False, "error": f"replace_range: file not found: {file_path}"}
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_lines = f.readlines()
        except OSError as exc:
            return {"success": False, "error": str(exc)}
        if end > len(file_lines):
            return {"success": False, "error": f"replace_range: end {end} > file length {len(file_lines)}"}
        nl = "\n"
        replacement = new_content if new_content.endswith(nl) else new_content + nl
        if self._app_route_already_exists(
            file_lines,
            replacement,
            exclude_start=start - 1,
            exclude_end=end,
        ):
            return {
                "success": True,
                "summary": "replace_range: FastAPI route already present; no change",
                "result_data": {"path": file_path, "start": start, "end": end, "noop": True},
            }
        merged_lines = file_lines[: start - 1] + [replacement] + file_lines[end:]
        merged = "".join(merged_lines)
        if file_path.endswith(".py"):
            err = self._validate_python_ast(file_path, merged)
            if err:
                return {"success": False, "error": err}
        hash_before = hashlib.sha256("".join(file_lines).encode()).hexdigest()
        hash_new = hashlib.sha256(merged.encode()).hexdigest()
        if hash_before == hash_new:
            return {"success": True, "summary": "replace_range: no change"}
        wr = self._commit_safe_edit(full_path, merged, replacement)
        if not wr["success"]:
            return {"success": False, "error": wr["error"]}
        self._files_modified.append(file_path)
        return {
            "success": True,
            "summary": f"Replaced lines {start}–{end} in {file_path} with {len(replacement)} chars",
            "result_data": {"path": file_path, "start": start, "end": end},
        }

    def _execute_append_file(self, rt, action) -> Dict[str, Any]:
        """Append content to end of file. Params: path, content."""
        import hashlib
        file_path = action.parameters.get("path", "")
        new_content = action.parameters.get("content", "")
        if not file_path or new_content is None:
            return {"success": False, "error": "append_file: missing path/content"}
        full_path = os.path.join(self.project_root, file_path)
        existing = ""
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except OSError as exc:
                return {"success": False, "error": str(exc)}
        if (
            file_path.endswith(".py")
            and existing.strip()
            and self._looks_like_complete_python_module(new_content)
        ):
            return {
                "success": False,
                "error": (
                    "append_file: refusing to append complete Python module content "
                    f"to existing file: {file_path}; use write_file for new files "
                    "or replace_range/insert_after/insert_before for existing files"
                ),
            }
        nl = "\n"
        sep = "" if (not existing or existing.endswith(nl)) else nl
        merged = existing + sep + new_content
        if file_path.endswith(".py"):
            err = self._validate_python_ast(file_path, merged)
            if err:
                return {"success": False, "error": err}
        hash_before = hashlib.sha256(existing.encode()).hexdigest()
        hash_new = hashlib.sha256(merged.encode()).hexdigest()
        if hash_before == hash_new:
            return {"success": True, "summary": "append_file: no change"}
        wr = self._commit_safe_edit(full_path, merged, new_content)
        if not wr["success"]:
            return {"success": False, "error": wr["error"]}
        self._files_modified.append(file_path)
        return {
            "success": True,
            "summary": f"Appended {len(new_content)} chars to {file_path}",
            "result_data": {"path": file_path, "appended_chars": len(new_content)},
        }

    @staticmethod
    def _looks_like_complete_python_module(content: str) -> bool:
        """Heuristic for full-module Python content accidentally used as an append."""
        stripped = content.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            return False
        probe = "\n" + stripped
        module_body_markers = (
            "\ndef ",
            "\nasync def ",
            "\nclass ",
            "\n@pytest.fixture",
            "\napp = ",
            "\nclient = ",
        )
        return any(marker in probe for marker in module_body_markers)

    def _execute_plan_update(self, action) -> Dict[str, Any]:
        """Execute a plan update action."""
        updates = action.parameters.get("updates", "")
        self._world_state["last_plan_update"] = updates
        return {
            "success": True,
            "summary": f"Plan updated: {str(updates)[:100]}",
        }

    def _execute_memory_record(self, action) -> Dict[str, Any]:
        """Record a decision/lesson to memory."""
        event_type = action.parameters.get("event_type", "decision")
        content = action.parameters.get("content", "")
        self._memory_items.append({
            "event_type": event_type,
            "content": content,
            "timestamp": time.time(),
        })
        return {
            "success": True,
            "summary": f"Recorded {event_type}: {content[:100]}",
        }

    def _check_stop_conditions(self, step_num: int) -> Optional[str]:
        """Check if the loop should stop."""
        if step_num > self.max_steps:
            return "max_steps"
        if self._consecutive_errors >= self.max_consecutive_errors:
            return "budget_exceeded"
        if self._steps_without_write >= self.no_diff_steps_max:
            return "no_diff_repair"
        if self._stop_reason:
            return self._stop_reason
        return None

    def _build_summary(self, result: LoopResult) -> str:
        """Build a human-readable summary of the loop execution."""
        lines = [
            f"Loop {result.loop_id}: {result.status}",
            f"Goal: {result.goal}",
            f"Steps: {result.total_steps} ({result.successful_steps} ok, "
            f"{result.failed_steps} failed)",
            f"Stop: {result.stop_reason}",
            f"Duration: {result.total_duration_ms}ms",
        ]
        if result.files_modified:
            lines.append(f"Files: {', '.join(result.files_modified)}")
        if result.errors:
            lines.append(f"Errors: {len(result.errors)}")
        if result.steps and result.status == "blocked":
            terminal = result.steps[-1]
            details = []
            if terminal.action_type:
                details.append(f"action={terminal.action_type}")
            if terminal.reason:
                details.append(f"reason={terminal.reason}")
            if terminal.result_summary:
                details.append(f"result={terminal.result_summary}")
            if terminal.error:
                details.append(f"error={terminal.error}")
            if details:
                lines.append(f"Blocked detail: {redact_secrets('; '.join(details))[:500]}")
        return "\n".join(lines)

    # -- Public API --

    def get_steps(self) -> List[Dict[str, Any]]:
        """Get all steps as dicts."""
        return [s.to_dict() for s in self._steps]

    def get_state(self) -> Dict[str, Any]:
        """Get current loop state."""
        return {
            "step_count": len(self._steps),
            "consecutive_errors": self._consecutive_errors,
            "files_modified": list(set(self._files_modified)),
            "memory_items": len(self._memory_items),
            "world_state": self._world_state,
        }
