"""Agent Reasoning Loop for IGRIS_GPT — Epic #61.

The cognitive core: observe -> reason -> act -> observe -> repeat.

Loop structure:
    1. build_context (Context Manager)
    2. model_orchestrator.complete (Model Orchestrator)
    3. validate action schema (Agent Action Schema)
    4. route action to gate:
       - code_navigation -> CodeNavigator (safe, no side effects)
       - tool_runtime -> ToolRuntime (governed)
       - command_risk_engine -> risk gate (future, blocks for now)
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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


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
    "governor_stop",
    "llm_unavailable",
)

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5


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
    errors: List[str] = field(default_factory=list)
    final_summary: str = ""

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
            "errors": [redact_secrets(e) for e in self.errors],
            "final_summary": redact_secrets(self.final_summary),
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
        role: str = "coder",
    ):
        import os
        self.project_root = project_root or os.environ.get("PROJECT_ROOT", ".")
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.role = role

        # State
        self._steps: List[LoopStep] = []
        self._recent_errors: List[Dict[str, Any]] = []
        self._files_modified: List[str] = []
        self._memory_items: List[Dict[str, Any]] = []
        self._world_state: Dict[str, Any] = {}
        self._consecutive_errors = 0
        self._stop_reason = ""

        # Anti-repeat guard: tracks (action_type, params_key) -> count
        self._action_history: List[Dict[str, Any]] = []
        self._repeat_threshold = 2  # block after 2 identical successes without consumption

    def run(
        self,
        goal: str = "",
        mission_id: str = "",
        initial_context: Optional[Dict[str, Any]] = None,
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
            step = self._execute_step(step_num, goal, mission_id)
            self._steps.append(step)
            result.steps.append(step)

            # Track outcomes
            if step.outcome == "success":
                result.successful_steps += 1
                self._consecutive_errors = 0
            elif step.outcome in ("failure", "error"):
                result.failed_steps += 1
                self._consecutive_errors += 1
                result.errors.append(step.error or f"Step {step_num} failed")
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

        return result

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
        for prev in reversed(self._action_history):
            if prev.get("signature") == sig and prev.get("outcome") == "success":
                last_result_data = prev.get("result_data")
                break

        if last_result_data and self._was_result_consumed(last_result_data):
            return None

        return (
            f"Anti-repeat guard: '{action_type}' with identical parameters "
            f"already succeeded {repeat_count} time(s) without its results "
            f"being consumed by a downstream action. Strategy shift required."
        )

    def _was_result_consumed(self, result_data: Any) -> bool:
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
        for prev in self._action_history:
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
                step.outcome = "blocked"
                step.error = repeat_diagnosis
                step.result_summary = (
                    "Governor anti-repeat: identical action repeated without "
                    "consuming previous results. Use the results from the "
                    "previous execution or choose a different action."
                )
                self._world_state["anti_repeat_triggered"] = True
                self._world_state["anti_repeat_diagnosis"] = repeat_diagnosis
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

            # 4b. Store structured result data
            result_data = exec_result.get("result_data")
            if result_data is not None:
                step.result_data = result_data
                self._store_tool_result(action.action_type, result_data)

            # 4c. Record for anti-repeat tracking
            self._record_action_history(
                action.action_type,
                action.parameters,
                step.outcome,
                result_data=result_data,
            )

            if not exec_result.get("success", False):
                step.error = exec_result.get("error", "Execution failed")
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
        return step

    def _build_context(self, goal: str, mission_id: str):
        """Build context packet for the current step."""
        from igris.core.context_manager import ContextManager
        ctx = ContextManager(project_root=self.project_root)
        return ctx.build_context(
            goal=goal,
            role=self.role,
            profile="cheap_cloud_reasoning",
            mission_id=mission_id,
            world_state=self._world_state,
            recent_actions=[s.to_dict() for s in self._steps[-10:]],
            recent_errors=self._recent_errors[-5:],
            memory_items=self._memory_items,
        )

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
        messages = [
            {"role": "user", "content": self._format_user_message(context_packet)},
        ]

        orch_result = orch.complete(
            task_type="code_reasoning",
            messages=messages,
            system_prompt=system_prompt,
            json_mode=True,
        )

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
                test_args = action.parameters.get("args", [])
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

    def _execute_write_file(self, rt, action) -> Dict[str, Any]:
        """Execute write_file with real verification.

        Checks:
        - File hash before/after to detect real change
        - Returns success=False if no actual change occurred
        - Tracks files_modified only on real diff
        """
        import hashlib

        file_path = action.parameters.get("path", "")
        content = action.parameters.get("content", "")

        if not file_path:
            return {"success": False, "error": "write_file: missing 'path' parameter"}
        if not content:
            return {"success": False, "error": "write_file: missing 'content' parameter"}

        # Hash before
        hash_before = None
        full_path = os.path.join(self.project_root, file_path)
        if os.path.isfile(full_path):
            try:
                with open(full_path, "rb") as f:
                    hash_before = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                pass

        # Check if content is identical (no-op write)
        hash_new = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if hash_before == hash_new:
            return {
                "success": False,
                "error": f"write_file: content identical to existing file "
                         f"'{file_path}' — no change made. Use propose_patch "
                         f"or apply_patch for targeted edits.",
                "summary": f"No change: {file_path} already has this content",
            }

        # Perform the write via ToolRuntime (with safety checks)
        tr = rt.fs_write(path=full_path, content=content)

        if not tr.success:
            return {
                "success": False,
                "error": tr.error,
                "summary": f"write_file failed: {tr.error}",
            }

        # Verify hash after
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
            "summary": f"Written {len(content)} chars to {file_path} "
                       f"(hash changed: {(hash_before or 'new')[:8]}→{hash_after[:8]})",
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
