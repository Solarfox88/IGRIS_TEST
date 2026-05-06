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
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
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

        if initial_context:
            self._world_state.update(initial_context)

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

            # 6. Track file modifications
            if action.action_type in ("write_file", "apply_patch"):
                file_path = action.parameters.get("path", "")
                if file_path:
                    self._files_modified.append(file_path)

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
        """Execute a code navigation action."""
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

            return {
                "success": result.success,
                "summary": f"{action.action_type}: {result.total_count} results",
                "error": result.error or "",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_tool_runtime(self, action) -> Dict[str, Any]:
        """Execute a tool runtime action.

        Uses the existing ToolRuntime from Epic #41 for governed execution.
        """
        try:
            if action.action_type == "git_status":
                from igris.core.tool_runtime import ToolRuntime
                rt = ToolRuntime()
                tr = rt.execute("git", "status")
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No output",
                    "error": tr.error,
                }
            elif action.action_type == "git_diff":
                from igris.core.tool_runtime import ToolRuntime
                rt = ToolRuntime()
                tr = rt.execute("git", "diff")
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No diff",
                    "error": tr.error,
                }
            elif action.action_type == "run_tests":
                from igris.core.tool_runtime import ToolRuntime
                rt = ToolRuntime()
                tr = rt.execute("shell", "run", command="python -m pytest -q --tb=short")
                return {
                    "success": tr.success,
                    "summary": tr.output[:500] if tr.output else "No output",
                    "error": tr.error,
                }
            elif action.action_type == "http_check":
                from igris.core.tool_runtime import ToolRuntime
                rt = ToolRuntime()
                tr = rt.execute("http", "check", url=action.parameters.get("url", ""))
                return {
                    "success": tr.success,
                    "summary": tr.output[:200] if tr.output else "No response",
                    "error": tr.error,
                }
            elif action.action_type in ("write_file", "propose_patch", "apply_patch"):
                # File modifications accepted — tracked for report
                return {
                    "success": True,
                    "summary": f"Action {action.action_type} accepted for "
                               f"path={action.parameters.get('path', 'unknown')}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Tool runtime action not yet integrated: {action.action_type}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
