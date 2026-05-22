"""GOAP-like Planner for IGRIS_GPT — Epic #43.

Goal-Oriented Action Planning based on state, preconditions, effects,
risk, cost, and success criteria. Replans after failure. Works without
LLM via deterministic fallback; LLM output must be schema-validated.

Key concepts:
    WorldState   — current state of the repo/environment
    GOAPAction   — action with preconditions, effects, risk, cost
    GOAPPlan     — ordered sequence of actions to reach a goal
    GOAPPlanner  — generates/validates plans, supports replanning
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# World State
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """Observable state of the environment.

    Each property is a string key with a known value.
    """
    properties: Dict[str, Any] = field(default_factory=dict)

    # Standard state keys
    REPO_CLEAN = "repo_clean"          # bool
    TESTS_PASS = "tests_pass"          # bool | "unknown"
    SERVICE_RUNNING = "service_running" # bool | "unknown"
    DOCKER_AVAILABLE = "docker_available"
    NGINX_AVAILABLE = "nginx_available"
    DOMAIN_RESOLVES = "domain_resolves"
    SSL_PRESENT = "ssl_present"
    PROVIDER_AVAILABLE = "provider_available"
    BUDGET_OK = "budget_ok"

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def satisfies(self, conditions: Dict[str, Any]) -> bool:
        """Check if all conditions are met by current state."""
        for k, v in conditions.items():
            if self.properties.get(k) != v:
                return False
        return True

    def apply_effects(self, effects: Dict[str, Any]) -> "WorldState":
        """Return new state with effects applied."""
        new_props = dict(self.properties)
        new_props.update(effects)
        return WorldState(properties=new_props)

    def to_dict(self) -> Dict[str, Any]:
        return {"properties": dict(self.properties)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldState":
        return cls(properties=data.get("properties", {}))

    @classmethod
    def default_state(cls) -> "WorldState":
        """Create a conservative default state."""
        return cls(properties={
            cls.REPO_CLEAN: "unknown",
            cls.TESTS_PASS: "unknown",
            cls.SERVICE_RUNNING: "unknown",
            cls.DOCKER_AVAILABLE: "unknown",
            cls.NGINX_AVAILABLE: "unknown",
            cls.DOMAIN_RESOLVES: "unknown",
            cls.SSL_PRESENT: "unknown",
            cls.PROVIDER_AVAILABLE: "unknown",
            cls.BUDGET_OK: True,
        })


# ---------------------------------------------------------------------------
# GOAP Action
# ---------------------------------------------------------------------------

ACTION_FAMILIES = (
    "observation", "synthesis", "repo_diff_discovery", "patch_strategy",
    "branch_pr_plan", "review_gate", "candidate_materialization",
    "mastery_cycle", "mastery_gate", "school_report", "grading_diagnosis",
    "stabilization_audit", "devops_deploy", "server_diagnosis",
    "test_repair", "code_patch", "documentation", "security_audit", "other",
)


@dataclass
class GOAPAction:
    """An action in the GOAP planner."""
    id: str = field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    title: str = ""
    family: str = "other"
    description: str = ""
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    risk: str = "low"
    cost: float = 1.0
    required_tools: List[str] = field(default_factory=list)
    requires_rollback: bool = False
    success_criteria: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    cooldown_family: str = ""
    blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "family": self.family,
            "description": self.description,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "risk": self.risk,
            "cost": self.cost,
            "required_tools": self.required_tools,
            "requires_rollback": self.requires_rollback,
            "success_criteria": self.success_criteria,
            "failure_modes": self.failure_modes,
            "cooldown_family": self.cooldown_family or self.family,
            "blocked": self.blocked,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GOAPAction":
        return cls(
            id=data.get("id", f"act-{uuid.uuid4().hex[:8]}"),
            title=data.get("title", ""),
            family=data.get("family", "other"),
            description=data.get("description", ""),
            preconditions=data.get("preconditions", {}),
            effects=data.get("effects", {}),
            risk=data.get("risk", "low"),
            cost=data.get("cost", 1.0),
            required_tools=data.get("required_tools", []),
            requires_rollback=data.get("requires_rollback", False),
            success_criteria=data.get("success_criteria", []),
            failure_modes=data.get("failure_modes", []),
            cooldown_family=data.get("cooldown_family", ""),
            blocked=data.get("blocked", False),
        )


# ---------------------------------------------------------------------------
# GOAP Plan
# ---------------------------------------------------------------------------

@dataclass
class GOAPPlan:
    """An ordered plan of actions."""
    id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    mission_id: str = ""
    goal: Dict[str, Any] = field(default_factory=dict)
    actions: List[GOAPAction] = field(default_factory=list)
    initial_state: Optional[WorldState] = None
    current_step: int = 0
    status: str = "created"  # created | executing | replanning | done | failed
    total_cost: float = 0.0
    replan_count: int = 0
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "goal": self.goal,
            "actions": [a.to_dict() for a in self.actions],
            "initial_state": self.initial_state.to_dict() if self.initial_state else None,
            "current_step": self.current_step,
            "status": self.status,
            "total_cost": self.total_cost,
            "replan_count": self.replan_count,
            "action_count": len(self.actions),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GOAPPlan":
        actions = [GOAPAction.from_dict(a) for a in data.get("actions", [])]
        initial = WorldState.from_dict(data["initial_state"]) if data.get("initial_state") else None
        return cls(
            id=data.get("id", f"plan-{uuid.uuid4().hex[:8]}"),
            mission_id=data.get("mission_id", ""),
            goal=data.get("goal", {}),
            actions=actions,
            initial_state=initial,
            current_step=data.get("current_step", 0),
            status=data.get("status", "created"),
            total_cost=data.get("total_cost", 0.0),
            replan_count=data.get("replan_count", 0),
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Standard action library (deterministic)
# ---------------------------------------------------------------------------

def _standard_actions() -> List[GOAPAction]:
    """Built-in library of standard GOAP actions."""
    return [
        GOAPAction(
            id="analyze-repo", title="Analyze repository state",
            family="observation", cost=1.0, risk="low",
            preconditions={}, effects={"repo_analyzed": True},
            success_criteria=["Git status checked", "File structure known"],
            required_tools=["git", "filesystem"],
        ),
        GOAPAction(
            id="run-tests", title="Run test suite",
            family="observation", cost=2.0, risk="low",
            preconditions={"repo_analyzed": True},
            effects={"tests_pass": True},
            success_criteria=["pytest returns 0"],
            required_tools=["test"],
        ),
        GOAPAction(
            id="identify-target", title="Identify target files",
            family="synthesis", cost=1.0, risk="low",
            preconditions={"repo_analyzed": True},
            effects={"target_identified": True},
            success_criteria=["Target files listed"],
            required_tools=["filesystem"],
        ),
        GOAPAction(
            id="generate-patch", title="Generate patch proposal",
            family="code_patch", cost=3.0, risk="medium",
            preconditions={"target_identified": True},
            effects={"patch_ready": True},
            success_criteria=["Patch generated", "Diff preview available"],
            required_tools=["filesystem"],
        ),
        GOAPAction(
            id="validate-patch", title="Validate patch proposal",
            family="review_gate", cost=1.0, risk="low",
            preconditions={"patch_ready": True},
            effects={"patch_validated": True},
            success_criteria=["No secrets in patch", "No binary files"],
            required_tools=["filesystem"],
        ),
        GOAPAction(
            id="apply-patch", title="Apply validated patch",
            family="code_patch", cost=2.0, risk="medium",
            preconditions={"patch_validated": True},
            effects={"patch_applied": True},
            success_criteria=["Files modified successfully"],
            required_tools=["filesystem"],
            requires_rollback=True,
        ),
        GOAPAction(
            id="run-tests-after", title="Run tests after changes",
            family="test_repair", cost=2.0, risk="low",
            preconditions={"patch_applied": True},
            effects={"tests_pass_after": True},
            success_criteria=["All tests pass"],
            required_tools=["test"],
        ),
        GOAPAction(
            id="update-docs", title="Update documentation",
            family="documentation", cost=1.0, risk="low",
            preconditions={"patch_applied": True},
            effects={"docs_updated": True},
            success_criteria=["Docs reflect changes"],
            required_tools=["filesystem"],
        ),
        GOAPAction(
            id="prepare-commit", title="Prepare git commit",
            family="branch_pr_plan", cost=1.0, risk="medium",
            preconditions={"tests_pass_after": True},
            effects={"commit_ready": True},
            success_criteria=["Commit message clear", "No secret files staged"],
            required_tools=["git"],
        ),
        GOAPAction(
            id="generate-report", title="Generate decision report",
            family="school_report", cost=1.0, risk="low",
            preconditions={"commit_ready": True},
            effects={"report_generated": True},
            success_criteria=["Report contains steps, outcomes, decisions"],
            required_tools=[],
        ),
    ]


# ---------------------------------------------------------------------------
# GOAP Planner
# ---------------------------------------------------------------------------

class GOAPPlanner:
    """GOAP-like planner that generates plans from state and goals.

    Plans are deterministic by default. LLM output, if used, must be
    schema-validated via validate_llm_plan().
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        action_library: Optional[List[GOAPAction]] = None,
    ):
        import os
        self.project_root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
        self._actions = action_library or _standard_actions()
        self._saturated_families: Dict[str, int] = {}
        self._blocked_actions: Set[str] = set()
        self._recent_families: List[str] = []

    # -- State --

    def get_current_state(self) -> WorldState:
        """Build world state from current environment."""
        state = WorldState.default_state()

        # Check git
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.project_root), capture_output=True, text=True, timeout=5,
            )
            state.set(WorldState.REPO_CLEAN, result.stdout.strip() == "")
        except Exception:
            state.set(WorldState.REPO_CLEAN, "unknown")

        return state

    # -- Action eligibility --

    def get_eligible_actions(self, state: WorldState) -> List[GOAPAction]:
        """Return actions whose preconditions are met and not blocked/saturated."""
        eligible: List[GOAPAction] = []
        for action in self._actions:
            if action.blocked:
                continue
            if action.id in self._blocked_actions:
                continue
            if not state.satisfies(action.preconditions):
                continue
            # Check family saturation
            family = action.cooldown_family or action.family
            if self._saturated_families.get(family, 0) >= 3:
                continue
            eligible.append(action)
        return eligible

    def _score_action(self, action: GOAPAction, goal: Dict[str, Any], state: WorldState) -> float:
        """Score an action based on how much it progresses toward goal."""
        score = 0.0
        for k, v in action.effects.items():
            if k in goal and goal[k] == v:
                score += 10.0
        # Penalize cost
        score -= action.cost
        # Penalize saturated families
        family = action.cooldown_family or action.family
        recent_count = self._recent_families.count(family)
        score -= recent_count * 2.0
        # Penalize high risk
        risk_penalty = {"low": 0, "medium": 1, "high": 3, "critical": 10}
        score -= risk_penalty.get(action.risk, 5)
        return score

    # -- Plan generation --

    def generate_plan(
        self,
        goal: Dict[str, Any],
        state: Optional[WorldState] = None,
        mission_id: str = "",
        max_steps: int = 20,
    ) -> GOAPPlan:
        """Generate a plan to achieve goal from current state.

        Uses forward chaining: starting from current state, greedily
        select actions that progress toward goal. Falls back to standard
        action sequence if no goal-specific plan can be found.
        """
        if state is None:
            state = self.get_current_state()

        plan = GOAPPlan(mission_id=mission_id, goal=goal, initial_state=state)
        current_state = WorldState(properties=dict(state.properties))
        used_actions: Set[str] = set()

        for _ in range(max_steps):
            if current_state.satisfies(goal):
                break

            eligible = self.get_eligible_actions(current_state)
            eligible = [a for a in eligible if a.id not in used_actions]

            if not eligible:
                # Fallback: use standard sequential plan
                break
            try:
                import os
                from igris.core.memory_graph import MemoryGraph
                graph = MemoryGraph(os.environ.get("PROJECT_ROOT", "."))
                for action in eligible:
                    history = graph.get_action_history(
                        goal_type=goal.get("type", ""),
                        action_family=action.family,
                    )
                    failed_count = sum(1 for h in history if h.get("content", {}).get("outcome") == "failure")
                    if failed_count >= 2:
                        action.cost = action.cost * 1.5
            except Exception:
                pass

            # Score and select best action
            scored = sorted(eligible, key=lambda a: self._score_action(a, goal, current_state), reverse=True)
            best = scored[0]
            plan.actions.append(best)
            plan.total_cost += best.cost
            used_actions.add(best.id)
            current_state = current_state.apply_effects(best.effects)

        # If plan is empty, use standard fallback sequence
        if not plan.actions:
            plan.actions = self._fallback_plan(goal, state)
            plan.total_cost = sum(a.cost for a in plan.actions)

        plan.status = "created"
        return plan

    def _fallback_plan(self, goal: Dict[str, Any], state: WorldState) -> List[GOAPAction]:
        """Generate a safe fallback plan using standard library order."""
        actions: List[GOAPAction] = []
        current = WorldState(properties=dict(state.properties))
        for action in self._actions:
            if action.blocked or action.id in self._blocked_actions:
                continue
            if current.satisfies(action.preconditions):
                actions.append(action)
                current = current.apply_effects(action.effects)
                if current.satisfies(goal):
                    break
        return actions

    # -- Replanning --

    def replan_after_failure(
        self,
        plan: GOAPPlan,
        failed_action_id: str,
        failure_reason: str = "",
        state: Optional[WorldState] = None,
    ) -> GOAPPlan:
        """Generate a new plan after a failure.

        Blocks the failed action and generates a fresh plan.
        """
        self._blocked_actions.add(failed_action_id)

        # Track family saturation
        for a in plan.actions:
            if a.id == failed_action_id:
                family = a.cooldown_family or a.family
                self._saturated_families[family] = self._saturated_families.get(family, 0) + 1
                self._recent_families.append(family)
                break

        new_state = state or self.get_current_state()
        new_plan = self.generate_plan(plan.goal, new_state, plan.mission_id)
        new_plan.replan_count = plan.replan_count + 1
        new_plan.status = "created"
        return new_plan

    # -- LLM plan validation --

    def validate_llm_plan(self, raw_plan: Any) -> Optional[GOAPPlan]:
        """Validate and parse LLM-generated plan output.

        Returns None if plan is invalid. LLM output must be JSON with:
        - actions: list of action objects with title, family, preconditions,
          effects, risk, cost, success_criteria
        """
        if not isinstance(raw_plan, dict):
            try:
                raw_plan = json.loads(raw_plan) if isinstance(raw_plan, str) else None
            except (json.JSONDecodeError, TypeError):
                return None

        if not raw_plan or "actions" not in raw_plan:
            return None

        actions_raw = raw_plan.get("actions", [])
        if not isinstance(actions_raw, list) or len(actions_raw) == 0:
            return None

        actions: List[GOAPAction] = []
        for a_raw in actions_raw:
            if not isinstance(a_raw, dict):
                return None
            if not a_raw.get("title"):
                return None
            if not a_raw.get("success_criteria"):
                return None  # success_criteria is mandatory

            # Validate risk
            risk = a_raw.get("risk", "low")
            if risk not in ("low", "medium", "high", "critical"):
                return None

            # Validate family
            family = a_raw.get("family", "other")
            if family not in ACTION_FAMILIES:
                family = "other"

            action = GOAPAction.from_dict({**a_raw, "family": family, "risk": risk})
            actions.append(action)

        plan = GOAPPlan(
            goal=raw_plan.get("goal", {}),
            actions=actions,
            total_cost=sum(a.cost for a in actions),
        )
        return plan

    # -- Explain --

    def explain_plan(self, plan: GOAPPlan) -> Dict[str, Any]:
        """Human-readable explanation of the plan."""
        steps_info: List[Dict[str, Any]] = []
        for i, action in enumerate(plan.actions):
            steps_info.append({
                "step": i,
                "title": action.title,
                "family": action.family,
                "risk": action.risk,
                "cost": action.cost,
                "preconditions": action.preconditions,
                "effects": action.effects,
                "success_criteria": action.success_criteria,
                "requires_rollback": action.requires_rollback,
            })
        return {
            "plan_id": plan.id,
            "mission_id": plan.mission_id,
            "goal": plan.goal,
            "total_steps": len(plan.actions),
            "total_cost": plan.total_cost,
            "replan_count": plan.replan_count,
            "status": plan.status,
            "steps": steps_info,
            "saturated_families": dict(self._saturated_families),
            "blocked_actions": list(self._blocked_actions),
        }

    def explain_next_action(self, plan: GOAPPlan, state: Optional[WorldState] = None) -> Dict[str, Any]:
        """Explain why the next action was chosen."""
        if plan.current_step >= len(plan.actions):
            return {"message": "Plan complete — no more actions", "plan_id": plan.id}

        action = plan.actions[plan.current_step]
        eligible = self.get_eligible_actions(state or WorldState.default_state())
        eligible_ids = [a.id for a in eligible]

        return {
            "plan_id": plan.id,
            "step": plan.current_step,
            "action": action.to_dict(),
            "reason": f"Selected '{action.title}' (family={action.family}, risk={action.risk}, cost={action.cost})",
            "eligible_actions": len(eligible_ids),
            "family_saturation": dict(self._saturated_families),
        }

    # -- Persistence --

    def save_plan(self, plan: GOAPPlan) -> Path:
        plan_dir = self.project_root / ".igris" / "goap" / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        path = plan_dir / f"{plan.id}.json"
        path.write_text(json.dumps(plan.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def load_plan(self, plan_id: str) -> Optional[GOAPPlan]:
        path = self.project_root / ".igris" / "goap" / "plans" / f"{plan_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return GOAPPlan.from_dict(data)
        except Exception:
            return None

    def list_plans(self, mission_id: str = "") -> List[Dict[str, Any]]:
        plan_dir = self.project_root / ".igris" / "goap" / "plans"
        if not plan_dir.exists():
            return []
        plans: List[Dict[str, Any]] = []
        for fp in sorted(plan_dir.glob("plan-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if mission_id and data.get("mission_id") != mission_id:
                    continue
                plans.append(data)
            except Exception:
                continue
        return plans
