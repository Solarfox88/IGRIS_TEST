"""VastAI Fleet Manager — ETC-gated multi-instance lifecycle management.

Manages a pool of Vast.ai GPU instances autonomously:
- ETC-gated scale-up: open new instances only when it's cheaper than waiting
- Warm-instance preference: reuse loaded-model VMs (0-min startup)
- Phase detection: tracks agent phases (coding / test_execution / idle / stuck)
- Stuck recovery: restart Ollama then terminate+replace if still stuck
- Idle eviction: terminate warm VMs idle > max_idle_minutes

Epic: https://github.com/Solarfox88/IGRIS_GPT/issues/593

Usage:
    from igris.layers.advisory.vastai_fleet import _SHARED_FLEET

    inst = _SHARED_FLEET.acquire(issue_number=543, task_type="hard_debugging")
    if inst:
        endpoint = inst.ollama_endpoint()
    _SHARED_FLEET.release(inst.instance_id, outcome="success")
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTUP_COLD_MINUTES: float = 10.0
STARTUP_WARM_MINUTES: float = 0.0
MARGIN_MINUTES: float = 3.0
RATE_USD_PER_MIN: float = 0.30 / 60           # deepseek-r1:32b ≈ $0.005/min
DEFAULT_HISTORY_AVG_MINUTES: float = 45.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InstanceStatus(str, Enum):
    PROVISIONING = "provisioning"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    STUCK = "stuck"
    TERMINATED = "terminated"


class AgentPhase(str, Enum):
    CODING = "coding"
    TEST_EXECUTION = "test_execution"
    IDLE = "idle"
    STUCK = "stuck"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FleetPolicy:
    min_instances: int = 1
    max_instances: int = 5
    max_idle_minutes: float = 10.0
    stuck_threshold_minutes: float = 5.0
    max_hourly_cost_usd: float = 2.50
    # Default matches SUPPORTED_MODELS["deepseek-r1:32b"]["estimated_cost_hr"] in vastai_manager.py
    # Override with VASTAI_INSTANCE_HOURLY_COST env var if using a different model/bid
    instance_hourly_cost_usd: float = 0.30

    @classmethod
    def from_env(cls) -> "FleetPolicy":
        """Build policy from env vars, falling back to defaults."""
        policy = cls()
        raw_max_cost = os.environ.get("VASTAI_MAX_HOURLY_COST", "")
        if raw_max_cost:
            try:
                policy.max_hourly_cost_usd = float(raw_max_cost)
            except ValueError:
                pass
        raw_max_inst = os.environ.get("VASTAI_MAX_INSTANCES", "")
        if raw_max_inst:
            try:
                policy.max_instances = int(raw_max_inst)
            except ValueError:
                pass
        raw_inst_cost = os.environ.get("VASTAI_INSTANCE_HOURLY_COST", "")
        if raw_inst_cost:
            try:
                policy.instance_hourly_cost_usd = float(raw_inst_cost)
            except ValueError:
                pass
        return policy

@dataclass
class QueuedTask:
    issue_number: int
    task_type: str = "code_reasoning"
    queued_at: datetime = field(default_factory=datetime.utcnow)
    callback_id: str = ""  # optional correlation id


@dataclass
class FleetInstance:
    instance_id: str
    host: str
    port: int
    status: InstanceStatus = InstanceStatus.PROVISIONING
    assigned_issue: Optional[int] = None
    task_type: str = "code_reasoning"
    model_loaded: bool = False
    phase: AgentPhase = AgentPhase.IDLE
    started_at: datetime = field(default_factory=datetime.utcnow)
    task_started_at: Optional[datetime] = None
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    cost_so_far_usd: float = 0.0
    _stuck_restart_attempted: bool = False
    secondary_issue: Optional[int] = None
    secondary_task_type: str = ""

    def elapsed_task_minutes(self) -> float:
        if self.task_started_at is None:
            return 0.0
        return (datetime.utcnow() - self.task_started_at).total_seconds() / 60.0

    def estimated_etc_minutes(
        self,
        history_avg_minutes: float = DEFAULT_HISTORY_AVG_MINUTES,
        avg_total_tokens: int = 8000,
    ) -> float:
        elapsed = self.elapsed_task_minutes()
        if self.tokens_per_sec > 0 and self.tokens_generated < avg_total_tokens:
            tokens_remaining = max(0, avg_total_tokens - self.tokens_generated)
            return tokens_remaining / self.tokens_per_sec / 60.0
        remaining_frac = max(0.0, 1.0 - elapsed / max(history_avg_minutes, 1.0))
        return history_avg_minutes * remaining_frac

    def is_warm(self) -> bool:
        return self.model_loaded and self.status in (InstanceStatus.READY, InstanceStatus.IDLE)

    def ollama_endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def uptime_minutes(self) -> float:
        return (datetime.utcnow() - self.started_at).total_seconds() / 60.0

    def idle_minutes(self) -> float:
        return (datetime.utcnow() - self.last_heartbeat).total_seconds() / 60.0

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "assigned_issue": self.assigned_issue,
            "task_type": self.task_type,
            "model_loaded": self.model_loaded,
            "phase": self.phase.value,
            "elapsed_task_minutes": round(self.elapsed_task_minutes(), 1),
            "estimated_etc_minutes": round(self.estimated_etc_minutes(), 1),
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "cost_so_far_usd": round(self.cost_so_far_usd, 5),
        }


@dataclass
class FleetState:
    instances: List[FleetInstance] = field(default_factory=list)
    task_queue: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def queue_depth(self) -> int:
        return len(self.task_queue)

    def enqueue(self, task: QueuedTask) -> None:
        self.task_queue.append(task)

    def dequeue(self) -> Optional[QueuedTask]:
        return self.task_queue.popleft() if self.task_queue else None

    @property
    def busy(self) -> List[FleetInstance]:
        return [i for i in self.instances if i.status == InstanceStatus.BUSY]

    @property
    def idle(self) -> List[FleetInstance]:
        return [i for i in self.instances if i.status == InstanceStatus.IDLE]

    @property
    def warm_idle(self) -> List[FleetInstance]:
        return [i for i in self.idle if i.model_loaded]

    @property
    def stuck(self) -> List[FleetInstance]:
        return [i for i in self.instances if i.status == InstanceStatus.STUCK]

    @property
    def active(self) -> List[FleetInstance]:
        return [i for i in self.instances if i.status != InstanceStatus.TERMINATED]

    def hourly_cost_usd(self, instance_hourly_cost: float = 0.30) -> float:
        """Compute current fleet hourly cost using the provided per-instance rate."""
        return len(self.active) * instance_hourly_cost

    def get(self, instance_id: str) -> Optional[FleetInstance]:
        for inst in self.instances:
            if inst.instance_id == instance_id:
                return inst
        return None


# ---------------------------------------------------------------------------
# Scale actions
# ---------------------------------------------------------------------------

@dataclass
class ScaleWait:
    reason: str = ""


@dataclass
class ScaleAssignWarm:
    instance: FleetInstance
    reason: str = "warm_idle_available"


@dataclass
class ScaleOpenNew:
    count: int
    reason: str = ""


# ---------------------------------------------------------------------------
# FleetScheduler
# ---------------------------------------------------------------------------

class FleetScheduler:
    """Stateless decision engine — pure functions of FleetState."""

    def finishing_soon_count(self, busy: List[FleetInstance]) -> int:
        threshold = STARTUP_COLD_MINUTES + MARGIN_MINUTES
        return sum(1 for inst in busy if inst.estimated_etc_minutes() <= threshold)

    def instances_to_open(self, state: FleetState, policy: FleetPolicy) -> int:
        if state.queue_depth <= 0:
            return 0
        finishing_soon = self.finishing_soon_count(state.busy)
        uncovered_queue = max(0, state.queue_depth - finishing_soon)
        if uncovered_queue <= 0:
            return 0
        available_slots = policy.max_instances - len(state.active)
        if available_slots <= 0:
            return 0
        hourly = state.hourly_cost_usd()
        budget_headroom = policy.max_hourly_cost_usd - hourly
        if budget_headroom <= 0:
            return 0
        budget_slots = max(0, int(budget_headroom / (policy.instance_hourly_cost_usd / 60 * STARTUP_COLD_MINUTES + 1e-9)))
        return min(uncovered_queue, available_slots, budget_slots)

    def scale_decision(self, state: FleetState, policy: FleetPolicy):
        if state.queue_depth > 0 and state.warm_idle:
            best_warm = state.warm_idle[0]
            return ScaleAssignWarm(
                instance=best_warm,
                reason=f"warm_idle instance {best_warm.instance_id} available",
            )
        n = self.instances_to_open(state, policy)
        if n > 0:
            finishing = self.finishing_soon_count(state.busy)
            return ScaleOpenNew(
                count=n,
                reason=(
                    f"queue={state.queue_depth} finishing_soon={finishing} "
                    f"uncovered={state.queue_depth - finishing} -> open {n}"
                ),
            )
        if state.queue_depth > 0:
            finishing = self.finishing_soon_count(state.busy)
            return ScaleWait(
                reason=f"queue={state.queue_depth} but {finishing} instances finishing soon - wait"
            )
        return ScaleWait(reason="no queue")


# ---------------------------------------------------------------------------
# SecondarySlotScheduler
# ---------------------------------------------------------------------------

class SecondarySlotScheduler:
    """Routes lightweight tasks to GPU idle windows during TEST_EXECUTION phase.

    When an instance enters TEST_EXECUTION (pytest running, GPU=0%),
    its GPU slot is free for up to ~90 seconds. A secondary task can be
    started during this window and preempted when tests finish.
    """

    def available_slots(self, state: FleetState) -> List[FleetInstance]:
        """Instances in TEST_EXECUTION with no secondary task assigned."""
        return [
            inst for inst in state.busy
            if inst.phase == AgentPhase.TEST_EXECUTION
            and inst.secondary_issue is None
        ]

    def assign_secondary(self, inst: FleetInstance, task: QueuedTask) -> None:
        """Assign a secondary task to a GPU idle window."""
        inst.secondary_issue = task.issue_number
        inst.secondary_task_type = task.task_type

    def clear_secondary(self, inst: FleetInstance) -> Optional[int]:
        """Clear secondary assignment (primary resumed). Returns the cleared issue number."""
        issue = inst.secondary_issue
        inst.secondary_issue = None
        inst.secondary_task_type = ""
        return issue


# ---------------------------------------------------------------------------
# FleetMonitor
# ---------------------------------------------------------------------------

class FleetMonitor:
    POLL_INTERVAL_SECONDS: float = 30.0

    def __init__(self, state, policy, scheduler, provision_fn, terminate_fn, restart_ollama_fn):
        self._state = state
        self._policy = policy
        self._scheduler = scheduler
        self._provision = provision_fn
        self._terminate = terminate_fn
        self._restart_ollama = restart_ollama_fn
        self._secondary_scheduler = SecondarySlotScheduler()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="fleet-monitor")
        self._thread.start()
        _log.info("FleetMonitor started (interval=%.0fs)", self.POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop_event.wait(self.POLL_INTERVAL_SECONDS):
            try:
                self._poll_cycle()
            except Exception:
                _log.exception("FleetMonitor poll cycle error")

    def _poll_cycle(self) -> None:
        with self._state._lock:
            self._update_costs()
            self._detect_stuck()
            self._recover_stuck()
            self._evict_idle()
            self._clear_finished_secondaries()
            decision = self._scheduler.scale_decision(self._state, self._policy)
        self._apply_decision(decision)
        self._assign_secondary_slots()
        self._emit_status_log()

    def _update_costs(self) -> None:
        for inst in self._state.active:
            inst.cost_so_far_usd = inst.uptime_minutes() * RATE_USD_PER_MIN

    def _detect_stuck(self) -> None:
        threshold = self._policy.stuck_threshold_minutes
        for inst in self._state.busy:
            if inst.tokens_per_sec == 0.0 and inst.idle_minutes() > threshold:
                _log.warning("FleetMonitor: instance %s STUCK (idle %.1f min)",
                             inst.instance_id, inst.idle_minutes())
                inst.status = InstanceStatus.STUCK
                inst.phase = AgentPhase.STUCK

    def _recover_stuck(self) -> None:
        for inst in self._state.stuck:
            if not inst._stuck_restart_attempted:
                _log.info("FleetMonitor: restarting Ollama on stuck instance %s", inst.instance_id)
                inst._stuck_restart_attempted = True
                ok = self._restart_ollama(inst)
                if ok:
                    inst.status = InstanceStatus.BUSY
                    inst.phase = AgentPhase.CODING
                    inst.last_heartbeat = datetime.utcnow()
                    _log.info("FleetMonitor: Ollama restart succeeded for %s", inst.instance_id)
                else:
                    _log.warning("FleetMonitor: restart failed for %s - terminating", inst.instance_id)
                    inst.status = InstanceStatus.TERMINATED
                    if inst.assigned_issue is not None:
                        self._state.enqueue(QueuedTask(issue_number=inst.assigned_issue, task_type=inst.task_type))
            else:
                inst.status = InstanceStatus.TERMINATED
                if inst.assigned_issue is not None:
                    self._state.enqueue(QueuedTask(issue_number=inst.assigned_issue, task_type=inst.task_type))

    def _evict_idle(self) -> None:
        for inst in self._state.idle:
            if inst.idle_minutes() > self._policy.max_idle_minutes:
                _log.info("FleetMonitor: evicting idle instance %s (idle %.1f min)",
                          inst.instance_id, inst.idle_minutes())
                inst.status = InstanceStatus.TERMINATED
                self._terminate(inst.instance_id)

    def _clear_finished_secondaries(self) -> None:
        for inst in self._state.busy:
            if inst.secondary_issue is not None and inst.phase != AgentPhase.TEST_EXECUTION:
                cleared = self._secondary_scheduler.clear_secondary(inst)
                if cleared:
                    _log.info("FleetMonitor: secondary task #%d preempted (primary resumed)", cleared)
                    # Re-queue the secondary task so it gets assigned elsewhere
                    self._state.enqueue(QueuedTask(issue_number=cleared, task_type=inst.secondary_task_type))

    def _assign_secondary_slots(self) -> None:
        with self._state._lock:
            slots = self._secondary_scheduler.available_slots(self._state)
            for slot in slots:
                task = self._state.dequeue()
                if task is None:
                    break
                self._secondary_scheduler.assign_secondary(slot, task)
                _log.info("FleetMonitor: secondary task #%d assigned to %s (TEST_EXECUTION window)",
                          task.issue_number, slot.instance_id)

    def _apply_decision(self, decision) -> None:
        if isinstance(decision, ScaleOpenNew):
            _log.info("FleetMonitor: opening %d new instance(s) - %s", decision.count, decision.reason)
            new_instances = self._provision(decision.count)
            with self._state._lock:
                self._state.instances.extend(new_instances)
        elif isinstance(decision, ScaleAssignWarm):
            _log.info("FleetMonitor: assign queued task to warm instance %s", decision.instance.instance_id)
            with self._state._lock:
                task = self._state.dequeue()
                if task:
                    inst = decision.instance
                    inst.status = InstanceStatus.BUSY
                    inst.assigned_issue = task.issue_number
                    inst.task_type = task.task_type
                    inst.task_started_at = datetime.utcnow()
                    inst.last_heartbeat = datetime.utcnow()

    def _emit_status_log(self) -> None:
        state = self._state
        payload = {
            "fleet_size": len(state.active),
            "busy": len(state.busy),
            "idle": len(state.idle),
            "stuck": len(state.stuck),
            "queue_depth": state.queue_depth,
            "hourly_cost_usd": round(state.hourly_cost_usd(), 4),
            "instances": [i.to_dict() for i in state.active],
        }
        _log.info("FleetMonitor status: %s", json.dumps(payload))


# ---------------------------------------------------------------------------
# HeartbeatReceiver
# ---------------------------------------------------------------------------

@dataclass
class AgentHeartbeat:
    instance_id: str
    issue_number: int
    action_type: str
    tokens_generated: int
    tokens_per_sec: float


_TEST_ACTIONS = frozenset(["run_tests", "run_pytest", "execute_tests"])
_CODING_ACTIONS = frozenset([
    "write_file", "edit_file", "read_file", "search_code",
    "apply_patch", "create_file",
])


class HeartbeatReceiver:
    def __init__(self, state: FleetState) -> None:
        self._state = state

    def record(self, hb: AgentHeartbeat) -> None:
        with self._state._lock:
            inst = self._state.get(hb.instance_id)
            if inst is None:
                return
            inst.last_heartbeat = datetime.utcnow()
            inst.tokens_generated = hb.tokens_generated
            inst.tokens_per_sec = hb.tokens_per_sec
            if hb.action_type in _TEST_ACTIONS:
                inst.phase = AgentPhase.TEST_EXECUTION
            elif hb.action_type in _CODING_ACTIONS:
                inst.phase = AgentPhase.CODING


# ---------------------------------------------------------------------------
# VastAIFleet
# ---------------------------------------------------------------------------

class VastAIFleet:
    """High-level API for ModelOrchestrator and Supervisor."""

    def __init__(self, policy: Optional[FleetPolicy] = None) -> None:
        self._policy = policy or FleetPolicy()
        self._state = FleetState()
        self._scheduler = FleetScheduler()
        self._heartbeat_receiver = HeartbeatReceiver(self._state)
        self._monitor: Optional[FleetMonitor] = None

    def start_monitor(self) -> None:
        self._monitor = FleetMonitor(
            state=self._state,
            policy=self._policy,
            scheduler=self._scheduler,
            provision_fn=self._provision_instances,
            terminate_fn=self._terminate_instance,
            restart_ollama_fn=self._restart_ollama,
        )
        self._monitor.start()

    def stop_monitor(self) -> None:
        if self._monitor:
            self._monitor.stop()

    def acquire(self, issue_number: int, task_type: str = "code_reasoning") -> Optional[FleetInstance]:
        with self._state._lock:
            for inst in self._state.warm_idle:
                inst.status = InstanceStatus.BUSY
                inst.assigned_issue = issue_number
                inst.task_type = task_type
                inst.task_started_at = datetime.utcnow()
                inst.last_heartbeat = datetime.utcnow()
                inst._stuck_restart_attempted = False
                _log.info("VastAIFleet.acquire: warm instance %s -> issue #%d",
                          inst.instance_id, issue_number)
                return inst
            for inst in self._state.instances:
                if inst.status == InstanceStatus.READY:
                    inst.status = InstanceStatus.BUSY
                    inst.assigned_issue = issue_number
                    inst.task_type = task_type
                    inst.task_started_at = datetime.utcnow()
                    inst.last_heartbeat = datetime.utcnow()
                    inst._stuck_restart_attempted = False
                    _log.info("VastAIFleet.acquire: ready instance %s -> issue #%d",
                              inst.instance_id, issue_number)
                    return inst
            self._state.enqueue(QueuedTask(issue_number=issue_number, task_type=task_type))
            _log.info("VastAIFleet.acquire: no instance available, queue=%d", self._state.queue_depth)
            return None

    def release(self, instance_id: str, outcome: str = "success") -> None:
        with self._state._lock:
            inst = self._state.get(instance_id)
            if inst is None:
                return
            _log.info("VastAIFleet.release: %s outcome=%s issue=#%s",
                      instance_id, outcome, inst.assigned_issue)
            inst.assigned_issue = None
            inst.task_started_at = None
            inst.phase = AgentPhase.IDLE
            inst.status = InstanceStatus.IDLE
            inst.last_heartbeat = datetime.utcnow()

    def record_heartbeat(self, hb: AgentHeartbeat) -> None:
        self._heartbeat_receiver.record(hb)

    def get_ready_endpoint(self) -> Optional[str]:
        with self._state._lock:
            for inst in self._state.instances:
                if inst.status in (InstanceStatus.READY, InstanceStatus.BUSY, InstanceStatus.IDLE):
                    if inst.host:
                        return inst.ollama_endpoint()
        return None

    def fleet_status(self) -> Dict:
        with self._state._lock:
            state = self._state
            return {
                "fleet_size": len(state.active),
                "busy": len(state.busy),
                "idle": len(state.idle),
                "stuck": len(state.stuck),
                "queue_depth": state.queue_depth,
                "hourly_cost_usd": round(state.hourly_cost_usd(), 4),
                "instances": [i.to_dict() for i in state.active],
                "queue": [{"issue": t.issue_number, "task_type": t.task_type} for t in list(state.task_queue)],
            }

    def register_instance(self, inst: FleetInstance) -> None:
        with self._state._lock:
            self._state.instances.append(inst)

    def _provision_instances(self, count: int) -> List[FleetInstance]:  # pragma: no cover
        from igris.layers.advisory.vastai_manager import _SHARED_MANAGER
        instances = []
        for _ in range(count):
            started = _SHARED_MANAGER.auto_provision_for_orchestrator()
            if started:
                inst = FleetInstance(
                    instance_id=f"auto-{int(time.time())}",
                    host="",
                    port=11434,
                    status=InstanceStatus.PROVISIONING,
                )
                instances.append(inst)
        return instances

    def _terminate_instance(self, instance_id: str) -> None:  # pragma: no cover
        _log.info("VastAIFleet._terminate: %s", instance_id)

    def _restart_ollama(self, inst: FleetInstance) -> bool:  # pragma: no cover
        import urllib.request
        try:
            urllib.request.urlopen(f"http://{inst.host}:{inst.port}/api/tags", timeout=5)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_SHARED_FLEET: VastAIFleet = VastAIFleet()
