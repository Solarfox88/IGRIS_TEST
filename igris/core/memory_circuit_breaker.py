"""
memory_circuit_breaker.py — Epic #1073

Circuit breaker for memory write operations.

When LongTermMemory or MemoryGraph writes fail repeatedly, a naive system
will keep retrying and spamming logs.  This circuit breaker:

  1. Counts consecutive write failures per subsystem.
  2. Opens (disables writes) after OPEN_THRESHOLD failures.
  3. After RECOVERY_WINDOW_SECONDS, transitions to half-open.
  4. One success in half-open → closes (re-enables writes).
  5. One failure in half-open → opens again.

All operations are thread-safe.  The breaker is a singleton per subsystem
name — import and use directly:

    breaker = MemoryCircuitBreaker.get("long_term")
    if breaker.allow():
        try:
            long_term_memory.save(...)
            breaker.record_success()
        except Exception as exc:
            breaker.record_failure(exc)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

_log = logging.getLogger("igris.memory.circuit_breaker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPEN_THRESHOLD: int = 3         # failures before opening
DEFAULT_RECOVERY_WINDOW: float = 120.0  # seconds before attempting half-open


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class BreakerState:
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # writes disabled
    HALF_OPEN = "half_open" # one trial allowed


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class MemoryCircuitBreaker:
    """Thread-safe circuit breaker for a named memory subsystem."""

    name: str
    open_threshold: int = DEFAULT_OPEN_THRESHOLD
    recovery_window_seconds: float = DEFAULT_RECOVERY_WINDOW

    # Internal state (not part of public API)
    _state: str = field(default=BreakerState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_ts: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # Registry of named breakers
    _registry: Dict[str, "MemoryCircuitBreaker"] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def get(
        cls,
        name: str,
        open_threshold: int = DEFAULT_OPEN_THRESHOLD,
        recovery_window_seconds: float = DEFAULT_RECOVERY_WINDOW,
    ) -> "MemoryCircuitBreaker":
        """Return the singleton MemoryCircuitBreaker for *name*."""
        if not hasattr(cls, "_global_registry"):
            cls._global_registry: Dict[str, "MemoryCircuitBreaker"] = {}
        if name not in cls._global_registry:
            cls._global_registry[name] = cls(
                name=name,
                open_threshold=open_threshold,
                recovery_window_seconds=recovery_window_seconds,
            )
        return cls._global_registry[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all breakers (useful in tests)."""
        if hasattr(cls, "_global_registry"):
            cls._global_registry.clear()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        self._maybe_recover()
        return self._state

    def allow(self) -> bool:
        """Return True if a write attempt is allowed right now."""
        self._maybe_recover()
        with self._lock:
            if self._state == BreakerState.CLOSED:
                return True
            if self._state == BreakerState.HALF_OPEN:
                return True  # one trial allowed
            # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful write. Closes the breaker if half-open."""
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                _log.info(
                    "MemoryCircuitBreaker[%s]: half-open trial succeeded → closed",
                    self.name,
                )
                self._state = BreakerState.CLOSED
                self._failure_count = 0
            elif self._state == BreakerState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, exc: Optional[Exception] = None) -> None:
        """Record a failed write. Opens the breaker after *open_threshold* failures."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_ts = time.monotonic()

            if self._state == BreakerState.HALF_OPEN:
                # Failed trial → back to open
                self._state = BreakerState.OPEN
                _log.warning(
                    "MemoryCircuitBreaker[%s]: half-open trial failed → opened again. "
                    "Error: %s",
                    self.name, exc,
                )
            elif self._state == BreakerState.CLOSED:
                if self._failure_count >= self.open_threshold:
                    self._state = BreakerState.OPEN
                    _log.error(
                        "MemoryCircuitBreaker[%s]: opened after %d consecutive failures. "
                        "Memory writes DISABLED for %.0fs. Last error: %s",
                        self.name, self._failure_count,
                        self.recovery_window_seconds, exc,
                    )
                else:
                    _log.warning(
                        "MemoryCircuitBreaker[%s]: failure %d/%d. Error: %s",
                        self.name, self._failure_count, self.open_threshold, exc,
                    )

    def is_open(self) -> bool:
        return self.state == BreakerState.OPEN

    def is_closed(self) -> bool:
        return self.state == BreakerState.CLOSED

    def status_dict(self) -> Dict[str, object]:
        """Return a serialisable status snapshot."""
        s = self.state
        return {
            "name": self.name,
            "state": s,
            "failure_count": self._failure_count,
            "open_threshold": self.open_threshold,
            "recovery_window_seconds": self.recovery_window_seconds,
            "last_failure_ts": self._last_failure_ts or None,
            "healthy": s == BreakerState.CLOSED,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_recover(self) -> None:
        """Transition OPEN → HALF_OPEN when recovery window has elapsed."""
        with self._lock:
            if self._state == BreakerState.OPEN:
                elapsed = time.monotonic() - self._last_failure_ts
                if elapsed >= self.recovery_window_seconds:
                    self._state = BreakerState.HALF_OPEN
                    _log.info(
                        "MemoryCircuitBreaker[%s]: %.0fs elapsed → half-open (trial allowed)",
                        self.name, elapsed,
                    )
