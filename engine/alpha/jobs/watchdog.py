"""Bounded daemon-thread watchdog helpers for blocking provider calls."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS = 10


class ProviderOutageCircuitBreaker(RuntimeError):
    """Raised when abandoned fetch workers exceed the configured safety cap."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload))
        self.payload = payload


@dataclass
class WatchdogState:
    """State shared by one shard/job to cap abandoned daemon workers."""

    max_outstanding_timeouts: int = DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS
    max_consecutive_timeouts: int | None = None
    total_timeouts: int = 0
    consecutive_timeouts: int = 0
    outstanding_timeouts: int = 0
    thread_start_failures: int = 0
    circuit_open: bool = False
    circuit_reason: str | None = None

    def __post_init__(self) -> None:
        if self.max_outstanding_timeouts < 1:
            raise ValueError("max_outstanding_timeouts must be >= 1")
        if self.max_consecutive_timeouts is not None and self.max_consecutive_timeouts < 1:
            raise ValueError("max_consecutive_timeouts must be >= 1")
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_timeouts = 0

    def record_timeout(self) -> bool:
        with self._lock:
            self.total_timeouts += 1
            self.consecutive_timeouts += 1
            self.outstanding_timeouts += 1
            return self._update_circuit_locked("watchdog_timeout")

    def record_thread_finished_after_timeout(self) -> None:
        with self._lock:
            if self.outstanding_timeouts > 0:
                self.outstanding_timeouts -= 1

    def record_thread_start_failure(self, exc: BaseException) -> None:
        with self._lock:
            self.total_timeouts += 1
            self.consecutive_timeouts += 1
            self.thread_start_failures += 1
            self.circuit_open = True
            self.circuit_reason = f"thread_start_failed:{type(exc).__name__}"

    def _update_circuit_locked(self, reason: str) -> bool:
        if self.outstanding_timeouts >= self.max_outstanding_timeouts:
            self.circuit_open = True
            self.circuit_reason = f"{reason}:max_outstanding_timeouts"
        if (
            self.max_consecutive_timeouts is not None
            and self.consecutive_timeouts >= self.max_consecutive_timeouts
        ):
            self.circuit_open = True
            self.circuit_reason = f"{reason}:max_consecutive_timeouts"
        return self.circuit_open

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_outstanding_fetch_timeouts": self.max_outstanding_timeouts,
                "max_consecutive_fetch_timeouts": self.max_consecutive_timeouts,
                "watchdog_timeouts": self.total_timeouts,
                "consecutive_watchdog_timeouts": self.consecutive_timeouts,
                "outstanding_fetch_timeouts": self.outstanding_timeouts,
                "thread_start_failures": self.thread_start_failures,
                "circuit_open": self.circuit_open,
                "circuit_reason": self.circuit_reason,
            }


def call_with_daemon_deadline(
    func: Callable[[], Any],
    *,
    timeout_seconds: float,
    thread_name: str,
    state: WatchdogState | None = None,
    context: dict[str, Any] | None = None,
) -> Any:
    """Run ``func`` in a daemon thread and bound the caller by wall-clock time."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    watchdog_state = state or WatchdogState(max_outstanding_timeouts=1_000_000)
    call_context = dict(context or {})
    if watchdog_state.circuit_open:
        raise _circuit_breaker_error(watchdog_state, call_context)
    results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
    call_lock = threading.Lock()
    call_state = {
        "timed_out": False,
        "finished": False,
        "decremented_after_timeout": False,
    }

    def _target() -> None:
        try:
            results.put_nowait((True, func()))
        except BaseException as exc:  # noqa: BLE001 - propagate worker failure to caller
            results.put_nowait((False, exc))
        finally:
            should_decrement = False
            with call_lock:
                call_state["finished"] = True
                if (
                    call_state["timed_out"]
                    and not call_state["decremented_after_timeout"]
                ):
                    call_state["decremented_after_timeout"] = True
                    should_decrement = True
            if should_decrement:
                watchdog_state.record_thread_finished_after_timeout()

    try:
        thread = threading.Thread(target=_target, name=thread_name, daemon=True)
        thread.start()
        thread.join(timeout_seconds)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - convert start/join failure to breaker
        watchdog_state.record_thread_start_failure(exc)
        raise _circuit_breaker_error(watchdog_state, call_context) from exc

    if thread.is_alive():
        with call_lock:
            if call_state["finished"]:
                timed_out = False
                circuit_open = False
            else:
                call_state["timed_out"] = True
                timed_out = True
                circuit_open = watchdog_state.record_timeout()
        if timed_out and circuit_open:
            raise _circuit_breaker_error(watchdog_state, call_context)
        if timed_out:
            raise FuturesTimeoutError()

    ok, value = results.get_nowait()
    watchdog_state.record_success()
    if ok:
        return value
    raise value


def _circuit_breaker_error(
    state: WatchdogState,
    context: dict[str, Any],
) -> ProviderOutageCircuitBreaker:
    payload = {
        **context,
        "error": "provider_outage_circuit_breaker",
        **state.snapshot(),
    }
    return ProviderOutageCircuitBreaker(payload)
