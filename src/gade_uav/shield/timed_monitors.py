"""Timed monitors for the GADe-UAV runtime shield.

Implements docs/02_shield_spec.md §6:
    T-Shield-1: G ( count(emit_from_decoy_i, [t, t+1s]) <= N_burst )
                -- per-decoy emission burst limit
    T-Shield-2: G ( shield_decide_completed_within(Delta_shield_max) )
                -- shield decision-latency bound; deadline miss => REJECT
    T-Shield-3: G ( post_reject -> no_new_generated_action_within(Delta_quiet) )
                -- post-fallback quiescence; prevents thrashing

All three monitors are STATEFUL across shield decisions (unlike the
per-action LTL properties S1..S7). They accept a monotonic wall-clock
timestamp in seconds (float) from the caller rather than reading the
clock themselves -- this keeps them deterministic under test and
identical between simulation replay and live operation.

Ho et al. bounded-MTL rewriting is the semantics reference (spec §6);
we implement the specific bounded operators with sliding-window
counters and per-decoy last-reject timestamps, which is equivalent
for the properties we actually need.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from gade_uav.shield.actions import Action, ActionType


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedStepResult:
    """Outcome of stepping one timed monitor on one shield decision."""

    holds: bool
    reason: str | None = None
    monitor: str | None = None  # "T1" | "T2" | "T3"

    @classmethod
    def ok(cls, monitor: str) -> "TimedStepResult":
        return cls(holds=True, reason=None, monitor=monitor)

    @classmethod
    def fail(cls, monitor: str, reason: str) -> "TimedStepResult":
        return cls(holds=False, reason=reason, monitor=monitor)


# ---------------------------------------------------------------------------
# T-Shield-1: per-decoy emission burst limit
# ---------------------------------------------------------------------------


def _decoy_id_of(action: Action) -> str | None:
    """Return the decoy identity an action is emitted from, if identifiable.

    - DECOY_TELEMETRY carries decoy_id in payload.
    - MAVLINK_RESPONSE carries an optional decoy_id in payload.
    - Other action types are not per-decoy and are excluded from T1.
    """
    if action.type == ActionType.DECOY_TELEMETRY:
        did = action.payload.get("decoy_id")
        return did if isinstance(did, str) else None
    if action.type == ActionType.MAVLINK_RESPONSE:
        did = action.payload.get("decoy_id")
        return did if isinstance(did, str) else None
    return None


class BurstLimitMonitor:
    """T-Shield-1: per-decoy count of emissions in the last `window_s`.

    Each admitted emission from a decoy identity is recorded with its
    timestamp. On each new decision the monitor evicts stamps older than
    `window_s` from the head of that decoy's deque, then checks the count
    against `max_emissions`.

    Admission logic: a proposed emission is REJECTED if, after including
    this proposed emission, the window count would exceed `max_emissions`.
    Only actions with an identifiable decoy_id are subject to T1;
    actions without a decoy_id are vacuously admitted (T1's antecedent
    `emit_from_decoy_i` is false).

    State is updated ONLY when `record_admission` is called by the caller
    after ShieldDecide has admitted the action -- the check itself is
    pure. This mirrors the S6 pattern in the LTL layer.
    """

    name = "T1"

    def __init__(self, window_s: float, max_emissions: int) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if max_emissions < 1:
            raise ValueError(f"max_emissions must be >= 1, got {max_emissions}")
        self.window_s = float(window_s)
        self.max_emissions = int(max_emissions)
        self._history: dict[str, deque[float]] = {}

    def _evict(self, decoy_id: str, now_s: float) -> None:
        dq = self._history.get(decoy_id)
        if dq is None:
            return
        cutoff = now_s - self.window_s
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def check(self, action: Action, now_s: float) -> TimedStepResult:
        """Evaluate T1 for a proposed action. Does not mutate state."""
        decoy_id = _decoy_id_of(action)
        if decoy_id is None:
            return TimedStepResult.ok(self.name)
        self._evict(decoy_id, now_s)
        current = len(self._history.get(decoy_id, ()))
        if current + 1 > self.max_emissions:
            return TimedStepResult.fail(
                self.name,
                f"T1 violated: decoy {decoy_id!r} would exceed "
                f"{self.max_emissions} emissions in {self.window_s}s window "
                f"(current in-window: {current})",
            )
        return TimedStepResult.ok(self.name)

    def record_admission(self, action: Action, now_s: float) -> None:
        """Record a successful admission. Caller invokes after ShieldDecide=DEPLOY."""
        decoy_id = _decoy_id_of(action)
        if decoy_id is None:
            return
        dq = self._history.setdefault(decoy_id, deque())
        dq.append(float(now_s))

    def in_window_count(self, decoy_id: str, now_s: float) -> int:
        """Debug helper: current count for a decoy after eviction."""
        self._evict(decoy_id, now_s)
        return len(self._history.get(decoy_id, ()))


# ---------------------------------------------------------------------------
# T-Shield-2: shield decision-latency bound
# ---------------------------------------------------------------------------


class LatencyDeadlineMonitor:
    """T-Shield-2: enforce `elapsed_ms <= Delta_shield_max` per decision.

    The caller measures the wall-clock elapsed time of its own
    ShieldDecide call and passes it here. On deadline miss the monitor
    returns a violation, and per spec §6 ShieldDecide must force
    REJECT+FALLBACK (the enforcement is the caller's; the monitor only
    reports).

    Rolling p50/p99 statistics are exposed for the reported-metrics
    table in §9.
    """

    name = "T2"

    def __init__(self, deadline_ms: float, history_size: int = 10_000) -> None:
        if deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be > 0, got {deadline_ms}")
        self.deadline_ms = float(deadline_ms)
        self._history: deque[float] = deque(maxlen=history_size)
        self._miss_count: int = 0

    def check(self, elapsed_ms: float) -> TimedStepResult:
        self._history.append(float(elapsed_ms))
        if elapsed_ms > self.deadline_ms:
            self._miss_count += 1
            return TimedStepResult.fail(
                self.name,
                f"T2 violated: shield decision took {elapsed_ms:.3f}ms "
                f"> deadline {self.deadline_ms:.3f}ms",
            )
        return TimedStepResult.ok(self.name)

    def percentile(self, p: float) -> float:
        """Return the p-th percentile latency in ms (0 <= p <= 1)."""
        if not self._history:
            return 0.0
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        import numpy as np

        return float(np.quantile(list(self._history), p))

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def sample_count(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# T-Shield-3: post-fallback quiet period
# ---------------------------------------------------------------------------


class QuietPeriodMonitor:
    """T-Shield-3: no new generated action within Delta_quiet after a REJECT.

    Semantics locked in spec §6: after ShieldDecide returns
    REJECT+FALLBACK, the next `Delta_quiet` seconds must contain zero
    additional generated actions. The quiet period is GLOBAL (not
    per-decoy) because thrashing is a system-level failure mode --
    the orchestrator, not a specific decoy, is the source.

    The monitor is checked BEFORE any per-decision work in
    ShieldDecide, since a violation here means the orchestrator is
    misbehaving and no action should be considered.
    """

    name = "T3"

    def __init__(self, quiet_ms: float) -> None:
        if quiet_ms <= 0:
            raise ValueError(f"quiet_ms must be > 0, got {quiet_ms}")
        self.quiet_s = float(quiet_ms) / 1000.0
        self._last_reject_s: float | None = None

    # Absolute tolerance for the boundary elapsed >= quiet_s check, in
    # seconds. IEEE-754 arithmetic makes constructs like (10.2 - 10.0)
    # evaluate to 0.19999999999999996 < 0.2; we treat elapsed times
    # within 1 microsecond of quiet_s as boundary-admissible. This is
    # tighter than the shield's own decision deadline (Delta_shield_max
    # = 5 ms) by four orders of magnitude, so it cannot mask real
    # violations.
    _BOUNDARY_TOL_S: float = 1e-6

    def check(self, now_s: float) -> TimedStepResult:
        if self._last_reject_s is None:
            return TimedStepResult.ok(self.name)
        elapsed = float(now_s) - self._last_reject_s
        # Half-open semantics for "within(Delta_quiet)": elapsed strictly
        # less than quiet_s is a violation; elapsed >= quiet_s (allowing
        # float-boundary tolerance) is admitted.
        if elapsed < self.quiet_s - self._BOUNDARY_TOL_S:
            return TimedStepResult.fail(
                self.name,
                f"T3 violated: new action offered {elapsed*1000:.2f}ms "
                f"after REJECT, quiet period {self.quiet_s*1000:.2f}ms",
            )
        return TimedStepResult.ok(self.name)

    def record_reject(self, now_s: float) -> None:
        """Caller invokes after ShieldDecide returns REJECT+FALLBACK."""
        self._last_reject_s = float(now_s)

    @property
    def last_reject_s(self) -> float | None:
        return self._last_reject_s


# ---------------------------------------------------------------------------
# Composite timed monitor bundle
# ---------------------------------------------------------------------------


@dataclass
class TimedMonitors:
    """Bundle of the three timed monitors used by ShieldDecide.

    Kept as a dataclass (not a class hierarchy) because each monitor
    has a different check signature -- T1 needs the action + now_s,
    T2 needs the elapsed time, T3 needs only now_s. The caller in
    decide.py invokes them at the appropriate points in the L1..L5
    procedure.
    """

    burst: BurstLimitMonitor
    latency: LatencyDeadlineMonitor
    quiet: QuietPeriodMonitor


__all__ = [
    "BurstLimitMonitor",
    "LatencyDeadlineMonitor",
    "QuietPeriodMonitor",
    "TimedMonitors",
    "TimedStepResult",
]
