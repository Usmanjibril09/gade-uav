"""Composite shield decision procedure (spec §7).

Ties together L1..L5:
    L1: type-level forbidden effector check
    L2: hard LTL safety monitor (S1..S7)
    L3: adaptive conformal resource gate
    L4: timed monitors T-Shield-1/2/3
    L5: admit

On L2 violation, at most ONE repair attempt via a caller-supplied
projector function. On L3 violation, at most ONE downscale attempt
via a caller-supplied downscaler function. Either may return None
to indicate no admissible repair exists; both are subject to a
proof obligation on the caller side that any transformation cannot
introduce a new S1..S7 violation (spec §7 projection whitelist).

The procedure is DETERMINISTIC and STATELESS across calls except
for the state carried inside the passed-in ShieldContext (S6),
TimedMonitors (T1/T3 histories), and MultiDimensionalConformalGate
(calibration + alphas). Callers own that state.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from gade_uav.shield.actions import COST_DIMENSIONS, Action, CostVector, ShieldContext
from gade_uav.shield.conformal_gate import (
    GateDecision,
    MultiDimensionalConformalGate,
)
from gade_uav.shield.cost_estimator import CostEstimator, GeneratorCostEstimator
from gade_uav.shield.fallback import FallbackAction, FallbackPolicy
from gade_uav.shield.ltl_monitor import MonitorResult, SafetyMonitor
from gade_uav.shield.propositions import evaluate_all
from gade_uav.shield.timed_monitors import TimedMonitors, TimedStepResult


logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    """Top-level shield verdict."""

    DEPLOY = "DEPLOY"
    MODIFY = "MODIFY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ShieldDecision:
    """Structured verdict + full audit trail (spec §9 metrics require this)."""

    verdict: Verdict
    action_out: Action | None  # DEPLOY or MODIFY payload; None for REJECT
    fallback: FallbackAction | None  # populated iff verdict == REJECT
    layer_failed: str | None  # "L1" | "L2" | "L3" | "L4" or None
    ltl_result: MonitorResult | None  # from L2
    gate_decision: GateDecision | None  # from L3
    timed_failures: tuple[TimedStepResult, ...] = field(default_factory=tuple)
    repair_attempted: bool = False  # true iff a MODIFY was tried
    repair_succeeded: bool = False  # true iff MODIFY produced admissible action
    decide_elapsed_ms: float = 0.0  # wall-clock T-Shield-2 measurement
    reason: str | None = None


# Type aliases for the two caller-supplied projection functions.
LTLRepairFn = Callable[[Action, ShieldContext, MonitorResult], Action | None]
CostDownscaleFn = Callable[[Action, GateDecision], Action | None]




def _bypass_gate_decision() -> "GateDecision":
    """Return a passing GateDecision for shield-mode=off / soft-only bypass."""
    from gade_uav.shield.conformal_gate import GateDecision
    return GateDecision(
        admitted=True,
        insufficient_calibration=False,
        failing_dimensions=(),
        per_dimension=(),
    )

class Shield:
    """Composite shield -- one instance per running defender.

    The shield does NOT own the clock; callers pass `now_s` for
    determinism under simulation replay. Wall-clock T-Shield-2
    measurement uses `time.perf_counter` internally, but that clock
    is used only for the intra-decision elapsed measurement (not for
    absolute event timestamps).
    """

    def __init__(
        self,
        ltl_monitor: SafetyMonitor,
        gate: MultiDimensionalConformalGate,
        timed: TimedMonitors,
        fallback_policy: FallbackPolicy,
        ltl_repair: LTLRepairFn | None = None,
        cost_downscale: CostDownscaleFn | None = None,
        cost_estimator: CostEstimator | None = None,
        enable_ltl: bool = True,
        enable_gate: bool = True,
    ) -> None:
        self._ltl = ltl_monitor
        self._gate = gate
        self._enable_ltl = enable_ltl
        self._enable_gate = enable_gate
        self._timed = timed
        self._fallback = fallback_policy
        self._ltl_repair = ltl_repair
        self._cost_downscale = cost_downscale
        self._cost_estimator = cost_estimator or GeneratorCostEstimator()

    # -- calibration feedback -------------------------------------------

    def observe_cost_outcome(
        self, decision: ShieldDecision, observed: "CostVector"
    ) -> None:
        """Feed the observed cost of an admitted action back to the L3 gate.

        Requires the ShieldDecision produced by decide() so per-dimension
        UCBs recorded at admission time are available for miscoverage
        checks and ACI updates.
        """
        # When the gate is bypassed (shield_mode=off / soft-only variants
        # that disable L3), there is no calibration state to update.
        if not self._enable_gate:
            return
        if decision.gate_decision is None:
            raise ValueError(
                "observe_cost_outcome requires a decision with gate_decision "
                "populated (i.e., an action that passed L3)"
            )
        obs_dict = {
            "cpu": observed.cpu,
            "ram": observed.ram,
            "lat": observed.lat,
            "bw": observed.bw,
            "energy": observed.energy,
            "mission_delay": observed.mission_delay,
        }
        self._gate.observe_outcome(decision=decision.gate_decision, observed=obs_dict)

    def _resource_decision(
        self, action: Action, ctx: ShieldContext, primary_attack_family: str | None
    ) -> GateDecision | None:
        try:
            estimates = self._cost_estimator.estimate(action, ctx, primary_attack_family)
            predicted = {dimension: float(estimates[dimension]) for dimension in COST_DIMENSIONS}
            if any(not math.isfinite(value) or value < 0.0 for value in predicted.values()):
                raise ValueError("Resource estimates must be finite and nonnegative")
        except Exception:
            logger.exception("Resource prediction failed during shield authorization")
            return None

        return self._gate.decide(predicted=predicted, phase=ctx.mission_phase)

    # -- public entry point ---------------------------------------------

    def decide(
        self,
        action: Action,
        ctx: ShieldContext,
        now_s: float,
        primary_attack_family: str | None = None,
    ) -> ShieldDecision:
        """Run L1..L5 for a single candidate action.

        Args:
            action:            The orchestrator's candidate d*.
            ctx:               Frozen shield context for this decision.
            now_s:             Monotonic timestamp (seconds) for T1/T3.
            primary_attack_family: Attack family key for fallback lookup
                                   (see fallback.KNOWN_ATTACK_FAMILIES).
                                   May be None; uses the "unknown" row.
        """
        t_start = time.perf_counter()

        expires_at_s = ctx.mission_state_expires_at_s
        if not ctx.mission_state_trusted or (
            expires_at_s is not None
            and (not math.isfinite(expires_at_s) or now_s >= expires_at_s)
        ):
            return self._reject(
                action,
                ctx,
                primary_attack_family,
                layer="L1",
                reason="Mission state is untrusted or expired",
                now_s=now_s,
                t_start=t_start,
            )

        # L4 pre-gate: T-Shield-3 (post-fallback quiet period). Checked
        # BEFORE any work because a violation means the orchestrator is
        # misbehaving and no candidate should be considered.
        t3 = self._timed.quiet.check(now_s)
        if not t3.holds:
            return self._reject(
                action,
                ctx,
                primary_attack_family,
                layer="L4",
                reason=t3.reason,
                timed_failures=(t3,),
                now_s=now_s,
                t_start=t_start,
            )

        # L1: type-level forbidden effector.
        props = evaluate_all(action, ctx)
        if self._enable_ltl and props["target_forbidden"]:
            return self._reject(
                action,
                ctx,
                primary_attack_family,
                layer="L1",
                reason=(
                    f"L1 violated: target_iface {action.target_iface.value!r} "
                    f"is in the forbidden set (effector routing bug)"
                ),
                now_s=now_s,
                t_start=t_start,
            )

        # L2: hard LTL safety monitor. (skipped when enable_ltl=False)
        if not self._enable_ltl:
            ltl_res = MonitorResult(holds=True, failing_property=None, reason=None)
        else:
            ltl_res = self._ltl.evaluate(props, action, ctx)
        repair_attempted = False
        repair_succeeded = False
        if not ltl_res.holds:
            repair_attempted = True
            if self._ltl_repair is not None:
                repaired = self._ltl_repair(action, ctx, ltl_res)
                if repaired is not None:
                    # Re-evaluate the repaired action from scratch. Only ONE
                    # repair attempt (spec §7 repair_max_attempts=1).
                    repaired_props = evaluate_all(repaired, ctx)
                    if repaired_props["target_forbidden"]:
                        return self._reject(
                            action,
                            ctx,
                            primary_attack_family,
                            layer="L1",
                            reason="L1 violated after LTL repair (rejected)",
                            ltl_result=ltl_res,
                            repair_attempted=True,
                            now_s=now_s,
                            t_start=t_start,
                        )
                    ltl_res2 = self._ltl.evaluate(repaired_props, repaired, ctx)
                    if ltl_res2.holds:
                        # Repair succeeded at L2; continue with L3.
                        action = repaired
                        props = repaired_props
                        ltl_res = ltl_res2
                        repair_succeeded = True

            if not repair_succeeded:
                return self._reject(
                    action,
                    ctx,
                    primary_attack_family,
                    layer="L2",
                    reason=ltl_res.reason,
                    ltl_result=ltl_res,
                    repair_attempted=repair_attempted,
                    now_s=now_s,
                    t_start=t_start,
                )

        # L3: conformal resource gate. (skipped when enable_gate=False)
        if not self._enable_gate:
            gate_res = _bypass_gate_decision()
        else:
            gate_res = self._resource_decision(action, ctx, primary_attack_family)
            if gate_res is None:
                return self._reject(
                    action, ctx, primary_attack_family,
                    layer="L3", reason="Resource prediction failed",
                    ltl_result=ltl_res, now_s=now_s, t_start=t_start,
                )
        cost_repair_attempted = False
        cost_repair_succeeded = False
        if not gate_res.admitted:
            if gate_res.insufficient_calibration:
                # Cannot gate below min_calibration_size (spec §5.1).
                # Route to REJECT; caller must accept a REJECT until
                # enough calibration is available.
                return self._reject(
                    action,
                    ctx,
                    primary_attack_family,
                    layer="L3",
                    reason="L3 violated: insufficient calibration",
                    ltl_result=ltl_res,
                    gate_decision=gate_res,
                    repair_attempted=repair_attempted,
                    repair_succeeded=repair_succeeded,
                    now_s=now_s,
                    t_start=t_start,
                )

            cost_repair_attempted = True
            if self._cost_downscale is not None:
                downscaled = self._cost_downscale(action, gate_res)
                if downscaled is not None:
                    # Re-run L1+L2 on the downscaled action (proof obligation:
                    # a valid downscale cannot introduce a new violation, but
                    # we verify anyway).
                    ds_props = evaluate_all(downscaled, ctx)
                    if ds_props["target_forbidden"]:
                        return self._reject(
                            action,
                            ctx,
                            primary_attack_family,
                            layer="L1",
                            reason="L1 violated after cost downscale (rejected)",
                            ltl_result=ltl_res,
                            gate_decision=gate_res,
                            repair_attempted=True,
                            now_s=now_s,
                            t_start=t_start,
                        )
                    ds_ltl = self._ltl.evaluate(ds_props, downscaled, ctx)
                    if not ds_ltl.holds:
                        return self._reject(
                            action,
                            ctx,
                            primary_attack_family,
                            layer="L2",
                            reason=f"L2 violated after cost downscale: {ds_ltl.reason}",
                            ltl_result=ds_ltl,
                            gate_decision=gate_res,
                            repair_attempted=True,
                            now_s=now_s,
                            t_start=t_start,
                        )
                    ds_gate = self._resource_decision(downscaled, ctx, primary_attack_family)
                    if ds_gate is None:
                        return self._reject(
                            action, ctx, primary_attack_family,
                            layer="L3", reason="Resource prediction failed after cost downscale",
                            ltl_result=ltl_res, repair_attempted=True,
                            now_s=now_s, t_start=t_start,
                        )
                    if ds_gate.admitted:
                        action = downscaled
                        props = ds_props
                        gate_res = ds_gate
                        cost_repair_succeeded = True
                    else:
                        # Preserve the repaired candidate's gate result so
                        # rejection diagnostics report its actual remaining
                        # failing dimensions rather than the original ones.
                        gate_res = ds_gate

            if not cost_repair_succeeded:
                return self._reject(
                    action,
                    ctx,
                    primary_attack_family,
                    layer="L3",
                    reason=(
                        f"L3 violated: dimensions over budget: "
                        f"{list(gate_res.failing_dimensions)}"
                    ),
                    ltl_result=ltl_res,
                    gate_decision=gate_res,
                    repair_attempted=repair_attempted or cost_repair_attempted,
                    repair_succeeded=repair_succeeded,
                    now_s=now_s,
                    t_start=t_start,
                )

        # L4: T-Shield-1 burst limit (per-decoy). T-Shield-2 is measured
        # at the very end; T-Shield-3 was pre-checked.
        t1 = self._timed.burst.check(action, now_s)
        if not t1.holds:
            return self._reject(
                action,
                ctx,
                primary_attack_family,
                layer="L4",
                reason=t1.reason,
                ltl_result=ltl_res,
                gate_decision=gate_res,
                timed_failures=(t1,),
                repair_attempted=repair_attempted or cost_repair_attempted,
                repair_succeeded=repair_succeeded or cost_repair_succeeded,
                now_s=now_s,
                t_start=t_start,
            )

        # L5: admit. Update T-Shield-1 state now that the action is
        # committed. T-Shield-2 records the elapsed time; a deadline
        # miss here does NOT retroactively reject an already-admitted
        # action but IS reported in the audit trail for the §9 metric.
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        t2 = self._timed.latency.check(elapsed_ms=elapsed_ms)
        self._timed.burst.record_admission(action, now_s)

        if not t2.holds:
            # Spec §6 T-Shield-2: on deadline miss, force REJECT+FALLBACK.
            # We honor that even though the action has been admitted at
            # L1..L4 -- the deadline miss is itself a safety-relevant
            # event.
            return self._reject(
                action,
                ctx,
                primary_attack_family,
                layer="L4",
                reason=t2.reason,
                ltl_result=ltl_res,
                gate_decision=gate_res,
                timed_failures=(t2,),
                repair_attempted=repair_attempted or cost_repair_attempted,
                repair_succeeded=repair_succeeded or cost_repair_succeeded,
                now_s=now_s,
                t_start=t_start,
            )

        verdict = (
            Verdict.MODIFY
            if (repair_succeeded or cost_repair_succeeded)
            else Verdict.DEPLOY
        )
        return ShieldDecision(
            verdict=verdict,
            action_out=action,
            fallback=None,
            layer_failed=None,
            ltl_result=ltl_res,
            gate_decision=gate_res,
            timed_failures=(),
            repair_attempted=repair_attempted or cost_repair_attempted,
            repair_succeeded=repair_succeeded or cost_repair_succeeded,
            decide_elapsed_ms=elapsed_ms,
            reason=None,
        )

    # -- internal reject helper -----------------------------------------

    def _reject(
        self,
        action: Action,
        ctx: ShieldContext,
        primary_attack_family: str | None,
        *,
        layer: str,
        reason: str | None,
        now_s: float,
        ltl_result: MonitorResult | None = None,
        gate_decision: GateDecision | None = None,
        timed_failures: tuple[TimedStepResult, ...] = (),
        repair_attempted: bool = False,
        repair_succeeded: bool = False,
        t_start: float,
    ) -> ShieldDecision:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        # Do NOT count a REJECT toward T-Shield-2's history: T2 measures
        # the latency of ADMIT decisions (they must finish within
        # Delta_shield_max). A REJECT can be faster or slower; its
        # timing is not the enforcement target. Update T-Shield-3.
        fallback = self._fallback.select(primary_attack_family, ctx.mission_phase)
        # Use the caller-supplied clock for T-Shield-3 arming; t_start is
        # a perf_counter timestamp used only for the elapsed-ms measurement
        # and MUST NOT leak into caller-visible timing state.
        self._timed.quiet.record_reject(now_s=now_s)
        return ShieldDecision(
            verdict=Verdict.REJECT,
            action_out=None,
            fallback=fallback,
            layer_failed=layer,
            ltl_result=ltl_result,
            gate_decision=gate_decision,
            timed_failures=timed_failures,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            decide_elapsed_ms=elapsed_ms,
            reason=reason,
        )


__all__ = ["Shield", "ShieldDecision", "Verdict"]
