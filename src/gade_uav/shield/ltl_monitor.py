"""Hand-coded LTL safety monitor for the GADe-UAV runtime shield.

Implements docs/02_shield_spec.md §4.2 (properties S1..S7) and §4.3
(hand-coded state machine, LTL semantics traceable). Each property is
one class exposing:

    step(propositions, action, ctx) -> StepResult

where StepResult carries (holds: bool, reason: str | None).
The composite `SafetyMonitor` runs all seven in parallel and reports
which property failed on the FIRST violation encountered (deterministic
ordering S1..S7 for reproducible failure attribution).

All properties in spec §4.2 are pure-safety (G ¬bad or G (a -> b)).
No property references history except S6 (per-decoy field-jump bounds,
delegated to the decoy_state_consistent proposition) and S7 (whose
history dependence is captured entirely by the current-step conjunction
`emit_mavlink AND NOT uncertainty_low`).

Because every property is `G ...`, the monitor is stateless w.r.t.
LTL semantics -- one violation is permanent for the trace but the
shield resets between decisions. State that MUST persist across
decisions (S6 last-emission dict) lives in the ShieldContext, updated
by the caller after each ADMITTED action (spec §7 procedure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gade_uav.shield.actions import Action, ShieldContext


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """Outcome of stepping one safety property on one action."""

    holds: bool
    reason: str | None = None  # human-readable when holds is False

    @classmethod
    def ok(cls) -> "StepResult":
        return cls(holds=True, reason=None)

    @classmethod
    def fail(cls, reason: str) -> "StepResult":
        return cls(holds=False, reason=reason)


@dataclass(frozen=True)
class MonitorResult:
    """Composite result across S1..S7."""

    holds: bool
    failing_property: str | None  # "S1".."S7" or None
    reason: str | None


# ---------------------------------------------------------------------------
# Property protocol
# ---------------------------------------------------------------------------


class SafetyProperty(Protocol):
    name: str

    def step(
        self,
        props: dict[str, bool],
        action: Action,
        ctx: ShieldContext,
    ) -> StepResult: ...


# ---------------------------------------------------------------------------
# S1..S7 (spec §4.2)
# ---------------------------------------------------------------------------


class S1_NoFlightCriticalEmission:
    """S1: G ¬(emit_setmode ∨ emit_reboot ∨ emit_rtl_land ∨ emit_arm
    ∨ emit_paramset ∨ emit_mission_write)."""

    name = "S1"

    _BANNED = (
        "emit_setmode",
        "emit_reboot",
        "emit_rtl_land",
        "emit_arm",
        "emit_paramset",
        "emit_mission_write",
    )

    def step(self, props, action, ctx):
        for p in self._BANNED:
            if props[p]:
                return StepResult.fail(f"S1 violated: {p} is true")
        return StepResult.ok()


class S2_NoRealTargeting:
    """S2: G ¬(target_real_fc ∨ target_real_gcs)."""

    name = "S2"

    def step(self, props, action, ctx):
        if props["target_real_fc"]:
            return StepResult.fail("S2 violated: target_real_fc")
        if props["target_real_gcs"]:
            return StepResult.fail("S2 violated: target_real_gcs")
        return StepResult.ok()


class S3_SchemaValid:
    """S3: G schema_valid."""

    name = "S3"

    def step(self, props, action, ctx):
        if not props["schema_valid"]:
            return StepResult.fail(
                f"S3 violated: payload fails schema for type {action.type.value}"
            )
        return StepResult.ok()


class S4_SigConsistent:
    """S4: G (emit_mavlink → sig_consistent)."""

    name = "S4"

    def step(self, props, action, ctx):
        if props["emit_mavlink"] and not props["sig_consistent"]:
            return StepResult.fail(
                "S4 violated: MAVLink emission without decoy-key signing"
            )
        return StepResult.ok()


class S5_NoSharedChannelInCriticalPhase:
    """S5: G (phase_critical → ¬(emit_mavlink ∧ on_shared_channel))."""

    name = "S5"

    def step(self, props, action, ctx):
        if (
            props["phase_critical"]
            and props["emit_mavlink"]
            and props["on_shared_channel"]
        ):
            return StepResult.fail(
                f"S5 violated: MAVLink on shared channel during "
                f"critical phase {ctx.mission_phase.value}"
            )
        return StepResult.ok()


class S6_DecoyStateConsistent:
    """S6: G decoy_state_consistent."""

    name = "S6"

    def step(self, props, action, ctx):
        if not props["decoy_state_consistent"]:
            return StepResult.fail(
                "S6 violated: decoy telemetry field jump exceeds declared bound"
            )
        return StepResult.ok()


class S7_UncertaintyBoundsTemplate:
    """S7: G ((emit_mavlink ∧ ¬uncertainty_low) →
    emitted_from_bounded_template_only)."""

    name = "S7"

    def step(self, props, action, ctx):
        if (
            props["emit_mavlink"]
            and not props["uncertainty_low"]
            and not props["emitted_from_bounded_template_only"]
        ):
            return StepResult.fail(
                "S7 violated: high-uncertainty MAVLink emission outside "
                "bounded template whitelist"
            )
        return StepResult.ok()


# ---------------------------------------------------------------------------
# Composite monitor
# ---------------------------------------------------------------------------

# Ordering matters for reproducible failure attribution -- once a property
# fails, later ones are not evaluated. S1..S7 as in the spec.
_DEFAULT_PROPERTIES: tuple[SafetyProperty, ...] = (
    S1_NoFlightCriticalEmission(),
    S2_NoRealTargeting(),
    S3_SchemaValid(),
    S4_SigConsistent(),
    S5_NoSharedChannelInCriticalPhase(),
    S6_DecoyStateConsistent(),
    S7_UncertaintyBoundsTemplate(),
)


class SafetyMonitor:
    """Runs S1..S7 in order, short-circuiting on first failure.

    Stateless across decisions -- persistent state lives in ShieldContext
    (S6's last_emission_per_decoy). Callers must update that dict AFTER
    an action is admitted, not inside the monitor.
    """

    def __init__(
        self,
        properties: tuple[SafetyProperty, ...] = _DEFAULT_PROPERTIES,
    ) -> None:
        self._properties = properties

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._properties)

    def evaluate(
        self,
        props: dict[str, bool],
        action: Action,
        ctx: ShieldContext,
    ) -> MonitorResult:
        """Evaluate the full φ_safe = S1 ∧ ... ∧ S7 conjunction.

        Returns MonitorResult(holds=True, ...) if every property holds,
        otherwise MonitorResult(holds=False, failing_property=<name>,
        reason=<why>) for the first failure in S1..S7 order.
        """
        for prop in self._properties:
            r = prop.step(props, action, ctx)
            if not r.holds:
                return MonitorResult(
                    holds=False,
                    failing_property=prop.name,
                    reason=r.reason,
                )
        return MonitorResult(holds=True, failing_property=None, reason=None)


__all__ = [
    "MonitorResult",
    "S1_NoFlightCriticalEmission",
    "S2_NoRealTargeting",
    "S3_SchemaValid",
    "S4_SigConsistent",
    "S5_NoSharedChannelInCriticalPhase",
    "S6_DecoyStateConsistent",
    "S7_UncertaintyBoundsTemplate",
    "SafetyMonitor",
    "SafetyProperty",
    "StepResult",
]
