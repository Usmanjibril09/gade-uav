"""Tests for the LTL safety monitor (spec §4.2, S1..S7).

Each property is tested in isolation for both admission and rejection.
The composite SafetyMonitor is tested for correct S1..S7 short-circuit
ordering: on multi-property violation, the earlier-numbered property
must be reported.
"""

from __future__ import annotations

import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    DecoyIdentity,
    Effector,
    GenMeta,
    MissionPhase,
    ShieldContext,
)
from gade_uav.shield.ltl_monitor import (
    S1_NoFlightCriticalEmission,
    S2_NoRealTargeting,
    S3_SchemaValid,
    S4_SigConsistent,
    S5_NoSharedChannelInCriticalPhase,
    S6_DecoyStateConsistent,
    S7_UncertaintyBoundsTemplate,
    SafetyMonitor,
)
from gade_uav.shield.propositions import evaluate_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cost() -> CostVector:
    return CostVector(0.1, 10.0, 1.0, 100.0, 0.5, 0.0)


def _meta(conf: float = 0.9, tid: str | None = None) -> GenMeta:
    return GenMeta(
        confidence=conf,
        schema_version="v1",
        generation_context_hash="deadbeef",
        template_id=tid,
    )


def _mavlink(
    msg_id: int = 0,
    command: int | None = None,
    target: Effector = Effector.DECOY_TELEMETRY_CHANNEL,
    conf: float = 0.9,
    tid: str | None = None,
    with_msg_id: bool = True,
) -> Action:
    payload: dict = {"fields": {}}
    if with_msg_id:
        payload["msg_id"] = msg_id
    if command is not None:
        payload["command"] = command
    return Action(
        type=ActionType.MAVLINK_RESPONSE,
        payload=payload,
        target_iface=target,
        predicted_cost=_cost(),
        gen_meta=_meta(conf=conf, tid=tid),
    )


def _telemetry(
    decoy_id: str = "d1",
    fields: dict | None = None,
    target: Effector = Effector.DECOY_TELEMETRY_CHANNEL,
) -> Action:
    return Action(
        type=ActionType.DECOY_TELEMETRY,
        payload={"decoy_id": decoy_id, "fields": fields or {}},
        target_iface=target,
        predicted_cost=_cost(),
        gen_meta=_meta(),
    )


def _ctx(
    phase: MissionPhase = MissionPhase.CRUISE,
    signing: bool = False,
    decoy_key: bool = False,
    known_decoys: dict | None = None,
    last_emissions: dict | None = None,
    whitelist: frozenset[str] = frozenset(),
    unc_thresh: float = 0.30,
) -> ShieldContext:
    return ShieldContext(
        mission_phase=phase,
        signing_active=signing,
        decoy_key_active=decoy_key,
        known_decoys=known_decoys or {},
        last_emission_per_decoy=last_emissions or {},
        uncertainty_threshold=unc_thresh,
        bounded_template_whitelist=whitelist,
    )


def _step(prop, action: Action, ctx: ShieldContext):
    return prop.step(evaluate_all(action, ctx), action, ctx)


# ---------------------------------------------------------------------------
# S1 -- no flight-critical emission
# ---------------------------------------------------------------------------


class TestS1:
    prop = S1_NoFlightCriticalEmission()

    def test_heartbeat_admitted(self):
        assert _step(self.prop, _mavlink(msg_id=0), _ctx()).holds

    @pytest.mark.parametrize(
        "msg_id,command,label",
        [
            (11, None, "SET_MODE"),
            (76, 176, "COMMAND_LONG DO_SET_MODE"),
            (23, None, "PARAM_SET"),
            (76, 246, "COMMAND_LONG REBOOT_SHUTDOWN"),
            (76, 20, "COMMAND_LONG RTL"),
            (76, 21, "COMMAND_LONG LAND"),
            (76, 400, "COMMAND_LONG ARM_DISARM"),
            (38, None, "MISSION_WRITE_PARTIAL_LIST"),
            (39, None, "MISSION_ITEM"),
            (44, None, "MISSION_COUNT"),
            (73, None, "MISSION_ITEM_INT"),
        ],
    )
    def test_flight_critical_rejected(self, msg_id, command, label):
        r = _step(self.prop, _mavlink(msg_id=msg_id, command=command), _ctx())
        assert not r.holds, f"S1 should reject {label}"
        assert "S1 violated" in (r.reason or "")


# ---------------------------------------------------------------------------
# S2 -- no real-system targeting
# ---------------------------------------------------------------------------


class TestS2:
    prop = S2_NoRealTargeting()

    def test_decoy_target_admitted(self):
        assert _step(
            self.prop, _mavlink(target=Effector.DECOY_MAVLINK_CHANNEL), _ctx()
        ).holds

    @pytest.mark.parametrize(
        "target",
        [
            Effector.FLIGHT_CONTROLLER_BUS,
            Effector.MOTOR_MIXER,
        ],
    )
    def test_real_fc_rejected(self, target):
        r = _step(self.prop, _mavlink(target=target), _ctx())
        assert not r.holds
        assert "target_real_fc" in (r.reason or "")

    @pytest.mark.parametrize(
        "target",
        [
            Effector.GCS_COMMAND_CHANNEL_PRIMARY,
            Effector.MISSION_UPLOAD_CHANNEL,
        ],
    )
    def test_real_gcs_rejected(self, target):
        r = _step(self.prop, _mavlink(target=target), _ctx())
        assert not r.holds
        assert "target_real_gcs" in (r.reason or "")


# ---------------------------------------------------------------------------
# S3 -- schema valid
# ---------------------------------------------------------------------------


class TestS3:
    prop = S3_SchemaValid()

    def test_valid_mavlink_admitted(self):
        assert _step(self.prop, _mavlink(), _ctx()).holds

    def test_missing_msg_id_rejected(self):
        r = _step(self.prop, _mavlink(with_msg_id=False), _ctx())
        assert not r.holds
        assert "S3 violated" in (r.reason or "")


# ---------------------------------------------------------------------------
# S4 -- signing consistency
# ---------------------------------------------------------------------------


class TestS4:
    prop = S4_SigConsistent()

    def test_signing_off_admitted(self):
        assert _step(self.prop, _mavlink(), _ctx(signing=False)).holds

    def test_signing_on_decoy_key_admitted(self):
        assert _step(
            self.prop,
            _mavlink(),
            _ctx(signing=True, decoy_key=True),
        ).holds

    def test_signing_on_operator_key_rejected(self):
        r = _step(self.prop, _mavlink(), _ctx(signing=True, decoy_key=False))
        assert not r.holds
        assert "S4 violated" in (r.reason or "")

    def test_signing_on_operator_key_but_non_mavlink_admitted(self):
        # S4 is `emit_mavlink -> sig_consistent` -- vacuous for non-MAVLink.
        a = Action(
            type=ActionType.NETWORK_RESPONSE,
            payload={"protocol": "tcp"},
            target_iface=Effector.NETWORK_LAYER_RATE_LIMITER,
            predicted_cost=_cost(),
            gen_meta=_meta(),
        )
        assert _step(self.prop, a, _ctx(signing=True, decoy_key=False)).holds


# ---------------------------------------------------------------------------
# S5 -- no shared channel during critical phase
# ---------------------------------------------------------------------------


class TestS5:
    prop = S5_NoSharedChannelInCriticalPhase()

    @pytest.mark.parametrize(
        "phase",
        [
            MissionPhase.CRUISE,
            MissionPhase.IDLE,
            MissionPhase.POST_LANDING,
        ],
    )
    def test_non_critical_shared_channel_admitted(self, phase):
        a = _mavlink(target=Effector.DECOY_MAVLINK_CHANNEL)
        assert _step(self.prop, a, _ctx(phase=phase)).holds

    @pytest.mark.parametrize(
        "phase",
        [
            MissionPhase.TAKEOFF,
            MissionPhase.LANDING,
            MissionPhase.PRECISION_HOVER,
            MissionPhase.EMERGENCY_RECOVERY,
        ],
    )
    def test_critical_shared_channel_rejected(self, phase):
        a = _mavlink(target=Effector.DECOY_MAVLINK_CHANNEL)
        r = _step(self.prop, a, _ctx(phase=phase))
        assert not r.holds
        assert "S5 violated" in (r.reason or "")

    def test_critical_isolated_channel_admitted(self):
        # DECOY_TELEMETRY_CHANNEL is NOT in SHARED_CHANNEL_EFFECTORS,
        # so it's admissible even during takeoff.
        a = _mavlink(target=Effector.DECOY_TELEMETRY_CHANNEL)
        assert _step(self.prop, a, _ctx(phase=MissionPhase.TAKEOFF)).holds


# ---------------------------------------------------------------------------
# S6 -- decoy state consistency
# ---------------------------------------------------------------------------


class TestS6:
    prop = S6_DecoyStateConsistent()

    def _identity(self) -> DecoyIdentity:
        return DecoyIdentity(
            decoy_id="d1",
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )

    def test_first_emission_admitted(self):
        a = _telemetry(fields={"alt_m": 100.0})
        ctx = _ctx(known_decoys={"d1": self._identity()})
        assert _step(self.prop, a, ctx).holds

    def test_bounded_jump_admitted(self):
        a = _telemetry(fields={"alt_m": 104.0})
        ctx = _ctx(
            known_decoys={"d1": self._identity()},
            last_emissions={"d1": {"alt_m": 100.0}},
        )
        assert _step(self.prop, a, ctx).holds

    def test_unbounded_jump_rejected(self):
        a = _telemetry(fields={"alt_m": 500.0})
        ctx = _ctx(
            known_decoys={"d1": self._identity()},
            last_emissions={"d1": {"alt_m": 100.0}},
        )
        r = _step(self.prop, a, ctx)
        assert not r.holds
        assert "S6 violated" in (r.reason or "")

    def test_non_telemetry_vacuous(self):
        assert _step(self.prop, _mavlink(), _ctx()).holds


# ---------------------------------------------------------------------------
# S7 -- uncertainty bounds template
# ---------------------------------------------------------------------------


class TestS7:
    prop = S7_UncertaintyBoundsTemplate()

    def test_low_uncertainty_no_template_admitted(self):
        # conf 0.9 -> uncertainty 0.1 <= threshold 0.3, so no template
        # restriction applies.
        a = _mavlink(conf=0.9, tid=None)
        assert _step(self.prop, a, _ctx()).holds

    def test_high_uncertainty_with_whitelisted_template_admitted(self):
        a = _mavlink(conf=0.4, tid="safe_hb_v1")
        ctx = _ctx(whitelist=frozenset({"safe_hb_v1"}))
        assert _step(self.prop, a, ctx).holds

    def test_high_uncertainty_without_template_rejected(self):
        a = _mavlink(conf=0.4, tid=None)
        r = _step(self.prop, a, _ctx())
        assert not r.holds
        assert "S7 violated" in (r.reason or "")

    def test_high_uncertainty_non_whitelisted_template_rejected(self):
        a = _mavlink(conf=0.4, tid="freeform")
        ctx = _ctx(whitelist=frozenset({"safe_hb_v1"}))
        r = _step(self.prop, a, ctx)
        assert not r.holds
        assert "S7 violated" in (r.reason or "")

    def test_non_mavlink_vacuous(self):
        # S7 antecedent requires emit_mavlink; non-MAVLink is vacuous.
        a = Action(
            type=ActionType.NETWORK_RESPONSE,
            payload={"protocol": "tcp"},
            target_iface=Effector.NETWORK_LAYER_RATE_LIMITER,
            predicted_cost=_cost(),
            gen_meta=_meta(conf=0.4, tid=None),
        )
        assert _step(self.prop, a, _ctx()).holds


# ---------------------------------------------------------------------------
# Composite SafetyMonitor
# ---------------------------------------------------------------------------


class TestSafetyMonitor:
    def test_property_ordering_is_s1_through_s7(self):
        mon = SafetyMonitor()
        assert mon.property_names == ("S1", "S2", "S3", "S4", "S5", "S6", "S7")

    def test_clean_action_admitted(self):
        mon = SafetyMonitor()
        a = _mavlink()
        r = mon.evaluate(evaluate_all(a, _ctx()), a, _ctx())
        assert r.holds
        assert r.failing_property is None

    def test_first_violation_attributed_to_s1(self):
        """Craft an action that violates S1 (flight-critical) AND S2
        (real-FC target). Monitor must report S1, the earlier property."""
        mon = SafetyMonitor()
        a = _mavlink(
            msg_id=76,
            command=176,  # SET_MODE via COMMAND_LONG (S1 fail)
            target=Effector.FLIGHT_CONTROLLER_BUS,  # (S2 fail)
        )
        r = mon.evaluate(evaluate_all(a, _ctx()), a, _ctx())
        assert not r.holds
        assert r.failing_property == "S1"

    def test_first_violation_attributed_to_s2_when_s1_ok(self):
        mon = SafetyMonitor()
        # HEARTBEAT (S1 ok) into a real FC (S2 fail).
        a = _mavlink(msg_id=0, target=Effector.FLIGHT_CONTROLLER_BUS)
        r = mon.evaluate(evaluate_all(a, _ctx()), a, _ctx())
        assert not r.holds
        assert r.failing_property == "S2"

    def test_first_violation_attributed_to_s3_when_s1_s2_ok(self):
        mon = SafetyMonitor()
        # Missing msg_id -> S3 fails; target is decoy so S2 ok; no
        # command emission so S1 ok.
        a = _mavlink(with_msg_id=False)
        r = mon.evaluate(evaluate_all(a, _ctx()), a, _ctx())
        assert not r.holds
        assert r.failing_property == "S3"

    def test_s4_reported_when_only_signing_bad(self):
        mon = SafetyMonitor()
        a = _mavlink()
        r = mon.evaluate(
            evaluate_all(a, _ctx(signing=True, decoy_key=False)),
            a,
            _ctx(signing=True, decoy_key=False),
        )
        assert not r.holds
        assert r.failing_property == "S4"

    def test_s5_reported_when_only_shared_channel_in_critical(self):
        mon = SafetyMonitor()
        a = _mavlink(target=Effector.DECOY_MAVLINK_CHANNEL)
        ctx = _ctx(phase=MissionPhase.TAKEOFF)
        r = mon.evaluate(evaluate_all(a, ctx), a, ctx)
        assert not r.holds
        assert r.failing_property == "S5"

    def test_s6_reported_when_only_decoy_jump_bad(self):
        mon = SafetyMonitor()
        identity = DecoyIdentity(
            decoy_id="d1",
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )
        a = _telemetry(fields={"alt_m": 500.0})
        ctx = _ctx(
            known_decoys={"d1": identity},
            last_emissions={"d1": {"alt_m": 100.0}},
        )
        r = mon.evaluate(evaluate_all(a, ctx), a, ctx)
        assert not r.holds
        assert r.failing_property == "S6"

    def test_s7_reported_when_only_uncertainty_bad(self):
        mon = SafetyMonitor()
        a = _mavlink(conf=0.4, tid=None)
        r = mon.evaluate(evaluate_all(a, _ctx()), a, _ctx())
        assert not r.holds
        assert r.failing_property == "S7"
