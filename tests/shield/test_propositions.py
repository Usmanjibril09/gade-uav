"""Unit tests for atomic propositions (spec §4.1).

Every proposition is exercised in both its holding and non-holding
form. Failures here indicate a spec-implementation mismatch and MUST
be fixed before any state-machine code is trusted.
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
from gade_uav.shield.propositions import (
    decoy_state_consistent,
    emit_arm,
    emit_mavlink,
    emit_mission_write,
    emit_paramset,
    emit_reboot,
    emit_rtl_land,
    emit_setmode,
    emitted_from_bounded_template_only,
    evaluate_all,
    on_shared_channel,
    phase_critical,
    schema_valid,
    sig_consistent,
    target_forbidden,
    target_real_fc,
    target_real_gcs,
    uncertainty_low,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cost() -> CostVector:
    return CostVector(
        cpu=0.1, ram=10.0, lat=1.0, bw=100.0, energy=0.5, mission_delay=0.0
    )


def _meta(conf: float = 0.9, tid: str | None = None) -> GenMeta:
    return GenMeta(
        confidence=conf,
        schema_version="v1",
        generation_context_hash="deadbeef",
        template_id=tid,
    )


def _mavlink_action(
    msg_id: int,
    command: int | None = None,
    target: Effector = Effector.DECOY_MAVLINK_CHANNEL,
    conf: float = 0.9,
    tid: str | None = None,
) -> Action:
    payload: dict = {"msg_id": msg_id, "fields": {}}
    if command is not None:
        payload["command"] = command
    return Action(
        type=ActionType.MAVLINK_RESPONSE,
        payload=payload,
        target_iface=target,
        predicted_cost=_cost(),
        gen_meta=_meta(conf=conf, tid=tid),
    )


def _decoy_telemetry_action(
    decoy_id: str,
    fields: dict,
    target: Effector = Effector.DECOY_TELEMETRY_CHANNEL,
) -> Action:
    return Action(
        type=ActionType.DECOY_TELEMETRY,
        payload={"decoy_id": decoy_id, "fields": fields},
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


# ---------------------------------------------------------------------------
# MAVLink emission propositions
# ---------------------------------------------------------------------------


class TestMavlinkEmissionProps:
    def test_emit_mavlink_true_for_mavlink(self):
        assert emit_mavlink(_mavlink_action(0), _ctx()) is True

    def test_emit_mavlink_false_for_network(self):
        a = Action(
            type=ActionType.NETWORK_RESPONSE,
            payload={"protocol": "tcp"},
            target_iface=Effector.NETWORK_LAYER_RATE_LIMITER,
            predicted_cost=_cost(),
            gen_meta=_meta(),
        )
        assert emit_mavlink(a, _ctx()) is False

    def test_emit_setmode_direct(self):
        # SET_MODE msg_id 11
        assert emit_setmode(_mavlink_action(11), _ctx()) is True

    def test_emit_setmode_via_command_long(self):
        # COMMAND_LONG (76) + MAV_CMD_DO_SET_MODE (176)
        assert emit_setmode(_mavlink_action(76, command=176), _ctx()) is True

    def test_emit_setmode_false_for_heartbeat(self):
        assert emit_setmode(_mavlink_action(0), _ctx()) is False

    def test_emit_paramset(self):
        assert emit_paramset(_mavlink_action(23), _ctx()) is True
        assert emit_paramset(_mavlink_action(0), _ctx()) is False

    def test_emit_reboot(self):
        # COMMAND_LONG + MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN (246)
        assert emit_reboot(_mavlink_action(76, command=246), _ctx()) is True
        assert emit_reboot(_mavlink_action(76, command=176), _ctx()) is False
        assert emit_reboot(_mavlink_action(0), _ctx()) is False

    def test_emit_rtl_land(self):
        # RTL (20) or LAND (21)
        assert emit_rtl_land(_mavlink_action(76, command=20), _ctx()) is True
        assert emit_rtl_land(_mavlink_action(76, command=21), _ctx()) is True
        assert emit_rtl_land(_mavlink_action(76, command=176), _ctx()) is False

    def test_emit_arm(self):
        # MAV_CMD_COMPONENT_ARM_DISARM (400)
        assert emit_arm(_mavlink_action(76, command=400), _ctx()) is True
        assert emit_arm(_mavlink_action(76, command=20), _ctx()) is False

    def test_emit_mission_write(self):
        for mid in (38, 39, 44, 73):
            assert emit_mission_write(_mavlink_action(mid), _ctx()) is True
        assert emit_mission_write(_mavlink_action(0), _ctx()) is False


# ---------------------------------------------------------------------------
# Effector-routing propositions
# ---------------------------------------------------------------------------


class TestRoutingProps:
    def test_target_real_fc(self):
        a = _mavlink_action(0, target=Effector.FLIGHT_CONTROLLER_BUS)
        assert target_real_fc(a, _ctx()) is True
        assert target_real_gcs(a, _ctx()) is False

    def test_target_real_gcs(self):
        a = _mavlink_action(0, target=Effector.GCS_COMMAND_CHANNEL_PRIMARY)
        assert target_real_gcs(a, _ctx()) is True
        assert target_real_fc(a, _ctx()) is False

    def test_target_forbidden_covers_all_forbidden(self):
        forbidden = [
            Effector.FLIGHT_CONTROLLER_BUS,
            Effector.MOTOR_MIXER,
            Effector.GCS_COMMAND_CHANNEL_PRIMARY,
            Effector.MISSION_UPLOAD_CHANNEL,
            Effector.PARAMETER_STORE_WRITE,
        ]
        for eff in forbidden:
            assert target_forbidden(_mavlink_action(0, target=eff), _ctx()) is True

    def test_target_forbidden_false_for_decoy(self):
        a = _mavlink_action(0, target=Effector.DECOY_MAVLINK_CHANNEL)
        assert target_forbidden(a, _ctx()) is False

    def test_on_shared_channel(self):
        a = _mavlink_action(0, target=Effector.DECOY_MAVLINK_CHANNEL)
        assert on_shared_channel(a, _ctx()) is True

    def test_not_on_shared_channel(self):
        a = _mavlink_action(0, target=Effector.DECOY_TELEMETRY_CHANNEL)
        assert on_shared_channel(a, _ctx()) is False


# ---------------------------------------------------------------------------
# Schema and signing propositions
# ---------------------------------------------------------------------------


class TestSchemaAndSigningProps:
    def test_schema_valid_mavlink(self):
        assert schema_valid(_mavlink_action(0), _ctx()) is True

    def test_schema_invalid_missing_msg_id(self):
        a = Action(
            type=ActionType.MAVLINK_RESPONSE,
            payload={"fields": {}},  # missing msg_id
            target_iface=Effector.DECOY_MAVLINK_CHANNEL,
            predicted_cost=_cost(),
            gen_meta=_meta(),
        )
        assert schema_valid(a, _ctx()) is False

    def test_schema_valid_decoy_telemetry(self):
        a = _decoy_telemetry_action("d1", {"alt_m": 100.0})
        assert schema_valid(a, _ctx()) is True

    def test_schema_invalid_decoy_telemetry_missing_id(self):
        a = Action(
            type=ActionType.DECOY_TELEMETRY,
            payload={"fields": {}},  # missing decoy_id
            target_iface=Effector.DECOY_TELEMETRY_CHANNEL,
            predicted_cost=_cost(),
            gen_meta=_meta(),
        )
        assert schema_valid(a, _ctx()) is False

    def test_sig_consistent_when_signing_off(self):
        a = _mavlink_action(0)
        assert sig_consistent(a, _ctx(signing=False)) is True

    def test_sig_inconsistent_when_signing_on_with_operator_key(self):
        a = _mavlink_action(0)
        assert sig_consistent(a, _ctx(signing=True, decoy_key=False)) is False

    def test_sig_consistent_when_signing_on_with_decoy_key(self):
        a = _mavlink_action(0)
        assert sig_consistent(a, _ctx(signing=True, decoy_key=True)) is True

    def test_sig_consistent_vacuous_for_nonmavlink(self):
        a = Action(
            type=ActionType.NETWORK_RESPONSE,
            payload={"protocol": "tcp"},
            target_iface=Effector.NETWORK_LAYER_RATE_LIMITER,
            predicted_cost=_cost(),
            gen_meta=_meta(),
        )
        # Even with signing on and no decoy key, non-mavlink is vacuously
        # sig_consistent -- S4 is `emit_mavlink -> sig_consistent`.
        assert sig_consistent(a, _ctx(signing=True, decoy_key=False)) is True


# ---------------------------------------------------------------------------
# Phase and template propositions
# ---------------------------------------------------------------------------


class TestPhaseAndTemplateProps:
    @pytest.mark.parametrize(
        "phase,expected",
        [
            (MissionPhase.CRUISE, False),
            (MissionPhase.IDLE, False),
            (MissionPhase.POST_LANDING, False),
            (MissionPhase.TAKEOFF, True),
            (MissionPhase.LANDING, True),
            (MissionPhase.PRECISION_HOVER, True),
            (MissionPhase.EMERGENCY_RECOVERY, True),
        ],
    )
    def test_phase_critical(self, phase, expected):
        a = _mavlink_action(0)
        assert phase_critical(a, _ctx(phase=phase)) is expected

    def test_uncertainty_low_true(self):
        # conf 0.9 -> uncertainty 0.1 <= threshold 0.3
        a = _mavlink_action(0, conf=0.9)
        assert uncertainty_low(a, _ctx(unc_thresh=0.3)) is True

    def test_uncertainty_low_false(self):
        # conf 0.5 -> uncertainty 0.5 > threshold 0.3
        a = _mavlink_action(0, conf=0.5)
        assert uncertainty_low(a, _ctx(unc_thresh=0.3)) is False

    def test_bounded_template_holds(self):
        a = _mavlink_action(0, tid="safe_heartbeat_v1")
        ctx = _ctx(whitelist=frozenset({"safe_heartbeat_v1"}))
        assert emitted_from_bounded_template_only(a, ctx) is True

    def test_bounded_template_fails_without_tid(self):
        a = _mavlink_action(0, tid=None)
        ctx = _ctx(whitelist=frozenset({"safe_heartbeat_v1"}))
        assert emitted_from_bounded_template_only(a, ctx) is False

    def test_bounded_template_fails_not_whitelisted(self):
        a = _mavlink_action(0, tid="freeform_response")
        ctx = _ctx(whitelist=frozenset({"safe_heartbeat_v1"}))
        assert emitted_from_bounded_template_only(a, ctx) is False


# ---------------------------------------------------------------------------
# Decoy-state proposition (S6 backing)
# ---------------------------------------------------------------------------


class TestDecoyStateConsistent:
    def _identity(self) -> DecoyIdentity:
        return DecoyIdentity(
            decoy_id="d1",
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )

    def test_first_emission_always_consistent(self):
        a = _decoy_telemetry_action("d1", {"alt_m": 100.0})
        ctx = _ctx(known_decoys={"d1": self._identity()})
        assert decoy_state_consistent(a, ctx) is True

    def test_altitude_jump_within_bound(self):
        a = _decoy_telemetry_action("d1", {"alt_m": 103.0})
        ctx = _ctx(
            known_decoys={"d1": self._identity()},
            last_emissions={"d1": {"alt_m": 100.0}},
        )
        assert decoy_state_consistent(a, ctx) is True

    def test_altitude_jump_exceeds_bound(self):
        a = _decoy_telemetry_action("d1", {"alt_m": 200.0})
        ctx = _ctx(
            known_decoys={"d1": self._identity()},
            last_emissions={"d1": {"alt_m": 100.0}},
        )
        assert decoy_state_consistent(a, ctx) is False

    def test_position_jump_within_bound(self):
        # ~1 meter jump
        a = _decoy_telemetry_action("d1", {"lat_deg": 43.6533, "lon_deg": -79.3832})
        ctx = _ctx(
            known_decoys={"d1": self._identity()},
            last_emissions={"d1": {"lat_deg": 43.6532, "lon_deg": -79.3832}},
        )
        assert decoy_state_consistent(a, ctx) is True

    def test_position_jump_exceeds_bound(self):
        # >100 km jump
        a = _decoy_telemetry_action("d1", {"lat_deg": 44.0, "lon_deg": -79.3832})
        ctx = _ctx(
            known_decoys={"d1": self._identity()},
            last_emissions={"d1": {"lat_deg": 43.6532, "lon_deg": -79.3832}},
        )
        assert decoy_state_consistent(a, ctx) is False

    def test_unknown_decoy_rejected(self):
        a = _decoy_telemetry_action("unknown_decoy", {"alt_m": 100.0})
        ctx = _ctx(known_decoys={})
        assert decoy_state_consistent(a, ctx) is False

    def test_vacuous_for_nonelemetry(self):
        a = _mavlink_action(0)
        ctx = _ctx()
        assert decoy_state_consistent(a, ctx) is True


# ---------------------------------------------------------------------------
# evaluate_all wiring smoke test
# ---------------------------------------------------------------------------


def test_evaluate_all_returns_17_keys():
    a = _mavlink_action(0)
    result = evaluate_all(a, _ctx())
    expected_keys = {
        "emit_mavlink",
        "emit_setmode",
        "emit_paramset",
        "emit_reboot",
        "emit_rtl_land",
        "emit_arm",
        "emit_mission_write",
        "target_real_fc",
        "target_real_gcs",
        "target_forbidden",
        "on_shared_channel",
        "schema_valid",
        "sig_consistent",
        "phase_critical",
        "decoy_state_consistent",
        "uncertainty_low",
        "emitted_from_bounded_template_only",
    }
    assert set(result.keys()) == expected_keys
    assert len(result) == 17
