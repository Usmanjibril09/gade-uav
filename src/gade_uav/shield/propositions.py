"""Atomic propositions over (Action, ShieldContext) for the safety shield.

Implements docs/02_shield_spec.md §4.1. Each proposition is a PURE
function returning bool. No side effects, no mutation of the context.
S6 (decoy_state_consistent) reads `context.last_emission_per_decoy` but
does not modify it -- the LTL monitor is responsible for updating that
state after each admitted action.

The 16 propositions:
    emit_setmode, emit_paramset, emit_reboot, emit_rtl_land, emit_arm,
    emit_mission_write, emit_mavlink,
    target_real_gcs, target_real_fc, on_shared_channel,
    schema_valid, sig_consistent, phase_critical,
    decoy_state_consistent, uncertainty_low,
    emitted_from_bounded_template_only

`budget_ucb_ok` from §4.1 is NOT here -- it depends on the conformal
gate's calibration state and lives in conformal_gate.py.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from gade_uav.shield.actions import (
    CRITICAL_PHASES,
    FORBIDDEN_EFFECTORS,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_RETURN_TO_LAUNCH,
    MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
    MSG_ID_COMMAND_LONG,
    MSG_ID_MISSION_COUNT,
    MSG_ID_MISSION_ITEM,
    MSG_ID_MISSION_ITEM_INT,
    MSG_ID_MISSION_WRITE_PARTIAL_LIST,
    MSG_ID_PARAM_SET,
    MSG_ID_SET_MODE,
    REAL_FC_EFFECTORS,
    REAL_GCS_EFFECTORS,
    SHARED_CHANNEL_EFFECTORS,
    Action,
    ActionType,
    ShieldContext,
)


# ---------------------------------------------------------------------------
# JSON schemas per action type (spec §4.1 schema_valid)
#
# Kept minimal by design -- these are shape checks, not deep semantic
# validation. Semantic validation is the generator's responsibility;
# the shield checks that whatever came out is at least parseable.
# ---------------------------------------------------------------------------

_MAVLINK_PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["msg_id", "fields"],
    "properties": {
        "msg_id": {"type": "integer", "minimum": 0, "maximum": 65535},
        "fields": {"type": "object"},
        # Optional: MAV_CMD command id when msg_id == COMMAND_LONG (76).
        "command": {"type": "integer", "minimum": 0, "maximum": 65535},
    },
    "additionalProperties": True,
}

_NETWORK_PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["protocol"],
    "properties": {
        "protocol": {"type": "string"},
        "src": {"type": "string"},
        "dst": {"type": "string"},
        "body": {},
    },
    "additionalProperties": True,
}

_DECOY_SERVICE_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service_name", "config"],
    "properties": {
        "service_name": {"type": "string"},
        "config": {"type": "object"},
    },
    "additionalProperties": True,
}

_DECOY_TELEMETRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["decoy_id", "fields"],
    "properties": {
        "decoy_id": {"type": "string"},
        "fields": {"type": "object"},
    },
    "additionalProperties": True,
}

_HONEY_BEACON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["beacon_id"],
    "properties": {
        "beacon_id": {"type": "string"},
        "advertise": {"type": "object"},
    },
    "additionalProperties": True,
}

_SCHEMAS_BY_TYPE: dict[ActionType, dict[str, Any]] = {
    ActionType.MAVLINK_RESPONSE: _MAVLINK_PAYLOAD_SCHEMA,
    ActionType.NETWORK_RESPONSE: _NETWORK_PAYLOAD_SCHEMA,
    ActionType.DECOY_SERVICE_CONFIG: _DECOY_SERVICE_CONFIG_SCHEMA,
    ActionType.DECOY_TELEMETRY: _DECOY_TELEMETRY_SCHEMA,
    ActionType.HONEY_BEACON: _HONEY_BEACON_SCHEMA,
}


# ---------------------------------------------------------------------------
# MAVLink emission propositions (spec §4.1)
# ---------------------------------------------------------------------------


def _is_mavlink(action: Action) -> bool:
    return action.type == ActionType.MAVLINK_RESPONSE


def _mav_msg_id(action: Action) -> int | None:
    if not _is_mavlink(action):
        return None
    mid = action.payload.get("msg_id")
    return int(mid) if isinstance(mid, int) else None


def _mav_command(action: Action) -> int | None:
    """Return the MAV_CMD id inside a COMMAND_LONG, or None if not applicable."""
    if _mav_msg_id(action) != MSG_ID_COMMAND_LONG:
        return None
    cmd = action.payload.get("command")
    return int(cmd) if isinstance(cmd, int) else None


def emit_mavlink(action: Action, ctx: ShieldContext) -> bool:
    """P: action is any MAVLink emission."""
    return _is_mavlink(action)


def emit_setmode(action: Action, ctx: ShieldContext) -> bool:
    """P: emits SET_MODE (11) or COMMAND_LONG with MAV_CMD_DO_SET_MODE (176)."""
    mid = _mav_msg_id(action)
    if mid == MSG_ID_SET_MODE:
        return True
    return _mav_command(action) == MAV_CMD_DO_SET_MODE


def emit_paramset(action: Action, ctx: ShieldContext) -> bool:
    """P: emits PARAM_SET (23)."""
    return _mav_msg_id(action) == MSG_ID_PARAM_SET


def emit_reboot(action: Action, ctx: ShieldContext) -> bool:
    """P: emits COMMAND_LONG with MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN (246)."""
    return _mav_command(action) == MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN


def emit_rtl_land(action: Action, ctx: ShieldContext) -> bool:
    """P: emits COMMAND_LONG with MAV_CMD_NAV_RETURN_TO_LAUNCH (20) or NAV_LAND (21)."""
    cmd = _mav_command(action)
    return cmd in (MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_CMD_NAV_LAND)


def emit_arm(action: Action, ctx: ShieldContext) -> bool:
    """P: emits COMMAND_LONG with MAV_CMD_COMPONENT_ARM_DISARM (400)."""
    return _mav_command(action) == MAV_CMD_COMPONENT_ARM_DISARM


_MISSION_WRITE_MSG_IDS: frozenset[int] = frozenset(
    {
        MSG_ID_MISSION_ITEM,
        MSG_ID_MISSION_ITEM_INT,
        MSG_ID_MISSION_WRITE_PARTIAL_LIST,
        MSG_ID_MISSION_COUNT,
    }
)


def emit_mission_write(action: Action, ctx: ShieldContext) -> bool:
    """P: emits any of MISSION_ITEM/MISSION_ITEM_INT/PARTIAL_LIST/COUNT."""
    mid = _mav_msg_id(action)
    return mid in _MISSION_WRITE_MSG_IDS


# ---------------------------------------------------------------------------
# Effector-routing propositions (spec §4.1)
# ---------------------------------------------------------------------------


def target_real_fc(action: Action, ctx: ShieldContext) -> bool:
    """P: effector is a real flight-controller endpoint."""
    return action.target_iface in REAL_FC_EFFECTORS


def target_real_gcs(action: Action, ctx: ShieldContext) -> bool:
    """P: effector is a real operator/GCS channel."""
    return action.target_iface in REAL_GCS_EFFECTORS


def target_forbidden(action: Action, ctx: ShieldContext) -> bool:
    """P: effector is anywhere in the forbidden set (§3).

    Not one of the 16 named propositions -- used by L1 in ShieldDecide.
    """
    return action.target_iface in FORBIDDEN_EFFECTORS


def on_shared_channel(action: Action, ctx: ShieldContext) -> bool:
    """P: effector shares RF spectrum with real mission traffic."""
    return action.target_iface in SHARED_CHANNEL_EFFECTORS


# ---------------------------------------------------------------------------
# Payload / signing propositions (spec §4.1)
# ---------------------------------------------------------------------------


def schema_valid(action: Action, ctx: ShieldContext) -> bool:
    """P: payload conforms to the JSON schema for its action type."""
    schema = _SCHEMAS_BY_TYPE.get(action.type)
    if schema is None:
        return False
    try:
        jsonschema.validate(instance=action.payload, schema=schema)
    except jsonschema.ValidationError:
        return False
    return True


def sig_consistent(action: Action, ctx: ShieldContext) -> bool:
    """P: if signing is active on a MAVLink emission, decoy key is used.

    Vacuously true when the action is not MAVLink or signing is inactive.
    Non-MAVLink actions cannot violate S4 (spec: S4 is `emit_mavlink -> sig_consistent`).
    """
    if not _is_mavlink(action):
        return True
    if not ctx.signing_active:
        return True
    return ctx.decoy_key_active


# ---------------------------------------------------------------------------
# Mission-phase and decoy-state propositions (spec §4.1)
# ---------------------------------------------------------------------------


def phase_critical(action: Action, ctx: ShieldContext) -> bool:
    """P: current mission phase is in the critical set."""
    return ctx.mission_phase in CRITICAL_PHASES


# Field keys the shield reads from decoy telemetry payloads for S6.
# Absence of a field in the payload is not a violation -- only present
# fields are checked against declared bounds.
_TELEMETRY_FIELDS_AND_BOUNDS: tuple[tuple[str, str], ...] = (
    ("alt_m", "tau_alt_m"),
    ("bat_pct", "tau_bat_pct"),
    ("att_rate_rad_s", "tau_att_rad_s"),
)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two (lat, lon) degree pairs."""
    from math import asin, cos, radians, sin, sqrt

    r = 6_371_000.0
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r * asin(sqrt(a))


def decoy_state_consistent(action: Action, ctx: ShieldContext) -> bool:
    """P: telemetry field jumps stay within per-decoy declared bounds (S6).

    Applies only to DECOY_TELEMETRY actions. First emission from a decoy
    is always consistent (no prior state to compare against). Missing
    fields in the payload are ignored; only present fields are checked.
    """
    if action.type != ActionType.DECOY_TELEMETRY:
        return True

    decoy_id = action.payload.get("decoy_id")
    if not isinstance(decoy_id, str):
        return False  # DECOY_TELEMETRY schema requires decoy_id

    identity = ctx.known_decoys.get(decoy_id)
    if identity is None:
        # Unknown decoy identity -- cannot verify bounds. Treat as violation.
        return False

    prev = ctx.last_emission_per_decoy.get(decoy_id)
    if prev is None:
        return True  # first emission, no bound to check

    fields = action.payload.get("fields")
    if not isinstance(fields, dict):
        return False

    # Scalar-field jump checks.
    for field_name, bound_attr in _TELEMETRY_FIELDS_AND_BOUNDS:
        if field_name in fields and field_name in prev:
            try:
                cur = float(fields[field_name])
                old = float(prev[field_name])
            except (TypeError, ValueError):
                return False
            bound = float(getattr(identity, bound_attr))
            if abs(cur - old) > bound:
                return False

    # Position jump via haversine when lat/lon present in both.
    if all(k in fields for k in ("lat_deg", "lon_deg")) and all(
        k in prev for k in ("lat_deg", "lon_deg")
    ):
        try:
            d = _haversine_m(
                float(prev["lat_deg"]),
                float(prev["lon_deg"]),
                float(fields["lat_deg"]),
                float(fields["lon_deg"]),
            )
        except (TypeError, ValueError):
            return False
        if d > float(identity.tau_pos_m):
            return False

    return True


# ---------------------------------------------------------------------------
# Uncertainty and template propositions (spec §4.1)
# ---------------------------------------------------------------------------


def uncertainty_low(action: Action, ctx: ShieldContext) -> bool:
    """Require both candidate and supplied cyber-state confidence to pass."""
    confidence = ctx.cyber_state_confidence
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        return False
    threshold = 1.0 - ctx.uncertainty_threshold
    return action.gen_meta.confidence >= threshold and (
        confidence is None or confidence >= threshold
    )


def emitted_from_bounded_template_only(action: Action, ctx: ShieldContext) -> bool:
    """P: action's template_id is in the bounded low-impact whitelist.

    An action without a template_id (free-form generator output) does
    NOT satisfy the proposition -- S7 requires template-restricted
    emission under high uncertainty.
    """
    tid = action.gen_meta.template_id
    if tid is None:
        return False
    return tid in ctx.bounded_template_whitelist


# ---------------------------------------------------------------------------
# Public API: evaluate all propositions at once (used by LTL monitor)
# ---------------------------------------------------------------------------


def evaluate_all(action: Action, ctx: ShieldContext) -> dict[str, bool]:
    """Evaluate every named proposition. Returns a dict keyed by prop name.

    Used by the LTL monitor to step its state machines and by tests to
    inspect the atomic decisions. Keep the key set stable -- state
    machines index it by string.
    """
    return {
        "emit_mavlink": emit_mavlink(action, ctx),
        "emit_setmode": emit_setmode(action, ctx),
        "emit_paramset": emit_paramset(action, ctx),
        "emit_reboot": emit_reboot(action, ctx),
        "emit_rtl_land": emit_rtl_land(action, ctx),
        "emit_arm": emit_arm(action, ctx),
        "emit_mission_write": emit_mission_write(action, ctx),
        "target_real_fc": target_real_fc(action, ctx),
        "target_real_gcs": target_real_gcs(action, ctx),
        "target_forbidden": target_forbidden(action, ctx),
        "on_shared_channel": on_shared_channel(action, ctx),
        "schema_valid": schema_valid(action, ctx),
        "sig_consistent": sig_consistent(action, ctx),
        "phase_critical": phase_critical(action, ctx),
        "decoy_state_consistent": decoy_state_consistent(action, ctx),
        "uncertainty_low": uncertainty_low(action, ctx),
        "emitted_from_bounded_template_only": emitted_from_bounded_template_only(
            action, ctx
        ),
    }
