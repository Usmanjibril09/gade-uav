"""Action representation for the GADe-UAV runtime safety shield.

Implements docs/02_shield_spec.md §2 (action tuple) and §3 (forbidden
effector set). All types are plain dataclasses -- no runtime dependencies
beyond the stdlib -- so the shield can be imported before torch/scapy
are available.

The Action tuple is the ONLY object the shield accepts. Every field is
required; downstream code must not add optional fields without a spec
amendment (see §12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Action types (spec §2)
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """Enumerated action types the orchestrator may emit.

    The generator's type system MUST restrict output to this set. Any
    other value is a build-time error, not a runtime rejection.
    """

    MAVLINK_RESPONSE = "mavlink_response"
    NETWORK_RESPONSE = "network_response"
    DECOY_SERVICE_CONFIG = "decoy_service_config"
    DECOY_TELEMETRY = "decoy_telemetry"
    HONEY_BEACON = "honey_beacon"


class MissionPhase(str, Enum):
    """Mission phases used by phase_critical proposition (spec §4.1).

    The set of phases considered critical is {TAKEOFF, LANDING,
    PRECISION_HOVER, EMERGENCY_RECOVERY}; all others are non-critical.
    """

    IDLE = "idle"
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    PRECISION_HOVER = "precision_hover"
    LANDING = "landing"
    EMERGENCY_RECOVERY = "emergency_recovery"
    POST_LANDING = "post_landing"


CRITICAL_PHASES: frozenset[MissionPhase] = frozenset(
    {
        MissionPhase.TAKEOFF,
        MissionPhase.LANDING,
        MissionPhase.PRECISION_HOVER,
        MissionPhase.EMERGENCY_RECOVERY,
    }
)


# ---------------------------------------------------------------------------
# Forbidden effector set (spec §3)
# ---------------------------------------------------------------------------


class Effector(str, Enum):
    """Effector endpoints the orchestrator may route actions to.

    The FORBIDDEN subset is enforced at construction time (spec §3):
    the generator's type system MUST refuse to emit an Action whose
    target_iface is in FORBIDDEN. Runtime L1 in ShieldDecide is a
    defense-in-depth backstop that treats any FORBIDDEN sighting as
    a CRITICAL routing bug.
    """

    # Decoy-side endpoints (permitted)
    DECOY_MAVLINK_CHANNEL = "decoy_mavlink_channel"
    DECOY_TELEMETRY_CHANNEL = "decoy_telemetry_channel"
    DECOY_SERVICE_PORT = "decoy_service_port"
    HONEY_BEACON_CHANNEL = "honey_beacon_channel"
    NETWORK_LAYER_RATE_LIMITER = "network_layer_rate_limiter"

    # Real-system endpoints (FORBIDDEN -- see FORBIDDEN_EFFECTORS)
    FLIGHT_CONTROLLER_BUS = "flight_controller_bus"
    MOTOR_MIXER = "motor_mixer"
    GCS_COMMAND_CHANNEL_PRIMARY = "gcs_command_channel_primary"
    MISSION_UPLOAD_CHANNEL = "mission_upload_channel"
    PARAMETER_STORE_WRITE = "parameter_store_write"


FORBIDDEN_EFFECTORS: frozenset[Effector] = frozenset(
    {
        Effector.FLIGHT_CONTROLLER_BUS,
        Effector.MOTOR_MIXER,
        Effector.GCS_COMMAND_CHANNEL_PRIMARY,
        Effector.MISSION_UPLOAD_CHANNEL,
        Effector.PARAMETER_STORE_WRITE,
    }
)


# Effectors classified as sharing RF with real mission traffic. Used by
# the `on_shared_channel` proposition (spec §4.1). Populated conservatively:
# any decoy channel not proven to be spectrum-isolated is treated as shared.
SHARED_CHANNEL_EFFECTORS: frozenset[Effector] = frozenset(
    {
        Effector.DECOY_MAVLINK_CHANNEL,
        Effector.HONEY_BEACON_CHANNEL,
    }
)


# Effectors that are real (not decoy) flight-controller / operator channels.
# Redundant with FORBIDDEN_EFFECTORS but named to match S2 propositions.
REAL_FC_EFFECTORS: frozenset[Effector] = frozenset(
    {
        Effector.FLIGHT_CONTROLLER_BUS,
        Effector.MOTOR_MIXER,
    }
)

REAL_GCS_EFFECTORS: frozenset[Effector] = frozenset(
    {
        Effector.GCS_COMMAND_CHANNEL_PRIMARY,
        Effector.MISSION_UPLOAD_CHANNEL,
    }
)


# ---------------------------------------------------------------------------
# Cost vector (spec §5, dimension units locked in shield-spec reconciliation)
# ---------------------------------------------------------------------------

# Locked units:
#   cpu           = fraction in [0.0, 1.0] per decision
#   ram           = megabytes
#   lat           = milliseconds wall-clock per generated action
#   bw            = bytes emitted
#   energy        = joules estimated per action
#   mission_delay = milliseconds of induced GCS<->UAV command-path delay
COST_DIMENSIONS: tuple[str, ...] = (
    "cpu",
    "ram",
    "lat",
    "bw",
    "energy",
    "mission_delay",
)


@dataclass(frozen=True)
class CostVector:
    """Predicted per-dimension cost vector (spec §2 predicted_cost).

    All fields are point predictions from the cost model M3. Conformal
    upper bounds are computed by the conformal gate; they are NOT stored
    here to keep the action tuple pure (a UCB depends on the calibration
    state, not on the action alone).
    """

    cpu: float
    ram: float
    lat: float
    bw: float
    energy: float
    mission_delay: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in COST_DIMENSIONS}


# ---------------------------------------------------------------------------
# Generator metadata (spec §2 gen_meta)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenMeta:
    """Provenance for the generated action.

    - confidence: joint generator+scorer uncertainty score in [0, 1].
      Used by the `uncertainty_low` proposition (S7).
    - schema_version: JSON schema version applied to `payload`.
    - generation_context_hash: hex digest of the belief/context vector
      the generator conditioned on. Enables per-context ablation and
      post-hoc audit.
    - template_id: identifier of the template used, or None if the
      generator sampled freely. Used by `emitted_from_bounded_template_only`
      (S7): only template_ids in the bounded whitelist satisfy the prop.
    """

    confidence: float
    schema_version: str
    generation_context_hash: str
    template_id: str | None = None


# ---------------------------------------------------------------------------
# The Action tuple (spec §2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One candidate action emitted by the orchestrator (d*_t in spec §2).

    Immutable by design (frozen=True). Every field is required; validation
    is by construction, not by post-hoc mutation.
    """

    type: ActionType
    payload: dict[str, Any]
    target_iface: Effector
    predicted_cost: CostVector
    gen_meta: GenMeta

    def __post_init__(self) -> None:
        # Fail-fast type-level checks. These are cheap and prevent whole
        # classes of downstream bug from ever hitting the shield.
        if not isinstance(self.type, ActionType):
            raise TypeError(f"type must be ActionType, got {type(self.type)}")
        if not isinstance(self.target_iface, Effector):
            raise TypeError(
                f"target_iface must be Effector, got {type(self.target_iface)}"
            )
        if not isinstance(self.payload, dict):
            raise TypeError(f"payload must be dict, got {type(self.payload)}")
        if not isinstance(self.predicted_cost, CostVector):
            raise TypeError(
                f"predicted_cost must be CostVector, got {type(self.predicted_cost)}"
            )
        if not isinstance(self.gen_meta, GenMeta):
            raise TypeError(f"gen_meta must be GenMeta, got {type(self.gen_meta)}")
        if not (0.0 <= self.gen_meta.confidence <= 1.0):
            raise ValueError(
                f"gen_meta.confidence must be in [0, 1], "
                f"got {self.gen_meta.confidence}"
            )


# ---------------------------------------------------------------------------
# Shield context (spec §4.1 -- propositions read these)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoyIdentity:
    """One decoy the shield tracks state for (used by S6).

    Bounds are declared per decoy at spawn time and cannot change during
    a mission. Used to detect impossible field-value jumps in emitted
    telemetry (spec §4.1 decoy_state_consistent).
    """

    decoy_id: str
    tau_alt_m: float  # max altitude jump per emission, meters
    tau_pos_m: float  # max position jump per emission, meters
    tau_bat_pct: float  # max battery percent jump per emission
    tau_att_rad_s: float  # max attitude rate change per emission


@dataclass(frozen=True)
class ShieldContext:
    """Read-only context passed alongside each Action to the shield.

    Contains everything a proposition may need to make its decision.
    Extending this requires a spec amendment.
    """

    mission_phase: MissionPhase
    signing_active: bool  # MAVLink v2 signing on?
    decoy_key_active: bool  # is the signing key a decoy key?
    known_decoys: dict[str, DecoyIdentity]  # decoy_id -> identity
    last_emission_per_decoy: dict[str, dict[str, float]] = field(default_factory=dict)
    # last_emission_per_decoy[decoy_id] = {"alt_m": ..., "lat_deg": ...,
    #   "lon_deg": ..., "bat_pct": ..., "att_rate_rad_s": ...}
    # Absent decoy_id => first emission for that decoy (no bound check).

    # Uncertainty threshold below which uncertainty_low holds.
    uncertainty_threshold: float = 0.3

    # Whitelist of template_ids allowed under S7 (bounded low-impact set).
    bounded_template_whitelist: frozenset[str] = field(default_factory=frozenset)

    # Optional upstream integrity and freshness metadata. Callers that receive
    # mission state over a link should stamp expires_at_s on the shield clock.
    mission_state_trusted: bool = True
    mission_state_expires_at_s: float | None = None
    cyber_state_confidence: float | None = None


# ---------------------------------------------------------------------------
# MAVLink message-ID constants (spec §4.1)
#
# These are the ONLY MAVLink IDs the shield references directly. Names
# and IDs from MAVLink common.xml, cross-checked against the GUIDE
# int_to_msg_id pickle observed in Phase 1.
# ---------------------------------------------------------------------------

MSG_ID_SET_MODE: int = 11
MSG_ID_PARAM_SET: int = 23
MSG_ID_MISSION_WRITE_PARTIAL_LIST: int = 38
MSG_ID_MISSION_ITEM: int = 39
MSG_ID_MISSION_COUNT: int = 44
MSG_ID_MISSION_ITEM_INT: int = 73
MSG_ID_COMMAND_LONG: int = 76

MAV_CMD_NAV_RETURN_TO_LAUNCH: int = 20
MAV_CMD_NAV_LAND: int = 21
MAV_CMD_DO_SET_MODE: int = 176
MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN: int = 246
MAV_CMD_COMPONENT_ARM_DISARM: int = 400
