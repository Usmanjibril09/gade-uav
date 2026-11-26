"""Fallback-action policy loader for the GADe-UAV shield.

Implements docs/02_shield_spec.md §8: rule-based lookup
    (mission_phase x primary_attack_family) -> FallbackAction

Source of truth: configs/fallback_policy.yaml.

Design invariants:
    - Fallback is RULE-BASED, not learned. The lookup table is fully
      populated at load time or construction fails. Missing cells are
      a spec error, not a runtime default.
    - The four fallback actions are the pre-verified set from spec §8;
      any addition requires a spec amendment.
    - `default_action` in the YAML is used ONLY when the attack family
      is `unknown` -- it never masks a missing (known-attack x phase)
      cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from gade_uav.shield.actions import MissionPhase


class FallbackAction(str, Enum):
    """Pre-verified fallback actions (spec §8)."""

    NO_OP = "NO_OP"
    STATIC_HONEY_RESPONSE = "STATIC_HONEY_RESPONSE"
    RATE_LIMIT_SOURCE = "RATE_LIMIT_SOURCE"
    HAND_OFF_TO_IDS = "HAND_OFF_TO_IDS"


# Canonical attack-family keys accepted in the YAML table. Kept in sync
# with docs/01_threat_model.md (T1..T18). The value here is the exact
# YAML key; do NOT rewrite these to canonical labels -- the fallback
# table addresses attack FAMILIES, not per-instance canonical labels.
KNOWN_ATTACK_FAMILIES: frozenset[str] = frozenset(
    {
        "T1_Reconnaissance",
        "T2_DoS",
        "T3_DDoS",
        "T4_UDP_Flooding",
        "T5_ICMP_Flooding",
        "T6_Deauthentication",
        "T7_MITM",
        "T8_Replay",
        "T9_BruteForce",
        "T10_Jamming",
        "T11_GPS_Spoofing",
        "T12_FakeLanding",
        "T13_MAVLink_Heartbeat_Flooding",
        "T14_MAVLink_Ping_Flooding",
        "T15_MAVLink_Request_Flooding",
        "T16_MAVLink_Corruption",
        "T17_MAVLink_Suspension",
        "T18_EvilTwin",
    }
)

# Special key for when the orchestrator cannot classify the attack.
UNKNOWN_FAMILY_KEY: str = "unknown"


@dataclass(frozen=True)
class FallbackPolicy:
    """Immutable lookup table for fallback selection.

    Load with `FallbackPolicy.from_yaml(path)`. Directly-constructed
    instances are unusual and reserved for tests.
    """

    default_action: FallbackAction
    # table[attack_family][phase] -> FallbackAction. Fully populated
    # for every (family in KNOWN_ATTACK_FAMILIES + "unknown") x
    # (phase in MissionPhase).
    table: dict[str, dict[MissionPhase, FallbackAction]]

    def __post_init__(self) -> None:
        expected_families = KNOWN_ATTACK_FAMILIES | {UNKNOWN_FAMILY_KEY}
        missing_families = expected_families - set(self.table.keys())
        if missing_families:
            raise ValueError(
                f"fallback policy missing attack families: {sorted(missing_families)}"
            )
        extra_families = set(self.table.keys()) - expected_families
        if extra_families:
            raise ValueError(
                f"fallback policy contains unknown attack families: "
                f"{sorted(extra_families)}"
            )
        expected_phases = set(MissionPhase)
        for family, per_phase in self.table.items():
            missing_phases = expected_phases - set(per_phase.keys())
            if missing_phases:
                raise ValueError(
                    f"fallback policy for {family!r} missing phases: "
                    f"{sorted(p.value for p in missing_phases)}"
                )

    def select(
        self,
        attack_family: str | None,
        phase: MissionPhase,
    ) -> FallbackAction:
        """Look up the fallback for (attack_family, phase).

        - If `attack_family` is None or not in KNOWN_ATTACK_FAMILIES,
          uses the `unknown` row. This is a deliberate lookup, not a
          default-fill: the `unknown` row is fully specified in the YAML.
        - If the `unknown` row itself somehow lacks the phase, returns
          `default_action` as a last resort.
        """
        key = (
            attack_family
            if attack_family in KNOWN_ATTACK_FAMILIES
            else UNKNOWN_FAMILY_KEY
        )
        row = self.table.get(key, {})
        return row.get(phase, self.default_action)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FallbackPolicy":
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level must be a mapping")

        default_str = raw.get("default_action")
        if not isinstance(default_str, str):
            raise ValueError(f"{path}: default_action must be a string")
        default_action = FallbackAction(default_str)

        raw_table = raw.get("table")
        if not isinstance(raw_table, dict):
            raise ValueError(f"{path}: table must be a mapping")

        table: dict[str, dict[MissionPhase, FallbackAction]] = {}
        for family, per_phase in raw_table.items():
            if not isinstance(per_phase, dict):
                raise ValueError(
                    f"{path}: table[{family!r}] must be a mapping of phase->action"
                )
            phase_map: dict[MissionPhase, FallbackAction] = {}
            for phase_str, action_str in per_phase.items():
                try:
                    phase = MissionPhase(phase_str)
                except ValueError as e:
                    raise ValueError(
                        f"{path}: table[{family!r}] has unknown phase {phase_str!r}"
                    ) from e
                try:
                    action = FallbackAction(action_str)
                except ValueError as e:
                    raise ValueError(
                        f"{path}: table[{family!r}][{phase_str!r}] has unknown "
                        f"action {action_str!r}"
                    ) from e
                phase_map[phase] = action
            table[family] = phase_map

        return cls(default_action=default_action, table=table)


__all__ = [
    "FallbackAction",
    "FallbackPolicy",
    "KNOWN_ATTACK_FAMILIES",
    "UNKNOWN_FAMILY_KEY",
]
