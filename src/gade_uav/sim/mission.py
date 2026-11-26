"""Mission: waypoint sequence + phase state machine.

A Mission owns a list of waypoints and a MissionPhase state. The
state machine is deterministic and driven by drone position, altitude,
and battery, evaluated once per SimulationLoop tick.

State machine (spec §14):

    IDLE --arm--> TAKEOFF --alt >= cruise_alt--> CRUISE
      --dist(next_wp) <= wp_tol--> (advance waypoint; stay in CRUISE)
      --last_wp reached--> PRECISION_HOVER
      --hover_time_elapsed--> LANDING
      --alt <= landing_alt_tol--> POST_LANDING

Battery <= emergency_bat_pct triggers EMERGENCY_RECOVERY from any
non-terminal phase; from there LANDING is forced.

Phases IDLE and POST_LANDING are terminal in either direction (start
vs end). CRUISE, TAKEOFF, LANDING, PRECISION_HOVER, EMERGENCY_RECOVERY
are transient.
"""

from __future__ import annotations

from dataclasses import dataclass

from gade_uav.shield.actions import MissionPhase

from .drone import MissionDrone
from .grid import Position2D


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float  # target altitude at this waypoint, meters


@dataclass
class Mission:
    """Waypoint-driven mission for a single MissionDrone.

    All thresholds are in SI units and can be overridden per mission.
    """

    waypoints: list[Waypoint]
    cruise_alt_m: float = 20.0
    landing_alt_tol_m: float = 0.5
    wp_tol_m: float = 3.0
    hover_duration_s: float = 2.0
    emergency_bat_pct: float = 15.0
    phase: MissionPhase = MissionPhase.IDLE
    _wp_index: int = 0
    _hover_start_s: float | None = None
    _armed: bool = False

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("Mission requires at least one waypoint")
        if self.cruise_alt_m <= 0:
            raise ValueError("cruise_alt_m must be > 0")
        if self.wp_tol_m <= 0:
            raise ValueError("wp_tol_m must be > 0")
        if not (0.0 <= self.emergency_bat_pct <= 100.0):
            raise ValueError("emergency_bat_pct must be in [0, 100]")

    # -- external API ----------------------------------------------------

    def arm(self) -> None:
        """Move IDLE -> TAKEOFF at next tick. Idempotent from IDLE."""
        if self.phase == MissionPhase.IDLE:
            self._armed = True

    @property
    def current_waypoint(self) -> Waypoint | None:
        if self._wp_index >= len(self.waypoints):
            return None
        return self.waypoints[self._wp_index]

    def step(self, drone: MissionDrone, now_s: float) -> MissionPhase:
        """Advance the state machine one tick. Returns the new phase.

        Also mutates drone.mission_phase so ShieldContext reads the
        same value.
        """
        # Emergency-recovery override from any active non-terminal phase.
        # Return immediately after promotion so the transition into
        # LANDING happens on the NEXT tick, giving observers (Shield,
        # S5) at least one tick where mission_phase == EMERGENCY_RECOVERY.
        if (
            self.phase
            not in (
                MissionPhase.IDLE,
                MissionPhase.POST_LANDING,
                MissionPhase.EMERGENCY_RECOVERY,
                MissionPhase.LANDING,
            )
            and drone.battery_pct <= self.emergency_bat_pct
        ):
            self.phase = MissionPhase.EMERGENCY_RECOVERY
            drone.mission_phase = self.phase
            return self.phase

        if self.phase == MissionPhase.IDLE and self._armed:
            self.phase = MissionPhase.TAKEOFF

        elif self.phase == MissionPhase.TAKEOFF:
            if drone.position.z >= self.cruise_alt_m:
                self.phase = MissionPhase.CRUISE

        elif self.phase == MissionPhase.CRUISE:
            wp = self.current_waypoint
            if wp is None:
                self.phase = MissionPhase.PRECISION_HOVER
                self._hover_start_s = now_s
            else:
                d = (
                    (drone.position.x - wp.x) ** 2 + (drone.position.y - wp.y) ** 2
                ) ** 0.5
                if d <= self.wp_tol_m:
                    self._wp_index += 1
                    if self._wp_index >= len(self.waypoints):
                        self.phase = MissionPhase.PRECISION_HOVER
                        self._hover_start_s = now_s

        elif self.phase == MissionPhase.PRECISION_HOVER:
            if (
                self._hover_start_s is not None
                and (now_s - self._hover_start_s) >= self.hover_duration_s
            ):
                self.phase = MissionPhase.LANDING

        elif self.phase == MissionPhase.EMERGENCY_RECOVERY:
            # From EMERGENCY_RECOVERY we descend to LANDING, but only after
            # the emergency-recovery phase has been visible for at least one
            # full tick (see promotion return above). We use _hover_start_s
            # as a general "phase entry time" marker for this transition.
            if self._hover_start_s is None:
                self._hover_start_s = now_s
            elif now_s > self._hover_start_s:
                self.phase = MissionPhase.LANDING
                self._hover_start_s = None

        elif self.phase == MissionPhase.LANDING:
            if drone.position.z <= self.landing_alt_tol_m:
                self.phase = MissionPhase.POST_LANDING

        drone.mission_phase = self.phase
        return self.phase

    @property
    def is_active(self) -> bool:
        """True unless IDLE (before arm) or POST_LANDING (mission complete)."""
        return self.phase not in (MissionPhase.IDLE, MissionPhase.POST_LANDING)

    def target_waypoint_position(self) -> Position2D | None:
        wp = self.current_waypoint
        if wp is None:
            return None
        return Position2D(x=wp.x, y=wp.y, z=wp.z)
