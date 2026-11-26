"""
Deterministic waypoint guidance controller for the GADe-UAV simulator.

This is NOT a flight-dynamics model and does not replace PX4/autopilot logic.
It provides bounded velocity commands to the existing kinematic MissionDrone
so that simulated missions can progress through:

    TAKEOFF -> CRUISE -> PRECISION_HOVER -> LANDING -> POST_LANDING

The controller contains no learned parameters and requires no training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gade_uav.shield.actions import MissionPhase

from .drone import MissionDrone
from .mission import Mission


@dataclass(frozen=True)
class GuidanceConfig:
    """Bounded kinematic guidance parameters."""

    cruise_speed_mps: float = 8.0
    climb_speed_mps: float = 3.0
    descent_speed_mps: float = 2.0
    precision_speed_mps: float = 2.0
    vertical_tolerance_m: float = 0.25

    def __post_init__(self) -> None:
        if self.cruise_speed_mps <= 0:
            raise ValueError("cruise_speed_mps must be > 0")
        if self.climb_speed_mps <= 0:
            raise ValueError("climb_speed_mps must be > 0")
        if self.descent_speed_mps <= 0:
            raise ValueError("descent_speed_mps must be > 0")
        if self.precision_speed_mps <= 0:
            raise ValueError("precision_speed_mps must be > 0")
        if self.vertical_tolerance_m <= 0:
            raise ValueError("vertical_tolerance_m must be > 0")


class MissionGuidanceController:
    """
    Simple phase-aware velocity controller.

    It only commands MissionDrone velocity. Mission phase transitions remain
    owned by Mission.step(), preserving the existing state-machine design.
    """

    def __init__(self, cfg: GuidanceConfig | None = None) -> None:
        self.cfg = cfg or GuidanceConfig()

    @staticmethod
    def _horizontal_velocity_toward(
        drone: MissionDrone,
        target_x: float,
        target_y: float,
        speed_mps: float,
    ) -> tuple[float, float]:
        dx = target_x - drone.position.x
        dy = target_y - drone.position.y
        dist = math.hypot(dx, dy)

        if dist <= 1e-9:
            return 0.0, 0.0

        # Do not command more speed than needed to preserve direction.
        vx = speed_mps * dx / dist
        vy = speed_mps * dy / dist
        return vx, vy

    def command(
        self,
        drone: MissionDrone,
        mission: Mission,
        *,
        speed_scale: float = 1.0,
    ) -> None:
        """Apply one bounded velocity command for the current mission phase.

        speed_scale is supplied by the mission-impact layer. A value of 1.0
        leaves nominal guidance unchanged; smaller values model bounded
        attack-induced progress degradation.
        """

        speed_scale = max(0.10, min(float(speed_scale), 1.0))
        phase = mission.phase

        if phase == MissionPhase.IDLE:
            drone.set_velocity(0.0, 0.0, 0.0)
            return

        if phase == MissionPhase.TAKEOFF:
            # Mission.step() transitions TAKEOFF -> CRUISE only when
            # position.z >= cruise_alt_m. Therefore do not stop inside a
            # tolerance band below the threshold.
            if drone.position.z < mission.cruise_alt_m:
                vz = self.cfg.climb_speed_mps * speed_scale
            else:
                vz = 0.0

            drone.set_velocity(0.0, 0.0, vz)
            return

        if phase == MissionPhase.CRUISE:
            wp = mission.current_waypoint

            if wp is None:
                drone.set_velocity(0.0, 0.0, 0.0)
                return

            vx, vy = self._horizontal_velocity_toward(
                drone,
                wp.x,
                wp.y,
                self.cfg.cruise_speed_mps * speed_scale,
            )

            dz = wp.z - drone.position.z

            if abs(dz) <= self.cfg.vertical_tolerance_m:
                vz = 0.0
            else:
                max_up = self.cfg.climb_speed_mps * speed_scale
                max_down = self.cfg.descent_speed_mps * speed_scale
                vz = max(
                    -max_down,
                    min(max_up, dz),
                )

            drone.set_velocity(vx, vy, vz)
            return

        if phase == MissionPhase.PRECISION_HOVER:
            # Mission.step() owns the hover timer.
            drone.set_velocity(0.0, 0.0, 0.0)
            return

        if phase in (
            MissionPhase.LANDING,
            MissionPhase.EMERGENCY_RECOVERY,
        ):
            if drone.position.z <= mission.landing_alt_tol_m:
                drone.set_velocity(0.0, 0.0, 0.0)
            else:
                drone.set_velocity(
                    0.0,
                    0.0,
                    -self.cfg.descent_speed_mps * speed_scale,
                )
            return

        if phase == MissionPhase.POST_LANDING:
            drone.set_velocity(0.0, 0.0, 0.0)
            return

        raise ValueError(f"Unhandled mission phase: {phase}")
