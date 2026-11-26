"""Drone entities: base + MissionDrone + DecoyDrone.

Kinematic model only (position += velocity * dt, velocity clamped).
Higher-fidelity dynamics are validated separately against PX4 SITL
per Phase 2 lock (offline validation, not in main loop).

Drones do not own their position in the world; Grid does. Drones own
velocity, battery, mission phase, and role. tick() returns the new
Position2D so the caller can update Grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gade_uav.shield.actions import MissionPhase

from .grid import Position2D


class DroneRole(str, Enum):
    MISSION = "MISSION"
    DECOY = "DECOY"


@dataclass
class Velocity2D:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class Drone:
    """Base drone. Do not instantiate directly."""

    drone_id: str
    role: DroneRole
    position: Position2D
    velocity: Velocity2D = field(default_factory=Velocity2D)
    battery_pct: float = 100.0
    mission_phase: MissionPhase = MissionPhase.IDLE
    max_speed_mps: float = 15.0  # typical multirotor cruise
    battery_drain_per_s: float = 0.05  # 0.05%/s -> ~33 min endurance

    def __post_init__(self) -> None:
        if not (0.0 <= self.battery_pct <= 100.0):
            raise ValueError(f"battery_pct out of [0,100]: {self.battery_pct}")
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be > 0")
        self._clamp_velocity()

    def _clamp_velocity(self) -> None:
        v = self.velocity
        speed = (v.vx * v.vx + v.vy * v.vy + v.vz * v.vz) ** 0.5
        if speed > self.max_speed_mps:
            scale = self.max_speed_mps / speed
            self.velocity = Velocity2D(v.vx * scale, v.vy * scale, v.vz * scale)

    def set_velocity(self, vx: float, vy: float, vz: float = 0.0) -> None:
        self.velocity = Velocity2D(vx, vy, vz)
        self._clamp_velocity()

    def tick(self, dt_s: float) -> Position2D:
        """Advance one dt. Returns the new intended position.

        Caller (SimulationLoop) is responsible for asking Grid to accept it.
        Battery decays deterministically; empty battery zeroes velocity.
        """
        if self.battery_pct <= 0.0:
            self.velocity = Velocity2D()
            return self.position
        new_pos = Position2D(
            x=self.position.x + self.velocity.vx * dt_s,
            y=self.position.y + self.velocity.vy * dt_s,
            z=max(0.0, self.position.z + self.velocity.vz * dt_s),
        )
        self.position = new_pos
        self.battery_pct = max(0.0, self.battery_pct - self.battery_drain_per_s * dt_s)
        return new_pos


@dataclass(kw_only=True)
class MissionDrone(Drone):
    """Real drone flying the mission. 15 of these per blueprint §14."""

    role: DroneRole = DroneRole.MISSION


@dataclass(kw_only=True)
class DecoyDrone(Drone):
    """Honey drone. 5 of these per blueprint §14.

    Registered with Shield via DecoyIdentity separately; this class
    only carries the physical/logical state, not the signing key or
    tolerance bands.
    """

    role: DroneRole = DroneRole.DECOY
    battery_drain_per_s: float = 0.02  # decoys move less, drain slower
