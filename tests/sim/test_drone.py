import pytest

from gade_uav.shield.actions import MissionPhase
from gade_uav.sim.drone import DecoyDrone, DroneRole, MissionDrone, Velocity2D
from gade_uav.sim.grid import Position2D


class TestConstruction:
    def test_mission_drone_role(self):
        d = MissionDrone(drone_id="m1", position=Position2D(0, 0))
        assert d.role == DroneRole.MISSION

    def test_decoy_drone_role(self):
        d = DecoyDrone(drone_id="h1", position=Position2D(0, 0))
        assert d.role == DroneRole.DECOY

    def test_bad_battery_rejected(self):
        with pytest.raises(ValueError):
            MissionDrone(drone_id="m1", position=Position2D(0, 0), battery_pct=150.0)

    def test_bad_max_speed_rejected(self):
        with pytest.raises(ValueError):
            MissionDrone(drone_id="m1", position=Position2D(0, 0), max_speed_mps=0)


class TestKinematics:
    def test_stationary_drone_does_not_move(self):
        d = MissionDrone(drone_id="m1", position=Position2D(100, 100))
        new = d.tick(0.01)
        assert new == Position2D(100, 100)

    def test_constant_velocity(self):
        d = MissionDrone(drone_id="m1", position=Position2D(0, 0))
        d.set_velocity(10, 0)
        new = d.tick(1.0)  # 10 m/s * 1 s
        assert new.x == pytest.approx(10.0)
        assert new.y == pytest.approx(0.0)

    def test_speed_clamped_to_max(self):
        d = MissionDrone(drone_id="m1", position=Position2D(0, 0), max_speed_mps=15.0)
        d.set_velocity(100, 0)  # over cap
        assert d.velocity.vx == pytest.approx(15.0)

    def test_battery_drains(self):
        d = MissionDrone(
            drone_id="m1",
            position=Position2D(0, 0),
            battery_pct=100.0,
            battery_drain_per_s=1.0,
        )
        d.tick(2.0)
        assert d.battery_pct == pytest.approx(98.0)

    def test_empty_battery_zeros_velocity(self):
        d = MissionDrone(
            drone_id="m1",
            position=Position2D(0, 0),
            battery_pct=0.0,
        )
        d.set_velocity(10, 0)
        d.tick(0.01)
        assert d.velocity == Velocity2D()

    def test_altitude_never_negative(self):
        d = MissionDrone(drone_id="m1", position=Position2D(0, 0, z=1.0))
        d.set_velocity(0, 0, -100)
        new = d.tick(1.0)
        assert new.z == 0.0

    def test_decoy_drains_slower(self):
        m = MissionDrone(drone_id="m", position=Position2D(0, 0))
        h = DecoyDrone(drone_id="h", position=Position2D(0, 0))
        assert h.battery_drain_per_s < m.battery_drain_per_s


class TestMissionPhaseState:
    def test_default_phase_idle(self):
        d = MissionDrone(drone_id="m1", position=Position2D(0, 0))
        assert d.mission_phase == MissionPhase.IDLE

    def test_phase_settable(self):
        d = MissionDrone(drone_id="m1", position=Position2D(0, 0))
        d.mission_phase = MissionPhase.CRUISE
        assert d.mission_phase == MissionPhase.CRUISE
