"""Mission state-machine tests."""

import pytest

from gade_uav.shield.actions import MissionPhase
from gade_uav.sim.drone import MissionDrone
from gade_uav.sim.grid import Position2D
from gade_uav.sim.mission import Mission, Waypoint


def _drone(x=0.0, y=0.0, z=0.0, bat=100.0) -> MissionDrone:
    return MissionDrone(
        drone_id="m1",
        position=Position2D(x, y, z),
        battery_pct=bat,
    )


class TestConstruction:
    def test_requires_waypoints(self):
        with pytest.raises(ValueError):
            Mission(waypoints=[])

    def test_default_phase_idle(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)])
        assert m.phase == MissionPhase.IDLE

    def test_bad_thresholds_rejected(self):
        with pytest.raises(ValueError):
            Mission(waypoints=[Waypoint(0, 0, 20)], cruise_alt_m=0)
        with pytest.raises(ValueError):
            Mission(waypoints=[Waypoint(0, 0, 20)], wp_tol_m=0)


class TestStateMachine:
    def test_idle_stays_until_armed(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)])
        d = _drone()
        for t in range(5):
            assert m.step(d, now_s=t * 0.01) == MissionPhase.IDLE

    def test_arm_transitions_to_takeoff(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)])
        d = _drone()
        m.arm()
        assert m.step(d, now_s=0.0) == MissionPhase.TAKEOFF

    def test_takeoff_transitions_to_cruise_at_altitude(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)], cruise_alt_m=20.0)
        d = _drone(z=5.0)
        m.arm()
        m.step(d, now_s=0.0)  # -> TAKEOFF
        assert m.phase == MissionPhase.TAKEOFF
        d.position = Position2D(0, 0, 20.0)
        m.step(d, now_s=0.01)
        assert m.phase == MissionPhase.CRUISE

    def test_cruise_advances_waypoints(self):
        wps = [Waypoint(100, 100, 20), Waypoint(200, 100, 20)]
        m = Mission(waypoints=wps, cruise_alt_m=20.0, wp_tol_m=3.0)
        d = _drone(z=20.0)
        m.arm()
        m.step(d, 0.0)
        m.step(d, 0.01)  # -> CRUISE
        d.position = Position2D(100.0, 100.0, 20.0)
        m.step(d, now_s=0.02)
        assert m._wp_index == 1
        assert m.phase == MissionPhase.CRUISE

    def test_last_waypoint_transitions_to_hover(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)], cruise_alt_m=20.0)
        d = _drone(z=20.0)
        m.arm()
        m.step(d, 0.0)
        m.step(d, 0.01)
        d.position = Position2D(100.0, 100.0, 20.0)
        m.step(d, now_s=0.02)
        assert m.phase == MissionPhase.PRECISION_HOVER

    def test_hover_transitions_to_landing_after_duration(self):
        m = Mission(
            waypoints=[Waypoint(100, 100, 20)],
            cruise_alt_m=20.0,
            hover_duration_s=1.0,
        )
        d = _drone(z=20.0)
        m.arm()
        m.step(d, 0.0)
        m.step(d, 0.01)
        d.position = Position2D(100, 100, 20)
        m.step(d, 0.02)  # PRECISION_HOVER, hover_start = 0.02
        m.step(d, 0.5)  # elapsed 0.48s -> still HOVER
        assert m.phase == MissionPhase.PRECISION_HOVER
        m.step(d, 1.03)  # elapsed 1.01s -> LANDING
        assert m.phase == MissionPhase.LANDING

    def test_landing_transitions_to_post_landing(self):
        m = Mission(
            waypoints=[Waypoint(100, 100, 20)],
            cruise_alt_m=20.0,
            hover_duration_s=0.0,
            landing_alt_tol_m=0.5,
        )
        d = _drone(z=20.0)
        m.arm()
        m.step(d, 0.0)
        m.step(d, 0.01)
        d.position = Position2D(100, 100, 20)
        m.step(d, 0.02)  # -> HOVER
        m.step(d, 0.03)  # hover_duration = 0 -> LANDING
        d.position = Position2D(100, 100, 0.3)
        m.step(d, 0.04)
        assert m.phase == MissionPhase.POST_LANDING


class TestEmergencyRecovery:
    def test_low_battery_triggers_emergency(self):
        m = Mission(
            waypoints=[Waypoint(100, 100, 20)],
            cruise_alt_m=20.0,
            emergency_bat_pct=15.0,
        )
        d = _drone(z=20.0, bat=10.0)
        m.arm()
        m.step(d, 0.0)
        m.step(d, 0.01)  # -> CRUISE
        m.step(d, 0.02)
        assert m.phase == MissionPhase.EMERGENCY_RECOVERY

    def test_emergency_leads_to_landing(self):
        m = Mission(
            waypoints=[Waypoint(100, 100, 20)],
            cruise_alt_m=20.0,
            emergency_bat_pct=15.0,
        )
        d = _drone(z=20.0, bat=10.0)
        m.arm()
        m.step(d, 0.0)
        m.step(d, 0.01)
        m.step(d, 0.02)  # -> EMERGENCY_RECOVERY
        m.step(d, 0.03)
        assert m.phase == MissionPhase.LANDING


class TestDroneMissionPhaseSync:
    def test_step_updates_drone_phase(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)])
        d = _drone()
        m.arm()
        m.step(d, now_s=0.0)
        assert d.mission_phase == MissionPhase.TAKEOFF


class TestIsActive:
    def test_idle_not_active(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)])
        assert not m.is_active

    def test_takeoff_active(self):
        m = Mission(waypoints=[Waypoint(100, 100, 20)])
        d = _drone()
        m.arm()
        m.step(d, 0.0)
        assert m.is_active
