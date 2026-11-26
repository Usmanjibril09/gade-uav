import pytest

from gade_uav.sim.clock import SimClock
from gade_uav.sim.drone import DecoyDrone, MissionDrone
from gade_uav.sim.grid import Grid, Position2D
from gade_uav.sim.loop import SimulationLoop


def _default_loop() -> SimulationLoop:
    return SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))


class TestLoopBasics:
    def test_add_drone_registers_in_grid(self):
        loop = _default_loop()
        d = MissionDrone(drone_id="m1", position=Position2D(100, 100))
        loop.add_drone(d)
        assert loop.grid.position_of("m1") == Position2D(100, 100)

    def test_duplicate_drone_rejected(self):
        loop = _default_loop()
        loop.add_drone(MissionDrone(drone_id="m1", position=Position2D(0, 0)))
        with pytest.raises(ValueError):
            loop.add_drone(MissionDrone(drone_id="m1", position=Position2D(1, 1)))

    def test_step_advances_clock(self):
        loop = _default_loop()
        loop.step()
        assert loop.clock.tick == 1
        assert loop.clock.now_s() == pytest.approx(0.01)

    def test_step_moves_drone_and_syncs_grid(self):
        loop = _default_loop()
        d = MissionDrone(drone_id="m1", position=Position2D(100, 100))
        d.set_velocity(10, 0)
        loop.add_drone(d)
        loop.step()
        # 10 m/s * 0.01 s = 0.1 m
        expected = Position2D(100.1, 100.0)
        assert loop.grid.position_of("m1").x == pytest.approx(expected.x)


class TestBoundaryClamping:
    def test_drone_clamped_to_grid_edge(self):
        loop = _default_loop()
        d = MissionDrone(drone_id="m1", position=Position2D(749.99, 100))
        d.set_velocity(100, 0)  # will overshoot 750
        loop.add_drone(d)
        for _ in range(10):
            loop.step()
        assert loop.grid.position_of("m1").x == pytest.approx(750.0)

    def test_drone_clamped_at_zero(self):
        loop = _default_loop()
        d = MissionDrone(drone_id="m1", position=Position2D(0.1, 100))
        d.set_velocity(-100, 0)
        loop.add_drone(d)
        loop.step()
        assert loop.grid.position_of("m1").x == 0.0


class TestMultiDroneScenario:
    def test_15_missions_5_decoys(self):
        """Blueprint §14 default topology."""
        loop = _default_loop()
        for i in range(15):
            loop.add_drone(
                MissionDrone(drone_id=f"m{i}", position=Position2D(100 + i * 30, 200))
            )
        for i in range(5):
            loop.add_drone(
                DecoyDrone(drone_id=f"h{i}", position=Position2D(400, 300 + i * 30))
            )
        assert len(loop.drones) == 20
        loop.run(n_ticks=100)  # 1 s of sim time
        assert loop.clock.now_s() == pytest.approx(1.0)

    def test_run_is_deterministic(self):
        def build():
            lp = _default_loop()
            d = MissionDrone(drone_id="m1", position=Position2D(100, 100))
            d.set_velocity(5, 3)
            lp.add_drone(d)
            return lp

        a, b = build(), build()
        a.run(500)
        b.run(500)
        assert a.grid.position_of("m1") == b.grid.position_of("m1")
        assert a.drones["m1"].battery_pct == b.drones["m1"].battery_pct
