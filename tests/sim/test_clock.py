import pytest

from gade_uav.sim.clock import SimClock


class TestSimClock:
    def test_default_dt_is_10ms(self):
        c = SimClock()
        assert c.dt_s == 0.01

    def test_now_starts_at_zero(self):
        assert SimClock().now_s() == 0.0

    def test_advance_increments_by_dt(self):
        c = SimClock(dt_s=0.01)
        assert c.advance() == pytest.approx(0.01)
        assert c.advance() == pytest.approx(0.02)

    def test_tick_counter(self):
        c = SimClock()
        for _ in range(100):
            c.advance()
        assert c.tick == 100
        assert c.now_s() == pytest.approx(1.0)

    def test_reset(self):
        c = SimClock()
        for _ in range(50):
            c.advance()
        c.reset()
        assert c.tick == 0
        assert c.now_s() == 0.0

    def test_zero_dt_rejected(self):
        with pytest.raises(ValueError):
            SimClock(dt_s=0.0)

    def test_negative_dt_rejected(self):
        with pytest.raises(ValueError):
            SimClock(dt_s=-0.01)

    def test_determinism_across_instances(self):
        a, b = SimClock(dt_s=0.01), SimClock(dt_s=0.01)
        for _ in range(1000):
            a.advance()
            b.advance()
        assert a.now_s() == b.now_s()
