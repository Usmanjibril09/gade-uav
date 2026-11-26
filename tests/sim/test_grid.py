import pytest

from gade_uav.sim.grid import Grid, Position2D


class TestGridConstruction:
    def test_default_is_750x750(self):
        g = Grid()
        assert g.width_m == 750.0 and g.height_m == 750.0

    def test_rejects_bad_dims(self):
        with pytest.raises(ValueError):
            Grid(width_m=0)
        with pytest.raises(ValueError):
            Grid(rf_range_m=-1)


class TestGridContainment:
    def test_contains_interior(self):
        g = Grid()
        assert g.contains(Position2D(100, 200))

    def test_contains_corners(self):
        g = Grid()
        assert g.contains(Position2D(0, 0))
        assert g.contains(Position2D(750, 750))

    def test_rejects_outside(self):
        g = Grid()
        assert not g.contains(Position2D(-1, 100))
        assert not g.contains(Position2D(100, 751))


class TestRegistration:
    def test_register_and_lookup(self):
        g = Grid()
        g.register("d1", Position2D(100, 100))
        assert g.position_of("d1") == Position2D(100, 100)
        assert len(g) == 1

    def test_duplicate_id_rejected(self):
        g = Grid()
        g.register("d1", Position2D(100, 100))
        with pytest.raises(ValueError):
            g.register("d1", Position2D(200, 200))

    def test_out_of_bounds_rejected(self):
        g = Grid()
        with pytest.raises(ValueError):
            g.register("d1", Position2D(1000, 100))


class TestRfNeighbors:
    def test_within_range_adjacent(self):
        g = Grid(rf_range_m=250)
        g.register("a", Position2D(100, 100))
        g.register("b", Position2D(200, 100))  # 100 m apart
        assert "b" in g.neighbors("a")
        assert "a" in g.neighbors("b")

    def test_out_of_range_not_adjacent(self):
        g = Grid(rf_range_m=250)
        g.register("a", Position2D(0, 0))
        g.register("b", Position2D(500, 0))  # 500 m apart
        assert g.neighbors("a") == []

    def test_move_updates_adjacency(self):
        g = Grid(rf_range_m=250)
        g.register("a", Position2D(0, 0))
        g.register("b", Position2D(500, 0))
        assert g.neighbors("a") == []
        g.update_position("b", Position2D(100, 0))
        assert "b" in g.neighbors("a")

    def test_boundary_of_range_adjacent(self):
        g = Grid(rf_range_m=250)
        g.register("a", Position2D(0, 0))
        g.register("b", Position2D(250, 0))
        assert "b" in g.neighbors("a")

    def test_multiple_neighbors(self):
        g = Grid(rf_range_m=250)
        g.register("hub", Position2D(375, 375))
        for i, (x, y) in enumerate([(400, 400), (350, 350), (500, 375)]):
            g.register(f"d{i}", Position2D(x, y))
        assert set(g.neighbors("hub")) == {"d0", "d1", "d2"}
