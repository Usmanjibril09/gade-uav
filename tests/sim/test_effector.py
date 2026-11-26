"""Effector tests: construction validation, determinism, per-dimension noise."""

from pathlib import Path

import numpy as np
import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    Effector as EffectorEnum,
    GenMeta,
)
from gade_uav.sim.effector import DimensionNoise, Effector, NoiseModel

CONFIG = Path(__file__).parents[2] / "configs" / "effector.yaml"


def _gen_meta() -> GenMeta:
    return GenMeta(
        confidence=0.9,
        schema_version="1.0",
        generation_context_hash="deadbeef",
        template_id="safe_heartbeat_v1",
    )


def _action(atype: ActionType = ActionType.HONEY_BEACON) -> Action:
    return Action(
        type=atype,
        payload={"template_id": "safe_heartbeat_v1"},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(
            cpu=0.01, ram=2.0, lat=0.5, bw=48.0, energy=0.01, mission_delay=0.1
        ),
        gen_meta=_gen_meta(),
    )


class TestConstruction:
    def test_loads_config(self):
        e = Effector(CONFIG, seed=0)
        assert set(e.noise_config.keys()) == {
            "cpu",
            "ram",
            "lat",
            "bw",
            "energy",
            "mission_delay",
        }

    def test_lat_is_lognormal(self):
        e = Effector(CONFIG, seed=0)
        assert e.noise_config["lat"].model is NoiseModel.LOGNORMAL

    def test_cpu_is_gaussian(self):
        e = Effector(CONFIG, seed=0)
        assert e.noise_config["cpu"].model is NoiseModel.GAUSSIAN

    def test_bad_sigma_rejected(self):
        with pytest.raises(ValueError):
            DimensionNoise(model=NoiseModel.GAUSSIAN, sigma=0.0)

    def test_missing_dim_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "noise:\n" "  cpu: { model: gaussian, sigma: 0.01 }\n" "base_cost: {}\n"
        )
        with pytest.raises(ValueError, match="missing dimension"):
            Effector(bad, seed=0)


class TestDeterminism:
    def test_same_seed_same_costs(self):
        a = Effector(CONFIG, seed=42)
        b = Effector(CONFIG, seed=42)
        act = _action()
        for _ in range(50):
            assert a.execute(act) == b.execute(act)

    def test_different_seed_diverges(self):
        a = Effector(CONFIG, seed=1)
        b = Effector(CONFIG, seed=2)
        act = _action()
        ca = [a.execute(act) for _ in range(20)]
        cb = [b.execute(act) for _ in range(20)]
        assert ca != cb


class TestNoiseStatistics:
    N = 5000

    def _samples(self, dim: str, atype: ActionType):
        e = Effector(CONFIG, seed=7)
        act = _action(atype)
        return np.array([getattr(e.execute(act), dim) for _ in range(self.N)])

    def test_gaussian_cpu_mean_near_base(self):
        s = self._samples("cpu", ActionType.HONEY_BEACON)
        assert 0.008 < s.mean() < 0.020

    def test_gaussian_cpu_non_negative(self):
        s = self._samples("cpu", ActionType.HONEY_BEACON)
        assert (s >= 0.0).all()

    def test_lognormal_lat_has_heavy_tail(self):
        s = self._samples("lat", ActionType.HONEY_BEACON)
        assert abs(s.mean() - 0.5 * np.exp(0.30**2 / 2)) < 0.02
        # Theoretical p99/mean for LogNormal(0, 0.30) ~= exp(2.326*0.30 - 0.045) ~= 1.92.
        # Assert clearly-heavier-than-Gaussian tail (Gaussian would give p99/mean ~ 1.5)
        assert np.quantile(s, 0.99) / s.mean() > 1.7

    def test_lognormal_mission_delay_positive(self):
        s = self._samples("mission_delay", ActionType.HONEY_BEACON)
        assert (s > 0.0).all()

    def test_gaussian_bw_mean_near_base(self):
        s = self._samples("bw", ActionType.HONEY_BEACON)
        assert abs(s.mean() - 48.0) < 2.0


class TestActionTypeCoverage:
    def test_all_action_types_have_base(self):
        e = Effector(CONFIG, seed=0)
        for atype in ActionType:
            c = e.execute(_action(atype))
            assert isinstance(c, CostVector)
            assert c.cpu >= 0 and c.bw >= 0
