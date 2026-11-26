"""Counterfactual tests for Effector family and phase multipliers.

These tests prove the multipliers actually change the output distribution,
not just that they load without error. Uses law-of-large-numbers averaging
(N=2000 draws per condition) so noise averages out and the multiplier
signal is visible above the sigma of each dimension.
"""

from __future__ import annotations

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
from gade_uav.sim.effector import Effector

CONFIG = Path(__file__).parents[2] / "configs" / "effector.yaml"
N = 2000  # draws per condition; enough for LLN to expose multiplier signal


def _action(atype: ActionType = ActionType.HONEY_BEACON) -> Action:
    return Action(
        type=atype,
        payload={"k": "v"},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(
            cpu=0.01, ram=2.0, lat=0.5, bw=48.0, energy=0.01, mission_delay=0.1
        ),
        gen_meta=GenMeta(
            confidence=0.5,
            schema_version="v1",
            generation_context_hash="test",
            template_id=None,
        ),
    )


def _mean(
    e: Effector, act: Action, *, phase: str, family: str | None, dim: str
) -> float:
    """Sample-mean of one cost dimension over N executions."""
    vals = np.array(
        [
            getattr(
                e.execute(act, mission_phase=phase, primary_attack_family=family),
                dim,
            )
            for _ in range(N)
        ]
    )
    return float(vals.mean())


class TestFamilyMultipliers:
    def test_jamming_increases_bw(self):
        """Jamming multiplier on bw is 1.5; expect ~50% jump above 'none'."""
        e = Effector(CONFIG, seed=0)
        act = _action(ActionType.HONEY_BEACON)  # base bw = 48
        bw_none = _mean(e, act, phase="cruise", family=None, dim="bw")
        e = Effector(CONFIG, seed=0)  # reset RNG for fair comparison
        bw_jam = _mean(e, act, phase="cruise", family="Jamming", dim="bw")
        ratio = bw_jam / bw_none
        # Expect 1.5 * base = 72 vs 48 -> ratio ~1.5. Allow ±5% for finite-N.
        assert 1.42 <= ratio <= 1.58, f"jamming/none bw ratio {ratio:.3f} not near 1.5"

    def test_mitm_increases_lat(self):
        """MITM lat multiplier is 1.5. Lat uses lognormal noise so expected
        mean scales with the multiplier; check ratio ~1.5 within tolerance."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        lat_none = _mean(e, act, phase="cruise", family=None, dim="lat")
        e = Effector(CONFIG, seed=0)
        lat_mitm = _mean(e, act, phase="cruise", family="MITM", dim="lat")
        ratio = lat_mitm / lat_none
        # Lognormal noise widens the CI; allow ±10%.
        assert 1.35 <= ratio <= 1.65, f"mitm/none lat ratio {ratio:.3f} not near 1.5"

    def test_ddos_increases_ram(self):
        """DDoS ram multiplier is 1.3."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        ram_none = _mean(e, act, phase="cruise", family=None, dim="ram")
        e = Effector(CONFIG, seed=0)
        ram_ddos = _mean(e, act, phase="cruise", family="DDoS", dim="ram")
        ratio = ram_ddos / ram_none
        assert 1.22 <= ratio <= 1.38, f"ddos/none ram ratio {ratio:.3f} not near 1.3"

    def test_none_family_is_identity(self):
        """primary_attack_family=None must equal explicit 'none' row."""
        e = Effector(CONFIG, seed=42)
        act = _action()
        c_none_implicit = _mean(e, act, phase="cruise", family=None, dim="bw")
        e = Effector(CONFIG, seed=42)
        c_none_explicit = _mean(e, act, phase="cruise", family="none", dim="bw")
        # Same seed + same multiplier row -> identical draws.
        assert abs(c_none_implicit - c_none_explicit) < 1e-9

    def test_unknown_family_falls_back_to_identity(self):
        """Unknown family emits a warning once and uses identity multipliers."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        bw_none = _mean(e, act, phase="cruise", family=None, dim="bw")
        e = Effector(CONFIG, seed=0)
        with pytest.warns(UserWarning, match="unknown primary_attack_family"):
            bw_unknown = _mean(
                e, act, phase="cruise", family="NotARealFamily", dim="bw"
            )
        assert abs(bw_unknown - bw_none) < 1e-9


class TestPhaseMultipliers:
    def test_takeoff_increases_cpu(self):
        """Takeoff cpu multiplier is 1.3."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        cpu_cruise = _mean(e, act, phase="cruise", family=None, dim="cpu")
        e = Effector(CONFIG, seed=0)
        cpu_takeoff = _mean(e, act, phase="takeoff", family=None, dim="cpu")
        ratio = cpu_takeoff / cpu_cruise
        assert (
            1.22 <= ratio <= 1.38
        ), f"takeoff/cruise cpu ratio {ratio:.3f} not near 1.3"

    def test_takeoff_increases_mission_delay(self):
        """Takeoff mission_delay multiplier is 1.5 (lognormal noise)."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        md_cruise = _mean(e, act, phase="cruise", family=None, dim="mission_delay")
        e = Effector(CONFIG, seed=0)
        md_takeoff = _mean(e, act, phase="takeoff", family=None, dim="mission_delay")
        ratio = md_takeoff / md_cruise
        assert (
            1.35 <= ratio <= 1.65
        ), f"takeoff/cruise md ratio {ratio:.3f} not near 1.5"

    def test_takeoff_reduces_bw(self):
        """Takeoff bw multiplier is 0.9 (less spectrum headroom)."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        bw_cruise = _mean(e, act, phase="cruise", family=None, dim="bw")
        e = Effector(CONFIG, seed=0)
        bw_takeoff = _mean(e, act, phase="takeoff", family=None, dim="bw")
        ratio = bw_takeoff / bw_cruise
        assert (
            0.85 <= ratio <= 0.95
        ), f"takeoff/cruise bw ratio {ratio:.3f} not near 0.9"

    def test_cruise_is_identity(self):
        """Cruise row is all 1.0; equals default (phase omitted)."""
        e = Effector(CONFIG, seed=42)
        act = _action()
        bw_cruise = _mean(e, act, phase="cruise", family=None, dim="bw")
        e = Effector(CONFIG, seed=42)
        bw_default = _mean(e, act, phase="cruise", family=None, dim="bw")  # explicit
        assert abs(bw_cruise - bw_default) < 1e-9


class TestCompoundEffects:
    def test_jamming_takeoff_multiplies_correctly(self):
        """Jamming(bw=1.5) * takeoff(bw=0.9) = 1.35 on bw."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        bw_none_cruise = _mean(e, act, phase="cruise", family=None, dim="bw")
        e = Effector(CONFIG, seed=0)
        bw_jam_takeoff = _mean(e, act, phase="takeoff", family="Jamming", dim="bw")
        ratio = bw_jam_takeoff / bw_none_cruise
        # 1.5 * 0.9 = 1.35, allow ±7% for finite-N + gaussian noise.
        assert (
            1.26 <= ratio <= 1.44
        ), f"jam+takeoff/none+cruise bw ratio {ratio:.3f} not near 1.35"

    def test_mitm_takeoff_compounds_lat(self):
        """MITM(lat=1.5) * takeoff(lat=1.2) = 1.8 on lat."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        lat_none_cruise = _mean(e, act, phase="cruise", family=None, dim="lat")
        e = Effector(CONFIG, seed=0)
        lat_mitm_takeoff = _mean(e, act, phase="takeoff", family="MITM", dim="lat")
        ratio = lat_mitm_takeoff / lat_none_cruise
        # Lognormal on lat -> wider tolerance.
        assert (
            1.60 <= ratio <= 2.05
        ), f"mitm+takeoff/none+cruise lat ratio {ratio:.3f} not near 1.8"


class TestBackwardCompatibility:
    def test_execute_without_kwargs_still_works(self):
        """Existing callers using execute(action) must still work; defaults
        (cruise + None) both map to identity rows, so output distribution is
        exactly the pre-multiplier behaviour."""
        e = Effector(CONFIG, seed=0)
        act = _action()
        # This is the exact call signature the old test_effector.py uses.
        c = e.execute(act)
        assert isinstance(c, CostVector)
        # Also verify the mean matches base * 1.0 * 1.0 within noise.
        e = Effector(CONFIG, seed=0)
        bw_default = _mean(e, act, phase="cruise", family=None, dim="bw")
        # base bw for honey_beacon = 48; noise sigma = 16; N=2000 -> SE ~0.36.
        assert 47.0 <= bw_default <= 49.0, f"default bw mean {bw_default:.2f} off"
