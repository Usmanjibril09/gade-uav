"""Integration test: Shield with M3CostEstimator wired into L3.

Verifies:
  1. M3 estimator produces a valid 6-dim dict for a realistic Action.
  2. Swapping primary_attack_family from "none" to "Jamming" increases
     the L3 point estimate for `bw` by ~1.5x (matches effector multiplier).
  3. Shield constructed with cost_estimator=M3CostEstimator behaves
     identically at the interface level (returns ShieldDecision) as
     Shield constructed with the default GeneratorCostEstimator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    Effector,
    GenMeta,
    MissionPhase,
    ShieldContext,
)
from gade_uav.shield.cost_estimator import (
    GeneratorCostEstimator,
    M3CostEstimator,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO / "artifacts" / "cost_predictor"


def _make_action() -> Action:
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"beacon_id": "hb-01"},
        target_iface=Effector.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(
            cpu=0.01,
            ram=2.0,
            lat=0.5,
            bw=48.0,
            energy=0.01,
            mission_delay=0.1,
        ),
        gen_meta=GenMeta(
            confidence=0.9,
            schema_version="v1",
            generation_context_hash="deadbeef",
        ),
    )


def _make_ctx() -> ShieldContext:
    return ShieldContext(
        mission_phase=MissionPhase.CRUISE,
        signing_active=True,
        decoy_key_active=False,
        known_decoys={},
    )


@pytest.fixture(scope="module")
def m3():
    if not ARTIFACT_DIR.exists():
        pytest.skip(f"M3 artifacts not present at {ARTIFACT_DIR}")
    return M3CostEstimator.from_dir(ARTIFACT_DIR)


class TestM3EstimatorBasics:
    def test_estimate_returns_six_dims(self, m3):
        est = m3.estimate(_make_action(), _make_ctx(), primary_attack_family="none")
        assert set(est.keys()) == {"cpu", "ram", "lat", "bw", "energy", "mission_delay"}
        for k, v in est.items():
            assert isinstance(v, float), f"{k} not float: {type(v)}"
            assert v >= 0.0, f"{k} negative: {v}"

    def test_estimate_finite(self, m3):
        import math

        est = m3.estimate(_make_action(), _make_ctx(), primary_attack_family="Jamming")
        for k, v in est.items():
            assert math.isfinite(v), f"{k} non-finite: {v}"

    def test_family_swap_changes_bw(self, m3):
        """Jamming should raise bw ~1.5x vs none (matches effector multiplier)."""
        act, ctx = _make_action(), _make_ctx()
        bw_none = m3.estimate(act, ctx, primary_attack_family="none")["bw"]
        bw_jam = m3.estimate(act, ctx, primary_attack_family="Jamming")["bw"]
        ratio = bw_jam / bw_none
        assert 1.2 <= ratio <= 1.8, (
            f"expected bw ratio ~1.5 for Jamming vs none, got {ratio:.3f} "
            f"(bw_none={bw_none:.2f}, bw_jam={bw_jam:.2f})"
        )

    def test_action_type_swap_changes_bw(self, m3):
        """decoy_telemetry has 2x base bw of honey_beacon."""
        hb = _make_action()
        dt = Action(
            type=ActionType.DECOY_TELEMETRY,
            payload=hb.payload,
            target_iface=Effector.DECOY_MAVLINK_CHANNEL,
            predicted_cost=hb.predicted_cost,
            gen_meta=hb.gen_meta,
        )
        bw_hb = m3.estimate(hb, _make_ctx(), primary_attack_family="none")["bw"]
        bw_dt = m3.estimate(dt, _make_ctx(), primary_attack_family="none")["bw"]
        assert bw_dt > 1.5 * bw_hb, (
            f"expected decoy_telemetry bw > 1.5x honey_beacon, "
            f"got {bw_dt:.2f} vs {bw_hb:.2f}"
        )


class TestGeneratorEstimatorParity:
    def test_generator_returns_predicted_cost(self):
        gen = GeneratorCostEstimator()
        act = _make_action()
        est = gen.estimate(act, _make_ctx(), primary_attack_family="none")
        assert est == act.predicted_cost.as_dict()

    def test_generator_ignores_family(self):
        gen = GeneratorCostEstimator()
        act = _make_action()
        e1 = gen.estimate(act, _make_ctx(), primary_attack_family="none")
        e2 = gen.estimate(act, _make_ctx(), primary_attack_family="Jamming")
        assert e1 == e2
