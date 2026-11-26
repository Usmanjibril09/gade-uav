"""SimulationLoop.deploy() integration tests: Shield -> Effector -> Recorder."""

from pathlib import Path

import polars as pl
import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    DecoyIdentity,
    Effector as EffectorEnum,
    GenMeta,
    MissionPhase,
)
from gade_uav.shield.conformal_gate import (
    BudgetSchedule,
    MultiDimensionalConformalGate,
)
from gade_uav.shield.decide import Shield, ShieldContext, Verdict
from gade_uav.shield.fallback import FallbackPolicy
from gade_uav.shield.ltl_monitor import SafetyMonitor
from gade_uav.shield.timed_monitors import (
    BurstLimitMonitor,
    LatencyDeadlineMonitor,
    QuietPeriodMonitor,
    TimedMonitors,
)
from gade_uav.sim.clock import SimClock
from gade_uav.sim.drone import MissionDrone
from gade_uav.sim.effector import Effector
from gade_uav.sim.grid import Grid, Position2D
from gade_uav.sim.loop import SimulationLoop
from gade_uav.sim.recorder import Recorder

REPO = Path(__file__).parents[2]
EFFECTOR_CFG = REPO / "configs" / "effector.yaml"
FALLBACK_CFG = REPO / "configs" / "fallback_policy.yaml"


def _gen_meta() -> GenMeta:
    return GenMeta(
        confidence=0.9,
        schema_version="1.0",
        generation_context_hash="deadbeef",
        template_id="safe_heartbeat_v1",
    )


def _budget_schedule() -> BudgetSchedule:
    base = {
        "cpu": 10.0,
        "ram": 10_000.0,
        "lat": 10_000.0,
        "bw": 10_000.0,
        "energy": 10.0,
        "mission_delay": 10_000.0,
    }
    slack = {phase.value: 1.0 for phase in MissionPhase}
    return BudgetSchedule(base_budget=base, phase_slack=slack)


def _gate() -> MultiDimensionalConformalGate:
    g = MultiDimensionalConformalGate(
        budgets=_budget_schedule(),
        alpha_target=0.10,
        aci_gamma=0.005,
        min_calibration_size=50,
    )
    for _ in range(60):
        g.add_calibration(
            predicted={
                "cpu": 0.01,
                "ram": 2.0,
                "lat": 0.5,
                "bw": 48.0,
                "energy": 0.01,
                "mission_delay": 0.1,
            },
            observed={
                "cpu": 0.011,
                "ram": 2.1,
                "lat": 0.55,
                "bw": 50.0,
                "energy": 0.011,
                "mission_delay": 0.11,
            },
        )
    return g


def _shield() -> Shield:
    return Shield(
        ltl_monitor=SafetyMonitor(),
        gate=_gate(),
        timed=TimedMonitors(
            burst=BurstLimitMonitor(window_s=1.0, max_emissions=10_000),
            latency=LatencyDeadlineMonitor(deadline_ms=1000.0),
            quiet=QuietPeriodMonitor(quiet_ms=200.0),
        ),
        fallback_policy=FallbackPolicy.from_yaml(FALLBACK_CFG),
    )


def _decoy_ctx() -> ShieldContext:
    return ShieldContext(
        mission_phase=MissionPhase.CRUISE,
        signing_active=True,
        decoy_key_active=True,
        known_decoys={
            "h1": DecoyIdentity(
                decoy_id="h1",
                tau_alt_m=5.0,
                tau_pos_m=15.0,
                tau_bat_pct=2.0,
                tau_att_rad_s=3.14,
            )
        },
        bounded_template_whitelist=frozenset({"safe_heartbeat_v1"}),
    )


def _beacon(decoy_id: str = "h1") -> Action:
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"beacon_id": f"beacon_{decoy_id}", "advertise": {"ssid": "test"}},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
        gen_meta=_gen_meta(),
    )


def _build_loop(tmp_path: Path) -> tuple[SimulationLoop, Recorder]:
    loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
    loop.add_drone(MissionDrone(drone_id="m1", position=Position2D(100, 100)))
    loop.attach_shield(_shield())
    loop.attach_effector(Effector(EFFECTOR_CFG, seed=0))
    rec = Recorder(tmp_path / "out.parquet")
    loop.attach_recorder(rec)
    return loop, rec


class TestDeployPreconditions:
    def test_deploy_without_shield_raises(self):
        loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
        with pytest.raises(RuntimeError, match="attach_shield"):
            loop.deploy(_beacon(), _decoy_ctx())

    def test_deploy_without_effector_raises(self):
        loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
        loop.attach_shield(_shield())
        with pytest.raises(RuntimeError, match="attach_effector"):
            loop.deploy(_beacon(), _decoy_ctx())


class TestDeployPathAdmitted:
    def test_beacon_deployed_and_recorded(self, tmp_path):
        loop, rec = _build_loop(tmp_path)
        d = loop.deploy(_beacon(), _decoy_ctx())
        assert d.verdict == Verdict.DEPLOY
        assert rec.total_rows == 1

    def test_observed_cost_written(self, tmp_path):
        loop, rec = _build_loop(tmp_path)
        loop.deploy(_beacon(), _decoy_ctx())
        rec.close()
        df = pl.read_parquet(tmp_path / "out.parquet")
        assert df.height == 1
        assert df["verdict"][0] == "DEPLOY"
        assert 0.0 <= df["obs_cpu"][0] < 0.1


class TestDeployPathRejected:
    def test_forbidden_effector_rejected_and_recorded(self, tmp_path):
        loop, rec = _build_loop(tmp_path)
        bad = Action(
            type=ActionType.MAVLINK_RESPONSE,
            payload={"msg_id": 0, "fields": {}},  # HEARTBEAT
            target_iface=EffectorEnum.FLIGHT_CONTROLLER_BUS,
            predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
            gen_meta=_gen_meta(),
        )
        d = loop.deploy(bad, _decoy_ctx(), primary_attack_family="T7_MITM")
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L1"
        rec.close()
        df = pl.read_parquet(tmp_path / "out.parquet")
        assert df["obs_cpu"][0] is None
        assert df["layer_failed"][0] == "L1"


class TestConformalFeedback:
    def test_observed_cost_feeds_gate(self, tmp_path):
        loop, _ = _build_loop(tmp_path)
        cal_before = loop._shield._gate._dims["cpu"].calibration_size
        for _ in range(5):
            loop.deploy(_beacon(), _decoy_ctx())
        cal_after = loop._shield._gate._dims["cpu"].calibration_size
        assert cal_after == cal_before + 5
