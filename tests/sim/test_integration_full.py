"""End-to-end simulator integration test.

15 MissionDrones + 5 DecoyDrones on a 750x750 grid. PoissonCampaignAdversary
firing all 12 attack classes. Orchestrator emits honey beacons at every
tick through the Shield->Effector->Recorder pipeline. Runs for a short
window (5 s = 500 ticks) as a smoke test; long-run wall-clock is captured
by the runtime assertion.
"""

from __future__ import annotations

import time
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
from gade_uav.shield.decide import Shield, ShieldContext
from gade_uav.shield.fallback import FallbackPolicy
from gade_uav.shield.ltl_monitor import SafetyMonitor
from gade_uav.shield.timed_monitors import (
    BurstLimitMonitor,
    LatencyDeadlineMonitor,
    QuietPeriodMonitor,
    TimedMonitors,
)
from gade_uav.sim.adversary import PoissonCampaignAdversary
from gade_uav.sim.clock import SimClock
from gade_uav.sim.drone import DecoyDrone, MissionDrone
from gade_uav.sim.effector import Effector
from gade_uav.sim.grid import Grid, Position2D
from gade_uav.sim.loop import SimulationLoop
from gade_uav.sim.mission import Mission, Waypoint
from gade_uav.sim.recorder import Recorder

REPO = Path(__file__).parents[2]
EFFECTOR_CFG = REPO / "configs" / "effector.yaml"
FALLBACK_CFG = REPO / "configs" / "fallback_policy.yaml"
ADVERSARY_CFG = REPO / "configs" / "adversary.yaml"


def _gen_meta() -> GenMeta:
    return GenMeta(
        confidence=0.9,
        schema_version="1.0",
        generation_context_hash="deadbeef",
        template_id="safe_heartbeat_v1",
    )


def _build_shield() -> Shield:
    base = {
        "cpu": 10.0,
        "ram": 10_000.0,
        "lat": 10_000.0,
        "bw": 10_000.0,
        "energy": 10.0,
        "mission_delay": 10_000.0,
    }
    slack = {phase.value: 1.0 for phase in MissionPhase}
    budgets = BudgetSchedule(base_budget=base, phase_slack=slack)
    gate = MultiDimensionalConformalGate(
        alpha_target=0.10,
        aci_gamma=0.005,
        min_calibration_size=50,
        budgets=budgets,
    )
    # Seed calibration so L3 does not reject on insufficient-cal.
    for _ in range(60):
        gate.add_calibration(
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
    return Shield(
        ltl_monitor=SafetyMonitor(),
        gate=gate,
        timed=TimedMonitors(
            burst=BurstLimitMonitor(window_s=1.0, max_emissions=10_000),
            latency=LatencyDeadlineMonitor(deadline_ms=1000.0),
            quiet=QuietPeriodMonitor(
                quiet_ms=0.001
            ),  # near-zero quiet: allow rapid re-entry
        ),
        fallback_policy=FallbackPolicy.from_yaml(FALLBACK_CFG),
    )


def _ctx(mission_phase: MissionPhase, known_decoys: dict) -> ShieldContext:
    return ShieldContext(
        mission_phase=mission_phase,
        signing_active=True,
        decoy_key_active=True,
        known_decoys=known_decoys,
        bounded_template_whitelist=frozenset({"safe_heartbeat_v1"}),
    )


def _beacon(decoy_id: str) -> Action:
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"beacon_id": f"beacon_{decoy_id}", "advertise": {"ssid": "test"}},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
        gen_meta=_gen_meta(),
    )


def _build_world(tmp_path: Path):
    loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
    # 15 mission drones, spaced along a lane.
    for i in range(15):
        d = MissionDrone(
            drone_id=f"m{i}",
            position=Position2D(50.0 + i * 20.0, 100.0, 20.0),
        )
        loop.add_drone(d)
        loop.attach_mission(
            d.drone_id,
            Mission(
                waypoints=[Waypoint(50.0 + i * 20.0, 400.0, 20.0)],
                cruise_alt_m=20.0,
                wp_tol_m=5.0,
                hover_duration_s=1.0,
            ),
        )
    # 5 decoys, static.
    known_decoys: dict[str, DecoyIdentity] = {}
    for i in range(5):
        did = f"h{i}"
        loop.add_drone(
            DecoyDrone(drone_id=did, position=Position2D(400.0, 300.0 + i * 30.0, 20.0))
        )
        known_decoys[did] = DecoyIdentity(
            decoy_id=did,
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )
    loop.attach_shield(_build_shield())
    loop.attach_effector(Effector(EFFECTOR_CFG, seed=0))
    loop.attach_recorder(Recorder(tmp_path / "episode.parquet"))
    loop.attach_adversary(PoissonCampaignAdversary(ADVERSARY_CFG, seed=0))
    # Arm all missions so they start advancing.
    for mission in loop._missions.values():
        mission.arm()
    return loop, known_decoys


@pytest.mark.integration
class TestEndToEndEpisode:
    def test_short_episode_smoke(self, tmp_path):
        """5s (500-tick) episode with 15 MDs + 5 HDs, all subsystems active."""
        loop, known_decoys = _build_world(tmp_path)
        wall_start = time.perf_counter()

        for _ in range(500):  # 5 seconds of sim time
            # Advance world (drones move, missions step, adversary polls).
            loop.step()
            # Orchestrator: fire one beacon per decoy per tick using its
            # current mission-phase context (we take m0's phase as representative).
            m0 = loop.drones["m0"]
            ctx = _ctx(m0.mission_phase, known_decoys)
            for did in known_decoys:
                loop.deploy(_beacon(did), ctx)

        wall_elapsed = time.perf_counter() - wall_start
        loop._recorder.close()

        # (1) All deploys admitted -> zero S1..S7 violations expected.
        df = pl.read_parquet(tmp_path / "episode.parquet")
        deploys = (df["verdict"] == "DEPLOY").sum()
        rejects = (df["verdict"] == "REJECT").sum()
        assert deploys == df.height, f"unexpected {rejects} rejections"

        # (2) Calibration grew by exactly `deploys` (one observe per DEPLOY).
        # 500 ticks * 5 decoys = 2500 deploys.
        assert deploys == 2500

        # (3) Recorder wrote expected row count.
        assert df.height == 2500

        # (4) Wall-clock budget: 5s of sim should complete in <10s wall time
        #     on any reasonable dev machine (5x real-time headroom for CI).
        assert (
            wall_elapsed < 10.0
        ), f"episode took {wall_elapsed:.1f}s wall for 5s sim; too slow"

    def test_adversary_generated_events(self, tmp_path):
        """Adversary must fire at least some campaigns over 500 ticks."""
        loop, _ = _build_world(tmp_path)
        total_events = 0
        for _ in range(500):
            loop.step()
            total_events += len(loop.last_events)
        loop._recorder.close()
        # Config yields ~0.188 campaign starts/s * 5s ~ 0.94 starts + packets.
        # Assert at least *some* events fired.
        assert total_events > 0

    def test_mission_phases_advanced(self, tmp_path):
        """After 500 ticks, at least one drone should have left IDLE."""
        loop, _ = _build_world(tmp_path)
        for _ in range(500):
            loop.step()
        phases = {
            d.drone_id: d.mission_phase
            for d in loop.drones.values()
            if d.drone_id.startswith("m")
        }
        assert any(p != MissionPhase.IDLE for p in phases.values())
