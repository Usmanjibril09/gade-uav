"""Warm-up simulator run: produces deployment_warmup parquets for M3 training.

Runs N episodes of the full Shield->Effector->Recorder pipeline built in
Phase 2 Track 3. Each episode: 12,000 ticks (120s at dt=10ms), 15 mission
drones + 5 decoys, PoissonCampaignAdversary firing all 12 attack classes,
one honey beacon per decoy per tick -> ~600k deployment rows/episode.

Reuses the fixture code path from tests/sim/test_integration_full.py to
guarantee production and tests share the same construction; when the
integration test is green, this script is too.

Outputs:
    data/interim/deployment_warmup/episode_{k:02d}.parquet
    data/interim/deployment_warmup/_warmup_manifest.json

Usage:
    uv run python scripts/run_warmup_sim.py --episodes 3
    uv run python scripts/run_warmup_sim.py --episodes 10
"""

from __future__ import annotations

import numpy as np

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from gade_uav.shield.actions import COST_DIMENSIONS
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
from gade_uav.shield.cost_estimator import CostEstimator, M3CostEstimator

REPO = Path(__file__).resolve().parents[1]
EFFECTOR_CFG = REPO / "configs" / "effector.yaml"
FALLBACK_CFG = REPO / "configs" / "fallback_policy.yaml"
ADVERSARY_CFG = REPO / "configs" / "adversary.yaml"
OUT_DIR = REPO / "data" / "interim" / "deployment_warmup"

TICKS_PER_EPISODE = 12_000  # 120s at dt=10ms
DT_S = 0.01
N_MISSION_DRONES = 15
N_DECOYS = 5


@dataclass
class EpisodeReport:
    episode: int
    seed: int
    rows_written: int
    verdict_counts: dict[str, int]
    calibration_growth_cpu: int
    campaign_events: int
    wall_seconds: float
    parquet_path: str


@dataclass
class WarmupManifest:
    n_episodes: int
    ticks_per_episode: int
    dt_s: float
    total_rows: int
    total_wall_seconds: float
    episodes: list[EpisodeReport] = field(default_factory=list)


def _gen_meta() -> GenMeta:
    return GenMeta(
        confidence=0.9,
        schema_version="1.0",
        generation_context_hash="warmup_v1",
        template_id="safe_heartbeat_v1",
    )



def _bounded_cost_downscale(
    action: Action,
    gate_decision,
) -> Action | None:
    """
    Single bounded L3 repair attempt.

    Preserves action semantics and changes only the declared resource-cost
    vector. The repaired action is rechecked by the existing shield logic.
    """
    if gate_decision.admitted:
        return None

    c = action.predicted_cost

    repaired_cost = CostVector(
        cpu=0.24 * c.cpu,
        ram=0.24 * c.ram,
        lat=0.24 * c.lat,
        bw=0.24 * c.bw,
        energy=0.24 * c.energy,
        mission_delay=0.24 * c.mission_delay,
    )

    return Action(
        type=action.type,
        payload=action.payload,
        target_iface=action.target_iface,
        predicted_cost=repaired_cost,
        gen_meta=action.gen_meta,
    )


def _build_shield(
    cost_estimator: CostEstimator | None = None,
    shield_mode: str = "full",
    calibration_npz: str | None = None,
    enable_cost_repair: bool = False,
    burst_max_emissions: int = 10_000,
    decision_deadline_ms: float = 1000.0,
    post_reject_quiet_ms: float = 0.001,
) -> Shield:
    # Realistic budgets: V1/V2 safe actions comfortably admit; V3
    # high_cost_beacon (cpu=0.08, ram=12, lat=3, bw=800, energy=0.08,
    # mission_delay=1.5) breaches 3-6 dimensions and gets L3 REJECT.
    # See design table in chat.
    base = {
        "cpu": 0.05,
        "ram": 8.0,
        "lat": 2.0,
        "bw": 200.0,
        "energy": 0.05,
        "mission_delay": 1.0,
    }
    slack = {phase.value: 1.0 for phase in MissionPhase}
    budgets = BudgetSchedule(base_budget=base, phase_slack=slack)
    calibration_data = None
    calibration_capacity = 10_000

    if calibration_npz is not None:
        with np.load(calibration_npz) as loaded:
            calibration_data = {
                k: np.asarray(
                    loaded[k],
                    dtype=np.float64,
                ).copy()
                for k in COST_DIMENSIONS
            }

        calibration_sizes = {
            k: int(v.size)
            for k, v in calibration_data.items()
        }

        if len(set(calibration_sizes.values())) != 1:
            raise ValueError(
                f"calibration sizes differ: {calibration_sizes}"
            )

        calibration_capacity = next(
            iter(calibration_sizes.values())
        )

    gate = MultiDimensionalConformalGate(
        alpha_target=0.10,
        aci_gamma=0.005,
        min_calibration_size=50,
        budgets=budgets,
        calibration_capacity=calibration_capacity,
    )

    if calibration_data is not None:
        gate.add_residual_calibration_batch(
            calibration_data
        )
    else:
        # Legacy initialization retained only for reproducibility of
        # previously completed experiments.
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
    enable_ltl = shield_mode in ("full", "hard-only")
    enable_gate = shield_mode in ("full", "soft-only")
    return Shield(
        enable_ltl=enable_ltl,
        enable_gate=enable_gate,
        ltl_monitor=SafetyMonitor(),
        gate=gate,
        timed=TimedMonitors(
            burst=BurstLimitMonitor(
                window_s=1.0,
                max_emissions=burst_max_emissions,
            ),
            latency=LatencyDeadlineMonitor(
                deadline_ms=decision_deadline_ms,
            ),
            quiet=QuietPeriodMonitor(
                quiet_ms=post_reject_quiet_ms,
            ),
        ),
        fallback_policy=FallbackPolicy.from_yaml(FALLBACK_CFG),
        cost_downscale=(
            _bounded_cost_downscale
            if enable_cost_repair
            else None
        ),
        cost_estimator=cost_estimator,
    )


def _ctx(mission_phase: MissionPhase, known_decoys: dict) -> ShieldContext:
    return ShieldContext(
        mission_phase=mission_phase,
        signing_active=True,
        decoy_key_active=True,
        known_decoys=known_decoys,
        bounded_template_whitelist=frozenset({"safe_heartbeat_v1"}),
    )


# Probing policy: 5 action variants per decoy per tick.
# Rationale (see chat design doc):
#   V1 safe_beacon           : baseline honey beacon (should DEPLOY).
#   V2 safe_decoy_telemetry  : DECOY_TELEMETRY, gives M3 that action_type.
#   V3 high_cost_beacon      : borderline L3 case (may DEPLOY or REJECT).
#   V4 malformed_beacon      : deterministic L2/S3 REJECT (null obs).
#   V5 forbidden_mavlink     : deterministic L1 REJECT (null obs).
#
# V4/V5 produce rows with observed_cost=None. They do not help M3 (which
# needs (features -> observed cost) pairs) but they DO give a downstream
# verdict/decision classifier full support, and they exercise Recorder
# schema paths that would otherwise be untested at scale.


def _safe_beacon(decoy_id: str) -> Action:
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"beacon_id": f"beacon_{decoy_id}", "advertise": {"ssid": "test"}},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
        gen_meta=_gen_meta(),
    )


def _safe_decoy_telemetry(decoy_id: str) -> Action:
    return Action(
        type=ActionType.DECOY_TELEMETRY,
        payload={"decoy_id": decoy_id, "fields": {"alt_m": 20.0, "bat_pct": 95.0}},
        target_iface=EffectorEnum.DECOY_MAVLINK_CHANNEL,
        predicted_cost=CostVector(0.015, 3.0, 0.8, 96.0, 0.015, 0.2),
        gen_meta=_gen_meta(),
    )


def _high_cost_beacon(decoy_id: str) -> Action:
    """Same beacon payload, inflated predicted_cost -> borderline L3."""
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"beacon_id": f"beacon_hi_{decoy_id}", "advertise": {"ssid": "test"}},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        # Inflated predictions bias L3 toward rejection under ACI drift.
        predicted_cost=CostVector(0.08, 12.0, 3.0, 800.0, 0.08, 1.5),
        gen_meta=_gen_meta(),
    )


def _malformed_beacon(decoy_id: str) -> Action:
    """Missing required 'beacon_id' -> L2/S3 REJECT every time."""
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"advertise": {"ssid": "test"}},  # no beacon_id
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
        gen_meta=_gen_meta(),
    )


def _forbidden_mavlink(decoy_id: str) -> Action:
    """MAVLINK_RESPONSE to FLIGHT_CONTROLLER_BUS -> L1 REJECT every time."""
    return Action(
        type=ActionType.MAVLINK_RESPONSE,
        payload={"msg_id": 0, "fields": {}},  # HEARTBEAT
        target_iface=EffectorEnum.FLIGHT_CONTROLLER_BUS,
        predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
        gen_meta=_gen_meta(),
    )


_PROBE_FACTORIES = (
    ("v1_safe_beacon", _safe_beacon),
    ("v2_safe_decoy_telemetry", _safe_decoy_telemetry),
    ("v3_high_cost_beacon", _high_cost_beacon),
    ("v4_malformed_beacon", _malformed_beacon),
    ("v5_forbidden_mavlink", _forbidden_mavlink),
)


def _build_world(
    seed: int, out_parquet: Path, cost_estimator: CostEstimator | None = None, shield_mode: str = "full"
):
    loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=DT_S))
    for i in range(N_MISSION_DRONES):
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
    known_decoys: dict[str, DecoyIdentity] = {}
    for i in range(N_DECOYS):
        did = f"h{i}"
        loop.add_drone(
            DecoyDrone(
                drone_id=did,
                position=Position2D(400.0, 300.0 + i * 30.0, 20.0),
            )
        )
        known_decoys[did] = DecoyIdentity(
            decoy_id=did,
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )
    loop.attach_shield(_build_shield(cost_estimator=cost_estimator, shield_mode=shield_mode))
    loop.attach_effector(Effector(EFFECTOR_CFG, seed=seed))
    loop.attach_recorder(Recorder(out_parquet, flush_every=10_000))
    loop.attach_adversary(PoissonCampaignAdversary(ADVERSARY_CFG, seed=seed))
    for mission in loop._missions.values():
        mission.arm()
    return loop, known_decoys


def run_episode(
    episode: int,
    seed: int,
    ticks: int,
    out_dir: Path = OUT_DIR,
    cost_estimator: CostEstimator | None = None,
    shield_mode: str = "full",
) -> EpisodeReport:
    out_parquet = out_dir / f"episode_{episode:02d}.parquet"
    loop, known_decoys = _build_world(seed, out_parquet, cost_estimator=cost_estimator, shield_mode=shield_mode)
    cal_before = loop._shield._gate._dims["cpu"].calibration_size

    campaign_events = 0
    # Track currently-active campaigns targeting mission drones, keyed by
    # campaign_id. Value: (start_tick, canonical_family). END events remove
    # entries (adversary END emission verified 17/17 in diagnostic).
    # The "primary_attack_family" for a probe action is derived globally:
    # if ANY mission drone is under attack this tick, use the earliest-
    # started active campaign's family/id. Matches the shield's semantics
    # of a single primary threat context per decision.
    active_campaigns: dict[str, tuple[int, str]] = {}
    wall_start = time.perf_counter()

    for tick in range(ticks):
        loop.step()
        campaign_events += len(loop.last_events)
        # Update active-campaign map from this tick's events.
        for ev in loop.last_events:
            if ev.kind.value == "CAMPAIGN_START":
                active_campaigns[ev.campaign_id] = (tick, ev.canonical_family)
            elif ev.kind.value == "CAMPAIGN_END":
                active_campaigns.pop(ev.campaign_id, None)
        # Derive the global "primary threat" for this tick: the earliest-
        # started active campaign, or (None, None) if idle.
        if active_campaigns:
            primary_cid = min(active_campaigns, key=lambda k: active_campaigns[k][0])
            primary_fam = active_campaigns[primary_cid][1]
        else:
            primary_cid, primary_fam = None, None
        m0 = loop.drones["m0"]
        ctx = _ctx(m0.mission_phase, known_decoys)
        for did in known_decoys:
            for _variant_name, factory in _PROBE_FACTORIES:
                loop.deploy(
                    factory(did),
                    ctx,
                    primary_attack_family=primary_fam,
                    campaign_id=primary_cid,
                )

        # Cheap progress dot every 2000 ticks.
        if (tick + 1) % 2000 == 0:
            elapsed = time.perf_counter() - wall_start
            print(
                f"  ep{episode:02d} tick {tick+1:>5}/{ticks}  " f"wall {elapsed:5.1f}s",
                flush=True,
            )

    loop._recorder.close()
    wall = time.perf_counter() - wall_start
    df = pl.read_parquet(out_parquet)
    verdicts = df.group_by("verdict").len().to_dicts() if df.height > 0 else []
    verdict_counts = {r["verdict"]: r["len"] for r in verdicts}
    cal_after = loop._shield._gate._dims["cpu"].calibration_size

    return EpisodeReport(
        episode=episode,
        seed=seed,
        rows_written=df.height,
        verdict_counts=verdict_counts,
        calibration_growth_cpu=cal_after - cal_before,
        campaign_events=campaign_events,
        wall_seconds=round(wall, 2),
        parquet_path=(
            str(out_parquet.relative_to(REPO))
            if out_parquet.is_relative_to(REPO)
            else str(out_parquet)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--ticks",
        type=int,
        default=TICKS_PER_EPISODE,
        help="Override ticks per episode (for smoke testing).",
    )
    parser.add_argument(
        "--cost-estimator",
        choices=("generator", "m3"),
        default="generator",
        help="Point cost estimator for L3 gate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for episode parquets and manifest.",
    )
    parser.add_argument(
        "--m3-artifact-dir",
        type=Path,
        default=REPO / "artifacts" / "cost_predictor",
        help="Directory containing M3 boosters (used only when --cost-estimator=m3).",
    )
    parser.add_argument(
        "--shield-mode",
        choices=["full", "off", "hard-only", "soft-only"],
        default="full",
        help="Shield configuration: full=L1+L2+L3+L4, off=bypass all, "
             "hard-only=L1+L2 only (no conformal gate), "
             "soft-only=L3 only (no LTL safety).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_estimator: CostEstimator | None = None
    if args.cost_estimator == "m3":
        if not args.m3_artifact_dir.exists():
            raise FileNotFoundError(
                f"--cost-estimator=m3 requires artifacts at {args.m3_artifact_dir}"
            )
        cost_estimator = M3CostEstimator.from_dir(args.m3_artifact_dir)
        print(
            f"[warmup] using M3 cost estimator from {args.m3_artifact_dir}", flush=True
        )
    else:
        print("[warmup] using generator cost estimator (default)", flush=True)

    manifest = WarmupManifest(
        n_episodes=args.episodes,
        ticks_per_episode=args.ticks,
        dt_s=DT_S,
        total_rows=0,
        total_wall_seconds=0.0,
    )

    total_start = time.perf_counter()
    for ep in range(args.episodes):
        seed = args.seed_base + ep
        print(f"\n=== episode {ep:02d} (seed={seed}) ===", flush=True)
        report = run_episode(
            ep, seed, args.ticks, out_dir=out_dir, cost_estimator=cost_estimator, shield_mode=args.shield_mode
        )
        manifest.episodes.append(report)
        manifest.total_rows += report.rows_written
        manifest.total_wall_seconds += report.wall_seconds
        print(
            f"  -> rows={report.rows_written} "
            f"verdicts={report.verdict_counts} "
            f"campaigns={report.campaign_events} "
            f"wall={report.wall_seconds}s",
            flush=True,
        )
    manifest.total_wall_seconds = round(time.perf_counter() - total_start, 2)

    (out_dir / "_warmup_manifest.json").write_text(
        json.dumps(
            {
                "n_episodes": manifest.n_episodes,
                "ticks_per_episode": manifest.ticks_per_episode,
                "dt_s": manifest.dt_s,
                "total_rows": manifest.total_rows,
                "total_wall_seconds": manifest.total_wall_seconds,
                "cost_estimator": args.cost_estimator,
                "shield_mode": args.shield_mode,
                "episodes": [asdict(e) for e in manifest.episodes],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"\nDone. {manifest.total_rows} rows across {manifest.n_episodes} "
        f"episodes in {manifest.total_wall_seconds}s. "
        f"Manifest: {out_dir / '_warmup_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
