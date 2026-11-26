"""
GADe-UAV mission-level evaluation.

NO MODEL TRAINING.

Scenarios:
    no_attack
    attack_only
    unshielded_defense
    generator_shield
    m3_shield

Uses:
- validated waypoint guidance;
- existing PoissonCampaignAdversary;
- bounded MissionImpactModel;
- existing shield implementation;
- frozen M3 model;
- identical adversary seeds across scenarios.

Important:
The current simulator/shield uses a single primary-threat abstraction.
When multiple campaigns overlap, the earliest-started active campaign
provides the family used for mission impact and shield context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from gade_uav.shield.cost_estimator import M3CostEstimator
from gade_uav.shield.actions import Action, CostVector, DecoyIdentity
from gade_uav.shield.decide import Verdict
from gade_uav.sim.adversary import (
    EventKind,
    PoissonCampaignAdversary,
)
from gade_uav.sim.clock import SimClock
from gade_uav.sim.drone import MissionDrone
from gade_uav.sim.effector import Effector
from gade_uav.sim.grid import Grid, Position2D
from gade_uav.sim.loop import SimulationLoop
from gade_uav.sim.mission import Mission, Waypoint
from gade_uav.sim.mission_impact import (
    ImpactSample,
    MissionImpact,
    MissionImpactModel,
    scalar_impact,
)

# Reuse the exact audited shield configuration and candidate constructors.
from run_warmup_sim import (
    ADVERSARY_CFG,
    EFFECTOR_CFG,
    _build_shield,
    _ctx,
    _high_cost_beacon,
    _safe_beacon,
    _safe_decoy_telemetry,
)


REPO = Path(__file__).resolve().parents[1]

M3_DIR = REPO / "artifacts" / "cost_predictor"

M3_CALIBRATION = (
    M3_DIR / "runtime_calibration.npz"
)

GENERATOR_CALIBRATION = (
    M3_DIR / "generator_runtime_calibration.npz"
)

OUT_ROOT = (
    REPO
    / "artifacts"
    / "mission_eval"
)

DT_S = 0.01
MAX_TICKS = 12_000      # 120 simulated seconds
N_WAYPOINTS = 3

SCENARIOS = (
    "no_attack",
    "attack_only",
    "unshielded_defense",
    "generator_shield",
    "m3_shield",
)

# Mission-evaluation candidates intentionally exclude malformed and
# forbidden probes. Those remain part of the functional shield experiment,
# but are not meaningful defensive candidates in a mission-effectiveness run.
#
# The mission experiment uses bounded synthetic variation in the declared
# resource-cost vectors. This is NOT a learned or GenAI generator. It prevents
# the mission study from degenerating into a fixed easy/easy/impossible
# three-action pattern and lets the frozen raw estimator and frozen M3
# encounter candidate costs near different resource boundaries.

CANDIDATE_VARIANTS = (
    "safe_beacon",
    "variable_beacon",
    "variable_telemetry",
    "high_cost_beacon",
)


def _replace_cost(
    action: Action,
    multipliers: np.ndarray,
) -> Action:
    """Return the same defensive action with a perturbed declared cost."""
    base = action.predicted_cost

    vals = np.asarray(
        [
            base.cpu,
            base.ram,
            base.lat,
            base.bw,
            base.energy,
            base.mission_delay,
        ],
        dtype=float,
    )

    vals = vals * np.asarray(multipliers, dtype=float)

    return Action(
        type=action.type,
        payload=action.payload,
        target_iface=action.target_iface,
        predicted_cost=CostVector(
            cpu=float(vals[0]),
            ram=float(vals[1]),
            lat=float(vals[2]),
            bw=float(vals[3]),
            energy=float(vals[4]),
            mission_delay=float(vals[5]),
        ),
        gen_meta=action.gen_meta,
    )


def _mission_candidates(
    rng: np.random.Generator,
    decoy_id: str,
) -> list[tuple[str, Action]]:
    """
    Produce one deterministic-under-seed candidate set.

    1. Safe beacon: fixed low-cost reference.
    2. Variable beacon: bounded cost perturbation.
    3. Variable telemetry: broader cost profile, including near-budget cases.
    4. High-cost beacon: explicit resource-stress reference.

    Perturbation ranges are simulation-study parameters, not measurements
    from physical UAV hardware.
    """

    safe = _safe_beacon(decoy_id)

    beacon_mult = rng.uniform(
        low=0.80,
        high=2.80,
        size=6,
    )

    telemetry_mult = rng.uniform(
        low=0.75,
        high=2.30,
        size=6,
    )

    variable_beacon = _replace_cost(
        _safe_beacon(decoy_id),
        beacon_mult,
    )

    variable_telemetry = _replace_cost(
        _safe_decoy_telemetry(decoy_id),
        telemetry_mult,
    )

    high = _high_cost_beacon(decoy_id)

    return [
        ("safe_beacon", safe),
        ("variable_beacon", variable_beacon),
        ("variable_telemetry", variable_telemetry),
        ("high_cost_beacon", high),
    ]


def make_world(
    seed: int,
    scenario: str,
    m3_estimator: M3CostEstimator | None,
):
    loop = SimulationLoop(
        grid=Grid(),
        clock=SimClock(dt_s=DT_S),
    )

    drone = MissionDrone(
        drone_id="m0",
        position=Position2D(
            50.0,
            100.0,
            0.0,
        ),
    )

    mission = Mission(
        waypoints=[
            Waypoint(
                50.0,
                180.0,
                20.0,
            ),
            Waypoint(
                120.0,
                220.0,
                20.0,
            ),
            Waypoint(
                180.0,
                160.0,
                20.0,
            ),
        ],
        cruise_alt_m=20.0,
        wp_tol_m=3.0,
        hover_duration_s=1.0,
    )

    loop.add_drone(drone)
    loop.attach_mission(
        drone.drone_id,
        mission,
    )
    mission.arm()

    # Logical decoy identity. A physical DecoyDrone is unnecessary here:
    # the shield requires identity/state metadata, while the Effector
    # operates on the defensive action itself.
    known_decoys = {
        "h0": DecoyIdentity(
            decoy_id="h0",
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )
    }

    if scenario != "no_attack":
        loop.attach_adversary(
            PoissonCampaignAdversary(
                ADVERSARY_CFG,
                seed=seed,
            )
        )

    if scenario == "generator_shield":
        shield = _build_shield(
            cost_estimator=None,
            shield_mode="full",
            calibration_npz=str(
                GENERATOR_CALIBRATION
            ),
        )
        loop.attach_shield(shield)
        loop.attach_effector(
            Effector(
                EFFECTOR_CFG,
                seed=seed,
            )
        )

    elif scenario == "m3_shield":
        if m3_estimator is None:
            raise RuntimeError(
                "m3_shield requires M3 estimator"
            )

        shield = _build_shield(
            cost_estimator=m3_estimator,
            shield_mode="full",
            calibration_npz=str(
                M3_CALIBRATION
            ),
        )
        loop.attach_shield(shield)
        loop.attach_effector(
            Effector(
                EFFECTOR_CFG,
                seed=seed,
            )
        )

    return (
        loop,
        drone,
        mission,
        known_decoys,
    )


def zero_sample() -> ImpactSample:
    z = MissionImpact.zero()

    return ImpactSample(
        family=None,
        pre_defense=z,
        post_defense=z,
        effectiveness=0.0,
        defense_active=False,
    )


def run_episode(
    *,
    scenario: str,
    episode: int,
    seed: int,
    intensity: float,
    m3_estimator: M3CostEstimator | None,
) -> dict:
    (
        loop,
        drone,
        mission,
        known_decoys,
    ) = make_world(
        seed,
        scenario,
        m3_estimator,
    )

    impact_model = MissionImpactModel(
        seed=seed + 50_000,
        intensity=intensity,
        noise_fraction=0.10,
    )

    # Independent RNG for candidate-cost variation.
    # Same seed => same candidate-cost stream across generator/M3 scenarios.
    candidate_rng = np.random.default_rng(
        seed + 90_000
    )

    active_campaigns: dict[
        str,
        tuple[int, str],
    ] = {}

    current_sample = zero_sample()
    loop.set_mission_impact(
        current_sample.post_defense
    )

    campaign_starts = 0
    attacked_ticks = 0
    defended_ticks = 0

    candidate_count = 0
    deploy_count = 0
    modify_count = 0
    reject_count = 0

    candidate_variant_counts = {
        name: 0 for name in CANDIDATE_VARIANTS
    }
    deploy_by_variant = {
        name: 0 for name in CANDIDATE_VARIANTS
    }
    reject_by_variant = {
        name: 0 for name in CANDIDATE_VARIANTS
    }
    modify_by_variant = {
        name: 0 for name in CANDIDATE_VARIANTS
    }

    decision_latencies: list[float] = []

    # Longitudinal M3 gate trace.
    # Diagnostic only: does not alter decisions or calibration.
    m3_safe_beacon_trace: list[dict] = []

    # Diagnostic counters for understanding shield behavior.
    reject_by_phase: dict[str, int] = {}
    deploy_by_phase: dict[str, int] = {}
    reject_by_family: dict[str, int] = {}
    deploy_by_family: dict[str, int] = {}
    gate_failure_dimensions: dict[str, int] = {}
    safe_beacon_reject_by_phase: dict[str, int] = {}
    safe_beacon_deploy_by_phase: dict[str, int] = {}

    # Time-integrated impact scores.
    # Multiplying by dt gives impact exposure over simulated time.
    cumulative_pre_impact = 0.0
    cumulative_post_impact = 0.0

    cumulative_command_delay_ms_s = 0.0
    cumulative_link_loss_s = 0.0

    attack_family_tick_counts: dict[str, int] = {}

    completed = False

    for tick in range(MAX_TICKS):
        # The currently configured post-defense impact is what this step
        # actually experiences.
        cumulative_pre_impact += (
            scalar_impact(
                current_sample.pre_defense
            )
            * DT_S
        )

        cumulative_post_impact += (
            scalar_impact(
                current_sample.post_defense
            )
            * DT_S
        )

        cumulative_command_delay_ms_s += (
            current_sample.post_defense.command_delay_ms
            * DT_S
        )

        cumulative_link_loss_s += (
            current_sample.post_defense.link_loss_prob
            * DT_S
        )

        if current_sample.family is not None:
            attacked_ticks += 1

            attack_family_tick_counts[
                current_sample.family
            ] = (
                attack_family_tick_counts.get(
                    current_sample.family,
                    0,
                )
                + 1
            )

        if current_sample.defense_active:
            defended_ticks += 1

        # Mission dynamics use current_sample.post_defense.
        loop.step()

        # Mission has completed; don't construct a next-tick impact state.
        if mission.phase.value == "post_landing":
            completed = True
            break

        # Update campaign state using events emitted by this step.
        for ev in loop.last_events:
            if ev.kind == EventKind.CAMPAIGN_START:
                active_campaigns[
                    ev.campaign_id
                ] = (
                    tick,
                    ev.canonical_family,
                )
                campaign_starts += 1

            elif ev.kind == EventKind.CAMPAIGN_END:
                active_campaigns.pop(
                    ev.campaign_id,
                    None,
                )

        if not active_campaigns:
            current_sample = zero_sample()
            loop.clear_mission_impact()
            continue

        # Existing GADe-UAV semantics:
        # earliest-active campaign is primary threat.
        primary_cid = min(
            active_campaigns,
            key=lambda cid:
                active_campaigns[cid][0],
        )

        primary_family = (
            active_campaigns[
                primary_cid
            ][1]
        )

        pre = impact_model.pre_defense_impact(
            primary_family
        )

        # ------------------------------------------------------------
        # Scenario-specific defensive behavior
        # ------------------------------------------------------------

        if scenario == "attack_only":
            current_sample = (
                impact_model.attenuate(
                    primary_family,
                    pre,
                    defense_active=False,
                )
            )

        elif scenario == "unshielded_defense":
            # Upper-bound scripted-defense condition:
            # all admissible candidate mechanisms are assumed available.
            current_sample = (
                impact_model.attenuate(
                    primary_family,
                    pre,
                    defense_active=True,
                    effectiveness_scale=1.0,
                )
            )

        elif scenario in (
            "generator_shield",
            "m3_shield",
        ):
            ctx = _ctx(
                mission.phase,
                known_decoys,
            )

            deployed_this_tick = 0

            candidates = _mission_candidates(
                candidate_rng,
                "h0",
            )

            for variant_name, action in candidates:
                candidate_count += 1
                candidate_variant_counts[variant_name] += 1

                decision = loop.deploy(
                    action,
                    ctx,
                    primary_attack_family=
                        primary_family,
                    campaign_id=
                        primary_cid,
                )

                decision_latencies.append(
                    float(
                        decision.decide_elapsed_ms
                    )
                )

                # Record the actual M3 gate state for the nominal safe
                # beacon. This is observational only.
                if (
                    scenario == "m3_shield"
                    and variant_name == "safe_beacon"
                ):
                    gate = decision.gate_decision

                    trace_row = {
                        "seed": seed,
                        "tick": tick,
                        "time_s": float(loop.clock.now_s()),
                        "phase": mission.phase.value,
                        "family": primary_family,
                        "verdict": decision.verdict.value,
                    }

                    if gate is not None:
                        for d in gate.per_dimension:
                            trace_row[
                                f"{d.dimension}_predicted"
                            ] = float(d.predicted)

                            trace_row[
                                f"{d.dimension}_ucb"
                            ] = float(d.ucb)

                            trace_row[
                                f"{d.dimension}_q"
                            ] = float(
                                d.ucb - d.predicted
                            )

                            trace_row[
                                f"{d.dimension}_budget"
                            ] = float(d.budget)

                            trace_row[
                                f"{d.dimension}_alpha"
                            ] = float(
                                d.alpha_current
                            )

                            trace_row[
                                f"{d.dimension}_admitted"
                            ] = bool(
                                d.admitted
                            )

                    m3_safe_beacon_trace.append(
                        trace_row
                    )

                phase_name = mission.phase.value
                family_name = primary_family or "NoActiveAttack"

                if decision.verdict == Verdict.DEPLOY:
                    deployed_this_tick += 1
                    deploy_count += 1
                    deploy_by_variant[variant_name] += 1

                    deploy_by_phase[phase_name] = (
                        deploy_by_phase.get(phase_name, 0) + 1
                    )
                    deploy_by_family[family_name] = (
                        deploy_by_family.get(family_name, 0) + 1
                    )

                    if variant_name == "safe_beacon":
                        safe_beacon_deploy_by_phase[phase_name] = (
                            safe_beacon_deploy_by_phase.get(phase_name, 0) + 1
                        )

                elif decision.verdict == Verdict.MODIFY:
                    deployed_this_tick += 1
                    modify_count += 1
                    modify_by_variant[variant_name] += 1

                else:
                    reject_count += 1
                    reject_by_variant[variant_name] += 1

                    reject_by_phase[phase_name] = (
                        reject_by_phase.get(phase_name, 0) + 1
                    )
                    reject_by_family[family_name] = (
                        reject_by_family.get(family_name, 0) + 1
                    )

                    if variant_name == "safe_beacon":
                        safe_beacon_reject_by_phase[phase_name] = (
                            safe_beacon_reject_by_phase.get(phase_name, 0) + 1
                        )

                    if decision.gate_decision is not None:
                        for dim in decision.gate_decision.failing_dimensions:
                            gate_failure_dimensions[dim] = (
                                gate_failure_dimensions.get(dim, 0) + 1
                            )

            defense_scale = (
                deployed_this_tick
                / len(candidates)
            )

            current_sample = (
                impact_model.attenuate(
                    primary_family,
                    pre,
                    defense_active=
                        deployed_this_tick > 0,
                    effectiveness_scale=
                        defense_scale,
                )
            )

        elif scenario == "no_attack":
            # Should not normally reach this branch because no adversary
            # is attached.
            current_sample = zero_sample()

        else:
            raise ValueError(
                f"unknown scenario {scenario}"
            )

        loop.set_mission_impact(
            current_sample.post_defense
        )

    completion_time_s = (
        loop.clock.now_s()
        if completed
        else None
    )

    waypoint_ratio = (
        mission._wp_index
        / len(mission.waypoints)
    )

    amr = (
        1.0
        - cumulative_post_impact
        / cumulative_pre_impact
        if cumulative_pre_impact > 0
        else None
    )

    latency = np.asarray(
        decision_latencies,
        dtype=float,
    )

    if latency.size:
        latency_stats = {
            "decision_latency_median_ms":
                float(
                    np.median(latency)
                ),
            "decision_latency_p95_ms":
                float(
                    np.quantile(
                        latency,
                        0.95,
                    )
                ),
            "decision_latency_p99_ms":
                float(
                    np.quantile(
                        latency,
                        0.99,
                    )
                ),
        }
    else:
        latency_stats = {
            "decision_latency_median_ms":
                None,
            "decision_latency_p95_ms":
                None,
            "decision_latency_p99_ms":
                None,
        }

    # Save longitudinal trace separately so the normal episode table
    # remains compact.
    trace_path = None

    if (
        scenario == "m3_shield"
        and m3_safe_beacon_trace
    ):
        trace_dir = (
            OUT_ROOT
            / "traces"
        )
        trace_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace_path = (
            trace_dir
            / f"m3_safe_beacon_seed_{seed}.csv"
        )

        pl.DataFrame(
            m3_safe_beacon_trace
        ).write_csv(
            trace_path
        )

    row = {
        "scenario": scenario,
        "episode": episode,
        "m3_trace_path":
            str(trace_path)
            if trace_path is not None
            else None,
        "seed": seed,
        "intensity": intensity,

        "completed": completed,
        "completion_time_s":
            completion_time_s,
        "waypoints_completed":
            mission._wp_index,
        "waypoint_completion_ratio":
            waypoint_ratio,
        "final_battery_pct":
            float(
                drone.battery_pct
            ),
        "emergency_recovery":
            (
                mission.phase.value
                == "emergency_recovery"
            ),

        "campaign_starts":
            campaign_starts,
        "attacked_ticks":
            attacked_ticks,
        "attacked_time_s":
            attacked_ticks * DT_S,
        "defended_ticks":
            defended_ticks,
        "defended_time_s":
            defended_ticks * DT_S,

        "candidate_count":
            candidate_count,
        "deploy_count":
            deploy_count,
        "modify_count":
            modify_count,
        "reject_count":
            reject_count,

        "candidate_variant_counts_json":
            json.dumps(
                candidate_variant_counts,
                sort_keys=True,
            ),
        "deploy_by_variant_json":
            json.dumps(
                deploy_by_variant,
                sort_keys=True,
            ),
        "modify_by_variant_json":
            json.dumps(
                modify_by_variant,
                sort_keys=True,
            ),
        "reject_by_variant_json":
            json.dumps(
                reject_by_variant,
                sort_keys=True,
            ),
        "reject_by_phase_json":
            json.dumps(
                reject_by_phase,
                sort_keys=True,
            ),
        "deploy_by_phase_json":
            json.dumps(
                deploy_by_phase,
                sort_keys=True,
            ),
        "reject_by_family_json":
            json.dumps(
                reject_by_family,
                sort_keys=True,
            ),
        "deploy_by_family_json":
            json.dumps(
                deploy_by_family,
                sort_keys=True,
            ),
        "gate_failure_dimensions_json":
            json.dumps(
                gate_failure_dimensions,
                sort_keys=True,
            ),
        "safe_beacon_reject_by_phase_json":
            json.dumps(
                safe_beacon_reject_by_phase,
                sort_keys=True,
            ),
        "safe_beacon_deploy_by_phase_json":
            json.dumps(
                safe_beacon_deploy_by_phase,
                sort_keys=True,
            ),

        "cumulative_pre_impact":
            cumulative_pre_impact,
        "cumulative_post_impact":
            cumulative_post_impact,
        "impact_mitigation_rate":
            amr,

        "integrated_command_delay_ms_s":
            cumulative_command_delay_ms_s,
        "integrated_link_loss_s":
            cumulative_link_loss_s,

        "attack_family_tick_counts_json":
            json.dumps(
                attack_family_tick_counts,
                sort_keys=True,
            ),

        **latency_stats,
    }

    return row


def print_summary(df: pl.DataFrame) -> None:
    print(
        "\n"
        "============================================================"
    )
    print(
        "MISSION EVALUATION SUMMARY"
    )
    print(
        "============================================================"
    )

    for scenario in SCENARIOS:
        sub = df.filter(
            pl.col("scenario")
            == scenario
        )

        if sub.height == 0:
            continue

        completed = (
            sub[
                "completed"
            ].cast(
                pl.Int64
            ).mean()
        )

        times = (
            sub.filter(
                pl.col(
                    "completion_time_s"
                ).is_not_null()
            )[
                "completion_time_s"
            ]
        )

        mean_time = (
            float(
                times.mean()
            )
            if len(times)
            else None
        )

        mean_battery = float(
            sub[
                "final_battery_pct"
            ].mean()
        )

        amr_values = sub.filter(
            pl.col(
                "impact_mitigation_rate"
            ).is_not_null()
        )[
            "impact_mitigation_rate"
        ]

        mean_amr = (
            float(
                amr_values.mean()
            )
            if len(amr_values)
            else None
        )

        print(
            f"\n{scenario}"
        )
        print(
            f"  episodes={sub.height}"
        )
        print(
            f"  completion_rate={completed:.3f}"
        )
        print(
            f"  mean_completion_time_s={mean_time}"
        )
        print(
            f"  mean_final_battery_pct={mean_battery:.3f}"
        )
        print(
            f"  mean_AMR={mean_amr}"
        )

        if scenario in (
            "generator_shield",
            "m3_shield",
        ):
            total_candidates = int(
                sub[
                    "candidate_count"
                ].sum()
            )

            total_deploy = int(
                sub[
                    "deploy_count"
                ].sum()
            )

            total_modify = int(
                sub[
                    "modify_count"
                ].sum()
            )

            print(
                f"  candidates={total_candidates:,}"
            )
            print(
                f"  deploy={total_deploy:,}"
            )
            print(
                f"  modify={total_modify:,}"
            )

            lat = sub[
                "decision_latency_p95_ms"
            ].drop_nulls()

            if len(lat):
                print(
                    "  mean_episode_p95_latency_ms="
                    f"{float(lat.mean()):.3f}"
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[410, 411],
    )

    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_ROOT / "smoke",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading frozen M3 estimator..."
    )

    m3_estimator = (
        M3CostEstimator.from_dir(
            M3_DIR
        )
    )

    rows = []

    total = (
        len(args.seeds)
        * len(SCENARIOS)
    )

    done = 0

    for ep, seed in enumerate(
        args.seeds
    ):
        for scenario in SCENARIOS:
            done += 1

            print(
                f"[{done}/{total}] "
                f"seed={seed} "
                f"scenario={scenario}",
                flush=True,
            )

            row = run_episode(
                scenario=scenario,
                episode=ep,
                seed=seed,
                intensity=
                    args.intensity,
                m3_estimator=
                    m3_estimator,
            )

            rows.append(
                row
            )

            print(
                "    "
                f"completed={row['completed']} "
                f"time={row['completion_time_s']} "
                f"battery={row['final_battery_pct']:.3f} "
                f"AMR={row['impact_mitigation_rate']}"
            )

    df = pl.DataFrame(
        rows
    )

    out_csv = (
        args.out_dir
        / "mission_eval.csv"
    )

    df.write_csv(
        out_csv
    )

    print_summary(
        df
    )

    print(
        f"\nSaved: {out_csv}"
    )


if __name__ == "__main__":
    main()
