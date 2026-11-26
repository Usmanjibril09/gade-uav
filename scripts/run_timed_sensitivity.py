"""
GADe-UAV isolated timed-monitor sensitivity.

NO MODEL TRAINING.
NO modification of locked nominal/OOD evaluations.

Each timed property is varied while the other two are relaxed so that
their effects cannot mask one another.

T1:
    burst thresholds = 10, 20, 40 emissions / 1 s

T2:
    deadlines = 2.5, 5.0, 10.0 ms

T3:
    quiet periods = 100, 200, 400 ms
    The simulated orchestrator obeys the quiet period after REJECT.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gade_uav.shield.actions import DecoyIdentity
from gade_uav.sim.mission import MissionPhase

from run_warmup_sim import (
    _build_shield,
    _ctx,
    _safe_beacon,
    _safe_decoy_telemetry,
    _high_cost_beacon,
)

REPO = Path(__file__).resolve().parents[1]

CALIBRATION = (
    REPO
    / "artifacts"
    / "cost_predictor"
    / "generator_runtime_calibration.npz"
)

OUT_DIR = REPO / "artifacts" / "timed_sensitivity_isolated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT_S = 0.01
N_STEPS = 5000


def _known_decoys():
    return {
        "h0": DecoyIdentity(
            decoy_id="h0",
            tau_alt_m=5.0,
            tau_pos_m=15.0,
            tau_bat_pct=2.0,
            tau_att_rad_s=3.14,
        )
    }


def _summarize(
    experiment,
    value,
    verdicts,
    layers,
    timed,
    latencies,
    offered,
    suppressed=0,
):
    a = np.asarray(latencies, dtype=float)

    return {
        "experiment": experiment,
        "threshold_value": float(value),
        "simulation_steps": N_STEPS,
        "candidate_offers": offered,
        "suppressed_candidates": suppressed,
        "suppressed_fraction": suppressed / N_STEPS,
        "deploy": verdicts["DEPLOY"],
        "modify": verdicts["MODIFY"],
        "reject": verdicts["REJECT"],
        "deploy_fraction_of_offered": (
            verdicts["DEPLOY"] / offered
            if offered
            else 0.0
        ),
        "reject_fraction_of_offered": (
            verdicts["REJECT"] / offered
            if offered
            else 0.0
        ),
        "l1_rejects": layers["L1"],
        "l2_rejects": layers["L2"],
        "l3_rejects": layers["L3"],
        "l4_rejects": layers["L4"],
        "t1_failures": timed["T1"],
        "t2_failures": timed["T2"],
        "t3_failures": timed["T3"],
        "latency_median_ms": (
            float(np.median(a))
            if len(a)
            else None
        ),
        "latency_p95_ms": (
            float(np.quantile(a, 0.95))
            if len(a)
            else None
        ),
        "latency_p99_ms": (
            float(np.quantile(a, 0.99))
            if len(a)
            else None
        ),
    }


def _record(decision, verdicts, layers, timed, latencies):
    verdicts[decision.verdict.value] += 1
    latencies.append(decision.decide_elapsed_ms)

    if decision.layer_failed:
        layers[decision.layer_failed] += 1

    for tf in decision.timed_failures:
        if not tf.holds and tf.monitor:
            timed[tf.monitor] += 1


def run_t1(max_emissions):
    """
    Isolate T1.

    T2 is relaxed to 1000 ms.
    T3 is effectively disabled with a 0.001 ms quiet interval.
    Only safe beacons are proposed.
    """
    shield = _build_shield(
        cost_estimator=None,
        shield_mode="full",
        calibration_npz=str(CALIBRATION),
        enable_cost_repair=False,
        burst_max_emissions=max_emissions,
        decision_deadline_ms=1000.0,
        post_reject_quiet_ms=0.001,
    )

    ctx = _ctx(
        MissionPhase.CRUISE,
        _known_decoys(),
    )

    verdicts = Counter()
    layers = Counter()
    timed = Counter()
    latencies = []

    for i in range(N_STEPS):
        now_s = i * DT_S

        decision = shield.decide(
            action=_safe_decoy_telemetry("h0"),
            ctx=ctx,
            now_s=now_s,
            primary_attack_family="Jamming",
        )

        _record(
            decision,
            verdicts,
            layers,
            timed,
            latencies,
        )

    return _summarize(
        "T1_burst_limit",
        max_emissions,
        verdicts,
        layers,
        timed,
        latencies,
        offered=N_STEPS,
    )


def run_t2(deadline_ms):
    """
    Isolate T2.

    T1 is effectively disabled.
    T3 is effectively disabled.
    """
    shield = _build_shield(
        cost_estimator=None,
        shield_mode="full",
        calibration_npz=str(CALIBRATION),
        enable_cost_repair=False,
        burst_max_emissions=1_000_000,
        decision_deadline_ms=deadline_ms,
        post_reject_quiet_ms=0.001,
    )

    ctx = _ctx(
        MissionPhase.CRUISE,
        _known_decoys(),
    )

    verdicts = Counter()
    layers = Counter()
    timed = Counter()
    latencies = []

    for i in range(N_STEPS):
        now_s = i * DT_S

        decision = shield.decide(
            action=_safe_beacon("h0"),
            ctx=ctx,
            now_s=now_s,
            primary_attack_family="Jamming",
        )

        _record(
            decision,
            verdicts,
            layers,
            timed,
            latencies,
        )

    return _summarize(
        "T2_latency_deadline",
        deadline_ms,
        verdicts,
        layers,
        timed,
        latencies,
        offered=N_STEPS,
    )


def run_t3(quiet_ms):
    """
    Isolate T3.

    Every 100 simulation steps, a deliberate high-cost beacon is offered
    to trigger a genuine L3 REJECT+FALLBACK.

    After rejection, the orchestrator respects T3 by suppressing generated
    candidates until the quiet period expires.

    The useful sensitivity measure is therefore candidate suppression /
    effective candidate throughput, not repeated T3 violations.
    """
    shield = _build_shield(
        cost_estimator=None,
        shield_mode="full",
        calibration_npz=str(CALIBRATION),
        enable_cost_repair=False,
        burst_max_emissions=1_000_000,
        decision_deadline_ms=1000.0,
        post_reject_quiet_ms=quiet_ms,
    )

    ctx = _ctx(
        MissionPhase.CRUISE,
        _known_decoys(),
    )

    verdicts = Counter()
    layers = Counter()
    timed = Counter()
    latencies = []

    quiet_until_s = -1.0
    offered = 0
    suppressed = 0

    for i in range(N_STEPS):
        now_s = i * DT_S

        # Orchestrator obeys the post-reject quiet contract.
        if now_s < quiet_until_s - 1e-12:
            suppressed += 1
            continue

        # Periodically trigger a genuine fallback event.
        if i % 100 == 99:
            action = _high_cost_beacon("h0")
        else:
            action = _safe_beacon("h0")

        decision = shield.decide(
            action=action,
            ctx=ctx,
            now_s=now_s,
            primary_attack_family="Jamming",
        )

        offered += 1

        _record(
            decision,
            verdicts,
            layers,
            timed,
            latencies,
        )

        if decision.verdict.value == "REJECT":
            quiet_until_s = now_s + quiet_ms / 1000.0

    return _summarize(
        "T3_quiet_period",
        quiet_ms,
        verdicts,
        layers,
        timed,
        latencies,
        offered=offered,
        suppressed=suppressed,
    )


def main():
    rows = []

    for n in (10, 20, 40):
        print(f"running isolated T1: {n} emissions/s ...", flush=True)
        rows.append(run_t1(n))

    for deadline in (2.5, 5.0, 10.0):
        print(f"running isolated T2: {deadline} ms ...", flush=True)
        rows.append(run_t2(deadline))

    for quiet in (100.0, 200.0, 400.0):
        print(f"running isolated T3: {quiet} ms ...", flush=True)
        rows.append(run_t3(quiet))

    df = pl.DataFrame(rows)

    csv_path = OUT_DIR / "timed_sensitivity_isolated.csv"
    json_path = OUT_DIR / "timed_sensitivity_isolated.json"

    df.write_csv(csv_path)
    json_path.write_text(
        json.dumps(rows, indent=2)
    )

    print()
    print(
        df.select(
            [
                "experiment",
                "threshold_value",
                "candidate_offers",
                "suppressed_candidates",
                "deploy",
                "reject",
                "l3_rejects",
                "l4_rejects",
                "t1_failures",
                "t2_failures",
                "t3_failures",
                "latency_median_ms",
                "latency_p95_ms",
                "latency_p99_ms",
            ]
        )
    )

    print()
    print("saved:", csv_path)
    print("saved:", json_path)


if __name__ == "__main__":
    main()
