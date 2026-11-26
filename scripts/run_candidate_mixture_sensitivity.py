"""
GADe-UAV candidate-composition sensitivity.

NO MODEL TRAINING.
Does not modify or overwrite the locked nominal/OOD mission evaluations.

Uses only the three exact resource profiles present in the M3
training/calibration support:
    safe_beacon
    safe_telemetry
    high_cost_beacon

Three candidates are still offered per defended tick. Only their
composition changes.

The mixture probabilities are simulation sensitivity settings,
not measured real-world UAV action frequencies.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_mission_eval as rme

from gade_uav.shield.cost_estimator import M3CostEstimator


REPO = Path(__file__).resolve().parents[1]

OUT_DIR = (
    REPO
    / "artifacts"
    / "candidate_mixture_sensitivity"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [
    42,
    123,
    2024,
    3407,
    7777,
]

SCENARIOS = (
    "generator_shield",
    "m3_shield",
)

MIXTURES = {
    "safe_heavy": {
        "safe_beacon": 0.60,
        "safe_telemetry": 0.30,
        "high_cost_beacon": 0.10,
    },
    "balanced_stochastic": {
        "safe_beacon": 1.0 / 3.0,
        "safe_telemetry": 1.0 / 3.0,
        "high_cost_beacon": 1.0 / 3.0,
    },
    "resource_heavy": {
        "safe_beacon": 0.25,
        "safe_telemetry": 0.25,
        "high_cost_beacon": 0.50,
    },
}

N_CANDIDATES_PER_TICK = 3


def make_candidate_sampler(probabilities):
    names = np.asarray(
        [
            "safe_beacon",
            "safe_telemetry",
            "high_cost_beacon",
        ],
        dtype=object,
    )

    p = np.asarray(
        [
            probabilities["safe_beacon"],
            probabilities["safe_telemetry"],
            probabilities["high_cost_beacon"],
        ],
        dtype=float,
    )

    if not np.isclose(p.sum(), 1.0):
        raise ValueError(
            f"mixture probabilities sum to {p.sum()}"
        )

    def sampled_candidates(
        rng: np.random.Generator,
        decoy_id: str,
    ):
        draws = rng.choice(
            names,
            size=N_CANDIDATES_PER_TICK,
            replace=True,
            p=p,
        )

        out = []

        for name in draws:
            if name == "safe_beacon":
                action = rme._safe_beacon(
                    decoy_id
                )

            elif name == "safe_telemetry":
                action = (
                    rme._safe_decoy_telemetry(
                        decoy_id
                    )
                )

            elif name == "high_cost_beacon":
                action = (
                    rme._high_cost_beacon(
                        decoy_id
                    )
                )

            else:
                raise RuntimeError(
                    f"unknown candidate variant: {name}"
                )

            out.append(
                (
                    str(name),
                    action,
                )
            )

        return out

    return sampled_candidates


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    rows = []

    for mixture in MIXTURES:
        for scenario in SCENARIOS:
            sub = df.filter(
                (pl.col("mixture") == mixture)
                & (pl.col("scenario") == scenario)
            )

            variant_totals = Counter()
            deploy_totals = Counter()
            reject_totals = Counter()

            for x in sub[
                "candidate_variant_counts_json"
            ].to_list():
                variant_totals.update(
                    json.loads(x)
                )

            for x in sub[
                "deploy_by_variant_json"
            ].to_list():
                deploy_totals.update(
                    json.loads(x)
                )

            for x in sub[
                "reject_by_variant_json"
            ].to_list():
                reject_totals.update(
                    json.loads(x)
                )

            total_candidates = int(
                sub["candidate_count"].sum()
            )

            total_deploy = int(
                sub["deploy_count"].sum()
            )

            total_modify = int(
                sub["modify_count"].sum()
            )

            total_reject = int(
                sub["reject_count"].sum()
            )

            amr_candidates = (
                "impact_mitigation_rate",
                "amr",
                "attack_mitigation_rate",
                "mission_amr",
            )

            amr_col = next(
                (
                    name
                    for name in amr_candidates
                    if name in sub.columns
                ),
                None,
            )

            if amr_col is None:
                raise RuntimeError(
                    "Could not locate AMR column. "
                    f"Available columns: {sub.columns}"
                )

            rows.append(
                {
                    "mixture": mixture,
                    "scenario": scenario,
                    "episodes": sub.height,
                    "candidate_count":
                        total_candidates,
                    "deploy_count":
                        total_deploy,
                    "modify_count":
                        total_modify,
                    "reject_count":
                        total_reject,
                    "admission_fraction":
                        (
                            (
                                total_deploy
                                + total_modify
                            )
                            / total_candidates
                            if total_candidates
                            else None
                        ),
                    "completion_rate":
                        float(
                            sub[
                                "completed"
                            ].mean()
                        ),
                    "mean_completion_time_s":
                        float(
                            sub[
                                "completion_time_s"
                            ].mean()
                        ),
                    "mean_final_battery_pct":
                        float(
                            sub[
                                "final_battery_pct"
                            ].mean()
                        ),
                    "mean_amr":
                        float(
                            sub[
                                amr_col
                            ].mean()
                        ),
                    "mean_episode_p95_latency_ms":
                        float(
                            sub[
                                "decision_latency_p95_ms"
                            ]
                            .drop_nulls()
                            .mean()
                        ),
                    "candidate_variants_json":
                        json.dumps(
                            dict(
                                variant_totals
                            ),
                            sort_keys=True,
                        ),
                    "deploy_variants_json":
                        json.dumps(
                            dict(
                                deploy_totals
                            ),
                            sort_keys=True,
                        ),
                    "reject_variants_json":
                        json.dumps(
                            dict(
                                reject_totals
                            ),
                            sort_keys=True,
                        ),
                }
            )

    return pl.DataFrame(rows)


def main():
    m3_estimator = (
        M3CostEstimator.from_dir(
            rme.M3_DIR
        )
    )

    original_sampler = (
        rme._mission_candidates
    )

    rows = []

    try:
        for mixture_name, probs in (
            MIXTURES.items()
        ):
            print(
                f"\n=== mixture: "
                f"{mixture_name} ==="
            )

            rme._mission_candidates = (
                make_candidate_sampler(
                    probs
                )
            )

            for seed in SEEDS:
                for scenario in SCENARIOS:
                    print(
                        f"  seed={seed} "
                        f"scenario={scenario}",
                        flush=True,
                    )

                    row = rme.run_episode(
                        scenario=scenario,
                        episode=seed,
                        seed=seed,
                        intensity=1.0,
                        m3_estimator=(
                            m3_estimator
                            if scenario
                            == "m3_shield"
                            else None
                        ),
                    )

                    row["mixture"] = (
                        mixture_name
                    )

                    row[
                        "mixture_probabilities_json"
                    ] = json.dumps(
                        probs,
                        sort_keys=True,
                    )

                    rows.append(row)

    finally:
        # Restore imported module state even
        # if an experiment fails.
        rme._mission_candidates = (
            original_sampler
        )

    df = pl.DataFrame(rows)

    raw_path = (
        OUT_DIR
        / "candidate_mixture_episodes.csv"
    )

    summary_path = (
        OUT_DIR
        / "candidate_mixture_summary.csv"
    )

    df.write_csv(
        raw_path
    )

    summary = summarize(
        df
    )

    summary.write_csv(
        summary_path
    )

    print()
    print(
        summary.select(
            [
                "mixture",
                "scenario",
                "candidate_count",
                "deploy_count",
                "reject_count",
                "admission_fraction",
                "completion_rate",
                "mean_completion_time_s",
                "mean_final_battery_pct",
                "mean_amr",
                "mean_episode_p95_latency_ms",
            ]
        )
    )

    print()
    print(
        "saved:",
        raw_path,
    )
    print(
        "saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
