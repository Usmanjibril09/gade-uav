"""
GADe-UAV C3b:
paired mission-level causal comparison of

    attack_only
    fixed_uniform
    genai_v2_cached

The deception policies reuse the exact C3a candidate-selection logic.

Causal mission coupling:
    if the current primary campaign is actively diverted,
    apply the existing MissionImpactModel attenuation with
    effectiveness_scale=1.0;
    otherwise apply no mission-impact attenuation.

Therefore AMR reported here is diversion-attributable within this
simulation model. No online LLM generation occurs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

import run_mission_eval as rme
import run_genai_causal_compare as c3a

from gade_uav.shield.cost_estimator import M3CostEstimator
from gade_uav.shield.decide import Verdict
from gade_uav.sim.adversary import EventKind
from gade_uav.sim.deception_response import DeceptionResponseModel
from gade_uav.sim.mission_impact import (
    MissionImpactModel,
    scalar_impact,
)


REPO = Path(__file__).resolve().parents[1]

OUT = (
    REPO
    / "artifacts"
    / "genai_mission_compare"
)

SEEDS = [
    42,
    123,
    2024,
    3407,
    7777,
]

METHODS = (
    "attack_only",
    "fixed_uniform",
    "genai_v2_cached",
)

INTENSITY = 1.0


def run_episode(
    *,
    seed: int,
    method: str,
    library: dict,
    m3_estimator: M3CostEstimator,
) -> dict:

    if method == "attack_only":
        world_scenario = "attack_only"
        estimator = None
    else:
        world_scenario = "m3_shield"
        estimator = m3_estimator

    (
        loop,
        drone,
        mission,
        known_decoys,
    ) = rme.make_world(
        seed,
        world_scenario,
        estimator,
    )

    impact_model = MissionImpactModel(
        seed=seed + 50_000,
        intensity=INTENSITY,
        noise_fraction=0.10,
    )

    response = None

    if method != "attack_only":
        response = DeceptionResponseModel(
            seed=seed + 120_000,
        )

    select_rng = np.random.default_rng(
        seed + 130_000
    )

    active_campaigns: dict[
        str,
        tuple[int, str],
    ] = {}

    attempted_campaigns: set[str] = set()

    current_sample = rme.zero_sample()

    loop.set_mission_impact(
        current_sample.post_defense
    )

    cumulative_pre_impact = 0.0
    cumulative_post_impact = 0.0

    attacked_ticks = 0
    diverted_ticks = 0

    candidate_count = 0
    admitted_count = 0
    rejected_count = 0

    cache_misses = 0
    unsupported_phase_opportunities = 0
    unsupported_campaign_ids: set[str] = set()
    unsupported_phase_counts = Counter()

    decision_latency_ms: list[float] = []

    variant_counts = Counter()
    theme_counts = Counter()
    profile_counts = Counter()
    candidate_type_counts = Counter()

    completed = False

    for tick in range(rme.MAX_TICKS):

        # ------------------------------------------------------------
        # Impact experienced during the tick that is about to advance.
        # This mirrors the current mission evaluator's current_sample
        # integration convention.
        # ------------------------------------------------------------

        cumulative_pre_impact += (
            scalar_impact(
                current_sample.pre_defense
            )
            * rme.DT_S
        )

        cumulative_post_impact += (
            scalar_impact(
                current_sample.post_defense
            )
            * rme.DT_S
        )

        if current_sample.family is not None:
            attacked_ticks += 1

        if current_sample.defense_active:
            diverted_ticks += 1

        # ------------------------------------------------------------
        # Advance simulator / mission / adversary.
        # ------------------------------------------------------------

        loop.step()

        now_s = float(
            loop.clock.now_s()
        )

        # ------------------------------------------------------------
        # Update campaign lifecycle.
        # ------------------------------------------------------------

        for event in loop.last_events:

            cid = str(
                event.campaign_id
            )

            if event.kind == EventKind.CAMPAIGN_START:

                family = str(
                    event.canonical_family
                )

                active_campaigns[
                    cid
                ] = (
                    tick,
                    family,
                )

                if response is not None:
                    response.register_campaign(
                        campaign_id=cid,
                        family=family,
                        now_s=now_s,
                    )

            elif event.kind == EventKind.CAMPAIGN_END:

                if response is not None:
                    response.end_campaign(
                        campaign_id=cid,
                        now_s=now_s,
                    )

                active_campaigns.pop(
                    cid,
                    None,
                )

        if response is not None:
            response.advance(
                now_s
            )

        # Mission terminal condition.
        if mission.phase.value == "post_landing":
            completed = True
            break

        # ------------------------------------------------------------
        # No active attack -> zero mission impact.
        # ------------------------------------------------------------

        if not active_campaigns:

            current_sample = (
                rme.zero_sample()
            )

            loop.set_mission_impact(
                current_sample.post_defense
            )

            continue

        # ------------------------------------------------------------
        # Preserve existing single-primary-threat abstraction:
        # earliest-started active campaign.
        # ------------------------------------------------------------

        primary_cid, (
            _start_tick,
            primary_family,
        ) = min(
            active_campaigns.items(),
            key=lambda kv: kv[1][0],
        )

        phase = str(
            mission.phase.value
        )

        # ------------------------------------------------------------
        # One candidate opportunity per campaign, exactly as C3a.
        # ------------------------------------------------------------

        if (
            method != "attack_only"
            and primary_cid
            not in attempted_campaigns
        ):

            if (
                phase
                not in c3a.SUPPORTED_PHASES
            ):
                unsupported_phase_opportunities += 1
                unsupported_campaign_ids.add(
                    primary_cid
                )
                unsupported_phase_counts[
                    phase
                ] += 1

            else:
                attempted_campaigns.add(
                    primary_cid
                )

                if method == "fixed_uniform":

                    variant = str(
                        select_rng.choice(
                            c3a.FIXED_VARIANTS
                        )
                    )

                    theme = (
                        c3a.FIXED_THEMES[
                            variant
                        ]
                    )

                    action = (
                        c3a.fixed_action(
                            variant
                        )
                    )

                    profile = (
                        "high"
                        if variant
                        == "high_cost_beacon"
                        else "fixed"
                    )

                    candidate_type = (
                        "decoy_telemetry"
                        if variant
                        == "safe_telemetry"
                        else "honey_beacon"
                    )

                    ctx = rme._ctx(
                        mission.phase,
                        known_decoys,
                    )

                elif method == "genai_v2_cached":

                    obj = library.get(
                        (
                            primary_family,
                            phase,
                        )
                    )

                    if obj is None:
                        cache_misses += 1
                        continue

                    (
                        action,
                        variant,
                    ) = (
                        c3a.genai_action_and_variant(
                            obj=obj,
                            family=primary_family,
                            phase=phase,
                        )
                    )

                    theme = str(
                        obj["theme"]
                    )

                    profile = str(
                        obj["profile"]
                    )

                    candidate_type = str(
                        obj["candidate_type"]
                    )

                    ctx = c3a.genai_context(
                        mission.phase,
                        known_decoys,
                    )

                else:
                    raise ValueError(
                        method
                    )

                candidate_count += 1

                variant_counts[
                    variant
                ] += 1

                theme_counts[
                    theme
                ] += 1

                profile_counts[
                    profile
                ] += 1

                candidate_type_counts[
                    candidate_type
                ] += 1

                decision = loop.deploy(
                    action,
                    ctx,
                    primary_attack_family=(
                        primary_family
                    ),
                    campaign_id=(
                        primary_cid
                    ),
                )

                decision_latency_ms.append(
                    float(
                        decision.decide_elapsed_ms
                    )
                )

                admitted = (
                    decision.verdict
                    in (
                        Verdict.DEPLOY,
                        Verdict.MODIFY,
                    )
                )

                if admitted:

                    admitted_count += 1

                    response.attempt(
                        campaign_id=primary_cid,
                        variant=variant,
                        theme=theme,
                        now_s=now_s,
                    )

                else:
                    rejected_count += 1

        # ------------------------------------------------------------
        # Generate raw attack impact for this primary family.
        # ------------------------------------------------------------

        pre = (
            impact_model.pre_defense_impact(
                primary_family
            )
        )

        # ------------------------------------------------------------
        # C3b causal bridge:
        #
        # The verified response API tells us whether the primary
        # campaign is CURRENTLY diverted at this simulated instant.
        #
        # Only active diversion attenuates mission impact.
        # ------------------------------------------------------------

        diverted_now = False

        if response is not None:

            diverted_now = (
                response.is_diverted(
                    primary_cid,
                    now_s=now_s,
                )
            )

        current_sample = (
            impact_model.attenuate(
                primary_family,
                pre,
                defense_active=(
                    diverted_now
                ),
                effectiveness_scale=(
                    1.0
                    if diverted_now
                    else 0.0
                ),
            )
        )

        loop.set_mission_impact(
            current_sample.post_defense
        )

    # ------------------------------------------------------------
    # Finalize any still-open campaign diversion durations.
    # ------------------------------------------------------------

    end_s = float(
        loop.clock.now_s()
    )

    if response is not None:

        for cid in list(
            active_campaigns
        ):
            response.end_campaign(
                campaign_id=cid,
                now_s=end_s,
            )

        deception = (
            response.summary()
        )

    else:

        deception = {
            "registered_campaigns":
                0,
            "eligible_deception_attempts":
                0,
            "diverted_campaigns":
                0,
            "sustained_engagements":
                0,
            "returned_to_target":
                0,
            "attack_diversion_probability":
                None,
            "deception_success_ratio":
                None,
            "return_rate_given_diversion":
                None,
            "mean_diversion_duration_s":
                None,
        }

    completion_time_s = (
        end_s
        if completed
        else None
    )

    amr = (
        1.0
        - (
            cumulative_post_impact
            / cumulative_pre_impact
        )
        if cumulative_pre_impact > 0.0
        else None
    )

    impact_reduction = (
        cumulative_pre_impact
        - cumulative_post_impact
    )

    return {
        "method":
            method,
        "seed":
            seed,

        "completed":
            completed,
        "completion_time_s":
            completion_time_s,
        "waypoints_completed":
            int(
                mission._wp_index
            ),
        "final_battery_pct":
            float(
                drone.battery_pct
            ),

        "attacked_ticks":
            attacked_ticks,
        "attacked_time_s":
            attacked_ticks
            * rme.DT_S,

        "diverted_ticks":
            diverted_ticks,
        "diverted_time_s":
            diverted_ticks
            * rme.DT_S,

        "cumulative_pre_impact":
            cumulative_pre_impact,
        "cumulative_post_impact":
            cumulative_post_impact,
        "diversion_attributable_impact_reduction":
            impact_reduction,
        "amr":
            amr,

        "candidate_count":
            candidate_count,
        "shield_admitted":
            admitted_count,
        "shield_rejected":
            rejected_count,
        "admission_rate":
            (
                admitted_count
                / candidate_count
                if candidate_count
                else None
            ),

        "registered_campaigns":
            deception[
                "registered_campaigns"
            ],
        "eligible_attempts":
            deception[
                "eligible_deception_attempts"
            ],
        "diverted_campaigns":
            deception[
                "diverted_campaigns"
            ],
        "sustained_engagements":
            deception[
                "sustained_engagements"
            ],
        "attack_diversion_probability":
            deception[
                "attack_diversion_probability"
            ],
        "dsr":
            deception[
                "deception_success_ratio"
            ],
        "return_rate_given_diversion":
            deception[
                "return_rate_given_diversion"
            ],
        "mean_diversion_duration_s":
            deception[
                "mean_diversion_duration_s"
            ],

        "mean_shield_latency_ms":
            (
                float(
                    np.mean(
                        decision_latency_ms
                    )
                )
                if decision_latency_ms
                else None
            ),

        "p95_shield_latency_ms":
            (
                float(
                    np.quantile(
                        decision_latency_ms,
                        0.95,
                    )
                )
                if decision_latency_ms
                else None
            ),

        "unsupported_phase_opportunities":
            unsupported_phase_opportunities,
        "unsupported_unique_campaigns":
            len(
                unsupported_campaign_ids
            ),
        "unsupported_phase_counts_json":
            json.dumps(
                dict(
                    unsupported_phase_counts
                ),
                sort_keys=True,
            ),
        "cache_misses":
            cache_misses,

        "variant_counts_json":
            json.dumps(
                dict(
                    variant_counts
                ),
                sort_keys=True,
            ),

        "theme_counts_json":
            json.dumps(
                dict(
                    theme_counts
                ),
                sort_keys=True,
            ),

        "profile_counts_json":
            json.dumps(
                dict(
                    profile_counts
                ),
                sort_keys=True,
            ),

        "candidate_type_counts_json":
            json.dumps(
                dict(
                    candidate_type_counts
                ),
                sort_keys=True,
            ),
    }


def build_summary(
    episodes: pl.DataFrame,
) -> pl.DataFrame:

    rows = []

    for method in METHODS:

        sub = episodes.filter(
            pl.col("method")
            == method
        )

        def mean_or_none(
            column: str,
        ):
            x = (
                sub[
                    column
                ]
                .drop_nulls()
            )

            if len(x) == 0:
                return None

            return float(
                x.mean()
            )

        rows.append(
            {
                "method":
                    method,
                "episodes":
                    sub.height,

                "completion_rate":
                    float(
                        sub[
                            "completed"
                        ].mean()
                    ),

                "mean_completion_time_s":
                    mean_or_none(
                        "completion_time_s"
                    ),

                "mean_final_battery_pct":
                    mean_or_none(
                        "final_battery_pct"
                    ),

                "mean_cumulative_pre_impact":
                    mean_or_none(
                        "cumulative_pre_impact"
                    ),

                "mean_cumulative_post_impact":
                    mean_or_none(
                        "cumulative_post_impact"
                    ),

                "mean_diversion_attributable_impact_reduction":
                    mean_or_none(
                        "diversion_attributable_impact_reduction"
                    ),

                "mean_amr":
                    mean_or_none(
                        "amr"
                    ),

                "total_candidates":
                    int(
                        sub[
                            "candidate_count"
                        ].sum()
                    ),

                "total_admitted":
                    int(
                        sub[
                            "shield_admitted"
                        ].sum()
                    ),

                "total_rejected":
                    int(
                        sub[
                            "shield_rejected"
                        ].sum()
                    ),

                "pooled_admission_rate":
                    (
                        float(
                            sub[
                                "shield_admitted"
                            ].sum()
                            / sub[
                                "candidate_count"
                            ].sum()
                        )
                        if sub[
                            "candidate_count"
                        ].sum()
                        > 0
                        else None
                    ),

                "total_eligible_attempts":
                    int(
                        sub[
                            "eligible_attempts"
                        ].sum()
                    ),

                "total_diverted_campaigns":
                    int(
                        sub[
                            "diverted_campaigns"
                        ].sum()
                    ),

                "total_sustained_engagements":
                    int(
                        sub[
                            "sustained_engagements"
                        ].sum()
                    ),

                "pooled_attack_diversion_probability":
                    (
                        float(
                            sub[
                                "diverted_campaigns"
                            ].sum()
                            / sub[
                                "eligible_attempts"
                            ].sum()
                        )
                        if sub[
                            "eligible_attempts"
                        ].sum()
                        > 0
                        else None
                    ),

                "pooled_dsr":
                    (
                        float(
                            sub[
                                "sustained_engagements"
                            ].sum()
                            / sub[
                                "eligible_attempts"
                            ].sum()
                        )
                        if sub[
                            "eligible_attempts"
                        ].sum()
                        > 0
                        else None
                    ),

                "mean_diverted_time_s":
                    mean_or_none(
                        "diverted_time_s"
                    ),

                "mean_p95_shield_latency_ms":
                    mean_or_none(
                        "p95_shield_latency_ms"
                    ),

                "unsupported_phase_opportunities":
                    int(
                        sub[
                            "unsupported_phase_opportunities"
                        ].sum()
                    ),

                "cache_misses":
                    int(
                        sub[
                            "cache_misses"
                        ].sum()
                    ),
            }
        )

    return pl.DataFrame(
        rows
    )


def build_paired_deltas(
    episodes: pl.DataFrame,
) -> pl.DataFrame:

    baseline = (
        episodes
        .filter(
            pl.col("method")
            == "attack_only"
        )
        .select(
            [
                "seed",
                pl.col(
                    "completion_time_s"
                ).alias(
                    "attack_completion_time_s"
                ),
                pl.col(
                    "final_battery_pct"
                ).alias(
                    "attack_final_battery_pct"
                ),
                pl.col(
                    "cumulative_post_impact"
                ).alias(
                    "attack_cumulative_post_impact"
                ),
            ]
        )
    )

    defended = (
        episodes
        .filter(
            pl.col("method")
            != "attack_only"
        )
    )

    paired = defended.join(
        baseline,
        on="seed",
        how="left",
    )

    paired = paired.with_columns(
        [
            (
                pl.col(
                    "attack_completion_time_s"
                )
                - pl.col(
                    "completion_time_s"
                )
            ).alias(
                "completion_time_saved_s"
            ),

            (
                pl.col(
                    "final_battery_pct"
                )
                - pl.col(
                    "attack_final_battery_pct"
                )
            ).alias(
                "battery_preserved_pct_points"
            ),

            (
                pl.col(
                    "attack_cumulative_post_impact"
                )
                - pl.col(
                    "cumulative_post_impact"
                )
            ).alias(
                "mission_impact_reduction_vs_attack_only"
            ),
        ]
    )

    return paired


def validate(
    episodes: pl.DataFrame,
):
    errors = []

    defended = episodes.filter(
        pl.col("method")
        != "attack_only"
    )

    for row in defended.iter_rows(
        named=True
    ):

        if (
            row["shield_admitted"]
            + row["shield_rejected"]
            != row["candidate_count"]
        ):
            errors.append(
                (
                    row["seed"],
                    row["method"],
                    "shield accounting",
                )
            )

        if (
            row["diverted_campaigns"]
            > row["eligible_attempts"]
        ):
            errors.append(
                (
                    row["seed"],
                    row["method"],
                    "diverted > eligible",
                )
            )

        if (
            row["sustained_engagements"]
            > row["diverted_campaigns"]
        ):
            errors.append(
                (
                    row["seed"],
                    row["method"],
                    "sustained > diverted",
                )
            )

        if (
            row["cumulative_post_impact"]
            >
            row["cumulative_pre_impact"]
            + 1e-12
        ):
            errors.append(
                (
                    row["seed"],
                    row["method"],
                    "post impact > pre impact",
                )
            )

    baseline = episodes.filter(
        pl.col("method")
        == "attack_only"
    )

    for row in baseline.iter_rows(
        named=True
    ):
        if (
            row["amr"] is not None
            and abs(
                row["amr"]
            ) > 1e-10
        ):
            errors.append(
                (
                    row["seed"],
                    "attack_only",
                    "AMR should be zero",
                )
            )

    if errors:
        raise RuntimeError(
            repr(
                errors
            )
        )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    library = (
        c3a.load_library()
    )

    print(
        "cached GenAI contexts:",
        len(library),
    )

    m3_estimator = (
        M3CostEstimator.from_dir(
            rme.M3_DIR
        )
    )

    rows = []

    for seed in SEEDS:

        for method in METHODS:

            print(
                f"seed={seed} "
                f"method={method}",
                flush=True,
            )

            row = run_episode(
                seed=seed,
                method=method,
                library=library,
                m3_estimator=m3_estimator,
            )

            rows.append(
                row
            )

    episodes = pl.DataFrame(
        rows
    )

    validate(
        episodes
    )

    summary = build_summary(
        episodes
    )

    paired = build_paired_deltas(
        episodes
    )

    episodes_path = (
        OUT
        / "genai_mission_episodes.csv"
    )

    summary_path = (
        OUT
        / "genai_mission_summary.csv"
    )

    paired_path = (
        OUT
        / "genai_mission_paired_deltas.csv"
    )

    config_path = (
        OUT
        / "genai_mission_config.json"
    )

    episodes.write_csv(
        episodes_path
    )

    summary.write_csv(
        summary_path
    )

    paired.write_csv(
        paired_path
    )

    config_path.write_text(
        json.dumps(
            {
                "seeds":
                    SEEDS,
                "methods":
                    METHODS,
                "intensity":
                    INTENSITY,
                "generator_runtime":
                    "offline_cached",
                "candidate_selection":
                    "identical to C3a",
                "mission_causal_bridge":
                    (
                        "MissionImpactModel attenuation is "
                        "active iff the current primary "
                        "campaign is_diverted()."
                    ),
                "diversion_effectiveness_scale":
                    1.0,
                "interpretation":
                    (
                        "AMR in this experiment is "
                        "diversion-attributable within "
                        "the simulator."
                    ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(summary)

    print()
    print("saved:", episodes_path)
    print("saved:", summary_path)
    print("saved:", paired_path)
    print("saved:", config_path)


if __name__ == "__main__":
    main()
