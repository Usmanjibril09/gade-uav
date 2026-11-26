"""
GADe-UAV C3a:
paired causal comparison of a context-independent fixed-template selector
against cached context-conditioned bounded GenAI-v2.

NO MODEL TRAINING.
NO ONLINE LLM GENERATION.

Both methods:
- see identical adversary seeds;
- select at most ONE candidate at a campaign's first supported opportunity;
- pass that candidate through the same runtime shield;
- use the same frozen family-theme compatibility mechanism;
- use paired response RNG seeds.

This evaluator measures causal deception behavior first.
Mission-impact integration is deliberately deferred to C3b until this
comparison passes its invariants.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl

import run_mission_eval as rme

from gade_uav.generator.bounded_genai import (
    BoundedGenAIGenerator,
    GENAI_TEMPLATE_ID,
)
from gade_uav.shield.cost_estimator import M3CostEstimator
from gade_uav.shield.decide import Verdict
from gade_uav.sim.adversary import EventKind
from gade_uav.sim.deception_response import DeceptionResponseModel


REPO = Path(__file__).resolve().parents[1]

OUT = (
    REPO
    / "artifacts"
    / "genai_causal_compare"
)

LIBRARY = (
    REPO
    / "artifacts"
    / "genai_candidate_library"
    / "bounded_candidates.jsonl"
)

SEEDS = [
    42,
    123,
    2024,
    3407,
    7777,
]

METHODS = (
    "fixed_uniform",
    "genai_v2_cached",
)

# C3a intentionally uses M3 shield in both conditions so the safety/cost
# estimator is held constant while candidate-selection policy changes.
WORLD_SCENARIO = "m3_shield"

SUPPORTED_PHASES = {
    "takeoff",
    "cruise",
    "landing",
}

FIXED_VARIANTS = (
    "safe_beacon",
    "safe_telemetry",
    "high_cost_beacon",
)

FIXED_THEMES = {
    "safe_beacon": "identity_decoy",
    "safe_telemetry": "telemetry_mimic",
    "high_cost_beacon": "identity_decoy",
}

# The frozen GenAI-v2 library was generated before we noticed that
# four prompt labels used convenient short names rather than the
# simulator's canonical attack-family taxonomy.
#
# This is ONLY a naming normalization. The cached model outputs,
# themes, profiles, confidence values, and compatibility coefficients
# remain unchanged.
LIBRARY_FAMILY_TO_CANONICAL = {
    "Deauth": "Deauthentication",
    "ICMP": "ICMP_Flooding",
    "Recon": "Reconnaissance",
    "UDP": "UDP_Flooding",
}


def load_library() -> dict[tuple[str, str], dict]:
    if not LIBRARY.exists():
        raise FileNotFoundError(
            f"Missing cached GenAI library: {LIBRARY}"
        )

    out: dict[tuple[str, str], dict] = {}

    for line in LIBRARY.read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if not row.get("valid", False):
            continue

        obj = row.get("parsed_json")

        if not isinstance(obj, dict):
            continue

        library_family = str(
            row["family"]
        )

        canonical_family = (
            LIBRARY_FAMILY_TO_CANONICAL.get(
                library_family,
                library_family,
            )
        )

        key = (
            canonical_family,
            str(row["phase"]),
        )

        if key in out:
            raise RuntimeError(
                f"Duplicate cached context {key}"
            )

        out[key] = obj

    return out


def fixed_action(
    variant: str,
):
    if variant == "safe_beacon":
        return rme._safe_beacon("h0")

    if variant == "safe_telemetry":
        return rme._safe_decoy_telemetry("h0")

    if variant == "high_cost_beacon":
        return rme._high_cost_beacon("h0")

    raise ValueError(variant)


def genai_action_and_variant(
    *,
    obj: dict,
    family: str,
    phase: str,
):
    action = BoundedGenAIGenerator._to_action(
        obj=obj,
        family=family,
        phase=phase,
        decoy_id="h0",
    )

    ctype = str(
        obj["candidate_type"]
    )

    profile = str(
        obj["profile"]
    )

    if ctype == "honey_beacon":
        variant = (
            "high_cost_beacon"
            if profile == "high"
            else "safe_beacon"
        )

    elif ctype == "decoy_telemetry":
        variant = "safe_telemetry"

    else:
        raise ValueError(
            f"Unsupported generated candidate_type {ctype!r}"
        )

    return action, variant


def genai_context(
    mission_phase,
    known_decoys,
):
    ctx = rme._ctx(
        mission_phase,
        known_decoys,
    )

    # Keep the existing whitelist and add only the bounded GenAI template.
    current = frozenset(
        ctx.bounded_template_whitelist
    )

    return replace(
        ctx,
        bounded_template_whitelist=(
            current
            | frozenset(
                {GENAI_TEMPLATE_ID}
            )
        ),
    )


def run_one(
    *,
    seed: int,
    method: str,
    library: dict,
    m3_estimator: M3CostEstimator,
):
    (
        loop,
        drone,
        mission,
        known_decoys,
    ) = rme.make_world(
        seed,
        WORLD_SCENARIO,
        m3_estimator,
    )

    response = DeceptionResponseModel(
        seed=seed + 120_000,
    )

    # Selection RNG is paired across conditions.
    select_rng = np.random.default_rng(
        seed + 130_000
    )

    active_campaigns: dict[
        str,
        tuple[int, str],
    ] = {}

    attempted_campaigns: set[str] = set()

    candidate_count = 0
    shield_admitted = 0
    shield_rejected = 0
    unsupported_phase_opportunities = 0
    cache_misses = 0

    chosen_variant_counts = Counter()
    chosen_theme_counts = Counter()
    chosen_profile_counts = Counter()
    chosen_type_counts = Counter()
    phase_counts = Counter()
    family_counts = Counter()

    decision_latency_ms: list[float] = []

    for tick in range(rme.MAX_TICKS):
        loop.step()

        now_s = float(
            loop.clock.now_s()
        )

        response.advance(
            now_s
        )

        for event in loop.last_events:
            cid = str(
                event.campaign_id
            )

            if event.kind == EventKind.CAMPAIGN_START:
                family = str(
                    event.canonical_family
                )

                active_campaigns[cid] = (
                    tick,
                    family,
                )

                response.register_campaign(
                    campaign_id=cid,
                    family=family,
                    now_s=now_s,
                )

            elif event.kind == EventKind.CAMPAIGN_END:
                active_campaigns.pop(
                    cid,
                    None,
                )

                response.end_campaign(
                    campaign_id=cid,
                    now_s=now_s,
                )

        if mission.phase.value == "post_landing":
            break

        if not active_campaigns:
            continue

        # Preserve current GADe-UAV single-primary-threat semantics.
        primary_cid, (
            _start_tick,
            primary_family,
        ) = min(
            active_campaigns.items(),
            key=lambda kv: kv[1][0],
        )

        if primary_cid in attempted_campaigns:
            continue

        phase = str(
            mission.phase.value
        )

        # The frozen v2 library contains these three mission phases only.
        # We do NOT silently remap an unsupported phase.
        if phase not in SUPPORTED_PHASES:
            unsupported_phase_opportunities += 1
            continue

        attempted_campaigns.add(
            primary_cid
        )

        if method == "fixed_uniform":
            variant = str(
                select_rng.choice(
                    FIXED_VARIANTS
                )
            )

            theme = FIXED_THEMES[
                variant
            ]

            action = fixed_action(
                variant
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
            key = (
                primary_family,
                phase,
            )

            obj = library.get(
                key
            )

            if obj is None:
                cache_misses += 1
                continue

            action, variant = (
                genai_action_and_variant(
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

            ctx = genai_context(
                mission.phase,
                known_decoys,
            )

        else:
            raise ValueError(
                method
            )

        candidate_count += 1

        chosen_variant_counts[
            variant
        ] += 1

        chosen_theme_counts[
            theme
        ] += 1

        chosen_profile_counts[
            profile
        ] += 1

        chosen_type_counts[
            candidate_type
        ] += 1

        phase_counts[
            phase
        ] += 1

        family_counts[
            primary_family
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
            shield_admitted += 1

            response.attempt(
                campaign_id=primary_cid,
                variant=variant,
                theme=theme,
                now_s=now_s,
            )

        else:
            shield_rejected += 1

    # Close any campaign still active when mission evaluation terminates.
    end_s = float(
        loop.clock.now_s()
    )

    for cid in list(
        active_campaigns
    ):
        response.end_campaign(
            campaign_id=cid,
            now_s=end_s,
        )

    campaign_rows = (
        response.snapshot()
    )

    for row in campaign_rows:
        row["seed"] = seed
        row["method"] = method

    eligible = sum(
        bool(
            row["attempted"]
        )
        for row in campaign_rows
    )

    diverted = sum(
        bool(
            row["diversion_success"]
        )
        for row in campaign_rows
    )

    sustained = sum(
        bool(
            row["sustained_success"]
        )
        for row in campaign_rows
    )

    returned = sum(
        bool(
            row["returned_to_target"]
        )
        for row in campaign_rows
    )

    durations = [
        float(
            row["total_diverted_s"]
        )
        for row in campaign_rows
        if float(
            row["total_diverted_s"]
        ) > 0.0
    ]

    compat = [
        float(
            row["theme_compatibility"]
        )
        for row in campaign_rows
        if row.get(
            "theme_compatibility"
        ) is not None
    ]

    effective_p = [
        float(
            row["diversion_probability"]
        )
        for row in campaign_rows
        if row.get(
            "diversion_probability"
        ) is not None
    ]

    episode_row = {
        "seed":
            seed,
        "method":
            method,
        "registered_campaigns":
            len(campaign_rows),
        "candidate_count":
            candidate_count,
        "shield_admitted":
            shield_admitted,
        "shield_rejected":
            shield_rejected,
        "admission_rate":
            (
                shield_admitted
                / candidate_count
                if candidate_count
                else None
            ),
        "eligible_attempts":
            eligible,
        "diverted_campaigns":
            diverted,
        "sustained_engagements":
            sustained,
        "returned_to_target":
            returned,
        "attack_diversion_probability":
            (
                diverted / eligible
                if eligible
                else None
            ),
        "dsr":
            (
                sustained / eligible
                if eligible
                else None
            ),
        "return_rate_given_diversion":
            (
                returned / diverted
                if diverted
                else None
            ),
        "mean_diversion_duration_s":
            (
                float(
                    np.mean(
                        durations
                    )
                )
                if durations
                else 0.0
            ),
        "mean_theme_compatibility":
            (
                float(
                    np.mean(
                        compat
                    )
                )
                if compat
                else None
            ),
        "mean_effective_diversion_probability":
            (
                float(
                    np.mean(
                        effective_p
                    )
                )
                if effective_p
                else None
            ),
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
        "cache_misses":
            cache_misses,
        "variant_counts_json":
            json.dumps(
                dict(
                    chosen_variant_counts
                ),
                sort_keys=True,
            ),
        "theme_counts_json":
            json.dumps(
                dict(
                    chosen_theme_counts
                ),
                sort_keys=True,
            ),
        "profile_counts_json":
            json.dumps(
                dict(
                    chosen_profile_counts
                ),
                sort_keys=True,
            ),
        "candidate_type_counts_json":
            json.dumps(
                dict(
                    chosen_type_counts
                ),
                sort_keys=True,
            ),
        "phase_counts_json":
            json.dumps(
                dict(
                    phase_counts
                ),
                sort_keys=True,
            ),
        "family_counts_json":
            json.dumps(
                dict(
                    family_counts
                ),
                sort_keys=True,
            ),
    }

    return (
        episode_row,
        campaign_rows,
    )


def aggregate(
    episode_df: pl.DataFrame,
) -> pl.DataFrame:
    rows = []

    for method in METHODS:
        sub = episode_df.filter(
            pl.col("method")
            == method
        )

        if sub.height == 0:
            continue

        eligible = int(
            sub[
                "eligible_attempts"
            ].sum()
        )

        diverted = int(
            sub[
                "diverted_campaigns"
            ].sum()
        )

        sustained = int(
            sub[
                "sustained_engagements"
            ].sum()
        )

        returned = int(
            sub[
                "returned_to_target"
            ].sum()
        )

        candidates = int(
            sub[
                "candidate_count"
            ].sum()
        )

        admitted = int(
            sub[
                "shield_admitted"
            ].sum()
        )

        rows.append(
            {
                "method":
                    method,
                "episodes":
                    sub.height,
                "candidate_count":
                    candidates,
                "shield_admitted":
                    admitted,
                "shield_rejected":
                    int(
                        sub[
                            "shield_rejected"
                        ].sum()
                    ),
                "admission_rate":
                    (
                        admitted
                        / candidates
                        if candidates
                        else None
                    ),
                "eligible_attempts":
                    eligible,
                "diverted_campaigns":
                    diverted,
                "sustained_engagements":
                    sustained,
                "attack_diversion_probability":
                    (
                        diverted
                        / eligible
                        if eligible
                        else None
                    ),
                "dsr":
                    (
                        sustained
                        / eligible
                        if eligible
                        else None
                    ),
                "return_rate_given_diversion":
                    (
                        returned
                        / diverted
                        if diverted
                        else None
                    ),
                "mean_episode_diversion_probability":
                    float(
                        sub[
                            "attack_diversion_probability"
                        ]
                        .drop_nulls()
                        .mean()
                    ),
                "mean_episode_dsr":
                    float(
                        sub[
                            "dsr"
                        ]
                        .drop_nulls()
                        .mean()
                    ),
                "mean_theme_compatibility":
                    float(
                        sub[
                            "mean_theme_compatibility"
                        ]
                        .drop_nulls()
                        .mean()
                    ),
                "mean_effective_diversion_probability":
                    float(
                        sub[
                            "mean_effective_diversion_probability"
                        ]
                        .drop_nulls()
                        .mean()
                    ),
                "mean_p95_shield_latency_ms":
                    float(
                        sub[
                            "p95_shield_latency_ms"
                        ]
                        .drop_nulls()
                        .mean()
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


def validate_invariants(
    episode_df: pl.DataFrame,
    campaign_df: pl.DataFrame,
):
    errors = []

    for row in episode_df.iter_rows(
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
                    "admission accounting",
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

    if campaign_df.height:
        bad_p = campaign_df.filter(
            pl.col(
                "diversion_probability"
            ).is_not_null()
            & (
                (
                    pl.col(
                        "diversion_probability"
                    )
                    < 0.0
                )
                | (
                    pl.col(
                        "diversion_probability"
                    )
                    > 1.0
                )
            )
        )

        if bad_p.height:
            errors.append(
                (
                    "campaign",
                    "all",
                    "probability outside [0,1]",
                )
            )

    if errors:
        raise RuntimeError(
            "Invariant failure(s): "
            + repr(
                errors
            )
        )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    library = load_library()

    print(
        "cached valid contexts:",
        len(library),
    )

    expected = {
        (family, phase)
        for family in {
            key[0]
            for key in library
        }
        for phase in SUPPORTED_PHASES
    }

    if len(library) != 33:
        print(
            "WARNING: expected 33 cached v2 contexts; "
            f"found {len(library)}"
        )

    m3_estimator = (
        M3CostEstimator.from_dir(
            rme.M3_DIR
        )
    )

    episode_rows = []
    campaign_rows = []

    for seed in SEEDS:
        for method in METHODS:
            print(
                f"seed={seed} "
                f"method={method}",
                flush=True,
            )

            erow, crows = run_one(
                seed=seed,
                method=method,
                library=library,
                m3_estimator=m3_estimator,
            )

            episode_rows.append(
                erow
            )

            campaign_rows.extend(
                crows
            )

    episode_df = pl.DataFrame(
        episode_rows
    )

    campaign_df = pl.DataFrame(
        campaign_rows
    )

    validate_invariants(
        episode_df,
        campaign_df,
    )

    summary_df = aggregate(
        episode_df
    )

    episode_path = (
        OUT
        / "genai_causal_episodes.csv"
    )

    campaign_path = (
        OUT
        / "genai_causal_campaigns.csv"
    )

    summary_path = (
        OUT
        / "genai_causal_summary.csv"
    )

    config_path = (
        OUT
        / "genai_causal_config.json"
    )

    episode_df.write_csv(
        episode_path
    )

    campaign_df.write_csv(
        campaign_path
    )

    summary_df.write_csv(
        summary_path
    )

    config_path.write_text(
        json.dumps(
            {
                "seeds":
                    SEEDS,
                "methods":
                    METHODS,
                "world_scenario":
                    WORLD_SCENARIO,
                "library":
                    str(
                        LIBRARY
                    ),
                "supported_phases":
                    sorted(
                        SUPPORTED_PHASES
                    ),
                "fixed_variants":
                    list(
                        FIXED_VARIANTS
                    ),
                "fixed_themes":
                    FIXED_THEMES,
                "library_family_aliases":
                    LIBRARY_FAMILY_TO_CANONICAL,
                "candidate_rule":
                    (
                        "one candidate selected at first "
                        "supported campaign opportunity"
                    ),
                "genai_runtime":
                    "cached_offline",
                "compatibility":
                    (
                        "frozen family-theme simulation "
                        "coefficients"
                    ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(summary_df)

    print()
    print(
        "saved:",
        episode_path,
    )
    print(
        "saved:",
        campaign_path,
    )
    print(
        "saved:",
        summary_path,
    )
    print(
        "saved:",
        config_path,
    )


if __name__ == "__main__":
    main()
