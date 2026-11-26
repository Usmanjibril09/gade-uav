"""Compare cached GenAI-v2 and hand-built libraries in the mission simulator."""

from __future__ import annotations

import json
from collections import Counter

import polars as pl

import run_genai_mission_compare as mission

OUT = mission.OUT.with_name("genai_matched_library")
GENERATED = "genai_v2_cached"
HAND_BUILT = "hand_built_cached"
PHASES = ("takeoff", "cruise", "landing")
MATCH_FIELDS = ("candidate_type", "profile", "theme", "confidence")

# The aggregate candidate mix is matched to the frozen generated library.
# The context assignments below are manual and use no generated row assignments.
CANDIDATE_PROFILES = {
    "low_beacon_telemetry": ("honey_beacon", "low", "telemetry_mimic", 0.95),
    "high_beacon_network": ("honey_beacon", "high", "network_presence", 0.95),
    "high_telemetry_network": ("decoy_telemetry", "high", "network_presence", 0.95),
    "medium_telemetry_cautious": ("decoy_telemetry", "medium", "telemetry_mimic", 0.88),
    "medium_telemetry": ("decoy_telemetry", "medium", "telemetry_mimic", 0.95),
    "medium_telemetry_service": ("decoy_telemetry", "medium", "service_lure", 0.95),
    "low_beacon_network": ("honey_beacon", "low", "network_presence", 0.95),
}

# Use lightweight beacons during critical phases for flood contexts, stronger
# network-presence profiles mainly during cruise, and telemetry-oriented
# profiles for integrity-related contexts. Columns follow PHASES.
HAND_ASSIGNMENTS = {
    "BruteForce": ("medium_telemetry", "medium_telemetry_service", "medium_telemetry"),
    "DDoS": ("low_beacon_telemetry", "high_telemetry_network", "low_beacon_telemetry"),
    "Deauthentication": ("low_beacon_network", "high_telemetry_network", "medium_telemetry_cautious"),
    "DoS": ("low_beacon_telemetry", "high_telemetry_network", "low_beacon_telemetry"),
    "FakeLanding": ("medium_telemetry", "medium_telemetry", "medium_telemetry"),
    "ICMP_Flooding": ("low_beacon_telemetry", "high_telemetry_network", "low_beacon_telemetry"),
    "Jamming": ("low_beacon_network", "high_beacon_network", "high_telemetry_network"),
    "MITM": ("medium_telemetry_cautious", "medium_telemetry", "medium_telemetry_cautious"),
    "Reconnaissance": ("low_beacon_telemetry", "medium_telemetry", "medium_telemetry"),
    "Replay": ("medium_telemetry_cautious", "medium_telemetry", "medium_telemetry_cautious"),
    "UDP_Flooding": ("medium_telemetry", "high_telemetry_network", "medium_telemetry_cautious"),
}


def build_hand_library(generated: dict) -> dict:
    library = {}
    for family, profiles in HAND_ASSIGNMENTS.items():
        for phase, profile in zip(PHASES, profiles, strict=True):
            candidate = dict(zip(MATCH_FIELDS, CANDIDATE_PROFILES[profile], strict=True))
            candidate["label"] = f"MANUAL_{family}_{phase}".upper()
            valid, reason = mission.c3a.BoundedGenAIGenerator._validate(candidate)
            if not valid:
                raise ValueError(f"Invalid manual candidate for {family}/{phase}: {reason}")
            library[family, phase] = candidate

    if library.keys() != generated.keys():
        raise ValueError("Generated and hand-built libraries have different contexts")
    manual_mix = Counter(tuple(row[field] for field in MATCH_FIELDS) for row in library.values())
    generated_mix = Counter(tuple(row[field] for field in MATCH_FIELDS) for row in generated.values())
    if manual_mix != generated_mix:
        raise ValueError(f"Library mix differs: {manual_mix - generated_mix=}, {generated_mix - manual_mix=}")
    return library


def main() -> None:
    generated = mission.c3a.load_library()
    hand_built = build_hand_library(generated)
    estimator = mission.M3CostEstimator.from_dir(mission.rme.M3_DIR)
    episodes = []
    for seed in mission.SEEDS:
        for method, library in ((GENERATED, generated), (HAND_BUILT, hand_built)):
            print(f"seed={seed} method={method}", flush=True)
            # Both libraries use the exact same contextual selection and shield path.
            row = mission.run_episode(
                seed=seed, method=GENERATED, library=library, m3_estimator=estimator
            )
            row["method"] = method
            episodes.append(row)

    frame = pl.DataFrame(episodes)
    mission.validate(frame)
    metrics = (
        "candidate_count", "eligible_attempts", "diverted_campaigns",
        "sustained_engagements", "admission_rate", "attack_diversion_probability",
        "dsr", "amr", "completion_time_s", "final_battery_pct",
    )
    generated_rows = frame.filter(pl.col("method") == GENERATED).select("seed", *metrics)
    hand_built_rows = frame.filter(pl.col("method") == HAND_BUILT).select("seed", *metrics)
    paired = generated_rows.join(hand_built_rows, on="seed", suffix="_hand_built", validate="1:1")
    paired = paired.with_columns(
        (pl.col(name) - pl.col(f"{name}_hand_built")).alias(f"{name}_genai_minus_hand_built")
        for name in metrics
    ).sort("seed")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.write_csv(OUT / "episodes.csv")
    paired.write_csv(OUT / "paired_deltas.csv")
    (OUT / "hand_built_library.json").write_text(
        json.dumps(
            [{"family": family, "phase": phase, **candidate}
             for (family, phase), candidate in hand_built.items()],
            indent=2,
        ) + "\n"
    )
    print(f"Saved matched results to {OUT}")


if __name__ == "__main__":
    main()
