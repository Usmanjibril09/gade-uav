"""
Build a cached bounded GenAI candidate library for GADe-UAV.

This is an OFFLINE generation step.

The resulting JSONL is later consumed by runtime evaluation without
placing the language model in the 5 ms shield decision path.
"""

from __future__ import annotations

import json
from pathlib import Path

from gade_uav.generator.bounded_genai import (
    BoundedGenAIGenerator,
)


REPO = Path(__file__).resolve().parents[1]

OUT_DIR = (
    REPO
    / "artifacts"
    / "genai_candidate_library"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_JSONL = (
    OUT_DIR
    / "bounded_candidates.jsonl"
)

SUMMARY_JSON = (
    OUT_DIR
    / "summary.json"
)


FAMILIES = [
    "BruteForce",
    "DDoS",
    "Deauth",
    "DoS",
    "FakeLanding",
    "ICMP",
    "Jamming",
    "MITM",
    "Recon",
    "Replay",
    "UDP",
]

PHASES = [
    "takeoff",
    "cruise",
    "landing",
]


def main():
    generator = (
        BoundedGenAIGenerator()
    )

    rows = []

    for family in FAMILIES:
        for phase in PHASES:
            print(
                f"generating "
                f"family={family} "
                f"phase={phase}",
                flush=True,
            )

            result = (
                generator.generate(
                    family=family,
                    phase=phase,
                    decoy_id="h0",
                )
            )

            row = {
                "family":
                    family,
                "phase":
                    phase,
                "valid":
                    result.valid,
                "raw_text":
                    result.raw_text,
                "parsed_json":
                    result.parsed_json,
                "generation_latency_ms":
                    result.generation_latency_ms,
                "error":
                    result.error,
            }

            rows.append(
                row
            )

            print(
                "  valid=",
                result.valid,
                " latency_ms=",
                round(
                    result.generation_latency_ms,
                    2,
                ),
                " output=",
                result.raw_text,
            )

    with OUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )

    valid_rows = [
        r
        for r in rows
        if r["valid"]
    ]

    latencies = [
        float(
            r[
                "generation_latency_ms"
            ]
        )
        for r in rows
    ]

    summary = {
        "model":
            "Qwen/Qwen2.5-0.5B-Instruct",
        "contexts":
            len(rows),
        "valid":
            len(valid_rows),
        "invalid":
            len(rows)
            - len(valid_rows),
        "valid_fraction":
            (
                len(valid_rows)
                / len(rows)
                if rows
                else None
            ),
        "mean_generation_latency_ms":
            (
                sum(latencies)
                / len(latencies)
                if latencies
                else None
            ),
        "min_generation_latency_ms":
            (
                min(latencies)
                if latencies
                else None
            ),
        "max_generation_latency_ms":
            (
                max(latencies)
                if latencies
                else None
            ),
        "runtime_mode":
            "offline_cached",
        "reason":
            (
                "Observed CPU generation latency "
                "is orders of magnitude above "
                "the 5 ms runtime shield deadline."
            ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "saved:",
        OUT_JSONL,
    )

    print(
        "saved:",
        SUMMARY_JSON,
    )


if __name__ == "__main__":
    main()
