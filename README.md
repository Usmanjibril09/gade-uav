# GADe-UAV

This repository accompanies *GADe-UAV: Mission-Aware Generative Cyber Deception with Runtime Safety Assurance for Autonomous UAV Systems*. It contains the simulator, runtime safety shield, frozen models, candidate library, configurations, and selected evaluation outputs. The evaluators run in the simulator and do not connect to aircraft or networks.

## Environment

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). Run commands from the repository root. The locked dependencies include PyTorch 2.2.2 and CPU LightGBM. On systems where LightGBM requires a separate OpenMP runtime, install the runtime through your operating system package manager.

```sh
uv sync --locked --python 3.11
uv run --no-sync pytest -q -m 'not integration' tests/shield tests/sim
```

## Reproduce the simulator evaluations

```sh
uv run --no-sync python scripts/run_genai_causal_compare.py
uv run --no-sync python scripts/run_genai_mission_compare.py
uv run --no-sync python scripts/run_genai_matched_library.py
uv run --no-sync python scripts/run_mission_safety_compare.py --out-dir results/mission_safety
uv run --no-sync python scripts/run_mission_eval.py --seeds 42 123 2024 3407 7777 --out-dir results/nominal
uv run --no-sync python scripts/run_mission_eval_ood.py --seeds 42 123 2024 3407 7777 --out-dir results/resource_shift
```

The cached candidate experiments require no model download. To regenerate the candidate cache, run `scripts/build_genai_candidate_library.py` with the locked environment and access to the Qwen checkpoint. Candidate generation replaces its output cache. Use a clean copy if you want to compare it with the archived outputs.

The three GenAI comparison scripts write under `artifacts/`. Compare new runs on a separate checkout if you want to preserve the tracked reference CSVs. Runtime timings depend on the host. Mission outcomes are simulator measurements. The fixed-window safety replay reports modeled costs and mitigation for common proposals, including a simulator-only bypass arm.

## Rebuild the six results figures

```sh
for name in fig02_belief_encoder fig03_m3_accuracy fig07_coverage fig08_selection_conditional fig10_ood_robustness fig11_mission_safety; do
  uv run --no-sync python -m "scripts.plotting.$name"
done
```

The expected PDFs and PNGs are written to `artifacts/figures/`. Figure inputs are the frozen encoder reports, the original M3 test-split prediction arrays, the conformal coverage summaries, the reported nominal and shift values, and the matched safety replay CSV. No raw datasets or preprocessing are required to build these figures.

## Optional encoder replay and timing

`artifacts/mission_safety_compare/reference/` contains the reported encoder replay and timing results. To rerun those modes, obtain the corresponding processed GUIDE window table from the authors and place it at `data/interim/features/belief_sequences.parquet`. The saved checkpoint expects the original row order. The processed table and the upstream raw captures are excluded from this repository.

```sh
OMP_NUM_THREADS=1 uv run --no-sync python scripts/run_mission_safety_compare.py --encoder-replay --out-dir results/encoder
OMP_NUM_THREADS=1 uv run --no-sync python scripts/run_mission_safety_compare.py --component-timing --out-dir results/timing
```

Training and upstream dataset preparation are outside the scope of this simulator artifact. The released evaluation uses frozen model checkpoints and calibration files.
