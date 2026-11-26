"""Matched simulator-only resource-estimator and shield ablation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import run_genai_mission_compare as library_eval
import run_mission_eval as mission_eval
from run_warmup_sim import _malformed_beacon

from gade_uav.shield.actions import ActionType, CostVector, Effector, MissionPhase
from gade_uav.shield.decide import Verdict
from gade_uav.shield.ltl_monitor import SafetyMonitor
from gade_uav.shield.propositions import evaluate_all

ARMS = ("generator_shield", "m3_shield", "simulator_bypass")
OUT = mission_eval.REPO / "artifacts" / "mission_safety_compare"
HORIZON_TICKS = 4000


def candidate_stream(rng):
    for action, scale in (
        (mission_eval._safe_beacon("h0"), rng.uniform(2.0, 4.5)),
        (mission_eval._safe_decoy_telemetry("h0"), rng.uniform(1.3, 2.6)),
    ):
        cost = CostVector(
            **{
                key: value * scale
                for key, value in action.predicted_cost.as_dict().items()
            }
        )
        yield replace(action, predicted_cost=cost)
    # One fixed malformed sentinel per attacked tick exercises the safety audit.
    # The bypass never invokes a dispatch or transmission interface.
    yield _malformed_beacon("h0")


def run_seed(seed, estimator):
    reference, _, reference_mission, decoys = mission_eval.make_world(
        seed, "attack_only", None
    )
    # The reference is the only simulated world. Arms score counterfactual
    # decisions against its shared contexts and never advance a separate world.
    shields = {
        "generator_shield": mission_eval._build_shield(
            calibration_npz=str(mission_eval.GENERATOR_CALIBRATION)
        ),
        "m3_shield": mission_eval._build_shield(
            cost_estimator=estimator, calibration_npz=str(mission_eval.M3_CALIBRATION)
        ),
    }
    if any(
        shield._ltl_repair is not None or shield._cost_downscale is not None
        for shield in shields.values()
    ):
        raise ValueError("Matched proposal audit requires repairs to be disabled")
    # This audit uses the same budget schedule as the shield, including phase slack.
    budgets = shields["generator_shield"]._gate._budgets
    cost_model = mission_eval.Effector(mission_eval.EFFECTOR_CFG, seed=seed)
    impact_model = mission_eval.MissionImpactModel(seed=seed + 50_000)
    monitor = SafetyMonitor()
    rng = np.random.default_rng(seed + 90_000)
    campaigns = {}
    counts = {arm: Counter() for arm in ARMS}
    disagreements = Counter()
    pre_impact = 0.0
    attacked_ticks = 0
    zero_feasible_ticks = 0

    for _ in range(HORIZON_TICKS):
        reference.step()
        for event in reference.last_events:
            if event.kind == mission_eval.EventKind.CAMPAIGN_START:
                campaigns[event.campaign_id] = event.canonical_family
            elif event.kind == mission_eval.EventKind.CAMPAIGN_END:
                campaigns.pop(event.campaign_id, None)

        if not campaigns:
            continue

        attacked_ticks += 1
        family = next(iter(campaigns.values()))
        phase = reference_mission.phase
        context = mission_eval._ctx(phase, decoys)
        now = reference.clock.now_s()
        before = impact_model.pre_defense_impact(family)
        pre_impact += mission_eval.scalar_impact(before) * mission_eval.DT_S
        actions = list(candidate_stream(rng))
        useful_admissions = Counter()

        for action in actions:
            props = evaluate_all(action, context)
            audit = monitor.evaluate(props, action, context)
            hard_violation = props["target_forbidden"] or not audit.holds
            # One shared counterfactual per proposal, including all rejected
            # proposals. This pure simulator operation returns a cost vector.
            observed = cost_model.execute(
                action, mission_phase=phase.value, primary_attack_family=family
            )
            over_budget = any(
                cost > budgets.budget_for(dimension, phase)
                for dimension, cost in observed.as_dict().items()
            )
            admitted = {"simulator_bypass": True}
            for arm, shield in shields.items():
                decision = shield.decide(
                    action, context, now_s=now, primary_attack_family=family
                )
                if decision.verdict == Verdict.MODIFY:
                    raise RuntimeError(
                        "Matched proposal audit requires repair to be disabled"
                    )
                admitted[arm] = decision.verdict == Verdict.DEPLOY
                if admitted[arm]:
                    # Each gate updates from its own prediction residual and
                    # admitted history. Shared draws preserve the pairing.
                    shield.observe_cost_outcome(decision, observed)

            for arm, accepted in admitted.items():
                counts[arm]["candidates"] += 1
                if accepted:
                    counts[arm]["admitted"] += 1
                    counts[arm]["deterministic_violations"] += int(hard_violation)
                    counts[arm]["budget_overruns"] += int(over_budget)
                    useful_admissions[arm] += int(
                        not hard_violation and not over_budget
                    )

            if admitted["m3_shield"] != admitted["generator_shield"]:
                arm = "m3_only" if admitted["m3_shield"] else "generator_only"
                disagreements[arm] += 1
                disagreements[f"{arm}_feasible"] += int(
                    not hard_violation and not over_budget
                )

        zero_feasible_ticks += int(useful_admissions["simulator_bypass"] == 0)
        for arm in ARMS:
            scale = useful_admissions[arm] / len(actions)
            after = impact_model.attenuate(
                family, before, defense_active=scale > 0, effectiveness_scale=scale
            )
            counts[arm]["residual_impact"] += (
                mission_eval.scalar_impact(after.post_defense) * mission_eval.DT_S
            )

    return [
        {
            "seed": seed,
            "arm": arm,
            "attacked_ticks": attacked_ticks,
            "zero_feasible_ticks": zero_feasible_ticks,
            **{
                key: counts[arm][key]
                for key in (
                    "candidates",
                    "admitted",
                    "deterministic_violations",
                    "budget_overruns",
                    "residual_impact",
                )
            },
            "modeled_mitigation": 1 - counts[arm]["residual_impact"] / pre_impact
            if pre_impact
            else None,
            **{
                key: disagreements[key]
                for key in (
                    "m3_only",
                    "m3_only_feasible",
                    "generator_only",
                    "generator_only_feasible",
                )
            },
        }
        for arm in ARMS
    ]


def run_encoder_replay(out):
    import torch

    from gade_uav.models.belief_dataset import LABELS, AuxStats, BeliefSequenceDataset
    from gade_uav.models.belief_encoder import BeliefEncoder, BeliefEncoderConfig

    repo = mission_eval.REPO
    checkpoint = torch.load(
        repo / "artifacts/belief_encoder/belief_encoder.pt", map_location="cpu"
    )
    indices = np.array(checkpoint["test_idx"])
    dataset = BeliefSequenceDataset(
        repo / "data/interim/features/belief_sequences.parquet",
        repo / checkpoint["vocab_path"],
        indices,
        AuxStats.from_dict(checkpoint["aux_stats"]),
    )
    sequence, auxiliary, truth, _ = zip(
        *(dataset[i] for i in range(len(dataset))), strict=True
    )
    sequences = torch.stack(sequence)
    features = torch.stack(auxiliary)
    encoder = BeliefEncoder(BeliefEncoderConfig(**checkpoint["model_config"])).eval()
    encoder.load_state_dict(checkpoint["model_state"])

    library_keys = {family for family, _ in library_eval.c3a.load_library()}
    _, _, _, decoys = mission_eval.make_world(42, "no_attack", None)
    context = mission_eval._ctx(MissionPhase.CRUISE, decoys)
    # Both comparison actions remain inert objects. No target system is contacted.
    beacon = mission_eval._safe_beacon("h0")
    heartbeat = replace(
        beacon,
        type=ActionType.MAVLINK_RESPONSE,
        payload={"msg_id": 0, "fields": {}},
        target_iface=Effector.DECOY_TELEMETRY_CHANNEL,
        gen_meta=replace(beacon.gen_meta, template_id=None),
    )
    incomplete = sequences.clone()
    incomplete[:, ::4] = 0
    rows = []
    for condition, messages in (
        ("held_out", sequences),
        ("missing_message_ids", incomplete),
    ):
        direct_shield, routed_shield = (
            mission_eval._build_shield(
                calibration_npz=str(mission_eval.GENERATOR_CALIBRATION)
            )
            for _ in range(2)
        )
        with torch.inference_mode():
            posterior = torch.cat(
                [
                    encoder(batch, aux)[0].softmax(dim=-1)
                    for batch, aux in zip(
                        messages.split(32), features.split(32), strict=True
                    )
                ]
            )
        confidence, predictions = posterior.max(dim=1)
        counts = Counter(
            dict.fromkeys(
                (
                    "windows",
                    "wrong_class",
                    "low_confidence",
                    "wrong_high_confidence",
                    "false_attack",
                    "missed_attack",
                    "attack_predictions",
                    "exact_cache_miss",
                    "changed_selection",
                    "direct_admitted",
                    "routed_admitted",
                ),
                0,
            )
        )
        for i, (actual, predicted, score) in enumerate(
            zip(truth, predictions.tolist(), confidence.tolist(), strict=True)
        ):
            counts["windows"] += 1
            counts["wrong_class"] += int(actual != predicted)
            uncertain = score < 1 - context.uncertainty_threshold
            counts["low_confidence"] += int(uncertain)
            counts["wrong_high_confidence"] += int(
                actual != predicted and not uncertain
            )
            counts["false_attack"] += int(actual == 0 and predicted != 0)
            counts["missed_attack"] += int(actual != 0 and predicted == 0)
            if predicted == 0:
                continue
            counts["attack_predictions"] += 1
            counts["exact_cache_miss"] += int(LABELS[predicted] not in library_keys)
            estimated_context = replace(context, cyber_state_confidence=score)
            # An explicit replay policy, independent of the campaign-family cache.
            # Low confidence selects a bounded beacon instead of a heartbeat.
            now = i * mission_eval.DT_S
            direct = direct_shield.decide(heartbeat, estimated_context, now_s=now)
            selected = beacon if uncertain else heartbeat
            routed = routed_shield.decide(selected, estimated_context, now_s=now)
            counts["changed_selection"] += int(uncertain)
            counts["direct_admitted"] += int(direct.verdict == Verdict.DEPLOY)
            counts["routed_admitted"] += int(routed.verdict == Verdict.DEPLOY)
        rows.append({"condition": condition, **counts})

    out.mkdir(parents=True, exist_ok=True)
    path = out / "encoder_replay.csv"
    if path.exists():
        raise FileExistsError(
            f"Choose a fresh --out-dir. Results already exist at {path}"
        )
    frame = pl.DataFrame(rows)
    frame.write_csv(path)
    print(frame)
    print(f"Saved replay results to {path}")


def run_component_timing(out):
    """Time a fixed-candidate CPU replay with a 5 ms shield deadline."""
    from contextlib import ExitStack
    from functools import wraps
    from time import perf_counter_ns
    from unittest.mock import patch

    import torch

    from gade_uav.models.belief_dataset import AuxStats, BeliefSequenceDataset
    from gade_uav.models.belief_encoder import BeliefEncoder, BeliefEncoderConfig
    from gade_uav.shield import decide as decision_module

    repo = mission_eval.REPO
    path = out / "component_timing.csv"
    if path.exists():
        raise FileExistsError(f"Timing results already exist at {path}")
    checkpoint = torch.load(
        repo / "artifacts/belief_encoder/belief_encoder.pt", map_location="cpu"
    )
    indices = np.random.default_rng(42).choice(
        checkpoint["test_idx"], size=200, replace=False
    )
    dataset = BeliefSequenceDataset(
        repo / "data/interim/features/belief_sequences.parquet",
        repo / checkpoint["vocab_path"],
        indices,
        AuxStats.from_dict(checkpoint["aux_stats"]),
    )
    encoder = BeliefEncoder(BeliefEncoderConfig(**checkpoint["model_config"])).eval()
    encoder.load_state_dict(checkpoint["model_state"])
    estimator = mission_eval.M3CostEstimator.from_dir(mission_eval.M3_DIR)
    _, _, _, decoys = mission_eval.make_world(42, "no_attack", None)
    context = mission_eval._ctx(MissionPhase.CRUISE, decoys)
    candidate = mission_eval._safe_beacon("h0")
    rows = []

    def measure_burst(burst):
        shield = mission_eval._build_shield(
            cost_estimator=estimator,
            calibration_npz=str(mission_eval.M3_CALIBRATION),
            decision_deadline_ms=5.0,
        )
        cost_model = mission_eval.Effector(mission_eval.EFFECTOR_CFG, seed=42)
        elapsed = Counter()

        def instrument(stack, owner, name, metric):
            original = getattr(owner, name)

            @wraps(original)
            def measured(*args, **kwargs):
                started = perf_counter_ns()
                try:
                    return original(*args, **kwargs)
                finally:
                    elapsed[metric] += perf_counter_ns() - started

            stack.enter_context(patch.object(owner, name, new=measured))

        def request(index, now, arrival):
            elapsed.clear()
            started = perf_counter_ns()
            sequence, features, *_ = dataset[index]
            with torch.inference_mode():
                logits, _ = encoder(sequence.unsqueeze(0), features.unsqueeze(0))
                logits.softmax(dim=-1).amax().item()
            encoded = perf_counter_ns()
            result = shield.decide(
                candidate, context, now_s=now, primary_attack_family="DoS"
            )
            decided = perf_counter_ns()
            updated = result.verdict == Verdict.DEPLOY
            if updated:
                observed = cost_model.execute(
                    candidate,
                    mission_phase=context.mission_phase.value,
                    primary_attack_family="DoS",
                )
                shield.observe_cost_outcome(result, observed)
            finished = perf_counter_ns()
            return {
                "burst": burst,
                "window": index,
                "verdict": result.verdict.value,
                "deadline_rejection": any(
                    "deadline" in str(item).lower() for item in result.timed_failures
                ),
                "encoder_ms": (encoded - started) / 1e6,
                "shield_ms": result.decide_elapsed_ms,
                "authorization_service_ms": (decided - started) / 1e6,
                "authorization_from_arrival_ms": (decided - arrival) / 1e6,
                "request_with_feedback_ms": (finished - started) / 1e6,
                "prediction_ms": elapsed["prediction"] / 1e6,
                "conformal_gate_ms": elapsed["gate"] / 1e6,
                "safety_verification_ms": elapsed["safety"] / 1e6,
                "calibration_update_ms": elapsed["update"] / 1e6 if updated else None,
            }

        with ExitStack() as stack:
            instrument(stack, shield._cost_estimator, "estimate", "prediction")
            instrument(stack, shield._gate, "decide", "gate")
            instrument(stack, decision_module, "evaluate_all", "safety")
            instrument(stack, shield._ltl, "evaluate", "safety")
            instrument(stack, shield._gate, "observe_outcome", "update")
            for index in range(10):
                request(index, float(index), perf_counter_ns())
            for start in range(0, len(dataset), burst):
                arrival = perf_counter_ns()
                for index in range(start, start + burst):
                    rows.append(request(index, float(index + 10), arrival))

    for burst in (1, 5):
        measure_burst(burst)

    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)
    print(f"Saved {len(rows)} timed requests to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=library_eval.SEEDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--encoder-replay", action="store_true")
    mode.add_argument("--component-timing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT / "runs")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed values must be unique")
    if args.component_timing:
        run_component_timing(args.out_dir)
        return
    if args.encoder_replay:
        run_encoder_replay(args.out_dir)
        return
    estimator = mission_eval.M3CostEstimator.from_dir(mission_eval.M3_DIR)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_paths = [args.out_dir / f"seed_{seed}.csv" for seed in args.seeds]
    path = args.out_dir / "episodes.csv"
    if path.exists() or any(seed_path.exists() for seed_path in seed_paths):
        parser.error("Results already exist. Choose a fresh --out-dir for this run.")
    rows = []
    for seed, seed_path in zip(args.seeds, seed_paths, strict=True):
        print(f"Running matched seed {seed}", flush=True)
        started = perf_counter()
        episode_rows = run_seed(seed, estimator)
        pl.DataFrame(episode_rows).write_csv(seed_path)
        rows.extend(episode_rows)
        print(f"Saved seed {seed} in {perf_counter() - started:.1f}s", flush=True)
    pl.DataFrame(rows).write_csv(path)
    print(f"Saved {len(rows)} episode rows to {path}")


if __name__ == "__main__":
    main()
