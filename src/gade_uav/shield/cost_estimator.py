"""Point cost estimators for the shield's L3 conformal gate.

The L3 gate (conformal_gate.MultiDimensionalConformalGate) requires a
per-dimension point estimate ``predicted[k]`` for each candidate action.
This module defines the estimator protocol and two implementations:

  * GeneratorCostEstimator — returns action.predicted_cost.as_dict();
    preserves the pre-M3 behaviour where the orchestrator's proposal
    carries its own cost self-estimate.

  * M3CostEstimator — returns a learned cost prediction from the M3
    LightGBM ensemble (models.cost_predictor.CostPredictor), built from
    the action + context + attack-family features that M3 was trained
    on.

The estimator is injected into Shield at construction time. The L3
conformal machinery (residual quantile, ACI, budgets) is unchanged: it
adapts online regardless of the point estimator. Swapping estimators is
therefore a clean ablation knob — one function, everything downstream
is measurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from gade_uav.models.cost_dataset import (
    ACTION_TYPES,
    ATTACK_FAMILIES,
    COST_DIMS,
    MISSION_PHASES,
    N_FEATURES,
    TARGET_IFACES,
)
from gade_uav.models.cost_predictor import CostPredictor
from gade_uav.shield.actions import Action, ShieldContext


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class CostEstimator(Protocol):
    """Point cost estimator interface consumed by Shield's L3 stage."""

    def estimate(
        self,
        action: Action,
        ctx: ShieldContext,
        primary_attack_family: str | None,
    ) -> dict[str, float]:
        """Return a mapping ``dim -> point estimate`` for every COST_DIMS entry."""
        ...


# ---------------------------------------------------------------------------
# Generator (baseline) — uses the orchestrator's self-estimate
# ---------------------------------------------------------------------------


class GeneratorCostEstimator:
    """Delegates to ``action.predicted_cost.as_dict()`` (pre-M3 behaviour)."""

    def estimate(
        self,
        action: Action,
        ctx: ShieldContext,
        primary_attack_family: str | None,
    ) -> dict[str, float]:
        # The generator's cost is attached to the Action itself. ctx and
        # primary_attack_family are ignored here but part of the protocol
        # so the Shield can swap estimators without conditionals.
        del ctx, primary_attack_family
        return action.predicted_cost.as_dict()


# ---------------------------------------------------------------------------
# M3 — LightGBM ensemble trained on 840k DEPLOY outcomes
# ---------------------------------------------------------------------------


class M3CostEstimator:
    """Wraps a fitted CostPredictor; builds the 26-d feature vector inline.

    The feature schema mirrors ``cost_dataset.build_features`` exactly —
    any change there requires a change here (there is a paired test).
    The final 6 dims are the generator's declared predicted_cost, so M3
    learns a residual correction rather than a full replacement (Path A).
    """

    def __init__(self, predictor: CostPredictor) -> None:
        if not predictor.boosters:
            raise ValueError("M3CostEstimator requires a fitted CostPredictor")
        self._predictor = predictor
        # Pre-compute category-to-index maps for O(1) one-hot writes.
        self._action_idx = {c: i for i, c in enumerate(ACTION_TYPES)}
        self._iface_idx = {c: i for i, c in enumerate(TARGET_IFACES)}
        self._phase_idx = {c: i for i, c in enumerate(MISSION_PHASES)}
        self._family_idx = {c: i for i, c in enumerate(ATTACK_FAMILIES)}

    @classmethod
    def from_dir(cls, artifact_dir: Path) -> "M3CostEstimator":
        return cls(CostPredictor.load(Path(artifact_dir)))

    def estimate(
        self,
        action: Action,
        ctx: ShieldContext,
        primary_attack_family: str | None,
    ) -> dict[str, float]:
        x = np.zeros((1, N_FEATURES), dtype=np.float32)
        off = 0

        # action_type one-hot
        i = self._action_idx.get(action.type.value)
        if i is not None:
            x[0, off + i] = 1.0
        off += len(ACTION_TYPES)

        # target_iface one-hot
        i = self._iface_idx.get(action.target_iface.value)
        if i is not None:
            x[0, off + i] = 1.0
        off += len(TARGET_IFACES)

        # mission_phase one-hot
        i = self._phase_idx.get(ctx.mission_phase.value)
        if i is not None:
            x[0, off + i] = 1.0
        off += len(MISSION_PHASES)

        # attack_family one-hot (null -> "none" sentinel)
        fam = primary_attack_family if primary_attack_family is not None else "none"
        i = self._family_idx.get(fam, self._family_idx["none"])
        x[0, off + i] = 1.0
        off += len(ATTACK_FAMILIES)

        # gen_confidence scalar
        x[0, off] = float(action.gen_meta.confidence)
        off += 1

        # generator-declared predicted_cost, in COST_DIMS order — must match
        # PRED_COLS ordering in cost_dataset.build_features exactly.
        pc = action.predicted_cost
        x[0, off + 0] = float(pc.cpu)
        x[0, off + 1] = float(pc.ram)
        x[0, off + 2] = float(pc.lat)
        x[0, off + 3] = float(pc.bw)
        x[0, off + 4] = float(pc.energy)
        x[0, off + 5] = float(pc.mission_delay)
        off += 6

        assert off == N_FEATURES, f"feature builder off by {N_FEATURES - off}"

        yhat = self._predictor.predict(x)[0]  # shape (6,)
        return {dim: float(yhat[i]) for i, dim in enumerate(COST_DIMS)}
