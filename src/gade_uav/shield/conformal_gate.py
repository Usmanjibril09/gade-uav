"""Adaptive conformal resource gate for the GADe-UAV shield.

Implements docs/02_shield_spec.md §5:
    §5.1 Split conformal per dimension: UCB_k(d) = c_hat_k(d) + q_hat_k
    §5.2 Adaptive Conformal Inference (Gibbs & Candes 2021):
             a_k^{t+1} = a_k^t + gamma * (a_target - 1{c_k^t > UCB_k^t})
    §5.3 Context-dependent budgets: budget_k(m_t) = base_k * slack(m_t)
    §5.4 Multi-dim composition: admit iff for all k, UCB_k(d) <= budget_k

Design invariants:
    - Calibration set (nonconformity scores) is disjoint from evaluation.
    - No dependency on torch/scapy -- pure numpy + stdlib.
    - Per-dimension state is independent; the multi-dim gate composes
      six per-dim gates without any Bonferroni correction (spec §5.4).
    - The gate REFUSES to decide below min_calibration_size (spec locks
      this at 200 in configs/shield.yaml). This is not an admission or
      rejection -- it is a distinct "insufficient calibration" outcome
      that the caller (ShieldDecide L3) must handle.

Nonconformity score convention:
    s_i = c_i_true - c_hat_k(d_i)   (residual, signed)
    UCB_k(d) = c_hat_k(d) + q_hat_k where q_hat_k is the
    ceil((n+1)(1-alpha))-th empirical quantile of the calibration
    residuals. Under exchangeability this gives
        P(c_k_true <= UCB_k) >= 1 - alpha   (spec §5.1).
    Only over-prediction is a safety failure, so a one-sided upper
    bound is exactly what we want.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass

import numpy as np

from gade_uav.shield.actions import COST_DIMENSIONS, MissionPhase


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerDimensionDecision:
    """Diagnostic record for one cost dimension on one gating decision."""

    dimension: str
    predicted: float
    ucb: float
    budget: float
    alpha_current: float
    admitted: bool


@dataclass(frozen=True)
class GateDecision:
    """Composite decision across all cost dimensions."""

    admitted: bool
    insufficient_calibration: (
        bool  # true iff any dim has < min_calibration_size samples
    )
    failing_dimensions: tuple[str, ...]  # dims where UCB > budget
    per_dimension: tuple[PerDimensionDecision, ...]


# ---------------------------------------------------------------------------
# Split-conformal per-dimension gate (spec §5.1)
# ---------------------------------------------------------------------------


class SplitConformalDimension:
    """Split-conformal upper confidence bound for one cost dimension.

    Maintains a calibration set of nonconformity residuals. The quantile
    is recomputed each time the calibration set changes. For efficiency
    with streaming updates the caller may batch calls to add_calibration.

    Miscoverage `alpha` is exposed as a public attribute so the ACI
    wrapper can mutate it after each observed outcome.
    """

    def __init__(
        self,
        dimension: str,
        alpha_initial: float,
        min_calibration_size: int,
        calibration_capacity: int = 10_000,
    ) -> None:
        if not (0.0 < alpha_initial < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha_initial}")
        if min_calibration_size < 1:
            raise ValueError(
                f"min_calibration_size must be >= 1, got {min_calibration_size}"
            )
        self.dimension = dimension
        self.alpha = float(alpha_initial)
        self.min_calibration_size = int(min_calibration_size)
        # Bounded deque so calibration doesn't grow without limit under long
        # missions. Exchangeability holds only if drift is slow enough for
        # the retained window to remain representative; ACI compensates
        # for the residual drift.
        self._residuals: deque[float] = deque(maxlen=calibration_capacity)

        # Exact sorted mirror of the same bounded residual window.
        #
        # quantile() can therefore return the required order statistic
        # directly instead of sorting the full calibration set on every
        # shield decision.
        #
        # This is NOT an approximate quantile structure.
        self._sorted_residuals: list[float] = []

    # -- calibration ------------------------------------------------------

    def _append_residual(self, residual: float) -> None:
        """Append one residual while keeping the exact sorted mirror."""

        residual = float(residual)

        # deque(maxlen=...) would silently evict the oldest value.
        # Remove that same value from the sorted mirror first so both
        # structures always represent exactly the same calibration window.
        if (
            self._residuals.maxlen is not None
            and len(self._residuals) == self._residuals.maxlen
        ):
            oldest = self._residuals[0]

            idx = bisect_left(
                self._sorted_residuals,
                oldest,
            )

            if (
                idx >= len(self._sorted_residuals)
                or self._sorted_residuals[idx] != oldest
            ):
                raise RuntimeError(
                    "sorted calibration mirror is inconsistent "
                    "with residual deque"
                )

            self._sorted_residuals.pop(idx)

        self._residuals.append(residual)

        insort(
            self._sorted_residuals,
            residual,
        )

    def add_calibration(self, predicted: float, observed: float) -> None:
        """Append one (prediction, ground-truth) pair to the calibration set."""

        self._append_residual(
            float(observed) - float(predicted)
        )

    def add_calibration_batch(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
    ) -> None:
        if predicted.shape != observed.shape:
            raise ValueError(
                f"shape mismatch: predicted {predicted.shape} vs observed {observed.shape}"
            )

        new_residuals = (
            np.asarray(observed, dtype=np.float64)
            - np.asarray(predicted, dtype=np.float64)
        ).reshape(-1)

        # Batch loading is used for the 270,001-row initial calibration.
        # Building it with repeated insort would be unnecessarily O(n^2).
        # Reconstruct the bounded chronological window once, then sort
        # exactly once.
        combined = (
            list(self._residuals)
            + new_residuals.tolist()
        )

        capacity = self._residuals.maxlen

        if (
            capacity is not None
            and len(combined) > capacity
        ):
            combined = combined[-capacity:]

        self._residuals.clear()
        self._residuals.extend(combined)

        self._sorted_residuals = sorted(combined)

    @property
    def calibration_size(self) -> int:
        return len(self._residuals)

    def has_sufficient_calibration(self) -> bool:
        return self.calibration_size >= self.min_calibration_size

    # -- quantile / UCB ---------------------------------------------------

    def quantile(self) -> float:
        """Return q_hat at the current alpha.

        Uses the ceil((n+1)(1-alpha))/n empirical quantile convention that
        gives finite-sample coverage 1 - alpha under exchangeability.
        Returns +inf if the calibration set is empty (no bound available).
        """
        n = self.calibration_size
        if n == 0:
            return math.inf
        # rank (1-indexed) of the desired empirical quantile
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        # clamp to [1, n]
        rank = max(1, min(rank, n))
        # Exact rank-th smallest residual, 0-indexed.
        # _sorted_residuals mirrors _residuals exactly.
        return float(
            self._sorted_residuals[rank - 1]
        )

    def ucb(self, predicted: float) -> float:
        """Upper confidence bound at the current alpha."""
        return float(predicted) + self.quantile()

    # -- ACI update (spec §5.2) ------------------------------------------

    def aci_update(
        self,
        miscovered: bool,
        gamma: float,
        alpha_target: float,
        alpha_min: float = 1e-4,
        alpha_max: float = 1.0 - 1e-4,
    ) -> None:
        """Gibbs & Candes ACI update.

            alpha <- alpha + gamma * (alpha_target - 1{miscovered})

        Clamped to (0, 1) to keep the quantile well-defined under
        adversarial drift.
        """
        indicator = 1.0 if miscovered else 0.0
        self.alpha = float(self.alpha + gamma * (alpha_target - indicator))
        if self.alpha < alpha_min:
            self.alpha = alpha_min
        elif self.alpha > alpha_max:
            self.alpha = alpha_max


# ---------------------------------------------------------------------------
# Budget schedule (spec §5.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetSchedule:
    """Per-dimension base budget and per-phase slack multiplier.

    budget_k(m_t) = base_budget[k] * phase_slack[m_t]

    Both dicts are mandatory in configs/budget_schedule.yaml. All
    six COST_DIMENSIONS must appear as keys in base_budget; every
    MissionPhase must appear as a key in phase_slack.
    """

    base_budget: dict[str, float]
    phase_slack: dict[MissionPhase, float]

    def __post_init__(self) -> None:
        missing_dims = set(COST_DIMENSIONS) - set(self.base_budget.keys())
        if missing_dims:
            raise ValueError(f"base_budget missing dimensions: {sorted(missing_dims)}")
        missing_phases = set(MissionPhase) - set(self.phase_slack.keys())
        if missing_phases:
            raise ValueError(
                f"phase_slack missing phases: {sorted(p.value for p in missing_phases)}"
            )

    def budget_for(self, dimension: str, phase: MissionPhase) -> float:
        return float(self.base_budget[dimension]) * float(self.phase_slack[phase])


# ---------------------------------------------------------------------------
# Multi-dimensional gate (spec §5.4)
# ---------------------------------------------------------------------------


class MultiDimensionalConformalGate:
    """Composes six per-dimension conformal gates.

    An action is admitted iff, for every dimension k in COST_DIMENSIONS,
    UCB_k(d) <= budget_k(mission_phase). Any dimension with insufficient
    calibration flags the whole gate as `insufficient_calibration`; the
    caller is expected to treat that as REJECT+FALLBACK (spec §7).
    """

    def __init__(
        self,
        alpha_target: float,
        aci_gamma: float,
        min_calibration_size: int,
        budgets: BudgetSchedule,
        calibration_capacity: int = 10_000,
    ) -> None:
        self.alpha_target = float(alpha_target)
        self.aci_gamma = float(aci_gamma)
        self.min_calibration_size = int(min_calibration_size)
        self._budgets = budgets
        self._dims: dict[str, SplitConformalDimension] = {
            k: SplitConformalDimension(
                dimension=k,
                alpha_initial=alpha_target,
                min_calibration_size=min_calibration_size,
                calibration_capacity=calibration_capacity,
            )
            for k in COST_DIMENSIONS
        }

    # -- calibration ------------------------------------------------------

    def add_calibration(
        self,
        predicted: dict[str, float],
        observed: dict[str, float],
    ) -> None:
        """Feed one (prediction_vector, observed_vector) pair."""
        for k in COST_DIMENSIONS:
            self._dims[k].add_calibration(predicted[k], observed[k])

    def add_residual_calibration_batch(
        self,
        residuals: dict[str, np.ndarray],
    ) -> None:
        """Load precomputed signed calibration residuals.

        Each residual must follow the gate's existing convention:

            observed - predicted

        This is mathematically equivalent to supplying calibration pairs
        with prediction=0 and observation=residual, while avoiding a slow
        row-by-row Python loop.
        """
        missing = set(COST_DIMENSIONS) - set(residuals)
        if missing:
            raise ValueError(
                f"missing residual dimensions: {sorted(missing)}"
            )

        sizes = {
            k: int(np.asarray(residuals[k]).size)
            for k in COST_DIMENSIONS
        }

        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"residual calibration sizes differ: {sizes}"
            )

        for k in COST_DIMENSIONS:
            arr = np.asarray(
                residuals[k],
                dtype=np.float64,
            ).reshape(-1)

            self._dims[k].add_calibration_batch(
                np.zeros_like(arr),
                arr,
            )

    def calibration_sizes(self) -> dict[str, int]:
        return {k: self._dims[k].calibration_size for k in COST_DIMENSIONS}

    def has_sufficient_calibration(self) -> bool:
        return all(self._dims[k].has_sufficient_calibration() for k in COST_DIMENSIONS)

    def current_alphas(self) -> dict[str, float]:
        return {k: self._dims[k].alpha for k in COST_DIMENSIONS}

    # -- gating -----------------------------------------------------------

    def decide(
        self,
        predicted: dict[str, float],
        phase: MissionPhase,
    ) -> GateDecision:
        """Evaluate the composite gate for one candidate action.

        Returns GateDecision with per-dimension diagnostics regardless of
        outcome. `admitted` implies not `insufficient_calibration` and
        empty `failing_dimensions`.
        """
        insufficient = not self.has_sufficient_calibration()
        per_dim: list[PerDimensionDecision] = []
        failing: list[str] = []

        for k in COST_DIMENSIONS:
            dim = self._dims[k]
            budget = self._budgets.budget_for(k, phase)
            pred = float(predicted[k])
            ucb = dim.ucb(pred)
            admitted_k = (not insufficient) and (ucb <= budget)
            if not admitted_k:
                failing.append(k)
            per_dim.append(
                PerDimensionDecision(
                    dimension=k,
                    predicted=pred,
                    ucb=ucb,
                    budget=budget,
                    alpha_current=dim.alpha,
                    admitted=admitted_k,
                )
            )

        admitted = (not insufficient) and (len(failing) == 0)
        return GateDecision(
            admitted=admitted,
            insufficient_calibration=insufficient,
            failing_dimensions=tuple(failing),
            per_dimension=tuple(per_dim),
        )

    # -- ACI update (spec §5.2) ------------------------------------------

    def observe_outcome(
        self,
        decision: GateDecision,
        observed: dict[str, float],
    ) -> None:
        """After an admitted action executes, record the observed cost.

        For each dimension: check whether the true cost exceeded its UCB
        at decision time (miscovered) and apply the ACI update to alpha_k.
        The observed cost is also appended to the calibration set so the
        quantile stays current under drift.
        """
        by_dim = {p.dimension: p for p in decision.per_dimension}
        for k in COST_DIMENSIONS:
            pdec = by_dim[k]
            true_cost = float(observed[k])
            miscovered = true_cost > pdec.ucb
            self._dims[k].aci_update(
                miscovered=miscovered,
                gamma=self.aci_gamma,
                alpha_target=self.alpha_target,
            )
            # Also refresh calibration with the new (prediction, truth) pair
            # so the empirical distribution tracks the operating regime.
            self._dims[k].add_calibration(pdec.predicted, true_cost)


__all__ = [
    "BudgetSchedule",
    "GateDecision",
    "MultiDimensionalConformalGate",
    "PerDimensionDecision",
    "SplitConformalDimension",
]
