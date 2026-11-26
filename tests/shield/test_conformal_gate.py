"""Tests for the adaptive conformal resource gate (spec §5)."""

from __future__ import annotations

import numpy as np
import pytest

from gade_uav.shield.actions import COST_DIMENSIONS, MissionPhase
from gade_uav.shield.conformal_gate import (
    BudgetSchedule,
    MultiDimensionalConformalGate,
    SplitConformalDimension,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_budgets() -> BudgetSchedule:
    return BudgetSchedule(
        base_budget={k: 1_000.0 for k in COST_DIMENSIONS},
        phase_slack={p: 1.0 for p in MissionPhase},
    )


def _pred_vec(v: float = 0.5) -> dict[str, float]:
    return {k: v for k in COST_DIMENSIONS}


# ---------------------------------------------------------------------------
# Split-conformal per-dimension coverage guarantee (spec §5.1)
# ---------------------------------------------------------------------------


class TestSplitConformalCoverage:
    """Under exchangeability, P(c_true <= UCB) >= 1 - alpha.

    We generate calibration and evaluation residuals from the SAME iid
    distribution and check empirical coverage on the evaluation set.
    Repeated across seeds to guard against sampling noise.
    """

    ALPHA = 0.10
    N_CAL = 500
    N_EVAL = 5_000
    N_TRIALS = 30
    # Tolerance under finite samples. With n=5000 and target 0.90 the
    # binomial std is sqrt(0.9*0.1/5000) ~ 0.0042; a 5-sigma envelope
    # is ~0.021. Use 0.03 to be safe across trials.
    COVERAGE_TOL = 0.03

    def test_marginal_coverage_gaussian(self):
        rng = np.random.default_rng(0)
        undercoverage_count = 0

        for trial in range(self.N_TRIALS):
            dim = SplitConformalDimension(
                dimension="cpu",
                alpha_initial=self.ALPHA,
                min_calibration_size=self.N_CAL,
            )
            # Calibration: prediction is always 0.0, ground truth ~ N(0, 1).
            # Residuals equal ground truth.
            cal_truths = rng.standard_normal(self.N_CAL)
            for t in cal_truths:
                dim.add_calibration(predicted=0.0, observed=float(t))

            eval_truths = rng.standard_normal(self.N_EVAL)
            ucb = dim.ucb(predicted=0.0)
            covered = np.mean(eval_truths <= ucb)

            if covered < (1.0 - self.ALPHA) - self.COVERAGE_TOL:
                undercoverage_count += 1

        # Trial-level coverage failure should be rare (<= 10% of trials).
        assert undercoverage_count <= 3, (
            f"{undercoverage_count}/{self.N_TRIALS} trials fell below "
            f"the coverage target."
        )

    def test_coverage_with_biased_predictor(self):
        """Predictions biased low; residuals still exchangeable.

        Split-conformal's guarantee is marginal over the JOINT calibration
        + evaluation distribution, not conditional on a single calibration
        draw. We average across trials to exercise the marginal claim, and
        require the mean empirical coverage across trials to meet the target
        within tolerance (matching test_marginal_coverage_gaussian).
        """
        rng = np.random.default_rng(1)
        coverages = []
        for _ in range(self.N_TRIALS):
            dim = SplitConformalDimension("cpu", self.ALPHA, self.N_CAL)
            # Truth ~ N(2, 1); prediction is constant 0 (bias = 2).
            for _ in range(self.N_CAL):
                dim.add_calibration(predicted=0.0, observed=float(rng.normal(2, 1)))
            eval_truths = rng.normal(2, 1, size=self.N_EVAL)
            ucb = dim.ucb(predicted=0.0)
            coverages.append(float(np.mean(eval_truths <= ucb)))

        mean_coverage = float(np.mean(coverages))
        assert mean_coverage >= (1.0 - self.ALPHA) - self.COVERAGE_TOL, (
            f"mean empirical coverage {mean_coverage:.3f} across "
            f"{self.N_TRIALS} trials below target"
        )


# ---------------------------------------------------------------------------
# ACI convergence under drift (spec §5.2)
# ---------------------------------------------------------------------------


class TestAdaptiveConformalConvergence:
    """Under drift, ACI keeps the long-run miscoverage frequency near alpha.

    We simulate a scenario where the residual distribution shifts every
    T steps. The static split-conformal quantile from an old calibration
    set becomes miscalibrated; the ACI update pulls alpha back so that
    the empirical miscover rate over the whole horizon is close to the
    target.
    """

    def test_aci_pulls_alpha_toward_target_under_shift(self):
        rng = np.random.default_rng(42)
        alpha_target = 0.10
        gamma = 0.005
        dim = SplitConformalDimension(
            dimension="cpu",
            alpha_initial=alpha_target,
            min_calibration_size=100,
        )
        # Seed with 500 samples from N(0,1).
        for _ in range(500):
            dim.add_calibration(0.0, float(rng.standard_normal()))

        # Shift the true distribution mid-run. Without ACI the static
        # quantile under-covers. ACI should widen the UCB (raise the
        # quantile, i.e. lower alpha) to restore coverage.
        n_steps = 5_000
        miscover_count = 0
        for t in range(n_steps):
            # Drift: heavier-tailed after step 1500.
            scale = 1.0 if t < 1500 else 2.0
            truth = float(rng.normal(0, scale))
            ucb = dim.ucb(0.0)
            miscovered = truth > ucb
            miscover_count += int(miscovered)
            dim.aci_update(
                miscovered=miscovered,
                gamma=gamma,
                alpha_target=alpha_target,
            )
            # Do NOT refresh calibration -- we want to isolate ACI's
            # contribution.

        emp_miscover = miscover_count / n_steps
        # ACI's long-run guarantee: |emp_miscover - alpha_target| -> 0
        # up to O(1/(gamma * n)) transient. With gamma=0.005 and n=5000
        # the transient bound is generous; use 0.05 tolerance.
        assert (
            abs(emp_miscover - alpha_target) < 0.05
        ), f"empirical miscover {emp_miscover:.3f}, target {alpha_target}"

    def test_alpha_clamped_in_open_interval(self):
        dim = SplitConformalDimension("cpu", 0.10, 10)
        # Force many miscover events; alpha should shrink but not below 0.
        for _ in range(10_000):
            dim.aci_update(miscovered=True, gamma=0.005, alpha_target=0.10)
        assert 0.0 < dim.alpha < 1.0

        # Force many well-covered events; alpha should grow but not above 1.
        for _ in range(10_000):
            dim.aci_update(miscovered=False, gamma=0.005, alpha_target=0.10)
        assert 0.0 < dim.alpha < 1.0


# ---------------------------------------------------------------------------
# BudgetSchedule
# ---------------------------------------------------------------------------


class TestBudgetSchedule:
    def test_missing_dimension_raises(self):
        with pytest.raises(ValueError, match="missing dimensions"):
            BudgetSchedule(
                base_budget={"cpu": 1.0},  # missing 5 dims
                phase_slack={p: 1.0 for p in MissionPhase},
            )

    def test_missing_phase_raises(self):
        with pytest.raises(ValueError, match="missing phases"):
            BudgetSchedule(
                base_budget={k: 1.0 for k in COST_DIMENSIONS},
                phase_slack={MissionPhase.CRUISE: 1.0},  # missing others
            )

    def test_budget_multiplies_slack(self):
        b = BudgetSchedule(
            base_budget={k: 100.0 for k in COST_DIMENSIONS},
            phase_slack={
                MissionPhase.IDLE: 1.0,
                MissionPhase.TAKEOFF: 0.5,
                MissionPhase.CRUISE: 1.5,
                MissionPhase.PRECISION_HOVER: 0.5,
                MissionPhase.LANDING: 0.5,
                MissionPhase.EMERGENCY_RECOVERY: 0.5,
                MissionPhase.POST_LANDING: 1.0,
            },
        )
        assert b.budget_for("cpu", MissionPhase.TAKEOFF) == 50.0
        assert b.budget_for("cpu", MissionPhase.CRUISE) == 150.0


# ---------------------------------------------------------------------------
# Multi-dimensional gate composition (spec §5.4)
# ---------------------------------------------------------------------------


class TestMultiDimensionalGate:
    def _fresh(self, min_cal: int = 100) -> MultiDimensionalConformalGate:
        return MultiDimensionalConformalGate(
            alpha_target=0.10,
            aci_gamma=0.005,
            min_calibration_size=min_cal,
            budgets=_default_budgets(),
        )

    def test_insufficient_calibration_rejects(self):
        gate = self._fresh(min_cal=100)
        d = gate.decide(_pred_vec(0.1), MissionPhase.CRUISE)
        assert d.insufficient_calibration is True
        assert d.admitted is False

    def _seed_calibration(
        self,
        gate: MultiDimensionalConformalGate,
        n: int,
        pred: float,
        truth: float,
    ) -> None:
        for _ in range(n):
            gate.add_calibration(
                predicted={k: pred for k in COST_DIMENSIONS},
                observed={k: truth for k in COST_DIMENSIONS},
            )

    def test_all_dimensions_admitted_when_ucb_below_budget(self):
        gate = self._fresh(min_cal=100)
        # Constant residual = 0. UCB = predicted. With budget 1000, easily admits.
        self._seed_calibration(gate, 100, pred=0.5, truth=0.5)
        d = gate.decide(_pred_vec(0.5), MissionPhase.CRUISE)
        assert d.admitted is True
        assert not d.failing_dimensions
        assert len(d.per_dimension) == len(COST_DIMENSIONS)

    def test_gate_rejects_when_any_dimension_exceeds_budget(self):
        # Tight budget on cpu only. Others slack.
        base = {k: 1_000.0 for k in COST_DIMENSIONS}
        base["cpu"] = 0.05
        budgets = BudgetSchedule(
            base_budget=base,
            phase_slack={p: 1.0 for p in MissionPhase},
        )
        gate = MultiDimensionalConformalGate(
            alpha_target=0.10,
            aci_gamma=0.005,
            min_calibration_size=100,
            budgets=budgets,
        )
        # Predict cpu=0.1 (> budget 0.05); other dims trivially fit.
        for _ in range(100):
            gate.add_calibration(
                predicted={k: 0.0 for k in COST_DIMENSIONS},
                observed={k: 0.0 for k in COST_DIMENSIONS},
            )
        pred = {k: 0.1 for k in COST_DIMENSIONS}
        d = gate.decide(pred, MissionPhase.CRUISE)
        assert d.admitted is False
        assert d.failing_dimensions == ("cpu",)

    def test_observe_outcome_refreshes_calibration_and_alpha(self):
        gate = self._fresh(min_cal=10)
        self._seed_calibration(gate, 10, pred=0.0, truth=0.0)
        pred = _pred_vec(0.5)
        d = gate.decide(pred, MissionPhase.CRUISE)
        alpha_before = gate.current_alphas()["cpu"]
        cal_size_before = gate.calibration_sizes()["cpu"]

        # Simulate a true cost above the UCB -> miscovered.
        observed = {k: 10_000.0 for k in COST_DIMENSIONS}
        gate.observe_outcome(d, observed)

        alpha_after = gate.current_alphas()["cpu"]
        cal_size_after = gate.calibration_sizes()["cpu"]
        # After a miscover event with target 0.10, ACI reduces alpha by
        # ~gamma * (0.10 - 1) = -0.0045, so alpha should have decreased.
        assert alpha_after < alpha_before
        # Calibration set grew by exactly one sample per dimension.
        assert cal_size_after == cal_size_before + 1
