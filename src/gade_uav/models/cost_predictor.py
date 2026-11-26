"""M3 cost predictor — six independent LightGBM regressors.

One LGBMRegressor per cost dimension (cpu, ram, lat, bw, energy, mission_delay).
Trained on raw obs_* targets (no normalization — trees are scale-invariant).
Saves as a single directory containing one .txt booster per dimension plus
a metadata JSON with dim order + hyperparameters for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gade_uav.models.cost_dataset import COST_DIMS, N_FEATURES, N_TARGETS

# ---------------------------------------------------------------------------
# Hyperparameters — locked defaults; overridable via CostPredictorConfig.
# ---------------------------------------------------------------------------


@dataclass
class CostPredictorConfig:
    """LightGBM hyperparameters, shared across all 6 regressors.

    Defaults chosen for 840k train rows × 21 features:
      - num_leaves=63: moderate capacity, avoids overfit on small feature space
      - learning_rate=0.05: standard for early-stopping regime
      - n_estimators=2000: high ceiling, early stopping selects actual count
      - min_data_in_leaf=100: 840k / 63 ≈ 13k rows/leaf ceiling; 100 floor
                              prevents pure-noise leaves in cpu/ram dims
      - feature_fraction=0.9, bagging_fraction=0.9: mild regularization
      - early_stopping_rounds=50 on val MAE
    """

    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 2000
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    lambda_l2: float = 0.1
    early_stopping_rounds: int = 50
    seed: int = 42
    n_jobs: int = -1
    verbose: int = -1

    def to_lgb_params(self) -> dict:
        return {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "num_threads": self.n_jobs if self.n_jobs > 0 else 0,
            "verbose": self.verbose,
        }


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


@dataclass
class CostPredictor:
    """Six LightGBM boosters, one per cost dimension.

    Booster order matches COST_DIMS. Save/load persists boosters as .txt files
    (LightGBM's native format) plus a metadata JSON.
    """

    config: CostPredictorConfig = field(default_factory=CostPredictorConfig)
    boosters: list[lgb.Booster] = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose_eval: int = 100,
    ) -> None:
        """Train one booster per cost dimension with early stopping on val MAE."""
        if X_train.shape[1] != N_FEATURES:
            raise ValueError(
                f"X_train has {X_train.shape[1]} cols, expected {N_FEATURES}"
            )
        if y_train.shape[1] != N_TARGETS:
            raise ValueError(
                f"y_train has {y_train.shape[1]} cols, expected {N_TARGETS}"
            )
        if X_val.shape[1] != N_FEATURES:
            raise ValueError(f"X_val has {X_val.shape[1]} cols, expected {N_FEATURES}")
        if y_val.shape[1] != N_TARGETS:
            raise ValueError(f"y_val has {y_val.shape[1]} cols, expected {N_TARGETS}")

        params = self.config.to_lgb_params()
        self.boosters = []
        self.best_iterations = []
        for i, dim in enumerate(COST_DIMS):
            dtrain = lgb.Dataset(X_train, label=y_train[:, i])
            dval = lgb.Dataset(X_val, label=y_val[:, i], reference=dtrain)
            callbacks = [
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=verbose_eval),
            ]
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=self.config.n_estimators,
                valid_sets=[dval],
                valid_names=[f"val_{dim}"],
                callbacks=callbacks,
            )
            self.boosters.append(booster)
            self.best_iterations.append(booster.best_iteration)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (n, 6) cost matrix in COST_DIMS order."""
        if not self.boosters:
            raise RuntimeError("predictor not fitted")
        if X.shape[1] != N_FEATURES:
            raise ValueError(f"X has {X.shape[1]} cols, expected {N_FEATURES}")
        out = np.zeros((X.shape[0], N_TARGETS), dtype=np.float32)
        for i, booster in enumerate(self.boosters):
            out[:, i] = booster.predict(X, num_iteration=booster.best_iteration).astype(
                np.float32
            )
        return out

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Save 6 boosters + metadata to out_dir."""
        if not self.boosters:
            raise RuntimeError("cannot save unfitted predictor")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for dim, booster in zip(COST_DIMS, self.boosters):
            booster.save_model(str(out_dir / f"booster_{dim}.txt"))
        meta = {
            "cost_dims": list(COST_DIMS),
            "n_features": N_FEATURES,
            "n_targets": N_TARGETS,
            "best_iterations": {
                d: int(it) for d, it in zip(COST_DIMS, self.best_iterations)
            },
            "config": {
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "min_data_in_leaf": self.config.min_data_in_leaf,
                "feature_fraction": self.config.feature_fraction,
                "bagging_fraction": self.config.bagging_fraction,
                "bagging_freq": self.config.bagging_freq,
                "lambda_l2": self.config.lambda_l2,
                "early_stopping_rounds": self.config.early_stopping_rounds,
                "seed": self.config.seed,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, in_dir: Path) -> "CostPredictor":
        """Load 6 boosters + metadata from in_dir."""
        in_dir = Path(in_dir)
        meta_path = in_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing {meta_path}")
        meta = json.loads(meta_path.read_text())
        if tuple(meta["cost_dims"]) != COST_DIMS:
            raise ValueError(
                f"saved cost_dims {meta['cost_dims']} != current {COST_DIMS}"
            )
        cfg = CostPredictorConfig(**meta["config"])
        boosters = []
        best_iters = []
        for dim in COST_DIMS:
            path = in_dir / f"booster_{dim}.txt"
            if not path.exists():
                raise FileNotFoundError(f"missing booster file: {path}")
            boosters.append(lgb.Booster(model_file=str(path)))
            best_iters.append(meta["best_iterations"][dim])
        return cls(config=cfg, boosters=boosters, best_iterations=best_iters)
