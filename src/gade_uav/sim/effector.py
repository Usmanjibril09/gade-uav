"""Effector: executes shield-admitted actions and produces observed cost.

The observed cost is a physics-lite closed form plus per-dimension noise.
Cost model rationale in configs/effector.yaml.

The Effector is deterministic under a fixed seed. It is NOT responsible
for updating the shield's conformal calibration; SimulationLoop does that
after collecting the observed cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from gade_uav.shield.actions import Action, ActionType, CostVector

_COST_DIMS = ("cpu", "ram", "lat", "bw", "energy", "mission_delay")


def _load_multiplier_block(
    raw: dict, dims: tuple[str, ...], block_name: str
) -> dict[str, dict[str, float]]:
    """Load a multiplier block (family or phase) with defensive defaults.

    A missing block gives an empty dict (all lookups fall through to the
    identity fallback in execute()). An entry present in the block must
    contain a "none" row for family, but per-dim keys inside each row
    default to 1.0 when missing. All values must be non-negative.
    """
    out: dict[str, dict[str, float]] = {}
    for key, row in (raw or {}).items():
        if not isinstance(row, dict):
            raise ValueError(
                f"{block_name}[{key!r}] must be a mapping, got {type(row).__name__}"
            )
        parsed: dict[str, float] = {}
        for dim in dims:
            v = float(row.get(dim, 1.0))
            if v < 0:
                raise ValueError(
                    f"{block_name}[{key!r}][{dim!r}] must be >= 0, got {v}"
                )
            parsed[dim] = v
        out[key] = parsed
    # For the family block, guarantee "none" identity exists so execute() can
    # always fall back safely.
    if block_name == "family_multipliers" and "none" not in out:
        out["none"] = {d: 1.0 for d in dims}
    return out


class NoiseModel(str, Enum):
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"


@dataclass(frozen=True)
class DimensionNoise:
    model: NoiseModel
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


class Effector:
    """Turns an Action into an observed CostVector.

    Parameters
    ----------
    config_path : YAML with `noise:` and `base_cost:` sections.
    seed        : integer seed for reproducible noise draws.
    """

    def __init__(self, config_path: str | Path, seed: int) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text())
        raw_noise = cfg["noise"]
        raw_base = cfg["base_cost"]

        self._noise: dict[str, DimensionNoise] = {}
        for dim in _COST_DIMS:
            if dim not in raw_noise:
                raise ValueError(f"noise config missing dimension {dim!r}")
            entry = raw_noise[dim]
            self._noise[dim] = DimensionNoise(
                model=NoiseModel(entry["model"]),
                sigma=float(entry["sigma"]),
            )

        # base_cost keyed by ActionType.value (string) for YAML friendliness.
        self._base: dict[ActionType, dict[str, float]] = {}
        for at in ActionType:
            if at.value not in raw_base:
                raise ValueError(f"base_cost config missing action type {at.value!r}")
            entry = raw_base[at.value]
            per_dim: dict[str, float] = {}
            for dim in _COST_DIMS:
                if dim not in entry:
                    raise ValueError(f"base_cost[{at.value}] missing dimension {dim!r}")
                v = float(entry[dim])
                if v < 0:
                    raise ValueError(
                        f"base_cost[{at.value}][{dim}] must be >= 0, got {v}"
                    )
                per_dim[dim] = v
            self._base[at] = per_dim

        # Load family and phase multipliers. Both blocks are optional for
        # backward compatibility; missing block -> all multipliers = 1.0
        # (identity, i.e. pre-multiplier behaviour is preserved exactly).
        self._family_mult: dict[str, dict[str, float]] = _load_multiplier_block(
            cfg.get("family_multipliers", {}),
            _COST_DIMS,
            block_name="family_multipliers",
        )
        self._phase_mult: dict[str, dict[str, float]] = _load_multiplier_block(
            cfg.get("phase_multipliers", {}), _COST_DIMS, block_name="phase_multipliers"
        )
        # Track missing keys to avoid log spam.
        self._warned_families: set[str] = set()
        self._warned_phases: set[str] = set()

        self._rng = np.random.default_rng(seed)

    def _draw(self, base: float, noise: DimensionNoise) -> float:
        if noise.model is NoiseModel.GAUSSIAN:
            return max(0.0, base + float(self._rng.normal(0.0, noise.sigma)))
        # log-normal: multiplicative, heavy right tail, mean = base * exp(sigma^2/2)
        mult = float(self._rng.lognormal(mean=0.0, sigma=noise.sigma))
        return base * mult

    def execute(
        self,
        action: Action,
        *,
        mission_phase: str = "cruise",
        primary_attack_family: str | None = None,
    ) -> CostVector:
        """Return the observed cost vector for this action.

        The observed cost is
            action.predicted_cost[dim]
              * family_multiplier[family][dim]
              * phase_multiplier[phase][dim]
              + noise[dim]
        Missing family/phase entries degrade to identity (multiplier 1.0)
        with a one-time warning per unknown key. `primary_attack_family=None`
        maps to the "none" row (idle-period identity).

        The generator's `predicted_cost` field is the causal mean of the
        observed cost (subject to adversarial multipliers and noise). This
        makes the pred_* fields genuine training signal for the M3 cost
        predictor rather than decorative metadata.
        """
        fam_key = primary_attack_family if primary_attack_family is not None else "none"
        fam_row = self._family_mult.get(fam_key)
        if fam_row is None:
            if fam_key not in self._warned_families:
                import warnings

                warnings.warn(
                    f"Effector: unknown primary_attack_family {fam_key!r}; "
                    f"using identity multipliers. Warned once per key.",
                    stacklevel=2,
                )
                self._warned_families.add(fam_key)
            fam_row = self._family_mult["none"]

        phase_row = self._phase_mult.get(mission_phase)
        if phase_row is None:
            if mission_phase not in self._warned_phases:
                import warnings

                warnings.warn(
                    f"Effector: unknown mission_phase {mission_phase!r}; "
                    f"using identity multipliers. Warned once per key.",
                    stacklevel=2,
                )
                self._warned_phases.add(mission_phase)
            phase_row = {d: 1.0 for d in _COST_DIMS}

        # Option B/3: the generator's declared predicted_cost is the causal
        # mean of observed cost, modulated by family/phase multipliers and
        # per-dimension noise. This makes pred_* a genuine feature for M3
        # (M3 learns E[obs|action] = pred * family_mult * phase_mult), which
        # a categorical-only feature set cannot represent. The YAML base_cost
        # is retained for backward compatibility as a fallback when the
        # generator omits a dimension (should never happen — CostVector
        # requires all six — but defensive).
        pc = action.predicted_cost
        pred = {
            "cpu": pc.cpu,
            "ram": pc.ram,
            "lat": pc.lat,
            "bw": pc.bw,
            "energy": pc.energy,
            "mission_delay": pc.mission_delay,
        }
        base_fallback = self._base[action.type]
        values = {
            dim: self._draw(
                pred.get(dim, base_fallback[dim]) * fam_row[dim] * phase_row[dim],
                self._noise[dim],
            )
            for dim in _COST_DIMS
        }
        return CostVector(
            cpu=values["cpu"],
            ram=values["ram"],
            lat=values["lat"],
            bw=values["bw"],
            energy=values["energy"],
            mission_delay=values["mission_delay"],
        )

    @property
    def noise_config(self) -> dict[str, DimensionNoise]:
        return dict(self._noise)
