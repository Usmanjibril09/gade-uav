"""Recorder: writes per-decision rows to Parquet.

One row per SimulationLoop.deploy() call. Columns cover:
  - simulator state (t_s, tick)
  - action metadata (action_type, target_iface, gen_confidence)
  - predicted vs observed cost per dimension
  - shield verdict + reason + failing property/dimension
  - context (mission_phase, primary_attack_family, campaign_id)

Rows are buffered in-memory and flushed on close() or when the buffer
exceeds `flush_every`. Parquet is written with zstd compression to match
Phase 1 outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from gade_uav.shield.actions import Action, CostVector
from gade_uav.shield.decide import ShieldDecision

_COST_DIMS = ("cpu", "ram", "lat", "bw", "energy", "mission_delay")


# Explicit Polars schema for DeploymentRow. Declared upfront so that
# type inference cannot be fooled by long null-prefixes on nullable
# columns (primary_attack_family and campaign_id are null while the
# adversary is idle; obs_* are null on REJECT rows). Keys MUST match
# DeploymentRow field names in order.
_DEPLOYMENT_SCHEMA: dict[str, pl.DataType] = {
    "t_s": pl.Float64,
    "tick": pl.Int64,
    "action_type": pl.Utf8,
    "target_iface": pl.Utf8,
    "gen_confidence": pl.Float64,
    "mission_phase": pl.Utf8,
    "primary_attack_family": pl.Utf8,
    "campaign_id": pl.Utf8,
    "verdict": pl.Utf8,
    "layer_failed": pl.Utf8,
    "reason": pl.Utf8,
    "ltl_failing_property": pl.Utf8,
    "gate_failing_dimensions": pl.Utf8,
    "timed_failing_monitors": pl.Utf8,
    "repair_attempted": pl.Boolean,
    "repair_succeeded": pl.Boolean,
    "decide_elapsed_ms": pl.Float64,
    "pred_cpu": pl.Float64,
    "pred_ram": pl.Float64,
    "pred_lat": pl.Float64,
    "pred_bw": pl.Float64,
    "pred_energy": pl.Float64,
    "pred_mission_delay": pl.Float64,
    "obs_cpu": pl.Float64,
    "obs_ram": pl.Float64,
    "obs_lat": pl.Float64,
    "obs_bw": pl.Float64,
    "obs_energy": pl.Float64,
    "obs_mission_delay": pl.Float64,
}


@dataclass
class DeploymentRow:
    t_s: float
    tick: int
    action_type: str
    target_iface: str
    gen_confidence: float
    mission_phase: str
    primary_attack_family: str | None
    campaign_id: str | None
    verdict: str
    layer_failed: str | None  # "L1"|"L2"|"L3"|"L4" or None
    reason: str | None
    ltl_failing_property: str | None  # from decision.ltl_result if any
    gate_failing_dimensions: str  # comma-joined; empty if none
    timed_failing_monitors: str  # comma-joined; empty if none
    repair_attempted: bool
    repair_succeeded: bool
    decide_elapsed_ms: float
    # predicted (from action.predicted_cost)
    pred_cpu: float
    pred_ram: float
    pred_lat: float
    pred_bw: float
    pred_energy: float
    pred_mission_delay: float
    # observed (from Effector); nulls if not deployed
    obs_cpu: float | None
    obs_ram: float | None
    obs_lat: float | None
    obs_bw: float | None
    obs_energy: float | None
    obs_mission_delay: float | None


class Recorder:
    def __init__(self, out_path: str | Path, flush_every: int = 1000) -> None:
        self._out_path = Path(out_path)
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        if flush_every <= 0:
            raise ValueError("flush_every must be > 0")
        self._flush_every = flush_every
        self._buffer: list[DeploymentRow] = []
        self._flushed_rows = 0
        self._flushed_frames: list[pl.DataFrame] = []

    def record(
        self,
        *,
        t_s: float,
        tick: int,
        action: Action,
        decision: ShieldDecision,
        observed_cost: CostVector | None,
        mission_phase: str,
        primary_attack_family: str | None,
        campaign_id: str | None,
    ) -> None:
        pc = action.predicted_cost
        ltl_prop: str | None = None
        if decision.ltl_result is not None and not decision.ltl_result.holds:
            ltl_prop = decision.ltl_result.failing_property
        gate_dims = ""
        if decision.gate_decision is not None:
            gate_dims = ",".join(sorted(decision.gate_decision.failing_dimensions))
        timed_mons = ",".join(
            sorted(
                tf.monitor
                for tf in decision.timed_failures
                if not tf.holds and tf.monitor is not None
            )
        )
        row = DeploymentRow(
            t_s=t_s,
            tick=tick,
            action_type=action.type.value,
            target_iface=action.target_iface.value,
            gen_confidence=action.gen_meta.confidence,
            mission_phase=mission_phase,
            primary_attack_family=primary_attack_family,
            campaign_id=campaign_id,
            verdict=decision.verdict.value,
            layer_failed=decision.layer_failed,
            reason=decision.reason,
            ltl_failing_property=ltl_prop,
            gate_failing_dimensions=gate_dims,
            timed_failing_monitors=timed_mons,
            repair_attempted=decision.repair_attempted,
            repair_succeeded=decision.repair_succeeded,
            decide_elapsed_ms=decision.decide_elapsed_ms,
            pred_cpu=pc.cpu,
            pred_ram=pc.ram,
            pred_lat=pc.lat,
            pred_bw=pc.bw,
            pred_energy=pc.energy,
            pred_mission_delay=pc.mission_delay,
            obs_cpu=observed_cost.cpu if observed_cost else None,
            obs_ram=observed_cost.ram if observed_cost else None,
            obs_lat=observed_cost.lat if observed_cost else None,
            obs_bw=observed_cost.bw if observed_cost else None,
            obs_energy=observed_cost.energy if observed_cost else None,
            obs_mission_delay=observed_cost.mission_delay if observed_cost else None,
        )
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_every:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        df = pl.DataFrame([asdict(r) for r in self._buffer], schema=_DEPLOYMENT_SCHEMA)
        self._flushed_frames.append(df)
        self._flushed_rows += len(self._buffer)
        self._buffer.clear()

    def close(self) -> Path:
        """Flush remaining rows and write the parquet file. Returns the path."""
        self._flush_buffer()
        if not self._flushed_frames:
            # Write an empty file with the correct schema.
            empty = pl.DataFrame(schema=_DEPLOYMENT_SCHEMA)
            empty.write_parquet(self._out_path, compression="zstd")
            return self._out_path
        combined = pl.concat(self._flushed_frames, how="vertical")
        combined.write_parquet(self._out_path, compression="zstd")
        return self._out_path

    @property
    def buffered_rows(self) -> int:
        return len(self._buffer)

    @property
    def flushed_rows(self) -> int:
        return self._flushed_rows

    @property
    def total_rows(self) -> int:
        return self._flushed_rows + len(self._buffer)
