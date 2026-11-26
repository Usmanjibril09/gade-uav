"""Recorder tests: schema, buffering, parquet round-trip."""

import polars as pl
import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    Effector as EffectorEnum,
    GenMeta,
    MissionPhase,
)
from gade_uav.shield.decide import ShieldDecision, Verdict
from gade_uav.shield.ltl_monitor import MonitorResult
from gade_uav.sim.recorder import DeploymentRow, Recorder


def _gen_meta() -> GenMeta:
    return GenMeta(
        confidence=0.9,
        schema_version="1.0",
        generation_context_hash="deadbeef",
        template_id="safe_heartbeat_v1",
    )


def _action() -> Action:
    return Action(
        type=ActionType.HONEY_BEACON,
        payload={"template_id": "safe_heartbeat_v1"},
        target_iface=EffectorEnum.HONEY_BEACON_CHANNEL,
        predicted_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
        gen_meta=_gen_meta(),
    )


def _deploy_decision() -> ShieldDecision:
    return ShieldDecision(
        verdict=Verdict.DEPLOY,
        action_out=_action(),
        fallback=None,
        layer_failed=None,
        ltl_result=None,
        gate_decision=None,
        timed_failures=(),
        repair_attempted=False,
        repair_succeeded=False,
        decide_elapsed_ms=0.42,
        reason=None,
    )


def _reject_decision_l2() -> ShieldDecision:
    return ShieldDecision(
        verdict=Verdict.REJECT,
        action_out=None,
        fallback=None,
        layer_failed="L2",
        ltl_result=MonitorResult(holds=False, failing_property="S1", reason="fake"),
        gate_decision=None,
        timed_failures=(),
        repair_attempted=False,
        repair_succeeded=False,
        decide_elapsed_ms=0.30,
        reason="S1 violated",
    )


class TestConstruction:
    def test_bad_flush_every(self, tmp_path):
        with pytest.raises(ValueError):
            Recorder(tmp_path / "x.parquet", flush_every=0)

    def test_creates_parent_dir(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "out.parquet"
        Recorder(p)
        assert p.parent.exists()


class TestRecording:
    def test_deploy_row_has_observed_cost(self, tmp_path):
        rec = Recorder(tmp_path / "out.parquet")
        obs = CostVector(0.011, 2.5, 0.6, 50.0, 0.012, 0.11)
        rec.record(
            t_s=1.0,
            tick=100,
            action=_action(),
            decision=_deploy_decision(),
            observed_cost=obs,
            mission_phase=MissionPhase.CRUISE.value,
            primary_attack_family=None,
            campaign_id=None,
        )
        assert rec.total_rows == 1

    def test_reject_row_has_null_observed(self, tmp_path):
        rec = Recorder(tmp_path / "out.parquet")
        rec.record(
            t_s=1.0,
            tick=100,
            action=_action(),
            decision=_reject_decision_l2(),
            observed_cost=None,
            mission_phase=MissionPhase.CRUISE.value,
            primary_attack_family="T7_MITM",
            campaign_id="c1",
        )
        rec.close()
        df = pl.read_parquet(tmp_path / "out.parquet")
        assert df.height == 1
        assert df["verdict"][0] == "REJECT"
        assert df["layer_failed"][0] == "L2"
        assert df["ltl_failing_property"][0] == "S1"
        assert df["obs_cpu"][0] is None


class TestBufferingAndFlush:
    def test_auto_flush_at_threshold(self, tmp_path):
        rec = Recorder(tmp_path / "out.parquet", flush_every=10)
        for i in range(25):
            rec.record(
                t_s=float(i) * 0.01,
                tick=i,
                action=_action(),
                decision=_deploy_decision(),
                observed_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
                mission_phase=MissionPhase.CRUISE.value,
                primary_attack_family=None,
                campaign_id=None,
            )
        assert rec.flushed_rows == 20
        assert rec.buffered_rows == 5

    def test_close_flushes_remainder(self, tmp_path):
        out = tmp_path / "out.parquet"
        rec = Recorder(out, flush_every=100)
        for i in range(7):
            rec.record(
                t_s=float(i) * 0.01,
                tick=i,
                action=_action(),
                decision=_deploy_decision(),
                observed_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
                mission_phase=MissionPhase.CRUISE.value,
                primary_attack_family=None,
                campaign_id=None,
            )
        rec.close()
        df = pl.read_parquet(out)
        assert df.height == 7


class TestSchema:
    def test_all_expected_columns_present(self, tmp_path):
        out = tmp_path / "out.parquet"
        rec = Recorder(out)
        rec.record(
            t_s=0.0,
            tick=0,
            action=_action(),
            decision=_deploy_decision(),
            observed_cost=CostVector(0.01, 2.0, 0.5, 48.0, 0.01, 0.1),
            mission_phase=MissionPhase.CRUISE.value,
            primary_attack_family=None,
            campaign_id=None,
        )
        rec.close()
        df = pl.read_parquet(out)
        assert set(df.columns) == set(DeploymentRow.__dataclass_fields__.keys())
