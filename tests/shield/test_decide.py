"""Integration tests for the composite Shield.decide() procedure (spec §7)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    COST_DIMENSIONS,
    CostVector,
    DecoyIdentity,
    Effector,
    GenMeta,
    MissionPhase,
    ShieldContext,
)
from gade_uav.shield.conformal_gate import (
    BudgetSchedule,
    MultiDimensionalConformalGate,
)
from gade_uav.shield.decide import Shield, Verdict
from gade_uav.shield.fallback import FallbackAction, FallbackPolicy
from gade_uav.shield.ltl_monitor import SafetyMonitor
from gade_uav.shield.timed_monitors import (
    BurstLimitMonitor,
    LatencyDeadlineMonitor,
    QuietPeriodMonitor,
    TimedMonitors,
)


REPO_ROOT = Path(__file__).parents[2]
YAML_PATH = REPO_ROOT / "configs" / "fallback_policy.yaml"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _cost(v: float = 0.1) -> CostVector:
    return CostVector(v, v * 10, v * 5, v * 100, v * 5, v * 5)


def _meta(conf: float = 0.9, tid: str | None = None) -> GenMeta:
    return GenMeta(
        confidence=conf,
        schema_version="v1",
        generation_context_hash="deadbeef",
        template_id=tid,
    )


def _mav(
    msg_id: int = 0,
    command: int | None = None,
    target: Effector = Effector.DECOY_TELEMETRY_CHANNEL,
    cost: float = 0.01,
    conf: float = 0.9,
) -> Action:
    payload = {"msg_id": msg_id, "fields": {}}
    if command is not None:
        payload["command"] = command
    return Action(
        type=ActionType.MAVLINK_RESPONSE,
        payload=payload,
        target_iface=target,
        predicted_cost=_cost(cost),
        gen_meta=_meta(conf=conf),
    )


def _ctx(phase: MissionPhase = MissionPhase.CRUISE) -> ShieldContext:
    return ShieldContext(
        mission_phase=phase,
        signing_active=False,
        decoy_key_active=False,
        known_decoys={},
    )


def _generous_budgets() -> BudgetSchedule:
    return BudgetSchedule(
        base_budget={k: 10_000.0 for k in COST_DIMENSIONS},
        phase_slack={p: 1.0 for p in MissionPhase},
    )


def _tight_cpu_budget() -> BudgetSchedule:
    base = {k: 10_000.0 for k in COST_DIMENSIONS}
    base["cpu"] = 0.005  # will fail on our default cost=0.01
    return BudgetSchedule(base_budget=base, phase_slack={p: 1.0 for p in MissionPhase})


def _shield(
    budgets: BudgetSchedule | None = None,
    ltl_repair=None,
    cost_downscale=None,
    calibration_size: int = 300,
    cost_estimator=None,
) -> Shield:
    budgets = budgets or _generous_budgets()
    gate = MultiDimensionalConformalGate(
        alpha_target=0.10,
        aci_gamma=0.005,
        min_calibration_size=200,
        budgets=budgets,
    )
    # Seed calibration so the gate is ready to decide. Zero residuals
    # means UCB == predicted, keeping decisions deterministic.
    for _ in range(calibration_size):
        gate.add_calibration(
            predicted={k: 0.0 for k in COST_DIMENSIONS},
            observed={k: 0.0 for k in COST_DIMENSIONS},
        )
    return Shield(
        ltl_monitor=SafetyMonitor(),
        gate=gate,
        timed=TimedMonitors(
            burst=BurstLimitMonitor(window_s=1.0, max_emissions=100),
            latency=LatencyDeadlineMonitor(deadline_ms=1_000.0),
            quiet=QuietPeriodMonitor(quiet_ms=200.0),
        ),
        fallback_policy=FallbackPolicy.from_yaml(YAML_PATH),
        ltl_repair=ltl_repair,
        cost_downscale=cost_downscale,
        cost_estimator=cost_estimator,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestDeploy:
    @pytest.mark.parametrize(
        "metadata",
        [
            {"mission_state_trusted": False},
            {"mission_state_expires_at_s": 0.5},
            {"mission_state_expires_at_s": 1.0},
            {"mission_state_expires_at_s": float("nan")},
        ],
    )
    def test_untrusted_or_delayed_mission_context_rejected(self, metadata):
        decision = _shield().decide(_mav(), replace(_ctx(), **metadata), now_s=1.0)
        assert decision.verdict == Verdict.REJECT
        assert decision.action_out is None
        assert decision.fallback is not None
        assert decision.reason == "Mission state is untrusted or expired"

    @pytest.mark.parametrize("confidence", [0.1, float("nan")])
    def test_uncertain_context_restricts_mavlink_candidates(self, confidence):
        ctx = replace(_ctx(), cyber_state_confidence=confidence, mission_state_expires_at_s=2.0)
        decision = _shield().decide(_mav(), ctx, now_s=1.0)
        assert decision.verdict == Verdict.REJECT
        assert decision.ltl_result.failing_property == "S7"

    def test_fresh_confident_context_is_eligible_for_deployment(self):
        ctx = replace(_ctx(), cyber_state_confidence=0.95, mission_state_expires_at_s=2.0)
        assert _shield().decide(_mav(), ctx, now_s=1.0).verdict == Verdict.DEPLOY

    def test_clean_action_deployed(self):
        s = _shield()
        d = s.decide(_mav(), _ctx(), now_s=0.0)
        assert d.verdict == Verdict.DEPLOY
        assert d.action_out is not None
        assert d.fallback is None
        assert d.layer_failed is None
        assert d.repair_attempted is False


# ---------------------------------------------------------------------------
# L1: forbidden effector
# ---------------------------------------------------------------------------


class TestLayer1Rejects:
    def test_forbidden_effector_rejected(self):
        s = _shield()
        a = _mav(target=Effector.FLIGHT_CONTROLLER_BUS)
        d = s.decide(a, _ctx(), now_s=0.0, primary_attack_family="T7_MITM")
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L1"
        assert d.fallback == FallbackAction.HAND_OFF_TO_IDS


# ---------------------------------------------------------------------------
# L2: LTL violation and repair
# ---------------------------------------------------------------------------


class TestLayer2Rejects:
    def test_flight_critical_rejected(self):
        s = _shield()
        # COMMAND_LONG + DO_SET_MODE = S1 violation
        a = _mav(msg_id=76, command=176)
        d = s.decide(a, _ctx(), now_s=0.0, primary_attack_family="T7_MITM")
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L2"
        assert d.ltl_result is not None
        assert d.ltl_result.failing_property == "S1"

    def test_ltl_repair_produces_modify(self):
        """A repair function that rewrites the payload into a benign
        heartbeat should let the shield emit MODIFY."""

        def repair(action, ctx, ltl_res):
            return _mav(msg_id=0)  # replace with heartbeat

        s = _shield(ltl_repair=repair)
        a = _mav(msg_id=76, command=176)
        d = s.decide(a, _ctx(), now_s=0.0)
        assert d.verdict == Verdict.MODIFY
        assert d.repair_attempted is True
        assert d.repair_succeeded is True
        assert d.action_out is not None
        assert d.action_out.payload["msg_id"] == 0

    def test_bad_repair_rejected(self):
        """A repair function that returns a still-violating action must
        result in REJECT (only one repair attempt per spec §7)."""

        def bad_repair(action, ctx, ltl_res):
            return _mav(msg_id=76, command=246)  # REBOOT_SHUTDOWN, still S1

        s = _shield(ltl_repair=bad_repair)
        a = _mav(msg_id=76, command=176)
        d = s.decide(a, _ctx(), now_s=0.0)
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L2"
        assert d.repair_attempted is True
        assert d.repair_succeeded is False

    def test_repair_returning_none_rejects(self):
        def refuse(action, ctx, ltl_res):
            return None

        s = _shield(ltl_repair=refuse)
        a = _mav(msg_id=76, command=176)
        d = s.decide(a, _ctx(), now_s=0.0)
        assert d.verdict == Verdict.REJECT
        assert d.repair_attempted is True
        assert d.repair_succeeded is False


# ---------------------------------------------------------------------------
# L3: conformal gate
# ---------------------------------------------------------------------------


class TestLayer3Rejects:
    @pytest.mark.parametrize("failure", [RuntimeError("predictor unavailable"), {}, {"cpu": float("nan")}])
    @pytest.mark.parametrize("during_repair", [False, True])
    def test_prediction_failure_rejects_with_fallback(self, failure, during_repair):
        estimates = {dimension: 0.01 for dimension in COST_DIMENSIONS}
        if isinstance(failure, dict) and failure:
            failure = {**estimates, **failure}
        responses = [estimates, failure] if during_repair else [failure]
        estimator = Mock()
        estimator.estimate.side_effect = responses
        shield = _shield(
            budgets=_tight_cpu_budget() if during_repair else None,
            cost_downscale=lambda action, gate: _mav(cost=0.0001),
            cost_estimator=estimator,
        )

        decision = shield.decide(_mav(), _ctx(), now_s=1.0)

        assert decision.verdict == Verdict.REJECT
        assert decision.layer_failed == "L3"
        assert decision.action_out is None
        assert decision.fallback is not None
        assert decision.gate_decision is None
        assert decision.repair_attempted == during_repair
        assert estimator.estimate.call_count == len(responses)

    def test_insufficient_calibration_rejects(self):
        s = _shield(calibration_size=50)  # below min 200
        d = s.decide(_mav(), _ctx(), now_s=0.0)
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L3"
        assert d.gate_decision is not None
        assert d.gate_decision.insufficient_calibration is True

    def test_over_budget_rejects(self):
        s = _shield(budgets=_tight_cpu_budget())
        d = s.decide(_mav(cost=0.01), _ctx(), now_s=0.0)
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L3"
        assert d.gate_decision is not None
        assert "cpu" in d.gate_decision.failing_dimensions

    def test_downscale_produces_modify(self):
        def downscale(action, gate_res):
            # Return an action with drastically lower cost.
            return _mav(cost=0.0001)

        s = _shield(budgets=_tight_cpu_budget(), cost_downscale=downscale)
        d = s.decide(_mav(cost=0.01), _ctx(), now_s=0.0)
        assert d.verdict == Verdict.MODIFY
        assert d.repair_succeeded is True

    def test_downscale_still_over_rejects(self):
        def bad_downscale(action, gate_res):
            return _mav(cost=0.02)  # even higher cost

        s = _shield(budgets=_tight_cpu_budget(), cost_downscale=bad_downscale)
        d = s.decide(_mav(cost=0.01), _ctx(), now_s=0.0)
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L3"


# ---------------------------------------------------------------------------
# L4: timed monitors
# ---------------------------------------------------------------------------


class TestLayer4Rejects:
    def _telemetry(self, decoy_id: str = "d1") -> Action:
        return Action(
            type=ActionType.DECOY_TELEMETRY,
            payload={"decoy_id": decoy_id, "fields": {}},
            target_iface=Effector.DECOY_TELEMETRY_CHANNEL,
            predicted_cost=_cost(0.01),
            gen_meta=_meta(),
        )

    def _ctx_with_decoy(self, decoy_id: str = "d1") -> ShieldContext:
        """Context that has registered the decoy identity, so S6 does
        not trip on DECOY_TELEMETRY emissions. Empty payload fields
        means no per-field jump check applies."""
        return ShieldContext(
            mission_phase=MissionPhase.CRUISE,
            signing_active=False,
            decoy_key_active=False,
            known_decoys={
                decoy_id: DecoyIdentity(
                    decoy_id=decoy_id,
                    tau_alt_m=5.0,
                    tau_pos_m=15.0,
                    tau_bat_pct=2.0,
                    tau_att_rad_s=3.14,
                ),
            },
        )

    def test_burst_limit_rejects(self):
        s = _shield()
        # Replace burst monitor with a limit of 2 for this test.
        s._timed.burst = BurstLimitMonitor(window_s=1.0, max_emissions=2)
        a = self._telemetry()
        ctx = self._ctx_with_decoy()
        for i in range(2):
            d = s.decide(a, ctx, now_s=float(i) * 0.1)
            assert d.verdict == Verdict.DEPLOY, d.reason
        d = s.decide(a, ctx, now_s=0.25)
        assert d.verdict == Verdict.REJECT
        assert d.layer_failed == "L4"

    def test_quiet_period_enforced_after_reject(self):
        s = _shield()
        # Force any reject.
        a = _mav(target=Effector.FLIGHT_CONTROLLER_BUS)
        d1 = s.decide(a, _ctx(), now_s=0.0, primary_attack_family="T7_MITM")
        assert d1.verdict == Verdict.REJECT
        # Immediate follow-up (within 200ms) must reject via L4/T3.
        d2 = s.decide(_mav(), _ctx(), now_s=0.05)
        assert d2.verdict == Verdict.REJECT
        assert d2.layer_failed == "L4"
        assert d2.timed_failures and d2.timed_failures[0].monitor == "T3"

    def test_admission_after_quiet_period(self):
        s = _shield()
        a = _mav(target=Effector.FLIGHT_CONTROLLER_BUS)
        d1 = s.decide(a, _ctx(), now_s=0.0, primary_attack_family="T7_MITM")
        assert d1.verdict == Verdict.REJECT
        # 300ms > 200ms quiet -> admissible.
        d2 = s.decide(_mav(), _ctx(), now_s=0.3)
        assert d2.verdict == Verdict.DEPLOY


# ---------------------------------------------------------------------------
# Fallback selection
# ---------------------------------------------------------------------------


class TestFallbackSelection:
    def test_reject_uses_configured_fallback(self):
        s = _shield()
        # L1 reject with attack family T13 (heartbeat flooding) in cruise
        # -> fallback_policy.yaml maps to RATE_LIMIT_SOURCE.
        a = _mav(target=Effector.FLIGHT_CONTROLLER_BUS)
        d = s.decide(
            a,
            _ctx(MissionPhase.CRUISE),
            now_s=0.0,
            primary_attack_family="T13_MAVLink_Heartbeat_Flooding",
        )
        assert d.verdict == Verdict.REJECT
        assert d.fallback == FallbackAction.RATE_LIMIT_SOURCE

    def test_reject_without_family_uses_unknown_row(self):
        s = _shield()
        a = _mav(target=Effector.FLIGHT_CONTROLLER_BUS)
        d = s.decide(a, _ctx(MissionPhase.CRUISE), now_s=0.0)
        # unknown/cruise in the shipped config -> HAND_OFF_TO_IDS.
        assert d.fallback == FallbackAction.HAND_OFF_TO_IDS


# ---------------------------------------------------------------------------
# Audit-trail invariants
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_deploy_populates_ltl_and_gate(self):
        s = _shield()
        d = s.decide(_mav(), _ctx(), now_s=0.0)
        assert d.verdict == Verdict.DEPLOY
        assert d.ltl_result is not None and d.ltl_result.holds
        assert d.gate_decision is not None and d.gate_decision.admitted

    def test_reject_reason_is_populated(self):
        s = _shield()
        a = _mav(target=Effector.FLIGHT_CONTROLLER_BUS)
        d = s.decide(a, _ctx(), now_s=0.0)
        assert d.reason is not None
        assert "L1 violated" in d.reason
