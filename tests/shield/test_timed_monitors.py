"""Tests for timed monitors T-Shield-1/2/3 (spec §6)."""

from __future__ import annotations

import pytest

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    Effector,
    GenMeta,
)
from gade_uav.shield.timed_monitors import (
    BurstLimitMonitor,
    LatencyDeadlineMonitor,
    QuietPeriodMonitor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cost() -> CostVector:
    return CostVector(0.1, 10.0, 1.0, 100.0, 0.5, 0.0)


def _meta() -> GenMeta:
    return GenMeta(
        confidence=0.9, schema_version="v1", generation_context_hash="deadbeef"
    )


def _telemetry(decoy_id: str = "d1") -> Action:
    return Action(
        type=ActionType.DECOY_TELEMETRY,
        payload={"decoy_id": decoy_id, "fields": {}},
        target_iface=Effector.DECOY_TELEMETRY_CHANNEL,
        predicted_cost=_cost(),
        gen_meta=_meta(),
    )


def _mavlink_with_decoy(decoy_id: str, msg_id: int = 0) -> Action:
    return Action(
        type=ActionType.MAVLINK_RESPONSE,
        payload={"msg_id": msg_id, "fields": {}, "decoy_id": decoy_id},
        target_iface=Effector.DECOY_MAVLINK_CHANNEL,
        predicted_cost=_cost(),
        gen_meta=_meta(),
    )


def _mavlink_no_decoy() -> Action:
    return Action(
        type=ActionType.MAVLINK_RESPONSE,
        payload={"msg_id": 0, "fields": {}},
        target_iface=Effector.DECOY_MAVLINK_CHANNEL,
        predicted_cost=_cost(),
        gen_meta=_meta(),
    )


def _network() -> Action:
    return Action(
        type=ActionType.NETWORK_RESPONSE,
        payload={"protocol": "tcp"},
        target_iface=Effector.NETWORK_LAYER_RATE_LIMITER,
        predicted_cost=_cost(),
        gen_meta=_meta(),
    )


# ---------------------------------------------------------------------------
# T-Shield-1: BurstLimitMonitor
# ---------------------------------------------------------------------------


class TestBurstLimitMonitor:
    def test_construction_rejects_bad_params(self):
        with pytest.raises(ValueError):
            BurstLimitMonitor(window_s=0, max_emissions=5)
        with pytest.raises(ValueError):
            BurstLimitMonitor(window_s=1.0, max_emissions=0)

    def test_admits_below_limit(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=3)
        for i in range(3):
            r = m.check(_telemetry(), now_s=float(i) * 0.1)
            assert r.holds
            m.record_admission(_telemetry(), now_s=float(i) * 0.1)

    def test_rejects_at_limit(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=3)
        for i in range(3):
            m.record_admission(_telemetry(), now_s=float(i) * 0.1)
        # 4th within window fails
        r = m.check(_telemetry(), now_s=0.35)
        assert not r.holds
        assert r.monitor == "T1"
        assert "d1" in (r.reason or "")

    def test_window_evicts_old_entries(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=3)
        # Three admissions at t=0, 0.1, 0.2
        for t in (0.0, 0.1, 0.2):
            m.record_admission(_telemetry(), now_s=t)
        # At t=1.5 (all three older than window), a new one must be admitted.
        r = m.check(_telemetry(), now_s=1.5)
        assert r.holds
        assert m.in_window_count("d1", now_s=1.5) == 0

    def test_per_decoy_isolation(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=2)
        m.record_admission(_telemetry("d1"), now_s=0.0)
        m.record_admission(_telemetry("d1"), now_s=0.1)
        # d1 is at capacity; d2 is fresh.
        assert not m.check(_telemetry("d1"), now_s=0.2).holds
        assert m.check(_telemetry("d2"), now_s=0.2).holds

    def test_mavlink_with_decoy_id_tracked(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=2)
        m.record_admission(_mavlink_with_decoy("d1"), now_s=0.0)
        m.record_admission(_mavlink_with_decoy("d1"), now_s=0.1)
        r = m.check(_mavlink_with_decoy("d1"), now_s=0.2)
        assert not r.holds

    def test_actions_without_decoy_id_are_vacuously_admitted(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=1)
        m.record_admission(_mavlink_no_decoy(), now_s=0.0)
        # Second call would still be admitted -- no decoy_id, T1 vacuous.
        r = m.check(_mavlink_no_decoy(), now_s=0.1)
        assert r.holds

    def test_non_decoy_action_types_vacuously_admitted(self):
        m = BurstLimitMonitor(window_s=1.0, max_emissions=1)
        assert m.check(_network(), now_s=0.0).holds
        m.record_admission(_network(), now_s=0.0)
        # No decoy_id, nothing recorded; window count for anything is 0.
        assert m.check(_network(), now_s=0.1).holds


# ---------------------------------------------------------------------------
# T-Shield-2: LatencyDeadlineMonitor
# ---------------------------------------------------------------------------


class TestLatencyDeadlineMonitor:
    def test_construction_rejects_bad_deadline(self):
        with pytest.raises(ValueError):
            LatencyDeadlineMonitor(deadline_ms=0)
        with pytest.raises(ValueError):
            LatencyDeadlineMonitor(deadline_ms=-1)

    def test_under_deadline_admits(self):
        m = LatencyDeadlineMonitor(deadline_ms=5.0)
        r = m.check(elapsed_ms=1.5)
        assert r.holds
        assert m.miss_count == 0

    def test_over_deadline_fails(self):
        m = LatencyDeadlineMonitor(deadline_ms=5.0)
        r = m.check(elapsed_ms=7.2)
        assert not r.holds
        assert r.monitor == "T2"
        assert m.miss_count == 1

    def test_history_accumulates(self):
        m = LatencyDeadlineMonitor(deadline_ms=5.0, history_size=100)
        for t in (0.5, 1.0, 2.0, 3.5, 4.0):
            m.check(elapsed_ms=t)
        assert m.sample_count == 5
        p50 = m.percentile(0.5)
        # median of [0.5,1.0,2.0,3.5,4.0] is 2.0
        assert p50 == pytest.approx(2.0)
        p99 = m.percentile(0.99)
        assert 3.5 <= p99 <= 4.0

    def test_ring_buffer_evicts_when_full(self):
        m = LatencyDeadlineMonitor(deadline_ms=5.0, history_size=3)
        for t in (1.0, 2.0, 3.0, 4.0):
            m.check(elapsed_ms=t)
        assert m.sample_count == 3  # oldest evicted
        # median of last 3 values [2.0, 3.0, 4.0] is 3.0
        assert m.percentile(0.5) == pytest.approx(3.0)

    def test_percentile_bounds(self):
        m = LatencyDeadlineMonitor(deadline_ms=5.0)
        m.check(elapsed_ms=1.0)
        with pytest.raises(ValueError):
            m.percentile(1.5)
        with pytest.raises(ValueError):
            m.percentile(-0.1)

    def test_percentile_empty(self):
        m = LatencyDeadlineMonitor(deadline_ms=5.0)
        assert m.percentile(0.5) == 0.0


# ---------------------------------------------------------------------------
# T-Shield-3: QuietPeriodMonitor
# ---------------------------------------------------------------------------


class TestQuietPeriodMonitor:
    def test_construction_rejects_bad_quiet(self):
        with pytest.raises(ValueError):
            QuietPeriodMonitor(quiet_ms=0)
        with pytest.raises(ValueError):
            QuietPeriodMonitor(quiet_ms=-1)

    def test_admits_before_any_reject(self):
        m = QuietPeriodMonitor(quiet_ms=200.0)
        assert m.check(now_s=0.0).holds
        assert m.check(now_s=100.0).holds

    def test_rejects_within_quiet_period(self):
        m = QuietPeriodMonitor(quiet_ms=200.0)
        m.record_reject(now_s=10.0)
        # 150ms later -> still in quiet period
        r = m.check(now_s=10.15)
        assert not r.holds
        assert r.monitor == "T3"

    def test_admits_after_quiet_period(self):
        m = QuietPeriodMonitor(quiet_ms=200.0)
        m.record_reject(now_s=10.0)
        # 250ms later -> quiet period elapsed
        r = m.check(now_s=10.25)
        assert r.holds

    def test_boundary_at_exact_quiet_admits(self):
        m = QuietPeriodMonitor(quiet_ms=200.0)
        m.record_reject(now_s=10.0)
        # exactly 200ms later: elapsed == quiet_s, not strictly less than
        r = m.check(now_s=10.2)
        assert r.holds

    def test_second_reject_resets_quiet_window(self):
        m = QuietPeriodMonitor(quiet_ms=200.0)
        m.record_reject(now_s=10.0)
        m.record_reject(now_s=10.5)  # a later reject re-arms the window
        r = m.check(now_s=10.65)
        assert not r.holds
