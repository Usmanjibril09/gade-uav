from pathlib import Path

import pytest

from gade_uav.sim.adversary import (
    AttackEvent,
    EventKind,
    PoissonCampaignAdversary,
    ScriptedAdversary,
)
from gade_uav.sim.clock import SimClock
from gade_uav.sim.drone import MissionDrone
from gade_uav.sim.grid import Grid, Position2D
from gade_uav.sim.loop import SimulationLoop

CONFIG = Path(__file__).parents[2] / "configs" / "adversary.yaml"


class TestScriptedAdversary:
    def test_empty_events(self):
        adv = ScriptedAdversary(events=[])
        assert adv.poll(0.0, 0.01, ["d1"]) == []

    def test_fires_event_in_window(self):
        e = AttackEvent(
            t_s=0.005,
            kind=EventKind.CAMPAIGN_START,
            attack_class="DDoS",
            canonical_family="DDoS",
            source_id="adv",
            target_drone_id="d1",
            campaign_id="c1",
        )
        adv = ScriptedAdversary(events=[e])
        assert adv.poll(0.0, 0.01, ["d1"]) == [e]

    def test_does_not_fire_after_window(self):
        e = AttackEvent(
            t_s=0.02,
            kind=EventKind.CAMPAIGN_START,
            attack_class="DDoS",
            canonical_family="DDoS",
            source_id="adv",
            target_drone_id="d1",
            campaign_id="c1",
        )
        adv = ScriptedAdversary(events=[e])
        assert adv.poll(0.0, 0.01, ["d1"]) == []

    def test_cursor_advances(self):
        events = [
            AttackEvent(
                t_s=t,
                kind=EventKind.CAMPAIGN_PACKET,
                attack_class="DoS",
                canonical_family="DoS",
                source_id="adv",
                target_drone_id="d1",
                campaign_id="c1",
                pkt_index=i + 1,
            )
            for i, t in enumerate([0.001, 0.005, 0.015, 0.025])
        ]
        adv = ScriptedAdversary(events=events)
        assert len(adv.poll(0.0, 0.01, ["d1"])) == 2
        assert len(adv.poll(0.01, 0.01, ["d1"])) == 1
        assert len(adv.poll(0.02, 0.01, ["d1"])) == 1

    def test_non_monotonic_events_rejected(self):
        e1 = AttackEvent(
            t_s=0.05,
            kind=EventKind.CAMPAIGN_START,
            attack_class="X",
            canonical_family="X",
            source_id="a",
            target_drone_id="d",
            campaign_id="c",
        )
        e2 = AttackEvent(
            t_s=0.01,
            kind=EventKind.CAMPAIGN_START,
            attack_class="X",
            canonical_family="X",
            source_id="a",
            target_drone_id="d",
            campaign_id="c",
        )
        with pytest.raises(ValueError):
            ScriptedAdversary(events=[e1, e2])


class TestPoissonAdversaryConstruction:
    def test_loads_config(self):
        adv = PoissonCampaignAdversary(CONFIG, seed=0)
        assert adv.active_campaign_count == 0

    def test_bad_rate_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "episode_duration_s: 120.0\n"
            "classes:\n"
            "  DoS:\n"
            "    campaign_rate_per_s: 0.0\n"
            "    intra_pkt_rate_per_s: 100.0\n"
            "    campaign_duration_s: 1.0\n"
            "    canonical_family: DoS\n"
        )
        with pytest.raises(ValueError):
            PoissonCampaignAdversary(bad, seed=0)

    def test_no_targets_returns_empty(self):
        adv = PoissonCampaignAdversary(CONFIG, seed=0)
        assert adv.poll(0.0, 0.01, []) == []


class TestPoissonAdversaryStatistics:
    """Statistical checks: campaign arrivals ~ Poisson process."""

    def test_campaign_count_matches_expected_rate(self):
        """Sum of per-class rates over 120s episode should match count."""
        adv = PoissonCampaignAdversary(CONFIG, seed=42)
        dt = 0.01
        n_ticks = 12_000  # 120s
        starts = 0
        for i in range(n_ticks):
            events = adv.poll(now_s=i * dt, dt_s=dt, targetable_drone_ids=["d1", "d2"])
            starts += sum(1 for e in events if e.kind == EventKind.CAMPAIGN_START)
        # Expected: sum of campaign_rate_per_s * 120s over all classes.
        # From config: 12 classes, rates 0.008-0.02 -> sum ~= 0.188/s
        #   E[starts] ~= 0.188 * 120 = ~22.5
        # Allow ±40% for a single seed; multi-seed convergence tested elsewhere.
        assert 12 <= starts <= 40, f"expected ~22 campaign starts, got {starts}"

    def test_seed_determinism(self):
        adv_a = PoissonCampaignAdversary(CONFIG, seed=7)
        adv_b = PoissonCampaignAdversary(CONFIG, seed=7)
        dt = 0.01
        for i in range(1000):
            ea = adv_a.poll(i * dt, dt, ["d1", "d2", "d3"])
            eb = adv_b.poll(i * dt, dt, ["d1", "d2", "d3"])
            assert len(ea) == len(eb)
            for a, b in zip(ea, eb):
                assert a == b

    def test_different_seeds_differ(self):
        adv_a = PoissonCampaignAdversary(CONFIG, seed=1)
        adv_b = PoissonCampaignAdversary(CONFIG, seed=2)
        dt = 0.01
        events_a, events_b = [], []
        for i in range(2000):
            events_a += adv_a.poll(i * dt, dt, ["d1"])
            events_b += adv_b.poll(i * dt, dt, ["d1"])
        # Same target only -> can't diverge on target; check timing.
        starts_a = [e.t_s for e in events_a if e.kind == EventKind.CAMPAIGN_START]
        starts_b = [e.t_s for e in events_b if e.kind == EventKind.CAMPAIGN_START]
        assert starts_a != starts_b


class TestCampaignLifecycle:
    def test_start_end_and_packets(self):
        adv = PoissonCampaignAdversary(CONFIG, seed=3)
        dt = 0.01
        starts, ends, packets = 0, 0, 0
        campaign_ids_started, campaign_ids_ended = set(), set()
        for i in range(12_000):
            for e in adv.poll(i * dt, dt, ["d1", "d2"]):
                if e.kind == EventKind.CAMPAIGN_START:
                    starts += 1
                    campaign_ids_started.add(e.campaign_id)
                elif e.kind == EventKind.CAMPAIGN_END:
                    ends += 1
                    campaign_ids_ended.add(e.campaign_id)
                elif e.kind == EventKind.CAMPAIGN_PACKET:
                    packets += 1
                    assert e.pkt_index >= 1
        # Ended campaigns are a subset of started (some may still be active).
        assert campaign_ids_ended.issubset(campaign_ids_started)
        assert starts >= ends
        # Started campaigns generate packets on average.
        if starts > 0:
            assert packets > 0

    def test_packets_target_matches_start(self):
        adv = PoissonCampaignAdversary(CONFIG, seed=11)
        dt = 0.01
        campaign_target: dict[str, str] = {}
        for i in range(12_000):
            for e in adv.poll(i * dt, dt, ["d1", "d2", "d3", "d4"]):
                if e.kind == EventKind.CAMPAIGN_START:
                    campaign_target[e.campaign_id] = e.target_drone_id
                elif e.kind == EventKind.CAMPAIGN_PACKET:
                    assert campaign_target[e.campaign_id] == e.target_drone_id


class TestLoopIntegration:
    def test_loop_records_events_via_adversary(self):
        loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
        loop.add_drone(MissionDrone(drone_id="m1", position=Position2D(100, 100)))
        e = AttackEvent(
            t_s=0.005,
            kind=EventKind.CAMPAIGN_START,
            attack_class="DDoS",
            canonical_family="DDoS",
            source_id="adv",
            target_drone_id="m1",
            campaign_id="c1",
        )
        loop.attach_adversary(ScriptedAdversary(events=[e]))
        loop.step()
        assert loop.last_events == [e]

    def test_loop_without_adversary_has_no_events(self):
        loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
        loop.add_drone(MissionDrone(drone_id="m1", position=Position2D(0, 0)))
        loop.step()
        assert loop.last_events == []

    def test_no_drones_means_no_poll(self):
        loop = SimulationLoop(grid=Grid(), clock=SimClock(dt_s=0.01))
        # Adversary attached but no drones registered -> loop must not poll.
        e = AttackEvent(
            t_s=0.005,
            kind=EventKind.CAMPAIGN_START,
            attack_class="X",
            canonical_family="X",
            source_id="a",
            target_drone_id="d1",
            campaign_id="c1",
        )
        loop.attach_adversary(ScriptedAdversary(events=[e]))
        loop.step()
        assert loop.last_events == []
