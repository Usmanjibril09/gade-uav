"""Two-level Poisson adversary for the simulator.

Level 1: campaign arrivals per attack class (Poisson process, rate λ_c).
Level 2: within an active campaign, packets fire at intra_pkt_rate until
         campaign_duration_s expires.

An AttackEvent is emitted every time a campaign starts OR a new packet
is generated inside an active campaign. Downstream consumers (Effector,
Recorder, Shield hook) decide what to do with each event.

Rates are calibrated in configs/adversary.yaml; see rationale there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml


class EventKind(str, Enum):
    CAMPAIGN_START = "CAMPAIGN_START"
    CAMPAIGN_PACKET = "CAMPAIGN_PACKET"
    CAMPAIGN_END = "CAMPAIGN_END"


@dataclass(frozen=True)
class AttackEvent:
    t_s: float
    kind: EventKind
    attack_class: str  # e.g. "DDoS", "Jamming"
    canonical_family: str  # e.g. "DDoS" (from labels.py)
    source_id: str  # attacker id (opaque)
    target_drone_id: str  # victim drone in the sim
    campaign_id: str  # groups PACKETs to their START
    pkt_index: int = 0  # 0 for START/END, 1..N for PACKET


@dataclass
class _ClassConfig:
    campaign_rate_per_s: float
    intra_pkt_rate_per_s: float
    campaign_duration_s: float
    canonical_family: str

    def __post_init__(self) -> None:
        if self.campaign_rate_per_s <= 0:
            raise ValueError("campaign_rate_per_s must be > 0")
        if self.intra_pkt_rate_per_s <= 0:
            raise ValueError("intra_pkt_rate_per_s must be > 0")
        if self.campaign_duration_s <= 0:
            raise ValueError("campaign_duration_s must be > 0")


@dataclass
class _ActiveCampaign:
    campaign_id: str
    attack_class: str
    canonical_family: str
    source_id: str
    target_drone_id: str
    start_s: float
    end_s: float  # deterministic end for this campaign
    next_pkt_s: float
    intra_pkt_rate: float
    pkt_index: int = 0


class Adversary(Protocol):
    """Contract for adversary implementations.

    poll(now_s, dt_s, targetable_drone_ids) is called by SimulationLoop
    every tick; it returns any AttackEvents that fall within
    [now_s, now_s + dt_s).
    """

    def poll(
        self, now_s: float, dt_s: float, targetable_drone_ids: list[str]
    ) -> list[AttackEvent]: ...


class PoissonCampaignAdversary:
    """Two-level Poisson adversary driven by configs/adversary.yaml.

    Deterministic under a fixed seed.
    """

    def __init__(
        self,
        config_path: str | Path,
        seed: int,
        source_id: str = "adv_0",
    ) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text())
        self._episode_s = float(cfg["episode_duration_s"])
        self._classes: dict[str, _ClassConfig] = {
            name: _ClassConfig(**params) for name, params in cfg["classes"].items()
        }
        if not self._classes:
            raise ValueError("adversary config has no classes")
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)  # for target selection
        self._source_id = source_id
        # Next scheduled campaign start per class (exponential inter-arrivals).
        self._next_campaign_at: dict[str, float] = {
            name: float(self._rng.exponential(1.0 / c.campaign_rate_per_s))
            for name, c in self._classes.items()
        }
        self._active: list[_ActiveCampaign] = []
        self._campaign_counter = 0

    def _new_campaign_id(self, attack_class: str) -> str:
        self._campaign_counter += 1
        return f"{attack_class}_{self._campaign_counter}"

    def _pick_target(self, targetable: list[str]) -> str:
        # Uniform random target; can be extended later.
        return self._py_rng.choice(targetable)

    def poll(
        self, now_s: float, dt_s: float, targetable_drone_ids: list[str]
    ) -> list[AttackEvent]:
        if not targetable_drone_ids:
            return []
        window_end = now_s + dt_s
        events: list[AttackEvent] = []

        # (1) Fire any scheduled campaign starts that fall in the window.
        for attack_class, cfg in self._classes.items():
            while self._next_campaign_at[attack_class] < window_end:
                start_s = self._next_campaign_at[attack_class]
                target = self._pick_target(targetable_drone_ids)
                cid = self._new_campaign_id(attack_class)
                # Deterministic duration draw at start; recorded on the campaign.
                duration = float(self._rng.exponential(cfg.campaign_duration_s))
                # First packet: exponential inter-arrival at intra rate.
                first_gap = float(self._rng.exponential(1.0 / cfg.intra_pkt_rate_per_s))
                camp = _ActiveCampaign(
                    campaign_id=cid,
                    attack_class=attack_class,
                    canonical_family=cfg.canonical_family,
                    source_id=self._source_id,
                    target_drone_id=target,
                    start_s=start_s,
                    end_s=start_s + duration,
                    next_pkt_s=start_s + first_gap,
                    intra_pkt_rate=cfg.intra_pkt_rate_per_s,
                )
                self._active.append(camp)
                events.append(
                    AttackEvent(
                        t_s=start_s,
                        kind=EventKind.CAMPAIGN_START,
                        attack_class=attack_class,
                        canonical_family=cfg.canonical_family,
                        source_id=self._source_id,
                        target_drone_id=target,
                        campaign_id=cid,
                    )
                )
                # Schedule next campaign of this class.
                gap = float(self._rng.exponential(1.0 / cfg.campaign_rate_per_s))
                self._next_campaign_at[attack_class] = start_s + gap

        # (2) For each active campaign, emit packet events up to window_end
        #     or campaign end, whichever comes first, then emit CAMPAIGN_END.
        still_active: list[_ActiveCampaign] = []
        for camp in self._active:
            horizon = min(window_end, camp.end_s)
            while camp.next_pkt_s < horizon:
                camp.pkt_index += 1
                events.append(
                    AttackEvent(
                        t_s=camp.next_pkt_s,
                        kind=EventKind.CAMPAIGN_PACKET,
                        attack_class=camp.attack_class,
                        canonical_family=camp.canonical_family,
                        source_id=camp.source_id,
                        target_drone_id=camp.target_drone_id,
                        campaign_id=camp.campaign_id,
                        pkt_index=camp.pkt_index,
                    )
                )
                camp.next_pkt_s += float(
                    self._rng.exponential(1.0 / camp.intra_pkt_rate)
                )
            if camp.end_s <= window_end:
                events.append(
                    AttackEvent(
                        t_s=camp.end_s,
                        kind=EventKind.CAMPAIGN_END,
                        attack_class=camp.attack_class,
                        canonical_family=camp.canonical_family,
                        source_id=camp.source_id,
                        target_drone_id=camp.target_drone_id,
                        campaign_id=camp.campaign_id,
                        pkt_index=camp.pkt_index,
                    )
                )
            else:
                still_active.append(camp)
        self._active = still_active

        # Return events in chronological order for downstream determinism.
        events.sort(key=lambda e: (e.t_s, e.kind.value, e.pkt_index))
        return events

    @property
    def active_campaign_count(self) -> int:
        return len(self._active)


@dataclass
class ScriptedAdversary:
    """Deterministic scripted adversary for unit tests.

    events: pre-built list of AttackEvents in chronological order.
    poll() returns events whose t_s falls in [now_s, now_s + dt_s).
    """

    events: list[AttackEvent] = field(default_factory=list)
    _cursor: int = 0

    def __post_init__(self) -> None:
        # Verify chronological order once.
        for a, b in zip(self.events, self.events[1:]):
            if b.t_s < a.t_s:
                raise ValueError(
                    "ScriptedAdversary events must be non-decreasing in t_s"
                )

    def poll(
        self, now_s: float, dt_s: float, targetable_drone_ids: list[str]
    ) -> list[AttackEvent]:
        window_end = now_s + dt_s
        out: list[AttackEvent] = []
        while (
            self._cursor < len(self.events)
            and self.events[self._cursor].t_s < window_end
        ):
            out.append(self.events[self._cursor])
            self._cursor += 1
        return out
