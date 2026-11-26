"""Deterministic discrete-time clock for the simulator.

dt is fixed at construction; tick() advances by exactly one dt.
Never uses wall-clock time. All monitors and recorders read now_s()
from this object to guarantee reproducibility across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimClock:
    dt_s: float = 0.01  # 10 ms default per Phase 2 lock
    _tick: int = 0

    def __post_init__(self) -> None:
        if self.dt_s <= 0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")

    @property
    def tick(self) -> int:
        return self._tick

    def now_s(self) -> float:
        return self._tick * self.dt_s

    def advance(self) -> float:
        self._tick += 1
        return self.now_s()

    def reset(self) -> None:
        self._tick = 0
