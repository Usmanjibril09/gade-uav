"""2D continuous grid with NetworkX-backed neighbor queries.

The world is a 750x750 m square (per blueprint §14). Drones live at
continuous (x, y) with a scalar altitude z. RF adjacency is computed
on demand from euclidean distance; NetworkX stores the current
snapshot for graph-based queries (shortest path, connected components).

Grid does NOT own drone state — it accepts positions and returns
neighbor lists. Drone objects live in DroneRegistry (drone.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class Position2D:
    x: float
    y: float
    z: float = 0.0  # scalar altitude, not a graph dimension

    def distance_xy(self, other: "Position2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class Grid:
    """750x750 m 2D world.

    Parameters
    ----------
    width_m, height_m : float
        World extents. Positions outside [0, width_m] x [0, height_m]
        raise ValueError on register.
    rf_range_m : float
        Two drones are RF-adjacent iff distance_xy <= rf_range_m.
        Default 250 m matches typical 2.4 GHz UAV telemetry range.
    """

    def __init__(
        self,
        width_m: float = 750.0,
        height_m: float = 750.0,
        rf_range_m: float = 250.0,
    ) -> None:
        if width_m <= 0 or height_m <= 0:
            raise ValueError("width_m and height_m must be > 0")
        if rf_range_m <= 0:
            raise ValueError("rf_range_m must be > 0")
        self.width_m = width_m
        self.height_m = height_m
        self.rf_range_m = rf_range_m
        self._positions: dict[str, Position2D] = {}
        self._graph = nx.Graph()

    def contains(self, pos: Position2D) -> bool:
        return 0.0 <= pos.x <= self.width_m and 0.0 <= pos.y <= self.height_m

    def register(self, drone_id: str, pos: Position2D) -> None:
        if not self.contains(pos):
            raise ValueError(
                f"position {pos} outside grid "
                f"[0,{self.width_m}]x[0,{self.height_m}]"
            )
        if drone_id in self._positions:
            raise ValueError(f"drone_id {drone_id!r} already registered")
        self._positions[drone_id] = pos
        self._graph.add_node(drone_id, pos=pos)
        self._recompute_edges_for(drone_id)

    def update_position(self, drone_id: str, pos: Position2D) -> None:
        if drone_id not in self._positions:
            raise KeyError(f"drone_id {drone_id!r} not registered")
        if not self.contains(pos):
            raise ValueError(f"position {pos} outside grid")
        self._positions[drone_id] = pos
        self._graph.nodes[drone_id]["pos"] = pos
        # Drop and recompute this node's edges; O(N) per move, fine for N<=20.
        self._graph.remove_edges_from(list(self._graph.edges(drone_id)))
        self._recompute_edges_for(drone_id)

    def _recompute_edges_for(self, drone_id: str) -> None:
        pos = self._positions[drone_id]
        for other_id, other_pos in self._positions.items():
            if other_id == drone_id:
                continue
            if pos.distance_xy(other_pos) <= self.rf_range_m:
                self._graph.add_edge(drone_id, other_id)

    def neighbors(self, drone_id: str) -> list[str]:
        if drone_id not in self._positions:
            raise KeyError(f"drone_id {drone_id!r} not registered")
        return list(self._graph.neighbors(drone_id))

    def position_of(self, drone_id: str) -> Position2D:
        return self._positions[drone_id]

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def __len__(self) -> int:
        return len(self._positions)
