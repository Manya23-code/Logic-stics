"""
traffic_simulator.py — Generates realistic, evolving traffic conditions
on a road network graph for the digital twin simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Disruption:
    """Represents a traffic disruption event."""
    node_id: int
    severity: float          # 0.0 = total block, 1.0 = no effect
    radius: int              # how many hops the disruption affects
    duration_steps: int      # how many sim steps it lasts
    remaining_steps: int = 0
    event_type: str = "accident"   # accident | weather | construction | congestion

    def __post_init__(self):
        if self.remaining_steps == 0:
            self.remaining_steps = self.duration_steps


class TrafficSimulator:
    """
    Maintains a live traffic state over a graph and evolves it with
    daily/weekly periodicity + random disruptions.
    """

    def __init__(self,
                 num_nodes: int,
                 adj_matrix: np.ndarray,
                 steps_per_day: int = 288,
                 seed: int = 42):
        self.num_nodes    = num_nodes
        self.adj          = adj_matrix
        self.steps_per_day = steps_per_day
        self.rng          = np.random.default_rng(seed)

        # Current traffic speed per node (km/h)
        self.base_speed   = self.rng.uniform(30, 80, size=num_nodes).astype(np.float32)
        self.current_speed = self.base_speed.copy()

        # Rolling history buffer for the GNN (lookback window)
        self.history_buffer: list[np.ndarray] = []
        self.max_history   = 60  # keep last 60 steps (~5 hours at 5-min intervals)

        # Active disruptions
        self.disruptions: list[Disruption] = []

        # Internal clock
        self.step = 0

    @property
    def time_of_day(self) -> float:
        """Fractional time of day [0, 1)."""
        return (self.step % self.steps_per_day) / self.steps_per_day

    @property
    def day_of_week(self) -> int:
        """Day of week [0=Mon … 6=Sun]."""
        return (self.step // self.steps_per_day) % 7

    def _daily_pattern(self) -> np.ndarray:
        """Sinusoidal rush-hour pattern (dips at 8AM and 6PM)."""
        tod = self.time_of_day
        # Two rush hours
        morning_rush = np.exp(-((tod - 0.333) ** 2) / 0.005)   # ~8 AM
        evening_rush = np.exp(-((tod - 0.75) ** 2) / 0.008)    # ~6 PM
        # Speed multiplier: 1.0 = free flow, lower = congested
        congestion = 1.0 - 0.4 * morning_rush - 0.35 * evening_rush
        # Weekend boost
        if self.day_of_week >= 5:
            congestion = congestion * 0.5 + 0.5
        return np.full(self.num_nodes, congestion, dtype=np.float32)

    def inject_disruption(self, node_id: int,
                          severity: float = 0.2,
                          radius: int = 3,
                          duration_steps: int = 24,
                          event_type: str = "accident"):
        """Add a disruption centered at node_id."""
        d = Disruption(
            node_id=node_id,
            severity=severity,
            radius=radius,
            duration_steps=duration_steps,
            event_type=event_type
        )
        self.disruptions.append(d)
        return d

    def _apply_disruptions(self, speeds: np.ndarray) -> np.ndarray:
        """Apply active disruptions with spatial decay."""
        active = []
        for d in self.disruptions:
            if d.remaining_steps <= 0:
                continue

            # BFS to find nodes within radius
            visited = {d.node_id: 0}
            queue   = [d.node_id]
            while queue:
                current = queue.pop(0)
                depth   = visited[current]
                if depth >= d.radius:
                    continue
                # Find neighbors from adjacency
                neighbors = np.where(self.adj[current] > 0)[0]
                for nb in neighbors:
                    if nb not in visited:
                        visited[nb] = depth + 1
                        queue.append(nb)

            # Apply severity with spatial decay
            for node, depth in visited.items():
                decay = d.severity ** (1.0 / (depth + 1))
                speeds[node] *= decay

            d.remaining_steps -= 1
            if d.remaining_steps > 0:
                active.append(d)

        self.disruptions = active
        return speeds

    def tick(self) -> np.ndarray:
        """Advance one simulation step. Returns current speed array."""
        pattern = self._daily_pattern()
        noise   = self.rng.normal(1.0, 0.05, size=self.num_nodes).astype(np.float32)

        speeds = self.base_speed * pattern * noise
        speeds = self._apply_disruptions(speeds)
        speeds = np.clip(speeds, 2.0, 120.0)

        self.current_speed = speeds
        self.history_buffer.append(speeds.copy())
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

        self.step += 1
        return speeds

    def get_history_tensor(self, lookback: int = 12) -> Optional[np.ndarray]:
        """
        Return the last `lookback` steps as a tensor (1, lookback, N, 1)
        ready for GNN inference. Returns None if not enough history.
        """
        if len(self.history_buffer) < lookback:
            return None
        window = np.stack(self.history_buffer[-lookback:], axis=0)  # (T, N)
        return window[np.newaxis, :, :, np.newaxis]  # (1, T, N, 1)

    def get_state_snapshot(self) -> dict:
        """Return the current simulation state as a JSON-serializable dict."""
        return {
            "step": self.step,
            "time_of_day": round(self.time_of_day * 24, 2),  # hours
            "day_of_week": self.day_of_week,
            "speeds": self.current_speed.tolist(),
            "disruptions": [
                {
                    "node_id": d.node_id,
                    "severity": d.severity,
                    "remaining_steps": d.remaining_steps,
                    "event_type": d.event_type,
                }
                for d in self.disruptions
            ],
        }
