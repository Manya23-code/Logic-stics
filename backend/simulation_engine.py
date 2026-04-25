"""
simulation_engine.py — Orchestrates the full digital twin simulation loop.
"""
import time, threading, json, numpy as np, asyncio
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.graph_builder import build_synthetic_grid
from data.traffic_simulator import TrafficSimulator
from data.dataset_loader import generate_synthetic_traffic
from model.predictor import Predictor
from routing.dynamic_router import DynamicRouter
from routing.fleet_manager import FleetManager


class SimulationEngine:
    """Full digital-twin simulation orchestrator."""

    def __init__(self, num_nodes_side: int = 15, num_vehicles: int = 30):
        n = num_nodes_side
        self.num_nodes = n * n

        # 1) Build graph
        print("[SimEngine] Building graph …")
        self.adj, self.node_features = build_synthetic_grid(n, n, out_dir="data/raw")

        # 2) Generate synthetic training data & train if needed
        if not os.path.exists("data/processed/train.npz"):
            print("[SimEngine] Generating synthetic traffic data …")
            generate_synthetic_traffic(self.num_nodes, num_steps=12 * 288,
                                       out_dir="data/processed")

        # 3) Load predictor
        print("[SimEngine] Loading predictor …")
        self.predictor = Predictor(
            checkpoint_path="model/checkpoints/best_model.pt",
            adj_path="data/raw/graph_data.pkl",
            scaler_path="data/processed/scaler.pkl")

        # 4) Traffic simulator
        self.traffic_sim = TrafficSimulator(self.num_nodes, self.adj)

        # 5) Router
        self.router = DynamicRouter(self.adj, self.node_features)

        # 6) Fleet
        self.fleet = FleetManager(num_vehicles, self.num_nodes, self.router)

        # State
        self.running = False
        self.speed_multiplier = 1  # 1x, 10x, 60x
        self.tick_interval = 1.0   # seconds between ticks at 1x
        self.current_prediction = None
        self.event_log: list[dict] = []
        self.websocket_clients: list = []
        self.step_count = 0

        # Warm up: run a few ticks to populate history
        for _ in range(15):
            self.traffic_sim.tick()

    def set_speed(self, multiplier: int):
        self.speed_multiplier = max(1, min(multiplier, 100))

    def inject_disruption(self, node_id: int, severity: float = 0.2,
                          radius: int = 3, duration: int = 24,
                          event_type: str = "accident") -> dict:
        node_id = node_id % self.num_nodes
        d = self.traffic_sim.inject_disruption(node_id, severity, radius, duration, event_type)
        event = {"type": "disruption_injected", "node_id": node_id,
                 "severity": severity, "event_type": event_type,
                 "step": self.step_count}
        self.event_log.append(event)
        return event

    def _run_tick(self) -> dict:
        """Execute one simulation tick."""
        # 1) Advance traffic
        speeds = self.traffic_sim.tick()
        self.step_count += 1

        # 2) Update router edge weights
        self.router.update_speeds(speeds)

        # 3) Run GNN prediction every 3 ticks
        bottleneck_nodes = []
        if self.step_count % 3 == 0:
            history = self.traffic_sim.get_history_tensor(lookback=12)
            if history is not None:
                self.current_prediction = self.predictor.predict(history)
                bottleneck_nodes = self.current_prediction["bottleneck_nodes"]
                if self.current_prediction.get("predicted_speeds"):
                    pred_array = np.array(self.current_prediction["predicted_speeds"])
                    self.router.update_speeds(speeds, pred_array)

        # 4) Advance fleet
        fleet_events = self.fleet.tick(speeds, bottleneck_nodes)
        self.event_log.extend(fleet_events)

        # 5) Build state snapshot
        state = {
            "step": self.step_count,
            "traffic": self.traffic_sim.get_state_snapshot(),
            "prediction": self.current_prediction,
            "fleet": self.fleet.get_state(),
            "events": fleet_events,
            "bottleneck_nodes": bottleneck_nodes,
        }
        return state

    async def run_loop(self):
        """Async simulation loop — pushes updates to WebSocket clients."""
        self.running = True
        while self.running:
            state = self._run_tick()
            # Broadcast to WebSocket clients
            msg = json.dumps(state, default=str)
            dead = []
            for ws in self.websocket_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.websocket_clients.remove(ws)

            await asyncio.sleep(self.tick_interval / self.speed_multiplier)

    def stop(self):
        self.running = False

    def get_snapshot(self) -> dict:
        """Get current state without advancing."""
        return {
            "step": self.step_count,
            "traffic": self.traffic_sim.get_state_snapshot(),
            "prediction": self.current_prediction,
            "fleet": self.fleet.get_state(),
            "num_nodes": self.num_nodes,
            "grid_size": int(np.sqrt(self.num_nodes)),
            "recent_events": self.event_log[-20:],
        }
