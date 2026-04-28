"""
simulation_engine.py — Orchestrates the full digital twin simulation loop.
UPGRADED: Real-time Sync (IST), Realistic Speed Fluctuations, and Hard Bottlenecks.
"""
import time, threading, json, numpy as np, asyncio, httpx
from datetime import datetime, timedelta, timezone
import sys, os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.graph_builder import build_synthetic_grid
from data.traffic_simulator import TrafficSimulator
from data.dataset_loader import generate_synthetic_traffic
from model.predictor import Predictor
from routing.dynamic_router import DynamicRouter
from routing.fleet_manager import FleetManager

TOMTOM_KEY = os.environ.get("TOMTOM_KEY", "po8RAcqnVRmWqMl9l4RRVp8IYZdv2P4X")

class SimulationEngine:
    """Full digital-twin simulation orchestrator."""

    def __init__(self, num_nodes_side: int = 15, num_vehicles: int = 30):
        n = num_nodes_side
        self.num_nodes = n * n

        # 1) Build graph
        print("[SimEngine] Building graph ...")
        self.adj, self.node_features = build_synthetic_grid(n, n, out_dir="data/raw")

        # 2) Generate synthetic training data
        if not os.path.exists("data/processed/train.npz"):
            print("[SimEngine] Generating synthetic traffic data ...")
            generate_synthetic_traffic(self.num_nodes, num_steps=12 * 288, out_dir="data/processed")

        # 3) Load predictor
        print("[SimEngine] Loading predictor ...")
        self.predictor = Predictor(
            checkpoint_path="model/checkpoints/best_model.pt",
            adj_path="data/raw/graph_data.pkl",
            scaler_path="data/processed/scaler.pkl")

        # 4) Components
        self.traffic_sim = TrafficSimulator(self.num_nodes, self.adj)
        self.router = DynamicRouter(self.adj, self.node_features)
        self.fleet = FleetManager(num_vehicles, self.num_nodes, self.router)

        # State
        self.running = False
        self.tick_interval = 2.0    
        self.step_count = 0
        self.websocket_clients: list = []
        
        # ✅ TIME SYNC FIX: Hardcoded to IST (UTC +5:30)
        self.ist_tz = timezone(timedelta(hours=5, minutes=30))
        self.time_offset_hours = 0  
        self.current_sim_time = datetime.now(self.ist_tz)
        self.is_live_synced = True  
        
        self.current_prediction = None
        self.event_log: list[dict] = []
        self.last_real_speed = 25.0 

        # Warm up
        for _ in range(15):
            self.traffic_sim.tick()

    async def _fetch_tomtom_speed(self):
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {"point": "28.6892,77.2106", "unit": "KMPH", "key": TOMTOM_KEY}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    speed = data['flowSegmentData']['currentSpeed']
                    return float(speed) + np.random.uniform(-0.5, 0.5)
        except Exception as e:
            print(f"[TomTom] Error: {e}")
        return self.last_real_speed

    def set_time_offset(self, hours: int):
        """Sets the simulation to a specific point in time (Past or Future)."""
        self.time_offset_hours = hours
        self.is_live_synced = (hours == 0)
        self.current_sim_time = datetime.now(self.ist_tz) + timedelta(hours=hours)
        print(f"[SimEngine] Time Travel initiated: {hours}h from now. Sim Time: {self.current_sim_time}")

    async def inject_disruption(self, node_id: int, severity: float = 0.8,
                                 radius: int = 3, duration: int = 24,
                                 event_type: str = "accident") -> dict:
        node_id = int(node_id) % self.num_nodes
        self.traffic_sim.inject_disruption(node_id, severity, radius, duration, event_type)
        
        event = {
            "type": "disruption_injected", 
            "node_id": node_id,
            "severity": severity, 
            "event_type": event_type,
            "step": self.step_count,
            "msg": f"Urgent: {event_type.capitalize()} at Node {node_id}!"
        }
        self.event_log.append(event)
        
        # Immediate Broadcast
        await self._broadcast({"events": [event], "alert": True})
        return {"status": "success", "event": event}

    async def _broadcast(self, data_dict: dict):
        msg = json.dumps(data_dict, default=str)
        dead = []
        for ws in self.websocket_clients:
            try:
                await ws.send_text(msg)
            except:
                dead.append(ws)
        for ws in dead:
            if ws in self.websocket_clients:
                self.websocket_clients.remove(ws)

    async def _run_tick(self) -> dict:
        # 1) SYNC TIME (IST)
        if self.is_live_synced:
            self.current_sim_time = datetime.now(self.ist_tz)
        else:
            self.current_sim_time += timedelta(seconds=self.tick_interval)

        time_decimal = self.current_sim_time.hour + (self.current_sim_time.minute / 60.0)
        
        # 2) FETCH TOMTOM
        if self.is_live_synced and self.step_count % 5 == 0:
            self.last_real_speed = await self._fetch_tomtom_speed()

        # 3) TRAFFIC & FLEET
        speeds = self.traffic_sim.tick()
        
        if not self.is_live_synced:
            base = 35.0 - (5.0 * np.sin(time_decimal * np.pi / 12))
            self.last_real_speed = base + np.random.uniform(-2, 2)
        
        speeds[0] = self.last_real_speed 
        self.step_count += 1

        # 4) PREDICTION
        bottleneck_nodes = []
        if self.step_count % 3 == 0:
            history = self.traffic_sim.get_history_tensor(lookback=12)
            if history is not None:
                self.current_prediction = self.predictor.predict(history)
                # Safely get bottlenecks from prediction
                bottleneck_nodes = self.current_prediction.get("bottleneck_nodes", [])

        # 🛑 REALISM PATCH: Force Speed Drops & Bottlenecks 🛑
        active_disruptions = getattr(self.traffic_sim, 'disruptions', [])
        safe_disruptions = []
        
        for d in active_disruptions:
            # Parse disruption safely
            n_id = d.get('node_id') if isinstance(d, dict) else getattr(d, 'node_id', None)
            if n_id is not None:
                safe_disruptions.append({"node_id": n_id, "severity": 0.8, "event_type": "accident"})
                # Force massive speed drop at disruption node (Traffic Jam)
                speeds[n_id] = 5.0 
                if n_id not in bottleneck_nodes:
                    bottleneck_nodes.append(n_id)

        # Add realistic noise to the whole grid so avg speed isn't a perfect 50
        noise = np.random.uniform(-7.0, 3.0, size=speeds.shape)
        speeds = np.clip(speeds + noise, 5.0, 65.0)
        
        # True dynamic average speed
        avg_speed_val = float(np.mean(speeds))

        # Update Router and Fleet
        self.router.update_speeds(speeds)
        fleet_events = self.fleet.tick(speeds, bottleneck_nodes)

        return {
            "step": self.step_count,
            "traffic": {
                "speeds": speeds.tolist(),
                "time_of_day": time_decimal,
                "current_speed": avg_speed_val,
                "disruptions": safe_disruptions # Ensures MapView sees disruptions
            },
            "prediction": self.current_prediction,
            "bottleneck_nodes": bottleneck_nodes, # Pushed to root for UI counters
            "fleet": self.fleet.get_state(),
            "events": fleet_events + self.event_log[-2:], 
            "is_live_synced": self.is_live_synced,
            "time_offset": self.time_offset_hours
        }

    async def run_loop(self):
        self.running = True
        while self.running:
            try:
                state = await self._run_tick()
                await self._broadcast(state)
            except Exception as e:
                print(f"[SimLoop Error] {e}")
            await asyncio.sleep(self.tick_interval)

    def stop(self):
        self.running = False