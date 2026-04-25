"""
fleet_manager.py — Simulates a fleet of delivery vehicles.
"""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Vehicle:
    id: int
    origin: int
    destination: int
    current_node: int
    route: list = field(default_factory=list)
    route_index: int = 0
    status: str = "en_route"  # en_route | delivered | rerouted | waiting
    total_time: float = 0.0
    total_distance: float = 0.0
    reroute_count: int = 0
    cargo_type: str = "general"

    def to_dict(self):
        return {"id": self.id, "origin": self.origin, "destination": self.destination,
                "current_node": self.current_node, "route": self.route,
                "route_index": self.route_index, "status": self.status,
                "total_time": round(self.total_time, 1),
                "total_distance": round(self.total_distance, 1),
                "reroute_count": self.reroute_count, "cargo_type": self.cargo_type,
                "progress": round(self.route_index / max(len(self.route) - 1, 1) * 100, 1)}


class FleetManager:
    def __init__(self, num_vehicles: int, num_nodes: int, router, seed: int = 42):
        self.router = router
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
        self.vehicles: list[Vehicle] = []
        self.stats = {"total_deliveries": 0, "total_reroutes": 0,
                      "total_time_saved": 0.0, "avg_delivery_time": 0.0}

        cargo_types = ["electronics", "food", "medicine", "industrial", "retail", "general"]
        for i in range(num_vehicles):
            o = self.rng.integers(0, num_nodes)
            d = self.rng.integers(0, num_nodes)
            while d == o:
                d = self.rng.integers(0, num_nodes)
            route_result = router.find_route(o, d)
            self.vehicles.append(Vehicle(id=i, origin=o, destination=d,
                current_node=o, route=route_result["path"],
                cargo_type=self.rng.choice(cargo_types)))

    def tick(self, current_speeds: np.ndarray, bottleneck_nodes: list[int]):
        """Advance all vehicles one step and reroute if needed."""
        events = []
        for v in self.vehicles:
            if v.status == "delivered":
                continue

            # Check if current route passes through bottleneck
            remaining = v.route[v.route_index:]
            needs_reroute = any(n in bottleneck_nodes for n in remaining[1:])

            if needs_reroute and v.route_index < len(v.route) - 1:
                comparison = self.router.compare_routes(v.current_node, v.destination)
                new_route = comparison["dynamic_route"]
                if new_route["path"] and new_route["total_time"] < comparison["static_route"]["total_time"]:
                    v.route = new_route["path"]
                    v.route_index = 0
                    v.reroute_count += 1
                    v.status = "rerouted"
                    self.stats["total_reroutes"] += 1
                    self.stats["total_time_saved"] += comparison["time_saved_seconds"]
                    events.append({"type": "reroute", "vehicle_id": v.id,
                        "time_saved": comparison["time_saved_seconds"],
                        "new_path": new_route["path"]})

            # Advance position
            if v.route_index < len(v.route) - 1:
                v.route_index += 1
                v.current_node = v.route[v.route_index]
                edge_data = self.router.G.edges.get((v.route[v.route_index-1], v.current_node))
                if edge_data:
                    v.total_distance += edge_data.get("length", 0)
                    v.total_time += edge_data.get("travel_time", 0)
                if v.status == "rerouted":
                    v.status = "en_route"

            # Check arrival
            if v.current_node == v.destination:
                v.status = "delivered"
                self.stats["total_deliveries"] += 1
                events.append({"type": "delivery", "vehicle_id": v.id,
                    "total_time": v.total_time})
                # Respawn with new destination
                new_dest = self.rng.integers(0, self.num_nodes)
                while new_dest == v.current_node:
                    new_dest = self.rng.integers(0, self.num_nodes)
                route_result = self.router.find_route(v.current_node, new_dest)
                v.origin = v.current_node
                v.destination = new_dest
                v.route = route_result["path"]
                v.route_index = 0
                v.status = "en_route"
                v.total_time = 0; v.total_distance = 0

        delivered_times = [v.total_time for v in self.vehicles if v.total_time > 0]
        self.stats["avg_delivery_time"] = float(np.mean(delivered_times)) if delivered_times else 0
        return events

    def get_state(self) -> dict:
        active = sum(1 for v in self.vehicles if v.status != "delivered")
        return {"vehicles": [v.to_dict() for v in self.vehicles],
                "active_count": active, **self.stats}
