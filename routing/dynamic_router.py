"""
dynamic_router.py — Time-dependent A* pathfinding with GNN-predicted edge weights.
"""
import heapq, math, numpy as np, networkx as nx

class DynamicRouter:
    """Maintains a road graph with dynamically-updated edge weights from GNN predictions."""

    def __init__(self, adj_matrix: np.ndarray, node_features: np.ndarray = None):
        self.num_nodes = adj_matrix.shape[0]
        self.adj = adj_matrix
        self.G = nx.DiGraph()

        # Build NetworkX graph from adjacency
        for i in range(self.num_nodes):
            speed = node_features[i, 1] if node_features is not None else 50.0
            length = node_features[i, 0] if node_features is not None else 500.0
            self.G.add_node(i, speed_limit=float(speed), length=float(length),
                            x=float(i % int(math.sqrt(self.num_nodes))),
                            y=float(i // int(math.sqrt(self.num_nodes))))
            for j in range(self.num_nodes):
                if adj_matrix[i, j] > 0:
                    seg_len = float(node_features[j, 0]) if node_features is not None else 500.0
                    base_speed = float(node_features[j, 1]) if node_features is not None else 50.0
                    self.G.add_edge(i, j, length=seg_len, base_speed=base_speed,
                                    current_speed=base_speed, travel_time=seg_len / (base_speed + 1e-8))

        # Time-dependent speed predictions: speeds[t][node] = predicted speed at future step t
        self.predicted_speeds = None  # shape: (T_horizon, N_nodes)

    def update_speeds(self, current_speeds: np.ndarray, predicted_speeds: np.ndarray = None):
        """Update edge weights from current (and optionally predicted future) speeds."""
        for u, v, data in self.G.edges(data=True):
            speed = float(max(current_speeds[v], 2.0))
            data["current_speed"] = speed
            data["travel_time"] = data["length"] / (speed * 1000 / 3600 + 1e-8)
        self.predicted_speeds = predicted_speeds

    def _heuristic(self, a: int, b: int) -> float:
        ax, ay = self.G.nodes[a].get("x", 0), self.G.nodes[a].get("y", 0)
        bx, by = self.G.nodes[b].get("x", 0), self.G.nodes[b].get("y", 0)
        dist = math.sqrt((ax - bx)**2 + (ay - by)**2) * 500
        return dist / (120 * 1000 / 3600)  # optimistic: max speed 120 km/h

    def _get_travel_time(self, u: int, v: int, steps_from_now: int = 0) -> float:
        """
        Get travel time for edge (u, v) at `steps_from_now` into the future.
        Uses GNN predictions when available, falls back to current travel time.
        """
        data = self.G.edges[u, v]
        if self.predicted_speeds is not None:
            horizon = len(self.predicted_speeds)
            # Clamp to prediction horizon — use last prediction if beyond
            t = min(steps_from_now, horizon - 1)
            pred_speed = float(max(self.predicted_speeds[t][v], 2.0))
            return data["length"] / (pred_speed * 1000 / 3600 + 1e-8)
        return data["travel_time"]

    def find_route(self, origin: int, destination: int, use_predictions: bool = True) -> dict:
        """Time-dependent A* pathfinding. Uses predicted speeds when use_predictions=True."""
        if origin == destination:
            return {"path": [origin], "total_time": 0, "total_distance": 0, "segments": []}
        if origin not in self.G or destination not in self.G:
            return {"path": [], "total_time": float("inf"), "total_distance": 0, "error": "Invalid nodes"}

        open_set = [(0, origin)]
        g_score = {origin: 0.0}
        came_from = {}
        node_steps = {origin: 0}  # steps from now (relative, not absolute)

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == destination:
                break

            for neighbor in self.G.successors(current):
                steps = node_steps.get(current, 0)
                if use_predictions:
                    tt = self._get_travel_time(current, neighbor, steps)
                else:
                    # Static: use base speed
                    data = self.G.edges[current, neighbor]
                    tt = data["length"] / (data["base_speed"] * 1000 / 3600 + 1e-8)

                tentative = g_score[current] + tt

                if tentative < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    # Estimate how many 5-min steps into the future we'll arrive
                    node_steps[neighbor] = int(tentative / 300)
                    f = tentative + self._heuristic(neighbor, destination)
                    heapq.heappush(open_set, (f, neighbor))

        if destination not in came_from and origin != destination:
            return {"path": [], "total_time": float("inf"), "total_distance": 0, "error": "No path found"}

        # Reconstruct path
        path = [destination]
        while path[-1] != origin:
            path.append(came_from[path[-1]])
        path.reverse()

        total_time = g_score[destination]
        total_dist = sum(self.G.edges[path[i], path[i+1]]["length"] for i in range(len(path)-1))
        segments = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            e = self.G.edges[u, v]
            segments.append({"from": u, "to": v, "length": e["length"],
                             "speed": e["current_speed"], "time": e["travel_time"]})

        return {"path": path, "total_time": round(total_time, 1),
                "total_distance": round(total_dist, 1), "segments": segments}

    def find_route_static(self, origin: int, destination: int) -> dict:
        """Static routing using base speeds (for comparison). Does NOT mutate edge data."""
        return self.find_route(origin, destination, use_predictions=False)

    def compare_routes(self, origin: int, destination: int) -> dict:
        """Compare static vs dynamic routing — key demo metric."""
        static = self.find_route_static(origin, destination)
        dynamic = self.find_route(origin, destination, use_predictions=True)
        time_saved = static["total_time"] - dynamic["total_time"]
        return {
            "static_route": static, "dynamic_route": dynamic,
            "time_saved_seconds": round(max(time_saved, 0), 1),
            "time_saved_percent": round(max(time_saved, 0) / (static["total_time"] + 1e-8) * 100, 1),
        }
