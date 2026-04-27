"""
main.py — FastAPI backend for Logic-stics Digital Twin.
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from backend.simulation_engine import SimulationEngine

app = FastAPI(title="Logic-stics", description="Predictive Logistics Digital Twin API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Global simulation engine (initialized on startup)
engine: Optional[SimulationEngine] = None

class DisruptionRequest(BaseModel):
    node_id: int
    severity: float = 0.2
    radius: int = 3
    duration: int = 24
    event_type: str = "accident"

class SpeedRequest(BaseModel):
    multiplier: int = 1

class RouteRequest(BaseModel):
    origin: int
    destination: int

@app.on_event("startup")
async def startup():
    global engine
    engine = SimulationEngine(num_nodes_side=15, num_vehicles=30)
    asyncio.create_task(engine.run_loop())

@app.get("/api/health")
def health():
    return {"status": "ok", "step": engine.step_count if engine else 0}

@app.get("/api/snapshot")
def get_snapshot():
    """Full current state of the simulation."""
    return engine.get_snapshot()

@app.get("/api/graph")
def get_graph():
    """Return graph topology for map rendering."""
    import numpy as np
    adj = engine.adj
    nodes = []
    grid = engine.get_snapshot()["grid_size"]
    for i in range(engine.num_nodes):
        row, col = divmod(i, grid)
        feat = engine.node_features[i] if engine.node_features is not None else [500, 50, 2, 5, 0]
        nodes.append({"id": i, "x": col, "y": row,
            "road_length": float(feat[0]), "speed_limit": float(feat[1]),
            "lanes": float(feat[2]), "road_class": float(feat[3])})
    edges = []
    for i in range(engine.num_nodes):
        for j in range(engine.num_nodes):
            if adj[i, j] > 0:
                edges.append({"source": i, "target": j})
    return {"nodes": nodes, "edges": edges, "grid_size": grid}

@app.get("/api/predict")
def get_prediction():
    """Latest GNN prediction results."""
    return engine.current_prediction or {"message": "No prediction yet"}

@app.get("/api/fleet")
def get_fleet():
    """Current fleet status."""
    return engine.fleet.get_state()

@app.post("/api/disruption")
def inject_disruption(req: DisruptionRequest):
    """Inject a disruption event."""
    return engine.inject_disruption(req.node_id, req.severity, req.radius,
                                    req.duration, req.event_type)

@app.post("/api/speed")
def set_speed(req: SpeedRequest):
    """Set simulation speed multiplier."""
    engine.set_speed(req.multiplier)
    return {"speed_multiplier": engine.speed_multiplier}

@app.post("/api/route")
def compute_route(req: RouteRequest):
    """Compare static vs dynamic routing."""
    return engine.router.compare_routes(req.origin, req.destination, engine.step_count)

@app.get("/api/events")
def get_events(limit: int = Query(default=50)):
    """Recent simulation events."""
    return {"events": engine.event_log[-limit:]}

@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    engine.websocket_clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        if ws in engine.websocket_clients:
            engine.websocket_clients.remove(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
