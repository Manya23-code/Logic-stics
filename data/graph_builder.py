"""
graph_builder.py — Extracts a city road network via OSMnx and converts it to
GNN-compatible PyTorch Geometric tensors (dual-graph formulation).
"""

import os
import pickle
import numpy as np
import networkx as nx

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

try:
    import torch
    from torch_geometric.utils import from_networkx
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Road class mapping (OSM highway tag → integer) ──────────────────────────
ROAD_CLASS_MAP = {
    "motorway": 0, "motorway_link": 0,
    "trunk": 1, "trunk_link": 1,
    "primary": 2, "primary_link": 2,
    "secondary": 3, "secondary_link": 3,
    "tertiary": 4, "tertiary_link": 4,
    "residential": 5,
    "living_street": 6,
    "service": 7,
    "unclassified": 8,
}

DEFAULT_SPEED = {
    0: 100, 1: 80, 2: 60, 3: 50,
    4: 40, 5: 30, 6: 20, 7: 20, 8: 30,
}


def extract_city_graph(place: str = "Los Angeles, California, USA",
                       network_type: str = "drive",
                       simplify: bool = True):
    """Download and simplify a driveable road network from OSM."""
    if not HAS_OSMNX:
        raise ImportError("osmnx is required – pip install osmnx")

    print(f"[GraphBuilder] Downloading road network for '{place}' …")
    G = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    G = ox.projection.project_graph(G)
    print(f"[GraphBuilder] Graph extracted: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def _road_class(highway_tag) -> int:
    """Map an OSM highway tag (may be a list) to an integer class."""
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    return ROAD_CLASS_MAP.get(str(highway_tag), 8)


def _parse_speed(maxspeed, road_cls: int) -> float:
    """Best-effort km/h speed from the maxspeed tag."""
    if maxspeed is None:
        return float(DEFAULT_SPEED.get(road_cls, 30))
    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0]
    try:
        return float(str(maxspeed).replace(" mph", "").replace(" km/h", ""))
    except (ValueError, TypeError):
        return float(DEFAULT_SPEED.get(road_cls, 30))


def build_dual_graph(G_primal):
    """
    Convert a primal OSMnx graph to a dual (line) graph where
    road *segments* become nodes and *intersections* become edges.
    Each dual-graph node carries features extracted from the road segment.
    """
    # Build line graph  (segment → node, shared-intersection → edge)
    L = nx.line_graph(G_primal, create_using=nx.DiGraph)

    # Attach features to each dual-node (which is an original edge)
    for dual_node in L.nodes():
        u, v, *key = dual_node  # original edge
        key = key[0] if key else 0
        edata = G_primal.edges[u, v, key] if G_primal.is_multigraph() else G_primal.edges[u, v]

        road_cls = _road_class(edata.get("highway", "unclassified"))
        speed    = _parse_speed(edata.get("maxspeed"), road_cls)
        length   = float(edata.get("length", 100.0))
        lanes    = float(edata.get("lanes", 1)) if edata.get("lanes") else 1.0
        oneway   = 1.0 if edata.get("oneway", False) else 0.0

        L.nodes[dual_node]["road_length"]  = length
        L.nodes[dual_node]["speed_limit"]  = speed
        L.nodes[dual_node]["lanes"]        = lanes
        L.nodes[dual_node]["road_class"]   = float(road_cls)
        L.nodes[dual_node]["oneway"]       = oneway

    return L


def graph_to_tensors(L, out_dir: str = "data/raw"):
    """Persist dual-graph as PyTorch Geometric tensors + pickle adjacency."""
    os.makedirs(out_dir, exist_ok=True)

    # Relabel nodes to contiguous integers for PyG
    mapping = {n: i for i, n in enumerate(L.nodes())}
    L_int = nx.relabel_nodes(L, mapping)

    N = L_int.number_of_nodes()
    feature_keys = ["road_length", "speed_limit", "lanes", "road_class", "oneway"]

    # Build numpy feature matrix
    x_np = np.zeros((N, len(feature_keys)), dtype=np.float32)
    for node_id in range(N):
        for j, key in enumerate(feature_keys):
            x_np[node_id, j] = float(L_int.nodes[node_id].get(key, 0.0))

    # Adjacency matrix (dense)
    adj = nx.adjacency_matrix(L_int).toarray().astype(np.float32)

    # Sensor-style ID map (for compatibility with METR-LA loaders)
    sensor_ids   = list(range(N))
    id_to_ind    = {sid: sid for sid in sensor_ids}

    # Save graph_data.pkl
    pkl_path = os.path.join(out_dir, "graph_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump((sensor_ids, id_to_ind, adj), f)
    print(f"[GraphBuilder] Saved adjacency ({N}×{N}) → {pkl_path}")

    # Save feature matrix
    feat_path = os.path.join(out_dir, "node_features.npy")
    np.save(feat_path, x_np)
    print(f"[GraphBuilder] Saved node features ({N}×{len(feature_keys)}) → {feat_path}")

    # Optionally save PyG Data object
    if HAS_TORCH:
        pyg_data = from_networkx(L_int)
        for j, key in enumerate(feature_keys):
            pass  # features already stored via from_networkx
        pyg_path = os.path.join(out_dir, "pyg_graph.pt")
        torch.save(pyg_data, pyg_path)
        print(f"[GraphBuilder] Saved PyG Data → {pyg_path}")

    return adj, x_np


# ── Fallback: Build a synthetic grid graph (for when OSMnx is unavailable) ──
def build_synthetic_grid(rows: int = 15, cols: int = 15, out_dir: str = "data/raw"):
    """
    Generate a synthetic grid road network for demos when OSMnx
    extraction is not practical (e.g. no internet, slow API).
    Returns adjacency matrix and node feature matrix.
    """
    os.makedirs(out_dir, exist_ok=True)

    G = nx.grid_2d_graph(rows, cols)
    G = G.to_directed()

    # Relabel to integers
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    N = G.number_of_nodes()

    rng = np.random.default_rng(42)
    for n in G.nodes():
        G.nodes[n]["road_length"]  = rng.uniform(100, 1500)
        G.nodes[n]["speed_limit"]  = rng.choice([30, 40, 50, 60, 80])
        G.nodes[n]["lanes"]        = rng.choice([1, 2, 3, 4])
        G.nodes[n]["road_class"]   = float(rng.integers(0, 9))
        G.nodes[n]["oneway"]       = float(rng.choice([0, 1]))

    feature_keys = ["road_length", "speed_limit", "lanes", "road_class", "oneway"]
    x_np = np.zeros((N, len(feature_keys)), dtype=np.float32)
    for node_id in range(N):
        for j, key in enumerate(feature_keys):
            x_np[node_id, j] = float(G.nodes[node_id].get(key, 0.0))

    adj = nx.adjacency_matrix(G).toarray().astype(np.float32)

    sensor_ids = list(range(N))
    id_to_ind  = {s: s for s in sensor_ids}
    pkl_path   = os.path.join(out_dir, "graph_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump((sensor_ids, id_to_ind, adj), f)

    np.save(os.path.join(out_dir, "node_features.npy"), x_np)
    print(f"[GraphBuilder] Synthetic grid {rows}×{cols} → {N} nodes saved to {out_dir}")
    return adj, x_np


if __name__ == "__main__":
    # Quick test: build a synthetic grid (no internet needed)
    build_synthetic_grid(15, 15)
