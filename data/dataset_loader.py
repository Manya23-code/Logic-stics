"""
dataset_loader.py — Downloads and preprocesses the METR-LA traffic dataset
into sliding-window tensors for ASTGCN training.
"""

import os
import pickle
import numpy as np


# ── Standard Scaler (channel-0 normalization) ────────────────────────────────
class StandardScaler:
    """
    Normalize features using mean/std computed on channel 0 of the training set.
    This is the standard protocol used by METR-LA / PeMS benchmarks.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std  = std

    def fit(self, data: np.ndarray):
        """Fit on channel 0: data shape (samples, nodes, features) or (timesteps, nodes)."""
        if data.ndim == 3:
            self.mean = data[:, :, 0].mean()
            self.std  = data[:, :, 0].std()
        else:
            self.mean = data.mean()
            self.std  = data.std()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.std + 1e-8) + self.mean


# ── METR-LA loader ───────────────────────────────────────────────────────────
def load_metr_la(raw_dir: str = "data/raw") -> tuple[np.ndarray, np.ndarray]:
    """
    Load METR-LA data.  Expected file: ``data/raw/metr-la.h5`` (or .npz).

    Returns
    -------
    data : np.ndarray  – shape (T, N) or (T, N, F), traffic speeds
    adj  : np.ndarray  – shape (N, N), adjacency matrix
    """
    h5_path  = os.path.join(raw_dir, "metr-la.h5")
    npz_path = os.path.join(raw_dir, "metr-la.npz")
    pkl_path = os.path.join(raw_dir, "graph_data.pkl")

    # Try loading traffic data
    data = None
    if os.path.exists(h5_path):
        import pandas as pd
        df = pd.read_hdf(h5_path)
        data = df.values.astype(np.float32)  # (T, N)
    elif os.path.exists(npz_path):
        data = np.load(npz_path)["data"].astype(np.float32)  # (T, N) or (T, N, F)

    if data is None:
        raise FileNotFoundError(
            f"METR-LA data not found. Place metr-la.h5 or metr-la.npz in {raw_dir}/")

    # Load adjacency
    adj = None
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            _, _, adj = pickle.load(f)

    if adj is None:
        N = data.shape[1]
        adj = np.eye(N, dtype=np.float32)

    print(f"[DatasetLoader] METR-LA loaded: data {data.shape}, adj {adj.shape}")
    return data, adj


# ── Sliding window generator ─────────────────────────────────────────────────
def create_sliding_windows(data: np.ndarray,
                           lookback: int = 12,
                           horizon: int = 12,
                           stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate input/output sliding windows from a (T, N [, F]) array.

    Returns x (samples, lookback, N, F) and y (samples, horizon, N, 1).
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]  # (T, N) → (T, N, 1)

    T, N, F = data.shape
    num_samples = (T - lookback - horizon) // stride + 1

    x_list, y_list = [], []
    for i in range(0, num_samples * stride, stride):
        x_list.append(data[i : i + lookback])
        y_list.append(data[i + lookback : i + lookback + horizon, :, :1])

    x = np.stack(x_list, axis=0)  # (S, lookback, N, F)
    y = np.stack(y_list, axis=0)  # (S, horizon,  N, 1)
    return x, y


def split_and_save(data: np.ndarray,
                   lookback: int = 12,
                   horizon: int = 12,
                   out_dir: str = "data/processed",
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.1):
    """
    Chronologically split data → train/val/test, build sliding windows,
    fit scaler on training set, and save .npz files.
    """
    os.makedirs(out_dir, exist_ok=True)

    T = data.shape[0]
    t_train = int(T * train_ratio)
    t_val   = int(T * (train_ratio + val_ratio))

    train_data = data[:t_train]
    val_data   = data[t_train:t_val]
    test_data  = data[t_val:]

    # Fit scaler on training set channel 0
    scaler = StandardScaler()
    scaler.fit(train_data if train_data.ndim == 3 else train_data[:, :, np.newaxis])

    splits = {"train": train_data, "val": val_data, "test": test_data}
    for name, split_data in splits.items():
        if split_data.ndim == 2:
            split_data = split_data[:, :, np.newaxis]
        # Normalize
        normed = scaler.transform(split_data)
        x, y = create_sliding_windows(normed, lookback, horizon)
        np.savez_compressed(
            os.path.join(out_dir, f"{name}.npz"), x=x, y=y)
        print(f"[DatasetLoader] {name}: x {x.shape}, y {y.shape}")

    # Save scaler parameters
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"mean": scaler.mean, "std": scaler.std}, f)
    print(f"[DatasetLoader] Scaler saved → {scaler_path}")

    return scaler


# ── Synthetic dataset generator (for demos without METR-LA) ──────────────────
def generate_synthetic_traffic(num_nodes: int = 225,
                               num_steps: int = 12 * 288,
                               out_dir: str = "data/processed",
                               lookback: int = 12,
                               horizon: int = 12):
    """
    Generate realistic-looking synthetic traffic data with daily periodicity,
    rush-hour patterns, and random disruptions. Perfect for demos.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    t = np.arange(num_steps)
    # Base daily pattern: speed dips during rush hours
    daily_cycle = 50 + 20 * np.sin(2 * np.pi * t / 288 - np.pi / 2)
    # Weekly modulation (weekends slightly faster)
    weekly_mod = 1 + 0.1 * np.sin(2 * np.pi * t / (288 * 7))

    data = np.zeros((num_steps, num_nodes), dtype=np.float32)
    for n in range(num_nodes):
        base = daily_cycle * weekly_mod
        noise = rng.normal(0, 3, size=num_steps)
        node_offset = rng.uniform(-10, 10)
        data[:, n] = np.clip(base + noise + node_offset, 5, 100)

    # Inject random disruptions (sudden speed drops)
    num_disruptions = 20
    for _ in range(num_disruptions):
        node = rng.integers(0, num_nodes)
        start = rng.integers(0, num_steps - 50)
        duration = rng.integers(10, 50)
        severity = rng.uniform(0.2, 0.5)
        data[start:start + duration, node] *= severity

    print(f"[DatasetLoader] Synthetic traffic: {data.shape}")
    scaler = split_and_save(data, lookback, horizon, out_dir)
    return data, scaler


if __name__ == "__main__":
    generate_synthetic_traffic()
