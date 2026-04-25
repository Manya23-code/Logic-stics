"""
predictor.py — Inference engine: loads trained ASTGCN, runs prediction,
detects bottlenecks, and returns structured results.
"""
import os, pickle, numpy as np, torch
from .astgcn import ASTGCN, scaled_laplacian, cheb_polynomials

class Predictor:
    def __init__(self, checkpoint_path="model/checkpoints/best_model.pt",
                 adj_path="data/raw/graph_data.pkl", scaler_path="data/processed/scaler.pkl",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load adjacency
        with open(adj_path, "rb") as f:
            _, _, self.adj = pickle.load(f)
        self.num_nodes = self.adj.shape[0]

        # Load scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                s = pickle.load(f)
                self.mean, self.std = s["mean"], s["std"]
        else:
            self.mean, self.std = 0.0, 1.0

        # Load model
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            cfg = ckpt["config"]
            L = scaled_laplacian(self.adj)
            self.cheb_polys = [p.to(self.device) for p in cheb_polynomials(L, cfg["K"])]
            self.model = ASTGCN(
                num_nodes=cfg["num_nodes"], in_features=cfg["in_features"],
                hidden_dim=cfg["hidden_dim"], num_timesteps_in=cfg["T_in"],
                num_timesteps_out=cfg["T_out"], K=cfg["K"], num_blocks=cfg["num_blocks"]
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.T_in = cfg["T_in"]
            self.T_out = cfg["T_out"]
            print(f"[Predictor] Loaded checkpoint (epoch {ckpt['epoch']})")
        else:
            self.model = None
            self.T_in = 12; self.T_out = 12
            print("[Predictor] No checkpoint found — using heuristic fallback")

    def normalize(self, data): return (data - self.mean) / (self.std + 1e-8)
    def denormalize(self, data): return data * (self.std + 1e-8) + self.mean

    def predict(self, history: np.ndarray) -> dict:
        """
        history: (1, T_in, N, 1) raw speed values
        Returns dict with predicted_speeds, bottleneck_nodes, confidence.
        """
        if self.model is not None:
            x = self.normalize(history)
            x_t = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                pred = self.model(x_t, self.cheb_polys)
            pred_np = self.denormalize(pred.cpu().numpy())  # (1, T_out, N, 1)
        else:
            # Heuristic fallback: exponential decay from last known state
            last = history[0, -1, :, 0]  # (N,)
            pred_np = np.zeros((1, self.T_out, self.num_nodes, 1), dtype=np.float32)
            for t in range(self.T_out):
                decay = 0.95 ** t
                noise = np.random.normal(1.0, 0.03, size=self.num_nodes)
                pred_np[0, t, :, 0] = last * decay * noise

        speeds = pred_np[0, :, :, 0]  # (T_out, N)

        # Bottleneck detection: nodes where min predicted speed < threshold
        min_speeds = speeds.min(axis=0)  # (N,)
        mean_speed = np.mean(min_speeds)
        std_speed = np.std(min_speeds)
        
        # A node is a bottleneck if it's slower than average by 1 std dev, 
        # OR if its absolute speed drops below 35 km/h (aggressive catching).
        threshold = max(mean_speed - 1.0 * std_speed, 35.0)

        bottleneck_mask = min_speeds < threshold
        bottleneck_nodes = np.where(bottleneck_mask)[0].tolist()
        
        # Severity relative to the threshold (0 = just at threshold, 1 = total stop)
        bottleneck_severity = {}
        for n in bottleneck_nodes:
            sev = float(1.0 - (min_speeds[n] / threshold))
            bottleneck_severity[int(n)] = max(0.1, min(1.0, sev))

        return {
            "predicted_speeds": speeds.tolist(),
            "bottleneck_nodes": bottleneck_nodes,
            "bottleneck_severity": bottleneck_severity,
            "threshold": float(threshold),
            "horizon_steps": self.T_out,
            "mean_predicted_speed": float(speeds.mean()),
        }
