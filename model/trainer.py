"""
trainer.py — Training loop for ASTGCN.
"""
import os, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .astgcn import build_model

class TrafficDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.x = torch.from_numpy(data["x"]).float()
        self.y = torch.from_numpy(data["y"]).float()
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def masked_mae(pred, target):
    mask = (target.abs() > 1e-6).float()
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-8)

def masked_rmse(pred, target):
    mask = (target.abs() > 1e-6).float()
    return torch.sqrt(((pred - target)**2 * mask).sum() / (mask.sum() + 1e-8))

def masked_mape(pred, target):
    mask = (target.abs() > 1e-6).float()
    return 100.0 * ((torch.abs(pred - target) / (target.abs() + 1e-8)) * mask).sum() / (mask.sum() + 1e-8)

def train_model(adj_matrix, data_dir="data/processed", checkpoint_dir="model/checkpoints",
                num_nodes=225, in_features=1, hidden_dim=64, T_in=12, T_out=12,
                K=3, num_blocks=2, batch_size=32, epochs=50, lr=1e-3, patience=10, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Trainer] Device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model, cheb_polys = build_model(adj_matrix, num_nodes, in_features, hidden_dim, T_in, T_out, K, num_blocks, device)
    train_loader = DataLoader(TrafficDataset(os.path.join(data_dir, "train.npz")), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TrafficDataset(os.path.join(data_dir, "val.npz")), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    best_val, wait = float("inf"), 0
    history = {"train_mae": [], "val_mae": [], "val_rmse": [], "val_mape": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, cheb_polys)
            loss = masked_mae(pred, yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); scheduler.step()
            losses.append(loss.item())
        avg_train = np.mean(losses)
        history["train_mae"].append(avg_train)

        model.eval()
        vm, vr, vp = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, cheb_polys)
                vm.append(masked_mae(pred, yb).item())
                vr.append(masked_rmse(pred, yb).item())
                vp.append(masked_mape(pred, yb).item())
        avg_vm, avg_vr, avg_vp = np.mean(vm), np.mean(vr), np.mean(vp)
        history["val_mae"].append(avg_vm); history["val_rmse"].append(avg_vr); history["val_mape"].append(avg_vp)

        print(f"  Epoch {epoch:3d}/{epochs} | Train MAE: {avg_train:.4f} | Val MAE: {avg_vm:.4f} RMSE: {avg_vr:.4f} MAPE: {avg_vp:.1f}% | {time.time()-t0:.1f}s")

        if avg_vm < best_val:
            best_val, wait = avg_vm, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "config": {"num_nodes": num_nodes, "in_features": in_features, "hidden_dim": hidden_dim,
                           "T_in": T_in, "T_out": T_out, "K": K, "num_blocks": num_blocks}},
                os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  ✓ Best model saved (MAE: {best_val:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"  ✗ Early stopping"); break

    print(f"\n[Trainer] Done. Best Val MAE: {best_val:.4f}")
    return model, cheb_polys, history
