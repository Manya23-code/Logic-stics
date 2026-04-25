"""
train_and_run.py — Quick-start script to generate data, train the model,
and launch the backend server in one go.
"""
import os, sys, pickle, numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  Logic-stics — Quick Start")
    print("=" * 60)

    # Step 1: Build synthetic graph
    print("\n[1/4] Building synthetic road graph …")
    from data.graph_builder import build_synthetic_grid
    adj, features = build_synthetic_grid(15, 15, out_dir="data/raw")
    print(f"  ✓ Graph: {adj.shape[0]} nodes")

    # Step 2: Generate synthetic traffic data
    print("\n[2/4] Generating synthetic traffic data …")
    from data.dataset_loader import generate_synthetic_traffic
    if not os.path.exists("data/processed/train.npz"):
        generate_synthetic_traffic(num_nodes=225, num_steps=12 * 288, out_dir="data/processed")
    else:
        print("  ✓ Data already exists, skipping")

    # Step 3: Train ASTGCN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[3/4] Training ASTGCN model … (device: {device})")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    from model.trainer import train_model
    if not os.path.exists("model/checkpoints/best_model.pt"):
        model, polys, history = train_model(
            adj_matrix=adj,
            data_dir="data/processed",
            checkpoint_dir="model/checkpoints",
            num_nodes=225,
            in_features=1,
            hidden_dim=64,   # larger dim — GPU can handle it
            T_in=12, T_out=12,
            K=3, num_blocks=2,
            batch_size=64,   # bigger batches on GPU
            epochs=50,       # more epochs for better convergence
            lr=1e-3,
            patience=10,
            device=device,
        )
        print(f"  ✓ Training complete. Best MAE: {min(history['val_mae']):.4f}")
    else:
        print("  ✓ Checkpoint exists, skipping training")

    # Step 4: Launch server
    print("\n[4/4] Launching FastAPI server …")
    print("  → Backend:  http://localhost:8000")
    print("  → API Docs: http://localhost:8000/docs")
    print("  → Frontend: Run 'cd frontend && npm run dev' in another terminal\n")

    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
