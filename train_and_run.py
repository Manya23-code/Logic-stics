"""
train_and_run.py — Quick-start script optimized for Cloud Deployment (Render).
Training steps are bypassed to prevent 60-second timeouts.
"""
import os, sys

# Path confusion door karne ke liye
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  Logic-stics — Cloud Deployment Start")
    print("=" * 60)

    # Step 1, 2, 3 (Graph, Data, Training) ko skip kiya hai kyunki model pehle se train hai!
    print("\n[✓] Bypassing training. Using pre-trained model from model/checkpoints/best_model.pt")

    # Step 4: Launch backend server
    print("\n[4/4] Launching backend server ...")
    import uvicorn
    
    # 🚨 SABSE BADA FIX: 'server.app' ki jagah 'backend.app' use kiya gaya hai 🚨
    # Agar aapki file ka naam main.py hai, toh isko 'from backend.main import app' kar dijiyega
    from server.app import app

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
