# Logic-stics: Predictive Logistics Digital Twin 
**Hackathon Presentation & Architecture Guide**

This document contains everything you need to explain your project confidently to judges, teammates, and mentors. It synthesizes your initial research (`resource.md`) with the actual codebase we built.

---

## 1. The Core Problem (The "Why")
**The Pitch:** Modern global supply chains manage millions of shipments, but they are entirely **reactive**. When a traffic accident or weather event happens, current GPS systems (like standard Google Maps) only reroute a truck *after* the traffic jam has already formed and the truck is stuck in it. 

**Our Solution:** We built **Logic-stics**, a "Predictive Digital Twin." Instead of waiting for congestion to happen, our system uses a state-of-the-art Graph Neural Network to look into the future, predict exactly where and when a bottleneck will form, and reroutes delivery fleets preemptively *before* they ever hit the traffic.

---

## 2. Unpacking your Research (`resource.md`)
Your `resource.md` file is a deep dive into the absolute cutting-edge of traffic prediction. Here are the key intellectual points to drop in your presentation:

*   **Why normal AI fails:** You can't use standard Deep Learning (like CNNs used for images) for traffic. Images are flat grids (Euclidean). Roads are complex webs of intersections (Non-Euclidean graphs). 
*   **The GNN Solution:** To solve this, researchers use **Spatio-Temporal Graph Neural Networks (ST-GNNs)**. 
    *   *Spatial* = It understands the physical layout of the roads (which roads connect to which).
    *   *Temporal* = It understands the flow of time (rush hour patterns).
*   **Our Specific Architecture:** Your research highlighted **ASTGCN** (Attention-Based Spatial-Temporal Graph Convolutional Network). This is exactly what we built in PyTorch. It uses mathematical attention mechanisms to figure out which intersections affect each other the most during a traffic jam.

---

## 3. How Our Code Actually Works (The Architecture)

If a judge asks how it works under the hood, here is the exact pipeline of our codebase:

### Phase 1: The Simulation Engine (`backend/simulation_engine.py`)
Since we don't have a real city to play with, we built a digital one.
*   We generate a **15x15 grid network** (225 intersections).
*   Our `TrafficSimulator` runs a virtual clock. It simulates daily rush hours (dips in speed at 8 AM and 6 PM) and injects random noise to simulate real-world driving.

### Phase 2: The Brain / ASTGCN (`model/astgcn.py` & `predictor.py`)
*   Every 15 virtual minutes, the simulation takes the last 60 minutes of traffic speed data and feeds it into our PyTorch ASTGCN model.
*   The model runs inference on your **NVIDIA RTX 4050 GPU** and predicts the traffic speeds for all 225 nodes for the *next* 60 minutes.
*   **Bottleneck Detection:** Our `predictor.py` script analyzes these future speeds. If it sees any intersection's predicted speed drop below 35 km/h (or severely below the average), it flags it as a **Predicted Bottleneck**.

### Phase 3: The Dynamic Router (`routing/dynamic_router.py`)
*   We use a time-dependent **A* (A-Star) Pathfinding algorithm**.
*   Standard algorithms calculate the shortest path based on physical distance. *Our* algorithm calculates the fastest path based on the ASTGCN's *future* speed predictions for the exact minute the truck is scheduled to arrive at that road.

### Phase 4: The Fleet Manager (`routing/fleet_manager.py`)
*   We simulate 30 active delivery trucks. 
*   If the Fleet Manager sees that a truck's current route will pass through a "Predicted Bottleneck", it immediately calculates a new dynamic route. 
*   If the new route saves time, the truck is instantly rerouted (turning Cyan on the dashboard), avoiding the traffic jam completely.

### Phase 5: The Frontend Dashboard (`frontend/src/`)
*   We built a premium, glassmorphic React dashboard.
*   It connects to the Python backend via **WebSockets** (`FastAPI`), allowing the 15x15 map canvas to update at 60 frames per second without any lag.

---

## 4. How to Demo This Perfectly
When you are showing this to a judge, follow this exact script:

1. **Start Normal:** Show them the dashboard running normally. Point out that the fleet (purple dots) is taking standard routes because the network is healthy (mostly green).
2. **Explain the AI:** Point to the top right. Explain that the ASTGCN model is constantly analyzing the grid, looking 60 minutes into the future.
3. **The "Aha!" Moment:** Click the pink **Inject Disruption** button. Click a node right in the path of a few vehicles. Explain: *"I just simulated a massive multi-car pileup."*
4. **The Result:** Point out the red halo that appears. *"The ASTGCN instantly predicted the cascading traffic jam. Notice how our vehicles just turned Cyan? The Fleet Manager intercepted them and dynamically rerouted them around the accident before they got stuck. Look at the 'Time Saved' KPI in the bottom right corner."*

---

## 5. Buzzwords to use (and what they mean)
*   **Non-Euclidean Graph:** Road networks aren't flat images; they are complex webs.
*   **Chebyshev Spectral Convolutions:** The advanced math our PyTorch model uses to "read" the road network.
*   **Time-Dependent A* Pathfinding:** An algorithm that finds the shortest path while accounting for the fact that road speeds change over time.
*   **Digital Twin:** A virtual simulation that exactly mirrors a real-world system (our city grid).
