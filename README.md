# Logic-stics 🧠🚛

> **Predictive Logistics Digital Twin** — Resilient Supply Chain Optimization powered by Spatio-Temporal Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white)]()
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=black)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)]()

## 🎯 What It Does

Logic-stics is a real-time digital twin that **predicts supply chain disruptions before they cascade** and **dynamically reroutes shipments** using deep learning.

1. **🗺️ Ingests** a road network graph and traffic telemetry
2. **🧠 Predicts** bottlenecks 15–60 minutes ahead using ASTGCN (Attention-based Spatio-Temporal Graph Convolutional Network)
3. **🔄 Reroutes** active shipments via time-dependent A* pathfinding
4. **📊 Visualizes** everything in a stunning interactive dashboard

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  OSMnx      │───▶│  ASTGCN      │───▶│  Dynamic    │
│  Graph      │    │  Predictor   │    │  A* Router  │
│  Builder    │    │  (PyTorch)   │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌──────────────────────────────────────────────────┐
│              FastAPI Backend                      │
│  WebSocket streaming · REST APIs · Sim Engine    │
└──────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│           React + Canvas Dashboard               │
│  Live Map · Fleet Tracking · KPI Metrics         │
└──────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) CUDA-capable GPU

### 1. Install Dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### 2. One-Command Launch

```bash
# Generates data → Trains model → Starts server
python train_and_run.py
```

### 3. Start Frontend (separate terminal)

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** and watch the digital twin come alive!

## 🎮 Demo Features

| Feature | Description |
|---------|-------------|
| **Live Traffic Map** | Road network colored by real-time speed (green → red) |
| **Bottleneck Prediction** | GNN detects future congestion 60 min ahead |
| **Dynamic Rerouting** | Vehicles auto-reroute around predicted bottlenecks |
| **Disruption Injection** | Click to simulate accidents, weather events |
| **KPI Dashboard** | Track deliveries, reroutes, time saved |
| **Fleet Tracking** | Watch 30 vehicles navigate the network |

## 📁 Project Structure

```
logic-stics/
├── data/                    # Data pipeline
│   ├── graph_builder.py     # OSMnx → GNN tensors
│   ├── dataset_loader.py    # METR-LA preprocessing
│   └── traffic_simulator.py # Synthetic traffic engine
├── model/                   # ML model
│   ├── astgcn.py           # ASTGCN architecture
│   ├── trainer.py          # Training loop
│   └── predictor.py        # Inference engine
├── routing/                 # Routing engine
│   ├── dynamic_router.py   # Time-dependent A*
│   └── fleet_manager.py    # Vehicle fleet sim
├── backend/                 # FastAPI server
│   ├── main.py             # API endpoints
│   └── simulation_engine.py # Orchestrator
├── frontend/               # React dashboard
│   └── src/
│       ├── components/     # UI components
│       └── hooks/          # WebSocket hook
└── train_and_run.py        # Quick-start script
```

## 🧠 Model: ASTGCN

**Attention-based Spatial-Temporal Graph Convolutional Network**

- **Spatial Attention** dynamically weights neighboring nodes
- **Temporal Attention** learns which historical timesteps matter
- **Chebyshev Graph Convolution** efficiently captures k-hop neighborhood features
- Pre-trained on traffic benchmark data (METR-LA compatible format)

## 📊 Key Metrics

- **Prediction Horizon:** 12 steps (60 minutes at 5-min intervals)
- **Lookback Window:** 12 steps (60 minutes of history)
- **Bottleneck Detection:** Nodes below μ - 2σ speed threshold
- **Routing:** Time-dependent A* with predictive edge weights

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/snapshot` | GET | Full simulation state |
| `/api/graph` | GET | Road network topology |
| `/api/predict` | GET | Latest GNN predictions |
| `/api/fleet` | GET | Vehicle fleet status |
| `/api/disruption` | POST | Inject disruption event |
| `/api/route` | POST | Compare static vs dynamic routes |
| `/api/speed` | POST | Set simulation speed |
| `/ws/live` | WS | Real-time state streaming |

## 🏆 Hackathon: Smart Supply Chains

Built for the **Resilient Logistics & Dynamic Supply Chain Optimization** challenge.

> *"Design a scalable system capable of continuously analyzing multifaceted transit data to preemptively detect and flag potential supply chain disruptions."*

---

Built with ❤️ using PyTorch, FastAPI, React, and Graph Neural Networks.
