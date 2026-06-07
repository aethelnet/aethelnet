# Aethelnet

**A Continuous Liquid Graph Neural Network for Decentralized Execution**

Aethelnet is an experimental neural architecture built on continuous Ordinary Differential Equations (ODEs) and dynamic topology (Liquid GNNs). Instead of a rigid, feed-forward matrix, Aethelnet models knowledge as a fluid, physics-informed graph where nodes ("Grains of Truth") and edges ("Synapses") decay, grow, and vibrate over time.

## 🧠 Core Architecture Highlights

*   **Physics-Informed Node Dynamics:** The graph isn't static. It uses a `ContinuousForge` that applies thermodynamic-like decay and Hebbian learning. Edges that fire together wire together, while idle pathways naturally decay over continuous time.
*   **P2P "Persona" Syncing:** The network implements a decentralized Mesh Gossip protocol (`p2p_sync.py`). Nodes periodically hunt for their peers' strongest signals (the highest-momentum subnetworks) and ingest them as isolated "Personas". This allows different execution nodes to share learned topological structures without needing centralized gradient syncing.
*   **Decoupled Reality Anchors:** The core is agnostic to its data source. It processes purely mathematical "truth" embeddings, allowing it to be mapped to financial data, language modeling, or arbitrary data streams by injecting external "Reality Anchors".

## 🚀 Usage Flows

You can run Aethelnet in different modes depending on whether you want to simulate the physics engine locally, or run a decentralized mesh node.

### 1. Run the Local ODE Simulation (The Forge)
Watch the graph topology evolve in real-time. This command spins up the Liquid GNN, creates a baseline structure, and applies continuous ODE dampening and excitation.
```bash
python3 scripts/run_simulation.py
```

### 2. Start a Mesh Node (Decentralized Server)
Start the FastAPI backend to expose your node's graph state to the mesh. Other nodes will be able to query your `/p2p/expertise` endpoint to ingest your strongest topological structures.
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Connect the CLI Diagnostics
If a mesh node is running, you can connect the command-line interface to read its current "Mind State", inspect edge weights, and view active Personas.
```bash
python3 cli/lgnn_cli.py --connect http://localhost:8000 --status
```

## 🛠 Installation

Requirements: Python 3.9+

```bash
git clone https://github.com/aethelnet/aethelnet.git
cd aethelnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🤝 Philosophy
Aethelnet moves away from massive, static parameter matrices towards smaller, highly dynamic graphs that *live* in continuous time. We believe the next step in generalized architectures relies on nodes that communicate via physical rules rather than arbitrary backward passes.
