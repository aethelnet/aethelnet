import asyncio
import logging
import torch
from fastapi import FastAPI, WebSocket
from typing import Dict, Any, List

from .liquid_graph import LiquidGraph

logger = logging.getLogger("Aethelnet.Engine")

class AethelEngine:
    """
    The orchestrator for the Liquid Graph Neural Network.
    Manages the continuous ignition loop and serves as the Swarm Node backend.
    """
    def __init__(self, hidden_dim: int = 768):
        self.graph = LiquidGraph(hidden_dim=hidden_dim)
        self.is_running = False
        self.connected_clients: List[WebSocket] = []

    async def ignition_loop(self, tick_interval: float = 1.0):
        """
        The continuous background loop that evolves the graph over time.
        This represents the 'Flow' or 'Meditation' state of the node.
        """
        self.is_running = True
        logger.info("[AethelEngine] Ignition sequence started. Entering continuous ODE flow.")
        
        while self.is_running:
            try:
                # Evolve topology through Neural ODEs
                self.graph.evolve_topology(compute_time=1.0)
                
                # Broadcast the new state to all connected swarm clients
                await self.broadcast_state()
                
            except Exception as e:
                logger.error(f"[AethelEngine] Error in ignition loop: {e}")
                
            await asyncio.sleep(tick_interval)

    async def broadcast_state(self):
        """
        Splatter the distilled 'Senf' (state) to the connected Swarm.
        """
        if not self.connected_clients:
            return
            
        state_summary = {
            "node_count": len(self.graph.nodes),
            "edge_count": self.graph.nx_graph.number_of_edges(),
            "speed_factor": self.graph.hardware_speed_factor
        }
        
        for ws in self.connected_clients:
            try:
                await ws.send_json(state_summary)
            except Exception:
                pass # Dead connection will be cleaned up by the websocket endpoint

# --- Minimal FastAPI Server Implementation ---

app = FastAPI(title="Aethelnet Swarm Node")
engine = AethelEngine()

@app.on_event("startup")
async def startup_event():
    # Start the continuous evolution in the background
    asyncio.create_task(engine.ignition_loop())

@app.websocket("/ws/swarm")
async def swarm_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the Swarm App (Federated Learning).
    Receives connections from mobile clients or other nodes.
    """
    await websocket.accept()
    engine.connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Here we would merge remote tensors from the Swarm
            pass
    except Exception:
        engine.connected_clients.remove(websocket)

@app.post("/api/inject")
async def inject_node(node_id: str):
    """
    Inject a new concept into the graph.
    """
    seed_emb = torch.randn(engine.graph.hidden_dim)
    seed_emb = seed_emb / (seed_emb.norm() + 1e-8)
    engine.graph.add_node(node_id, seed_emb)
    return {"status": "injected", "node_id": node_id}
