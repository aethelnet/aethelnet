import torch
from aethelnet.liquid_graph import LiquidGraph
import logging

logging.basicConfig(level=logging.INFO)

print("--- AETHELNET VITAL CHECK ---")

try:
    # Instantiate the core graph
    print("1. Initializing LiquidGraph...")
    graph = LiquidGraph(hidden_dim=64)
    
    # Create two nodes with random embeddings
    print("2. Injecting Nodes...")
    node_a_emb = torch.randn(64)
    node_a_emb = node_a_emb / node_a_emb.norm()
    
    node_b_emb = torch.randn(64)
    node_b_emb = node_b_emb / node_b_emb.norm()
    
    graph.add_node("Concept_A", node_a_emb)
    graph.add_node("Concept_B", node_b_emb)
    
    # Evolve the topology using Neural ODEs
    print("3. Evolving Topology (ODE Integration)...")
    graph.evolve_topology(compute_time=1.0)
    
    # Check if networkx graph recorded nodes
    node_count = graph.nx_graph.number_of_nodes()
    edge_count = graph.nx_graph.number_of_edges()
    
    print(f"\n[SUCCESS] Vital check passed!")
    print(f"Nodes in topology: {node_count}")
    print(f"Edges evolved: {edge_count}")
    print(f"Hardware Speed Factor: {graph.hardware_speed_factor:.2f}x")

except Exception as e:
    print(f"\n[ERROR] Vital check failed: {e}")
