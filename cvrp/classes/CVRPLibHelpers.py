# CVRPLibHelpers.py
# Helper functions to load CVRPLib instances

import torch
import vrplib
import re
from pathlib import Path
from tensordict import TensorDict
from rl4co.data.utils import load_npz_to_tensordict


def normalize_coord(coords: torch.Tensor) -> torch.Tensor:
    """Normalize coordinates to [0, 1] range."""
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    return (coords - mins) / (maxs - mins + 1e-8)


def vrp_to_td(vrp_path: str) -> TensorDict:
    """Read raw .vrp -> RL4CO-style TensorDict with batch dim = 1."""
    prob = vrplib.read_instance(vrp_path)
    coords = torch.tensor(prob["node_coord"], dtype=torch.float32)
    coords = normalize_coord(coords)

    depot = coords[0]
    locs  = coords[1:]

    demand = torch.tensor(prob["demand"][1:], dtype=torch.float32)
    capacity = float(prob["capacity"])

    td = TensorDict(
        {
            "locs": locs,
            "depot": depot,
            "demand": demand / capacity,  # normalized demand expected by RL4CO CVRP
        },
        batch_size=[]
    )
    return batchify_td(td)


def batchify_td(td: TensorDict) -> TensorDict:
    """Ensure a leading batch dimension of 1 for env.reset(batch)."""
    if td["locs"].ndim == 2:   # (n_loc,2) -> (1,n_loc,2)
        td = TensorDict(
            {k: v.unsqueeze(0) for k, v in td.items()},
            batch_size=[1],
        )
    return td


def load_val_instance(path: str) -> TensorDict:
    """
    Load a validation instance from either .npz or .vrp file.
    
    Args:
        path: Path to the instance file (.npz or .vrp)
        
    Returns:
        TensorDict with batch dimension
    """
    p = Path(path)
    if p.suffix == ".npz":
        td = load_npz_to_tensordict(str(p))  # assumes first axis is batch
        # if you saved per-instance without batch, fix it:
        if td["locs"].ndim == 2:
            td = batchify_td(td)
        return td
    elif p.suffix == ".vrp":
        return vrp_to_td(str(p))
    else:
        raise ValueError(f"Unsupported validation file type: {p}")


def load_bks_cost(instance_name: str, sol_base_dir: str = "cvrplib_instances/X") -> float:
    """
    Load BKS (Best Known Solution) cost from .sol file.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n101-k25")
        sol_base_dir: Directory containing .sol files
        
    Returns:
        BKS cost from the solution file, or None if not found
    """
    sol_path = Path(sol_base_dir) / f"{instance_name}.sol"
    if not sol_path.exists():
        return None
    
    with open(sol_path, 'r') as f:
        for line in f:
            if 'cost' in line.lower():
                # Extract the cost number
                match = re.search(r'(\d+(?:\.\d+)?)', line)
                if match:
                    return float(match.group(1))
    return None


def calculate_normalized_bks(instance_name: str, vrp_base_dir: str = "cvrplib_instances/X", 
                            sol_base_dir: str = "cvrplib_instances/X") -> float:
    """
    Calculate BKS cost in normalized coordinates.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n101-k25")
        vrp_base_dir: Directory containing .vrp files
        sol_base_dir: Directory containing .sol files
        
    Returns:
        BKS cost in normalized coordinate space, or None if files not found
    """
    vrp_path = Path(vrp_base_dir) / f"{instance_name}.vrp"
    sol_path = Path(sol_base_dir) / f"{instance_name}.sol"
    
    if not vrp_path.exists() or not sol_path.exists():
        return None
    
    # Load instance
    prob = vrplib.read_instance(str(vrp_path))
    coords = torch.tensor(prob["node_coord"], dtype=torch.float32)
    
    # Normalize coordinates the same way as in training
    normalized_coords = normalize_coord(coords)
    
    # Load solution routes
    solution = vrplib.read_solution(str(sol_path))
    routes = solution['routes']
    
    # Calculate cost in normalized space
    total_cost = 0.0
    depot_coords = normalized_coords[0]
    
    for route in routes:
        # Each route is a list of customer indices (1-based)
        # Convert to 0-based and add depot at start and end
        route_nodes = [0] + [idx for idx in route] + [0]
        
        # Calculate route distance
        for i in range(len(route_nodes) - 1):
            node_a = normalized_coords[route_nodes[i]]
            node_b = normalized_coords[route_nodes[i+1]]
            dist = torch.sqrt(((node_a - node_b) ** 2).sum()).item()
            total_cost += dist
    
    return total_cost
