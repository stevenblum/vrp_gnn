"""
Action utilities for Custom POMO Policy

Helper functions for:
- Decoding action indices to (type, i, j)
- Computing action masks (ADD/DELETE/DONE)
- Extracting edge information from state
"""

import torch
from typing import Tuple
import logging

log = logging.getLogger(__name__)


def decode_action_index(
    action_idx: torch.Tensor,
    num_add_actions: int,
    num_delete_actions: int,
    edge_indices: torch.Tensor,
    edge_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode flat action indices to (action_type, node_i, node_j).
    
    Action layout: [ADD actions | DELETE actions | DONE]
    
    Args:
        action_idx: Action indices (batch,)
        num_add_actions: Number of ADD actions (N*(N-1)/2)
        num_delete_actions: Number of DELETE actions per batch item (variable)
        edge_indices: Edge pairs for ADD actions (num_add_actions, 2)
        edge_list: Current edges for DELETE actions (batch, max_edges, 2)
        
    Returns:
        action_type: 0=ADD, 1=DELETE, 2=DONE (batch,)
        node_i: First node index (batch,)
        node_j: Second node index (batch,)
    """
    batch_size = action_idx.shape[0]
    device = action_idx.device
    
    action_type = torch.zeros(batch_size, dtype=torch.long, device=device)
    node_i = torch.zeros(batch_size, dtype=torch.long, device=device)
    node_j = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        idx = action_idx[b].item()
        
        if idx < num_add_actions:
            # ADD action
            action_type[b] = 0
            node_i[b] = edge_indices[idx, 0]
            node_j[b] = edge_indices[idx, 1]
            
        elif idx < num_add_actions + num_delete_actions:
            # DELETE action
            action_type[b] = 1
            delete_idx = idx - num_add_actions
            # Get edge from edge_list
            node_i[b] = edge_list[b, delete_idx, 0]
            node_j[b] = edge_list[b, delete_idx, 1]
            
        else:
            # DONE action
            action_type[b] = 2
            node_i[b] = -1
            node_j[b] = -1
    
    return action_type, node_i, node_j


def extract_edge_list(adjacency: torch.Tensor, max_edges: int) -> torch.Tensor:
    """
    Extract list of existing edges from adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (batch, N, N)
        max_edges: Maximum number of edges to extract
        
    Returns:
        edge_list: Tensor of edges (batch, max_edges, 2)
                  Padded with -1 for missing edges
    """
    batch_size, num_loc, _ = adjacency.shape
    device = adjacency.device
    
    edge_list = torch.full(
        (batch_size, max_edges, 2),
        -1,
        dtype=torch.long,
        device=device
    )
    
    for b in range(batch_size):
        edge_count = 0
        for i in range(num_loc):
            for j in range(i + 1, num_loc):
                if adjacency[b, i, j] == 1:
                    if edge_count < max_edges:
                        edge_list[b, edge_count, 0] = i
                        edge_list[b, edge_count, 1] = j
                        edge_count += 1
    
    return edge_list


def compute_action_masks(
    td,
    num_add_actions: int,
    max_delete_actions: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute action masks from TensorDict state.
    
    The environment provides a single mask with layout:
    [ADD actions | DELETE actions | DONE]
    
    We need to:
    1. Extract the global mask
    2. Split it appropriately for the three decoder heads
    
    Args:
        td: TensorDict with 'action_mask' key
        num_add_actions: Number of ADD actions
        max_delete_actions: Maximum number of DELETE actions
        
    Returns:
        mask_add: Mask for ADD actions (batch, num_add_actions)
        mask_del: Mask for DELETE actions (batch, max_delete_actions)
        mask_done: Mask for DONE action (batch,)
    """
    # Get global mask from environment
    # Shape: (batch, num_add_actions + max_delete_actions + 1)
    global_mask = td["action_mask"]
    
    batch_size = global_mask.shape[0]
    
    # Split mask
    mask_add = global_mask[:, :num_add_actions]  # (batch, num_add_actions)
    mask_del = global_mask[:, num_add_actions:num_add_actions + max_delete_actions]  # (batch, max_delete)
    mask_done = global_mask[:, -1]  # (batch,)
    
    return mask_add, mask_del, mask_done


def create_node_features(td, num_loc: int) -> torch.Tensor:
    """
    Create node features from TensorDict state.
    
    Features per node:
    - x, y coordinates (2)
    - degree (1)
    - adjacency row (N)  <- connections to other nodes
    - step counter normalized (1)
    - deletion counter normalized (1)
    
    Total: 2 + 1 + N + 1 + 1 = N + 5 features
    
    For N=20: 25 features
    For N=50: 55 features
    For N=100: 105 features
    
    Args:
        td: TensorDict with state
        num_loc: Number of nodes
        
    Returns:
        node_features: (batch, N, feat_dim)
    """
    batch_size = td["locs"].shape[0]
    device = td["locs"].device
    
    # Extract state
    locs = td["locs"]  # (batch, N, 2)
    adjacency = td["adjacency"]  # (batch, N, N)
    degrees = td["degrees"]  # (batch, N)
    current_step = td["current_step"]  # (batch, 1)
    num_deletions = td["num_deletions"]  # (batch, 1)
    
    # Normalize step and deletion counters
    max_steps = 2 * num_loc
    step_norm = current_step.float() / max_steps  # (batch, 1)
    deletion_norm = num_deletions.float() / num_loc  # (batch, 1)
    
    # Combine features
    features_list = [
        locs,  # (batch, N, 2)
        degrees.unsqueeze(-1).float(),  # (batch, N, 1)
        adjacency.float(),  # (batch, N, N) - each row shows connections
        step_norm.unsqueeze(1).expand(batch_size, num_loc, 1),  # (batch, N, 1)
        deletion_norm.unsqueeze(1).expand(batch_size, num_loc, 1),  # (batch, N, 1)
    ]
    
    node_features = torch.cat(features_list, dim=-1)  # (batch, N, N+5)
    
    return node_features


if __name__ == '__main__':
    # Test action utilities
    print("Testing action utilities...")
    
    batch_size = 2
    num_loc = 10
    num_add_actions = num_loc * (num_loc - 1) // 2
    max_delete_actions = num_loc
    
    # Create dummy edge indices (for ADD actions)
    edge_indices = torch.triu_indices(num_loc, num_loc, offset=1).t()
    
    # Create dummy edge list (for DELETE actions)
    edge_list = torch.randint(0, num_loc, (batch_size, max_delete_actions, 2))
    edge_list[:, -5:, :] = -1  # Pad last 5
    
    # Test different action types
    action_idx = torch.tensor([5, num_add_actions + 3])  # ADD for batch 0, DELETE for batch 1
    
    action_type, node_i, node_j = decode_action_index(
        action_idx, num_add_actions, max_delete_actions, edge_indices, edge_list
    )
    
    print(f"Action indices: {action_idx}")
    print(f"Decoded:")
    print(f"  action_type: {action_type}")
    print(f"  node_i: {node_i}")
    print(f"  node_j: {node_j}")
    print(f"  Batch 0: ADD edge ({node_i[0]}, {node_j[0]})")
    print(f"  Batch 1: DELETE edge ({node_i[1]}, {node_j[1]})")
    
    print("\nTest passed!")
