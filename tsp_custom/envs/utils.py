"""
Utility functions for Custom TSP Environment
Helper functions for action encoding/decoding, tour length calculation, etc.
"""

import torch
from typing import Tuple
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def decode_action(
    action_idx: torch.Tensor,
    adjacency: torch.Tensor,
    num_loc: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode flat action index to (action_type, node_i, node_j).
    
    Action space layout:
    - Indices [0, num_add_actions): ADD actions for edges (i,j) where i < j
    - Indices [num_add_actions, num_add_actions + max_edges): DELETE actions
    - Index [total - 1]: DONE action
    
    Args:
        action_idx: Tensor of action indices (batch_size,) or (batch_size, 1)
        adjacency: Adjacency matrix (batch_size, num_loc, num_loc)
        num_loc: Number of nodes
        
    Returns:
        Tuple of (action_type, node_i, node_j):
        - action_type: 0=ADD, 1=DELETE, 2=DONE
        - node_i, node_j: Node indices (or -1 for DONE)
    """
    # Handle different input shapes
    if len(action_idx.shape) == 2:
        action_idx = action_idx.squeeze(-1)
    
    batch_size = action_idx.shape[0]
    device = action_idx.device
    
    num_add_actions = num_loc * (num_loc - 1) // 2
    max_delete_actions = num_loc
    
    # Initialize outputs
    action_type = torch.zeros(batch_size, dtype=torch.long, device=device)
    node_i = torch.zeros(batch_size, dtype=torch.long, device=device)
    node_j = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        idx = action_idx[b].item()
        
        if idx < num_add_actions:
            # ADD action - decode which edge
            action_type[b] = 0
            # Convert flat index to (i, j) pair where i < j
            i, j = index_to_edge_pair(idx, num_loc)
            node_i[b] = i
            node_j[b] = j
        
        elif idx < num_add_actions + max_delete_actions:
            # DELETE action - find which existing edge
            action_type[b] = 1
            delete_idx = idx - num_add_actions
            
            # Find the delete_idx-th existing edge
            edge_count = 0
            found = False
            for i in range(num_loc):
                for j in range(i + 1, num_loc):
                    if adjacency[b, i, j] == 1:
                        if edge_count == delete_idx:
                            node_i[b] = i
                            node_j[b] = j
                            found = True
                            break
                        edge_count += 1
                if found:
                    break
            
            if not found:
                # This shouldn't happen if masking is correct
                # Default to invalid indices
                node_i[b] = -1
                node_j[b] = -1
                log.warning(f"Batch {b}: DELETE action {delete_idx} not found in adjacency")
        
        else:
            # DONE action
            action_type[b] = 2
            node_i[b] = -1
            node_j[b] = -1
    
    return action_type, node_i, node_j


def encode_action(
    action_type: int,
    node_i: int,
    node_j: int,
    adjacency: torch.Tensor,
    num_loc: int
) -> int:
    """
    Encode (action_type, node_i, node_j) to flat action index.
    Useful for testing and debugging.
    
    Args:
        action_type: 0=ADD, 1=DELETE, 2=DONE
        node_i: First node index
        node_j: Second node index
        adjacency: Adjacency matrix for DELETE actions
        num_loc: Number of nodes
        
    Returns:
        Flat action index
    """
    num_add_actions = num_loc * (num_loc - 1) // 2
    
    if action_type == 0:  # ADD
        # Convert (i, j) pair to flat index
        return edge_pair_to_index(node_i, node_j, num_loc)
    
    elif action_type == 1:  # DELETE
        # Find which existing edge this is
        edge_count = 0
        for i in range(num_loc):
            for j in range(i + 1, num_loc):
                if adjacency[i, j] == 1:
                    if i == node_i and j == node_j:
                        return num_add_actions + edge_count
                    edge_count += 1
        # Edge not found
        raise ValueError(f"Edge ({node_i}, {node_j}) not in adjacency matrix")
    
    else:  # DONE
        return num_add_actions + num_loc  # Last action index


def index_to_edge_pair(idx: int, num_loc: int) -> Tuple[int, int]:
    """
    Convert flat index to (i, j) edge pair where i < j.
    
    Uses the formula for upper triangular matrix indexing:
    For n nodes, edge (i, j) where i < j maps to index:
    idx = i * n - i * (i + 1) / 2 + (j - i - 1)
    
    We invert this to recover i and j from idx.
    
    Args:
        idx: Flat index
        num_loc: Number of nodes
        
    Returns:
        Tuple (i, j) where i < j
    """
    # Solve for i using quadratic formula
    # idx ≈ i * (num_loc - 1) - i² / 2
    # Rearranged: i² - i * (2 * num_loc - 1) + 2 * idx = 0
    
    # Simpler approach: iterate through pairs
    edge_count = 0
    for i in range(num_loc):
        for j in range(i + 1, num_loc):
            if edge_count == idx:
                return i, j
            edge_count += 1
    
    raise ValueError(f"Index {idx} out of range for {num_loc} nodes")


def edge_pair_to_index(i: int, j: int, num_loc: int) -> int:
    """
    Convert (i, j) edge pair to flat index.
    Assumes i < j.
    
    Args:
        i: First node (smaller index)
        j: Second node (larger index)
        num_loc: Number of nodes
        
    Returns:
        Flat index
    """
    if i >= j:
        # Swap if needed
        i, j = min(i, j), max(i, j)
    
    # Count edges before (i, j)
    idx = 0
    for ii in range(i):
        idx += (num_loc - ii - 1)
    idx += (j - i - 1)
    
    return idx


def compute_tour_length(
    locs: torch.Tensor,
    adjacency: torch.Tensor
) -> float:
    """
    Compute the length of a tour from node locations and adjacency matrix.
    
    Args:
        locs: Node coordinates (num_loc, 2)
        adjacency: Adjacency matrix (num_loc, num_loc)
        
    Returns:
        Total tour length
    """
    num_loc = locs.shape[0]
    total_length = 0.0
    
    # Sum distances for all selected edges
    # Each edge appears twice in adjacency (symmetric), so divide by 2
    for i in range(num_loc):
        for j in range(i + 1, num_loc):
            if adjacency[i, j]:
                dist = torch.norm(locs[i] - locs[j])
                total_length += dist.item()
    
    return total_length


def check_tour_validity(
    adjacency: torch.Tensor,
    num_loc: int
) -> bool:
    """
    Check if adjacency matrix represents a valid TSP tour.
    
    A valid tour must:
    1. Have all nodes with degree 2
    2. Form a single connected cycle
    
    Args:
        adjacency: Adjacency matrix (num_loc, num_loc)
        num_loc: Number of nodes
        
    Returns:
        True if valid tour, False otherwise
    """
    # Check degrees
    degrees = adjacency.sum(dim=0)
    if not (degrees == 2).all():
        return False
    
    # Check connectivity
    if not is_graph_connected(adjacency):
        return False
    
    return True


def is_graph_connected(adjacency: torch.Tensor) -> bool:
    """
    Check if graph is connected using BFS.
    
    Args:
        adjacency: Adjacency matrix (num_loc, num_loc)
        
    Returns:
        True if graph is connected, False otherwise
    """
    num_loc = adjacency.shape[0]
    
    # If no edges, not connected (unless single node)
    if adjacency.sum() == 0:
        return num_loc == 1
    
    # BFS from node 0
    visited = torch.zeros(num_loc, dtype=torch.bool, device=adjacency.device)
    queue = [0]
    visited[0] = True
    
    while queue:
        node = queue.pop(0)
        # Find neighbors
        for neighbor in range(num_loc):
            if adjacency[node, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    # Graph is connected if all nodes visited
    return visited.all().item()

def is_sub_tour(adjacency: torch.Tensor, node: int) -> bool:
    """
    Check if graph is a subtour by .
    
    Args:
        adjacency: Adjacency matrix (num_loc, num_loc)
        
    Returns:
        True if graph is connected, False otherwise
    """
    num_loc = adjacency.shape[0]
    
    # If no edges, not connected (unless single node)
    if adjacency.sum() == 0:
        return num_loc == 1
    
    # BFS from node 0
    visited = torch.zeros(num_loc, dtype=torch.bool, device=adjacency.device)
    queue = [0]
    visited[0] = True
    
    while queue:
        node = queue.pop(0)
        # Find neighbors
        for neighbor in range(num_loc):
            if adjacency[node, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    # Graph is connected if all nodes visited
    return visited.all().item()

def has_subtours(adj_matrix: torch.Tensor) -> bool:
    """
    Checks if a partially built TSP tour (represented by an adjacency matrix) 
    contains any subtours.

    Args:
        adj_matrix: A square torch.Tensor adjacency matrix of the current tour. 
                    A value of 1 indicates an edge exists between two nodes.

    Returns:
        True if subtours exist, False otherwise.
    """
    num_nodes = adj_matrix.shape[0]
    # Keep track of visited nodes
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    # Count the number of connected components
    component_count = 0

    for i in range(num_nodes):
        if not visited[i]:
            component_count += 1
            # Start BFS from the unvisited node
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop(0) # Dequeue
                # Find neighbors using the adjacency matrix
                # adj_matrix[u] > 0 finds all nodes j connected to u
                neighbors = torch.nonzero(adj_matrix[u] > 0.5).squeeze()
                
                # Handle case where neighbors might be a single index tensor or empty
                if neighbors.dim() == 0:
                    neighbors = neighbors.unsqueeze(0)
                
                for v in neighbors:
                    # Convert tensor to int for indexing
                    v_int = v.item()
                    if not visited[v_int]:
                        visited[v_int] = True
                        queue.append(v_int)
    
    # A valid complete TSP tour has exactly one component. 
    # During building, if more than one component exists, it means edges added 
    # so far form separate cycles (subtours) if all nodes are meant to be in a single tour.
    # Note: If the graph is not fully built (not all nodes are connected yet), 
    # this function detects *any* closed loops that are separate from other nodes/loops.
    
    # Check if the number of components is greater than 1
    # or if not all nodes have been visited, but a cycle was detected in a subset. 
    # More generally, if we're at a stage where a single tour is expected, 
    # multiple components indicate a problem.
    if component_count > 1 and torch.all(adj_matrix.sum(dim=1) == 2): # If all nodes have 2 edges, multiple components = subtours
         return True
    elif component_count > 1:
        # This means we have multiple separate partial paths/cycles
        return True
    
    return False


def get_avg_distance(locs: torch.Tensor) -> float:
    """
    Compute average pairwise distance between nodes.
    Used for deletion penalty calculation.
    
    Args:
        locs: Node coordinates (num_loc, 2)
        
    Returns:
        Average pairwise distance
    """
    num_loc = locs.shape[0]
    
    # Compute all pairwise distances
    dists = torch.cdist(locs.unsqueeze(0), locs.unsqueeze(0)).squeeze(0)
    
    # Sum all distances (excluding diagonal)
    total_dist = dists.sum() - dists.diag().sum()
    
    # Average over all pairs
    avg_dist = total_dist / (num_loc * (num_loc - 1))
    
    return avg_dist.item()


def greedy_nearest_neighbor_action(
    locs: torch.Tensor,
    adjacency: torch.Tensor,
    degrees: torch.Tensor,
    num_loc: int
) -> int:
    """
    Greedy nearest neighbor heuristic for selecting next edge to add.
    
    Strategy:
    1. Find all valid edges (both nodes have degree < 2, edge doesn't exist, no disconnected subtours)
    2. Among valid edges, choose the shortest one
    3. If no valid edges, return DONE action
    
    Args:
        locs: Node coordinates (num_loc, 2)
        adjacency: Adjacency matrix (num_loc, num_loc)
        degrees: Node degrees (num_loc,)
        num_loc: Number of nodes
        
    Returns:
        Action index (int)
    """
    device = locs.device
    num_add_actions = num_loc * (num_loc - 1) // 2
    num_done_action_idx = num_add_actions + num_loc  # DONE action index
    
    # Check if tour is complete (all degrees are 2)
    if (degrees == 2).all():
        # Tour complete - return DONE action
        return num_done_action_idx
    
    # Compute pairwise distances
    dists = torch.cdist(locs.unsqueeze(0), locs.unsqueeze(0)).squeeze(0)
    
    best_dist = float('inf')
    best_action_idx = 0  # Default to first edge (will be clamped to valid range)
    
    # Count current edges
    num_edges = (adjacency.sum() / 2).int().item()
    
    action_idx = 0
    for i in range(num_loc):
        for j in range(i + 1, num_loc):
            # Check basic validity
            is_valid = (
                adjacency[i, j] == 0 and
                degrees[i] < 2 and
                degrees[j] < 2
            )
            
            if is_valid:
                # Check for disconnected subtours
                would_violate = False
                
                # If adding this edge would complete a cycle (both nodes have degree 1)
                if degrees[i] == 1 and degrees[j] == 1:
                    # Check if we're completing the full tour or making a subtour
                    if num_edges < num_loc - 1:
                        # Not the final edge - need to check if this creates a disconnected subtour
                        # Simulate adding the edge
                        test_adjacency = adjacency.clone()
                        test_adjacency[i, j] = 1
                        test_adjacency[j, i] = 1
                        
                        # Check if i and j are in the same connected component
                        # If they are, adding this edge creates a cycle
                        # Use BFS to check connectivity from i
                        visited = torch.zeros(num_loc, dtype=torch.bool, device=device)
                        queue = [i]
                        visited[i] = True
                        
                        while queue:
                            node = queue.pop(0)
                            # Check neighbors in ORIGINAL adjacency (before adding edge)
                            neighbors = torch.where(adjacency[node] > 0)[0]
                            for neighbor in neighbors:
                                neighbor_idx = neighbor.item()
                                if not visited[neighbor_idx]:
                                    visited[neighbor_idx] = True
                                    queue.append(neighbor_idx)
                        
                        # If j is reachable from i, they're in the same component
                        # Adding edge would create a cycle before tour is complete
                        if visited[j]:
                            would_violate = True
                
                if not would_violate:
                    dist = dists[i, j].item()
                    if dist < best_dist:
                        best_dist = dist
                        best_action_idx = action_idx
            
            action_idx += 1
    
    # If no valid edge was found (best_dist still infinity), return DONE
    # This can happen if the greedy algorithm gets stuck
    if best_dist == float('inf'):
        # Log details about why no valid edge was found
        import logging
        log = logging.getLogger(__name__)
        log.warning(
            f"Greedy stuck: num_edges={num_edges}/{num_loc}, "
            f"degrees={degrees.tolist()}, "
            f"all_degree_2={(degrees == 2).all().item()}"
        )
        # No valid edges available - return DONE action
        return num_done_action_idx
    
    # Ensure we return a valid ADD action index (0 to num_add_actions-1)
    if best_action_idx >= num_add_actions:
        best_action_idx = 0  # Fallback to first edge
    
    return best_action_idx
