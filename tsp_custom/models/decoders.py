"""
Decoder Heads for Custom TSP Environment

Three specialized decoders:
1. AddEdgeDecoder: Scores potential edges to add (N²/2 actions)
2. DeleteEdgeDecoder: Scores existing edges to delete (variable number)
3. DoneDecoder: Scores DONE action (1 action)

Each decoder outputs logits for its action subset, which are then
concatenated and masked by the policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddEdgeDecoder(nn.Module):
    """
    Decoder for ADD edge actions.
    
    Scores all potential edges (i,j) where i < j based on:
    - Node embeddings for i and j
    - Euclidean distance between nodes
    
    Args:
        embed_dim: Dimension of node embeddings (default: 128)
        hidden_dim: Hidden dimension for MLP (default: 128)
    """
    
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Edge scoring MLP: concatenate(node_i_emb, node_j_emb, distance) -> score
        # Input: 2*embed_dim + 1 (two node embeddings + distance)
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embeddings, locs):
        """
        Compute logits for all potential edges.
        
        Args:
            node_embeddings: Node embeddings (batch, N, embed_dim)
            locs: Node locations (batch, N, 2)
            
        Returns:
            logits_add: Logits for all edges i<j (batch, N*(N-1)/2)
            edge_indices: Tensor of (i,j) pairs corresponding to logits (N*(N-1)/2, 2)
        """
        batch, N, embed_dim = node_embeddings.shape
        
        # Compute pairwise distances
        # Broadcasting: locs.unsqueeze(2) - locs.unsqueeze(1) gives all pairwise differences
        dist = torch.cdist(locs, locs, p=2)  # (batch, N, N)
        
        # Create pairwise node embeddings
        # node_embeddings.unsqueeze(2).expand: (batch, N, 1, embed_dim) -> (batch, N, N, embed_dim)
        # node_embeddings.unsqueeze(1).expand: (batch, 1, N, embed_dim) -> (batch, N, N, embed_dim)
        emb_i = node_embeddings.unsqueeze(2).expand(batch, N, N, embed_dim)
        emb_j = node_embeddings.unsqueeze(1).expand(batch, N, N, embed_dim)
        
        # Concatenate: (emb_i, emb_j, distance)
        edge_features = torch.cat([
            emb_i,
            emb_j,
            dist.unsqueeze(-1)  # Add feature dimension
        ], dim=-1)  # (batch, N, N, 2*embed_dim + 1)
        
        # Score all edges
        logits = self.edge_scorer(edge_features).squeeze(-1)  # (batch, N, N)
        
        # Extract upper triangle (i < j only) to avoid duplicates
        # Get indices for upper triangle
        triu_indices = torch.triu_indices(N, N, offset=1, device=logits.device)
        edge_indices = triu_indices.t()  # (N*(N-1)/2, 2)
        
        # Extract logits for upper triangle edges
        logits_add = logits[:, triu_indices[0], triu_indices[1]]  # (batch, N*(N-1)/2)
        
        return logits_add, edge_indices


class DeleteEdgeDecoder(nn.Module):
    """
    Decoder for DELETE edge actions.
    
    Scores existing edges in the current partial tour based on:
    - Node embeddings for endpoints
    - Edge distance
    - Delete bias (scheduled during training)
    
    Args:
        embed_dim: Dimension of node embeddings (default: 128)
        hidden_dim: Hidden dimension for MLP (default: 128)
    """
    
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Edge scoring MLP (same structure as ADD for consistency)
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embeddings, locs, edge_list, delete_bias=0.0):
        """
        Compute logits for deleting existing edges.
        
        Args:
            node_embeddings: Node embeddings (batch, N, embed_dim)
            locs: Node locations (batch, N, 2)
            edge_list: List of existing edges (batch, max_edges, 2)
                      Padded with -1 for invalid entries
            delete_bias: Scalar bias added to all delete logits (default: 0.0)
                        Schedule from -5.0 to 0.0 during training
            
        Returns:
            logits_del: Logits for deleting each edge (batch, max_edges)
        """
        batch, max_edges, _ = edge_list.shape
        _, N, embed_dim = node_embeddings.shape
        
        # Get embeddings and distances for each edge
        # We'll process each batch item separately due to variable edge lists
        
        logits_del_list = []
        
        for b in range(batch):
            edges = edge_list[b]  # (max_edges, 2)
            
            # Identify valid edges (not padded with -1)
            valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)  # (max_edges,)
            
            # Get node embeddings for edge endpoints
            # For invalid edges, we'll use index 0 (will be masked out anyway)
            edges_clamped = edges.clamp(min=0)
            emb_i = node_embeddings[b, edges_clamped[:, 0]]  # (max_edges, embed_dim)
            emb_j = node_embeddings[b, edges_clamped[:, 1]]  # (max_edges, embed_dim)
            
            # Compute distances
            locs_i = locs[b, edges_clamped[:, 0]]  # (max_edges, 2)
            locs_j = locs[b, edges_clamped[:, 1]]  # (max_edges, 2)
            dist = torch.norm(locs_i - locs_j, dim=-1, keepdim=True)  # (max_edges, 1)
            
            # Concatenate features
            edge_features = torch.cat([emb_i, emb_j, dist], dim=-1)  # (max_edges, 2*embed_dim + 1)
            
            # Score edges
            logits = self.edge_scorer(edge_features).squeeze(-1)  # (max_edges,)
            
            # Set invalid edges to -inf (will be masked out)
            logits = logits.masked_fill(~valid_mask, float('-inf'))
            
            logits_del_list.append(logits)
        
        logits_del = torch.stack(logits_del_list)  # (batch, max_edges)
        
        # Apply delete bias (encourages/discourages deletion during training)
        # Bias is scheduled: start at -5.0 (strongly discourage) -> 0.0 (neutral)
        logits_del = logits_del + delete_bias
        
        return logits_del


class DoneDecoder(nn.Module):
    """
    Decoder for DONE action.
    
    Predicts whether to terminate the tour construction based on:
    - Global graph context (mean pooling of node embeddings)
    
    Args:
        embed_dim: Dimension of node embeddings (default: 128)
        hidden_dim: Hidden dimension for MLP (default: 64)
    """
    
    def __init__(self, embed_dim=128, hidden_dim=64):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # DONE scoring MLP
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embeddings):
        """
        Compute logit for DONE action.
        
        Args:
            node_embeddings: Node embeddings (batch, N, embed_dim)
            
        Returns:
            logit_done: Logit for DONE action (batch, 1)
        """
        # Global context: mean pooling over all nodes
        global_context = node_embeddings.mean(dim=1)  # (batch, embed_dim)
        
        # Score DONE action
        logit_done = self.scorer(global_context)  # (batch, 1)
        
        return logit_done


if __name__ == '__main__':
    # Test the decoders
    batch_size = 4
    num_nodes = 20
    embed_dim = 128
    max_edges = 30
    
    # Create dummy inputs
    node_embeddings = torch.randn(batch_size, num_nodes, embed_dim)
    locs = torch.rand(batch_size, num_nodes, 2)  # Random locations in [0,1]
    
    # Create dummy edge list (some edges with padding)
    edge_list = torch.randint(0, num_nodes, (batch_size, max_edges, 2))
    # Simulate padding: set last 10 edges to -1
    edge_list[:, -10:, :] = -1
    
    print("Testing AddEdgeDecoder...")
    add_decoder = AddEdgeDecoder(embed_dim=embed_dim)
    logits_add, edge_indices = add_decoder(node_embeddings, locs)
    expected_num_edges = num_nodes * (num_nodes - 1) // 2
    print(f"  Input: node_embeddings={node_embeddings.shape}, locs={locs.shape}")
    print(f"  Output: logits_add={logits_add.shape}, edge_indices={edge_indices.shape}")
    print(f"  Expected: logits_add=({batch_size}, {expected_num_edges})")
    print(f"  Test passed: {logits_add.shape == (batch_size, expected_num_edges)}")
    
    print("\nTesting DeleteEdgeDecoder...")
    del_decoder = DeleteEdgeDecoder(embed_dim=embed_dim)
    logits_del = del_decoder(node_embeddings, locs, edge_list, delete_bias=-2.0)
    print(f"  Input: node_embeddings={node_embeddings.shape}, edge_list={edge_list.shape}")
    print(f"  Output: logits_del={logits_del.shape}")
    print(f"  Expected: logits_del=({batch_size}, {max_edges})")
    print(f"  Test passed: {logits_del.shape == (batch_size, max_edges)}")
    print(f"  Padded entries are -inf: {torch.all(logits_del[:, -10:] == float('-inf'))}")
    
    print("\nTesting DoneDecoder...")
    done_decoder = DoneDecoder(embed_dim=embed_dim)
    logit_done = done_decoder(node_embeddings)
    print(f"  Input: node_embeddings={node_embeddings.shape}")
    print(f"  Output: logit_done={logit_done.shape}")
    print(f"  Expected: logit_done=({batch_size}, 1)")
    print(f"  Test passed: {logit_done.shape == (batch_size, 1)}")
