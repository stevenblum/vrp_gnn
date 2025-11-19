import torch
import torch.nn as nn
from tensordict import TensorDict  # or from tensordict.tensordict import TensorDict

class CustomTSPInitEmbedding(nn.Module):
    """
    Initial embedding for TSP that uses:
      - locs: (x, y)
      - relative vectors to the K nearest neighbors for each node
    """
    def __init__(self, embedding_dim: int, k_neighbors: int = 5, linear_bias: bool = True):
        super().__init__()
        self.k = k_neighbors

        # Base features: (x, y) + K neighbor vectors, each 2D
        node_dim = 2 + 2 * self.k
        self.init_embed = nn.Linear(node_dim, embedding_dim, bias=linear_bias)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        td["locs"]: (B, N, 2)
        Returns: (B, N, embedding_dim)
        """
        locs = td["locs"]          # (B, N, 2)
        B, N, _ = locs.shape
        K = self.k
        assert N > K, f"Need num_loc > k_neighbors (got N={N}, k={K})"

        # diff[b, i, j, :] = loc_j - loc_i  (vector from node i to node j)
        diff = locs.unsqueeze(1) - locs.unsqueeze(2)    # (B, N, N, 2)

        # Pairwise distances ||loc_j - loc_i||
        dists = diff.norm(dim=-1)                       # (B, N, N)

        # Do not count self as neighbor
        eye = torch.eye(N, device=locs.device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
        dists = dists.masked_fill(eye, float("inf"))

        # Indices of K nearest neighbors for each node
        # neighbors_idx[b, i, k] = index of k-th nearest neighbor of node i
        neighbors_idx = dists.topk(k=K, dim=-1, largest=False).indices  # (B, N, K)

        # Get the relative vectors to those neighbors
        # diff[b, i, j, :] already stores loc_j - loc_i
        # Take along neighbor dim (j) using neighbors_idx
        neighbors_rel = torch.take_along_dim(
            diff,                          # (B, N, N, 2)
            neighbors_idx.unsqueeze(-1),   # (B, N, K, 1)
            dim=2,
        )                                  # (B, N, K, 2)

        # Flatten neighbor vectors into one feature vector per node
        neighbors_rel_flat = neighbors_rel.reshape(B, N, K * 2)  # (B, N, 2K)

        # Concatenate original coordinates with neighbor-relative vectors
        node_feats = torch.cat([locs, neighbors_rel_flat], dim=-1)  # (B, N, 2 + 2K)

        # Project to embedding space
        return self.init_embed(node_feats)  # (B, N, embedding_dim)
