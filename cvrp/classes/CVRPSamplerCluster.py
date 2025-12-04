# samplers.py
import torch
from typing import Sequence, Union, Optional

class SamplerCluster:
    """
    Instance-level mixture for RL4CO CVRPGenerator.

    - with prob (1 - p_cluster): uniform nodes
    - with prob p_cluster: clustered nodes
        where k is drawn uniformly from n_clusters_list *per clustered instance*

    Supports RL4CO's expectation:
        loc_sampler.sample(shape) -> tensor of shape = shape
    """

    def __init__(
        self,
        p_cluster: float = 0.5,
        n_clusters_list: Union[int, Sequence[int]] = (3,),
        cluster_std: float = 0.05,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
    ):
        self.p_cluster = float(p_cluster)

        if isinstance(n_clusters_list, int):
            n_clusters_list = [n_clusters_list]
        n_clusters_list = list(n_clusters_list)

        if len(n_clusters_list) == 0:
            raise ValueError("n_clusters_list must be non-empty")
        if any(k < 1 for k in n_clusters_list):
            raise ValueError("all entries in n_clusters_list must be >= 1")

        self.n_clusters_list = n_clusters_list
        self.cluster_std = float(cluster_std)
        self.min_loc = float(min_loc)
        self.max_loc = float(max_loc)

    # ---- RL4CO hook ----
    def sample(self, shape, device=None):
        """
        shape is typically (*batch_size, num_loc+1, 2)
        Returns tensor with EXACTLY this shape.
        """
        if isinstance(shape, torch.Size):
            shape = tuple(shape)

        *batch_shape, n_nodes, d = shape
        if d != 2:
            raise ValueError(f"Expected last dim 2 for (x,y), got {d}")

        if device is None:
            device = torch.device("cpu")

        # flatten batch dims to one B for easy generation
        B = 1
        for s in batch_shape:
            B *= int(s)

        locs_flat = self._generate_flat(B, n_nodes, device=device)

        # reshape back to batch_shape
        return locs_flat.view(*batch_shape, n_nodes, 2)

    # ---- optional standalone callable style ----
    def __call__(self, batch_size, num_loc: Optional[int] = None, device=None, **kwargs):
        """
        Convenience wrapper if you ever call it yourself.
        Generates (B, num_loc, 2) points.
        """
        if isinstance(batch_size, torch.Size):
            B = int(batch_size[0])
        elif isinstance(batch_size, (tuple, list)):
            B = int(batch_size[0])
        else:
            B = int(batch_size)

        if num_loc is None:
            num_loc = kwargs.get("num_loc", None)
        if num_loc is None:
            raise ValueError("num_loc must be provided")

        if device is None:
            device = kwargs.get("device", torch.device("cpu"))

        return self._generate_flat(B, num_loc, device=device)

    # ---- core logic ----
    def _generate_flat(self, B: int, n_nodes: int, device):
        span = self.max_loc - self.min_loc

        # start uniform everywhere
        locs = self.min_loc + span * torch.rand(B, n_nodes, 2, device=device)

        # which instances are clustered?
        use_cluster = (torch.rand(B, device=device) < self.p_cluster)
        clustered_idx = use_cluster.nonzero(as_tuple=False).flatten()

        if clustered_idx.numel() == 0:
            return locs

        for i in clustered_idx.tolist():
            k = self.n_clusters_list[
                torch.randint(0, len(self.n_clusters_list), (1,), device=device).item()
            ]

            centers = self.min_loc + span * torch.rand(k, 2, device=device)
            assign = torch.randint(0, k, (n_nodes,), device=device)
            clustered = centers[assign]

            clustered = clustered + (self.cluster_std * span) * torch.randn_like(clustered)
            clustered = clustered.clamp(self.min_loc, self.max_loc)

            locs[i] = clustered

        return locs
