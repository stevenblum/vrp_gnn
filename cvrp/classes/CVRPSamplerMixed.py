# CVRPSamplerMixed.py
import torch
from typing import Sequence, Union, Optional

class SamplerMixed:
    """
    Mixed instance sampler for RL4CO CVRPGenerator with depot and distribution control.
    
    Depot position:
    - with prob p_center: depot at center (0.5, 0.5)
    - with prob (1 - p_center): depot offset to corner
    
    Customer distribution:
    - with prob p_cluster: clustered customers
    - with prob (1 - p_cluster): uniform random customers
    
    Supports RL4CO's expectation:
        loc_sampler.sample(shape) -> tensor of shape = shape
        where shape[-2] includes depot (index 0)
    """

    def __init__(
        self,
        p_center: float = 0.33,      # 1/3 centered depot, 2/3 offset
        p_cluster: float = 0.5,      # 1/2 clustered, 1/2 random
        n_clusters_list: Union[int, Sequence[int]] = (5, 30),
        cluster_std: Union[float, tuple, list] = 0.04,  # single value or (min, max) range
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        offset_depot_pos: tuple = (0.0, 0.0),  # where to place offset depot
    ):
        self.p_center = float(p_center)
        self.p_cluster = float(p_cluster)

        if isinstance(n_clusters_list, int):
            n_clusters_list = [n_clusters_list]
        elif isinstance(n_clusters_list, range):
            n_clusters_list = list(n_clusters_list)
        else:
            n_clusters_list = list(n_clusters_list)

        if len(n_clusters_list) == 0:
            raise ValueError("n_clusters_list must be non-empty")
        if any(k < 1 for k in n_clusters_list):
            raise ValueError("all entries in n_clusters_list must be >= 1")

        self.n_clusters_list = n_clusters_list
        
        # Handle cluster_std as either a single value or a range
        if isinstance(cluster_std, (tuple, list)):
            if len(cluster_std) != 2:
                raise ValueError("cluster_std range must have exactly 2 values (min, max)")
            self.cluster_std_min = float(cluster_std[0])
            self.cluster_std_max = float(cluster_std[1])
            self.cluster_std_is_range = True
        else:
            self.cluster_std_min = float(cluster_std)
            self.cluster_std_max = float(cluster_std)
            self.cluster_std_is_range = False
            
        self.min_loc = float(min_loc)
        self.max_loc = float(max_loc)
        self.offset_depot_pos = tuple(offset_depot_pos)

    def sample(self, shape, device=None):
        """
        shape is typically (*batch_size, num_loc+1, 2)
        where node 0 is depot, nodes 1..num_loc are customers.
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

    def __call__(self, batch_size, num_loc: Optional[int] = None, device=None, **kwargs):
        """
        Convenience wrapper. Generates (B, num_loc+1, 2) points (depot + customers).
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

        # num_loc is customers only, so total nodes = num_loc + 1 (depot)
        n_nodes = num_loc + 1
        return self._generate_flat(B, n_nodes, device=device)

    def _generate_flat(self, B: int, n_nodes: int, device):
        """
        Generate B instances, each with n_nodes total (depot + customers).
        Node 0 = depot, nodes 1..n_nodes-1 = customers.
        """
        span = self.max_loc - self.min_loc
        center = self.min_loc + span / 2.0

        # Initialize all locations
        locs = torch.zeros(B, n_nodes, 2, device=device)

        # ---- DEPOT PLACEMENT ----
        # Determine which instances get centered vs offset depot
        use_center = (torch.rand(B, device=device) < self.p_center)
        
        for b in range(B):
            if use_center[b]:
                # Center depot
                locs[b, 0, :] = center
            else:
                # Offset depot
                locs[b, 0, 0] = self.offset_depot_pos[0]
                locs[b, 0, 1] = self.offset_depot_pos[1]

        # ---- CUSTOMER PLACEMENT ----
        # Determine which instances get clustered vs random customers
        use_cluster = (torch.rand(B, device=device) < self.p_cluster)
        
        # Generate customers for all instances (start uniform)
        locs[:, 1:, :] = self.min_loc + span * torch.rand(B, n_nodes - 1, 2, device=device)

        # Apply clustering to selected instances
        clustered_idx = use_cluster.nonzero(as_tuple=False).flatten()
        
        if len(clustered_idx) > 0:
            for b in clustered_idx:
                b_int = int(b.item())
                n_customers = n_nodes - 1
                
                # Pick number of clusters for this instance
                k = self.n_clusters_list[
                    torch.randint(0, len(self.n_clusters_list), (1,), device=device).item()
                ]
                
                # Sample cluster_std for this instance if using a range
                if self.cluster_std_is_range:
                    cluster_std = self.cluster_std_min + (self.cluster_std_max - self.cluster_std_min) * torch.rand(1, device=device).item()
                else:
                    cluster_std = self.cluster_std_min
                
                # Generate cluster centers
                centers = self.min_loc + span * torch.rand(k, 2, device=device)
                
                # Assign each customer to a cluster
                assignments = torch.randint(0, k, (n_customers,), device=device)
                
                # Generate clustered positions
                for i in range(n_customers):
                    cluster_id = assignments[i]
                    center_pos = centers[cluster_id]
                    # Add Gaussian noise around cluster center
                    noise = torch.randn(2, device=device) * cluster_std
                    new_pos = center_pos + noise
                    # Clamp to bounds
                    new_pos = torch.clamp(new_pos, self.min_loc, self.max_loc)
                    locs[b_int, i + 1, :] = new_pos

        return locs
