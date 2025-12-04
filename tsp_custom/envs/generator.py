"""
Generator for Custom TSP Environment with Edge Selection/Deletion
Generates random TSP instances following rl4co patterns
"""

import torch
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CustomTSPGenerator(Generator):
    """
    Data generator for Custom TSP environment.
    Generates TSP instances with random node coordinates.
    
    Args:
        num_loc: Number of nodes in the TSP instance
        min_loc: Minimum coordinate value
        max_loc: Maximum coordinate value
        loc_distribution: Distribution for sampling locations (uniform or normal)
    """
    
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: str = "uniform",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_distribution = loc_distribution
        
        log.info(
            f"Initialized CustomTSPGenerator with num_loc={num_loc}, "
            f"distribution={loc_distribution}"
        )
    
    def _generate(self, batch_size) -> TensorDict:
        """
        Generate batch of TSP instances
        
        Args:
            batch_size: List or int specifying batch dimensions
            
        Returns:
            TensorDict containing node locations
        """
        # Sample node coordinates
        if self.loc_distribution == "uniform":
            locs = torch.rand(
                (*batch_size, self.num_loc, 2),
                dtype=torch.float32
            ) * (self.max_loc - self.min_loc) + self.min_loc
        elif self.loc_distribution == "normal":
            locs = torch.randn(
                (*batch_size, self.num_loc, 2),
                dtype=torch.float32
            ) * 0.25 + 0.5  # Mean 0.5, std 0.25
            locs = torch.clamp(locs, self.min_loc, self.max_loc)
        else:
            raise ValueError(
                f"Unknown distribution: {self.loc_distribution}. "
                "Use 'uniform' or 'normal'."
            )
        
        log.debug(f"Generated batch with shape {locs.shape}")
        
        return TensorDict(
            {
                "locs": locs,
            },
            batch_size=batch_size,
        )
