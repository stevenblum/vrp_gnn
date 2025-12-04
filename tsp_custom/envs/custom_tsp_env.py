"""
Custom TSP Environment with Global Edge Selection and Deletion
Minimal viable implementation for Step 3 of development plan
"""

from typing import Optional
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from .generator import CustomTSPGenerator
from .utils import (
    compute_tour_length,
    check_tour_validity,
    is_graph_connected,
    decode_action,
    encode_action,
)

log = get_pylogger(__name__)


class CustomTSPEnv(RL4COEnvBase):
    """
    Custom TSP Environment with global edge selection and deletion.
    
    At each step, the agent can:
    - ADD an edge between any two nodes (global selection)
    - DELETE a previously added edge
    - Finish (DONE) when a valid tour is constructed
    
    Observations:
        - Node locations (x, y coordinates)
        - Adjacency matrix (selected edges)
        - Node degrees
        - Step counter and deletion counter
        - Action mask (feasible actions)
    
    Constraints:
        - Each node must have degree exactly 2 (hard constraint via masking)
        - Graph must be connected when DONE is selected
        - Maximum 2*N steps (hard limit)
    
    Reward:
        - Sparse: only at episode end
        - -(tour_length + deletion_penalty + limit_penalty)
        - deletion_penalty = 0.002 * avg_distance * num_deletions
    
    Args:
        generator: CustomTSPGenerator instance
        generator_params: Parameters for the generator
        max_steps_multiplier: Maximum steps as multiple of N (default 2)
        deletion_penalty_factor: Penalty factor for deletions (default 0.002 = 0.2%)
        delete_every_n_steps: Only allow deletions every N steps (default 4, set to 1 to allow always)
    """
    
    name = "custom_tsp"
    
    def __init__(
        self,
        generator: CustomTSPGenerator = None,
        generator_params: dict = {},
        max_steps_multiplier: int = 2,
        deletion_penalty_factor: float = 0.002,
        delete_every_n_steps: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if generator is None:
            generator = CustomTSPGenerator(**generator_params)
        
        self.generator = generator
        self.max_steps_multiplier = max_steps_multiplier
        self.deletion_penalty_factor = deletion_penalty_factor
        self.delete_every_n_steps = delete_every_n_steps
        
        # Calculate maximum edges possible (for action space sizing)
        self.num_loc = generator.num_loc
        self.max_edges = self.num_loc  # Complete tour has N edges
        self.max_steps = self.max_steps_multiplier * self.num_loc
        
        # Pre-calculate number of possible ADD actions (combinations of 2 nodes)
        self.num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        
        self._make_spec(generator)
        
        log.info(
            f"Initialized CustomTSPEnv: N={self.num_loc}, "
            f"max_steps={self.max_steps}, "
            f"num_add_actions={self.num_add_actions}, "
            f"delete_every_n_steps={self.delete_every_n_steps}"
        )
    
    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        """
        Execute one step of the environment.
        
        Args:
            td: TensorDict containing current state and action
            
        Returns:
            Updated TensorDict with new state
        """
        # Extract current state - detach and clone to avoid gradient issues with in-place ops
        current_step = td["current_step"]
        adjacency = td["adjacency"].detach().clone()
        degrees = td["degrees"].detach().clone()
        num_edges = td["num_edges"].detach().clone()
        num_deletions = td["num_deletions"].detach().clone()
        action_idx = td["action"]
        
        batch_size = adjacency.shape[0]
        num_loc = adjacency.shape[1]
        max_steps = num_loc  # !!! Need to change if we allow deletions
        
        # Decode action from index to (action_type, node_i, node_j)
        action_type, node_i, node_j = decode_action(
            action_idx, adjacency, num_loc
        )
        
        # Log the action (only for first item in batch to avoid clutter)
        if batch_size > 0:
            log.debug(
                f"Step {current_step[0].item()}: action_type={action_type[0].item()}, "
                f"node_i={node_i[0].item()}, node_j={node_j[0].item()}"
            )
        
        # Initialize done flag
        done = torch.zeros(batch_size, dtype=torch.bool, device=adjacency.device)
        
        # Track constraint violations for batch statistics
        num_degree_violations = 0
        num_invalid_done_degrees = 0
        num_invalid_done_disconnected = 0
        
        # Process each action type
        for b in range(batch_size):
            a_type = action_type[b].item()
            i = node_i[b].item()
            j = node_j[b].item()
            
            if a_type == 0:  # ADD
                if i >= 0 and j >= 0:  # Valid node indices
                    # Execute action without validation - model learns from penalties
                    adjacency[b, i, j] = 1
                    adjacency[b, j, i] = 1
                    degrees[b, i] += 1
                    degrees[b, j] += 1
                    num_edges[b] += 1
                    log.debug(f"Batch {b}: Added edge ({i}, {j})")
            
            elif a_type == 1:  # DELETE
                if i >= 0 and j >= 0:
                    adjacency[b, i, j] = 0
                    adjacency[b, j, i] = 0
                    degrees[b, i] -= 1
                    degrees[b, j] -= 1
                    num_edges[b] -= 1
                    num_deletions[b] += 1
                    log.debug(f"Batch {b}: Deleted edge ({i}, {j})")
            
            elif a_type == 2:  # DONE
                done[b] = True
                log.debug(f"Batch {b}: DONE selected")
                
                # Track invalid DONE actions (should be prevented by masking)
                all_degree_2 = (degrees[b] == 2).all().item()
                if not all_degree_2:
                    num_invalid_done_degrees += 1
                    log.debug(f"Batch {b}: DONE with invalid degrees")
                elif all_degree_2:
                    # Only check connectivity if all degrees are 2 (expensive operation)
                    if not is_graph_connected(adjacency[b]):
                        num_invalid_done_disconnected += 1
                        log.debug(f"Batch {b}: DONE with disconnected graph")
        
        # Increment step counter
        current_step = current_step + 1
        
        # Store violation counts with batch dimension for TensorDict compatibility
        # Create tensors with batch dimension (same value repeated)
        device = adjacency.device

        # Update TensorDict with current state
        td.update({
            "adjacency": adjacency,
            "degrees": degrees,
            "num_edges": num_edges,
            "num_deletions": num_deletions,
            "current_step": current_step,
            "done": done,
        })
        
        # Compute reward (sparse - only when episode is done)
        # current_step is a tensor, check if any episodes are done
        if done.any():
            # Compute reward for finished episodes
            rewards_full = CustomTSPEnv._get_reward_static(td)
            # Zero out rewards for episodes that aren't done yet
            reward = torch.where(
                done.unsqueeze(-1),
                rewards_full.unsqueeze(-1),
                torch.zeros(batch_size, 1, dtype=torch.float32, device=adjacency.device)
            )
        else:
            reward = torch.zeros(batch_size, 1, dtype=torch.float32, device=adjacency.device)
        
        td.set("reward", reward)
        
        # Update action mask for next step
        # Note: This will use delete_every_n_steps from td
        td.set("action_mask", CustomTSPEnv._get_action_mask_static(td))
        
        return td
    
    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None
    ) -> TensorDict:
        """
        Reset environment to initial state.
        
        Args:
            td: TensorDict with problem instances (contains 'locs')
            batch_size: Batch size for the environment
            
        Returns:
            TensorDict with initial state
        """
        device = td.device
        locs = td["locs"]
        batch_dims = locs.shape[:-2]
        num_loc = locs.shape[-2]
        
        # Initialize adjacency matrix (all zeros = no edges selected)
        adjacency = torch.zeros(
            (*batch_dims, num_loc, num_loc),
            dtype=torch.bool,
            device=device
        )
        
        # Initialize degrees (all zeros = no connections)
        degrees = torch.zeros(
            (*batch_dims, num_loc),
            dtype=torch.long,
            device=device
        )
        
        # Initialize counters
        current_step = torch.zeros((*batch_dims, 1), dtype=torch.long, device=device)
        num_deletions = torch.zeros((*batch_dims, 1), dtype=torch.long, device=device)
        num_edges = torch.zeros((*batch_dims, 1), dtype=torch.long, device=device)
        
        # Initialize done and hit_step_limit flags
        done = torch.zeros(*batch_dims, dtype=torch.bool, device=device)
        hit_step_limit = torch.zeros(*batch_dims, dtype=torch.bool, device=device)
        
        # Store delete frequency as a parameter
        delete_every_n_steps = torch.full(
            (*batch_dims, 1), 
            self.delete_every_n_steps, 
            dtype=torch.long, 
            device=device
        )
        
        # Create initial TensorDict
        td_reset = TensorDict(
            {
                "locs": locs,
                "adjacency": adjacency,
                "degrees": degrees,
                "num_edges": num_edges,
                "current_step": current_step,
                "num_deletions": num_deletions,
                "done": done,
                "hit_step_limit": hit_step_limit,
                "delete_every_n_steps": delete_every_n_steps,
                "reward": torch.zeros((*batch_dims, 1), dtype=torch.float32, device=device),
            },
            batch_size=batch_dims,
        )
        
        # Compute initial action mask
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        
        # log.info(f"Reset environment: batch_size={batch_dims}, num_loc={num_loc}")
        
        return td_reset
    
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask for current state.
        
        Masking rules:
        - ADD(i,j): allowed if edge doesn't exist and neither node has degree 2
        - DELETE(i,j): allowed if edge exists AND step % delete_every_n_steps == 0
        - DONE: allowed if valid tour exists (all degrees == 2 and connected)
        
        Returns:
            Boolean tensor where True = action is feasible
        """
        adjacency = td["adjacency"]
        degrees = td["degrees"]
        locs = td["locs"]
        current_step = td["current_step"]
        
        # Get delete frequency - from td if available (for static call), else from self
        if "delete_every_n_steps" in td.keys():
            delete_every_n_steps = td["delete_every_n_steps"][0, 0].item()
        else:
            delete_every_n_steps = self.delete_every_n_steps
        
        batch_size = adjacency.shape[0]
        num_loc = adjacency.shape[1]
        device = adjacency.device
        
        # Calculate number of each action type
        num_add_actions = num_loc * (num_loc - 1) // 2
        num_delete_actions = adjacency.sum(dim=(-2, -1)) // 2  # Each edge counted twice
        max_delete_actions = num_loc  # Maximum possible edges
        
        # Total action space: ADD actions + DELETE actions + DONE
        total_actions = num_add_actions + max_delete_actions + 1
        
        # Initialize mask (all actions masked by default)
        mask = torch.zeros(
            (batch_size, total_actions),
            dtype=torch.bool,
            device=device
        )
        
        # Process each batch element
        for b in range(batch_size):
            action_idx = 0
            
            # Mask ADD actions
            for i in range(num_loc):
                for j in range(i + 1, num_loc):
                    # ADD is feasible if:
                    # 1. Edge doesn't exist
                    # 2. Neither node has degree 2
                    can_add = (
                        adjacency[b, i, j] == 0 and
                        degrees[b, i] < 2 and
                        degrees[b, j] < 2
                    )
                    mask[b, action_idx] = can_add
                    action_idx += 1
            
            # Mask DELETE actions
            # Only allow deletions every N steps to prevent add/delete loops
            can_delete_this_step = (current_step[b, 0].item() % delete_every_n_steps == 0)
            
            # Track which edges exist for this batch element
            edge_count = 0
            for i in range(num_loc):
                for j in range(i + 1, num_loc):
                    if adjacency[b, i, j] == 1:
                        # Safety check: don't exceed allocated delete action slots
                        if edge_count < max_delete_actions:
                            mask[b, num_add_actions + edge_count] = can_delete_this_step
                            edge_count += 1
                        else:
                            log.warning(
                                f"Instance {b}: Too many edges ({edge_count+1}) for delete action space ({max_delete_actions}). "
                                f"This indicates degree constraints were violated."
                            )
                            break
                if edge_count >= max_delete_actions:
                    break
            
            # DONE action - check if valid tour exists
            # Valid tour: all degrees == 2 AND graph is connected
            all_degree_2 = (degrees[b] == 2).all()
            if all_degree_2:
                # Only check connectivity if degree condition met (optimization)
                if is_graph_connected(adjacency[b]):
                    mask[b, -1] = True  # DONE action is feasible
        
        # Safety: If no actions are feasible, force DONE to be available
        # This prevents getting stuck in invalid states
        for b in range(batch_size):
            if not mask[b].any():
                log.warning(
                    f"Instance {b}: No valid actions available! Forcing DONE. "
                    f"Degrees: {degrees[b].tolist()}, Edges: {adjacency[b].sum().item() // 2}"
                )
                mask[b, -1] = True
        
        log.debug(f"Action mask computed: {mask.sum(dim=-1)} feasible actions")
        
        return mask
    
    @staticmethod
    def _get_action_mask_static(td: TensorDict) -> torch.Tensor:
        """
        Static version of get_action_mask for use in _step.
        
        This is needed because _step is static but we need access to delete_every_n_steps.
        The parameter is stored in the TensorDict during reset.
        """
        adjacency = td["adjacency"]
        degrees = td["degrees"]
        locs = td["locs"]
        current_step = td["current_step"]
        delete_every_n_steps = td["delete_every_n_steps"][0, 0].item()
        
        batch_size = adjacency.shape[0]
        num_loc = adjacency.shape[1]
        device = adjacency.device
        
        # Calculate number of each action type
        num_add_actions = num_loc * (num_loc - 1) // 2
        max_delete_actions = num_loc
        
        # Total action space
        total_actions = num_add_actions + max_delete_actions + 1
        
        # Initialize mask
        mask = torch.zeros(
            (batch_size, total_actions),
            dtype=torch.bool,
            device=device
        )
        
        # Process each batch element
        for b in range(batch_size):
            action_idx = 0
            
            # Mask ADD actions
            for i in range(num_loc):
                for j in range(i + 1, num_loc):
                    can_add = (
                        adjacency[b, i, j] == 0 and
                        degrees[b, i] < 2 and
                        degrees[b, j] < 2
                    )
                    mask[b, action_idx] = can_add
                    action_idx += 1
            
            # Mask DELETE actions (with frequency check)
            can_delete_this_step = (current_step[b, 0].item() % delete_every_n_steps == 0)
            
            edge_count = 0
            for i in range(num_loc):
                for j in range(i + 1, num_loc):
                    if adjacency[b, i, j] == 1:
                        # Safety check: don't exceed allocated delete action slots
                        if edge_count < max_delete_actions:
                            mask[b, num_add_actions + edge_count] = can_delete_this_step
                            edge_count += 1
                        else:
                            # Silently skip (warning already issued during action execution)
                            break
                if edge_count >= max_delete_actions:
                    break
            
            # DONE action
            all_degree_2 = (degrees[b] == 2).all()
            if all_degree_2:
                if is_graph_connected(adjacency[b]):
                    mask[b, -1] = True
        
        # Safety: If no actions are feasible, force DONE to be available
        for b in range(batch_size):
            if not mask[b].any():
                mask[b, -1] = True
        
        return mask
    
    @staticmethod
    def _get_reward_static(td: TensorDict, deletion_penalty_factor: float = 0.002) -> torch.Tensor:
        """
        Static version of reward computation for use in _step.
        
        Compute reward for the episode.
        
        Reward is sparse - only computed at episode termination.
        - If valid tour: -(tour_length + deletion_penalty)
        - If invalid tour: large negative penalty
        - If hit step limit: additional penalty
        
        Args:
            td: TensorDict containing final state
            deletion_penalty_factor: Penalty factor for deletions
            
        Returns:
            Reward tensor
        """
        locs = td["locs"]
        adjacency = td["adjacency"]
        num_deletions = td["num_deletions"]
        hit_step_limit = td["hit_step_limit"]
        
        batch_size = locs.shape[0]
        device = locs.device
        
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Track statistics for summary
        num_invalid = 0
        num_step_limit = 0
        
        # Check tour validity and compute rewards
        for b in range(batch_size):
            is_valid = check_tour_validity(adjacency[b], locs.shape[1])
            
            # Debug: check what's actually in the tour
            num_edges = (adjacency[b].sum() / 2).int().item()
            degrees_list = adjacency[b].sum(dim=0).tolist()
            
            if is_valid:
                # Compute tour length
                tour_length = compute_tour_length(locs[b], adjacency[b])
                
                # Compute average pairwise distance for deletion penalty
                # avg_dist = mean distance between all pairs of nodes
                dists = torch.cdist(locs[b], locs[b])
                avg_dist = dists.sum() / (locs.shape[1] * (locs.shape[1] - 1))
                
                # Deletion penalty: 0.2% of average distance per deletion
                deletion_penalty = deletion_penalty_factor * avg_dist * num_deletions[b].item()
                
                # Reward is negative cost (we want to minimize)
                rewards[b] = -(tour_length + deletion_penalty)
                
                log.debug(
                    f"Instance {b}: Valid tour - length={tour_length:.4f}, "
                    f"deletions={num_deletions[b].item()}, "
                    f"penalty={deletion_penalty:.4f}, edges={num_edges}"
                )
            else:
                # Invalid tour: penalty must be worse than any possible valid tour
                # -100 is sufficient for normalized coordinates (tours typically < 10)
                rewards[b] = -100.0
                num_invalid += 1
                log.debug(f"Instance {b}: Invalid tour - edges={num_edges}, degrees={degrees_list}")
            
            # Additional penalty for hitting step limit
            # Smaller penalty than invalid tour to encourage completion attempts
            #if hit_step_limit[b]:
            #    rewards[b] += -50.0
            #    num_step_limit += 1
            #    log.debug(f"Instance {b}: Hit step limit")
        
        # Get violation counts from TensorDict (take first element since they're the same)
        num_degree_violations = td.get("num_degree_violations", torch.zeros(1))[0].item() if "num_degree_violations" in td.keys() else 0
        num_invalid_done_degrees = td.get("num_invalid_done_degrees", torch.zeros(1))[0].item() if "num_invalid_done_degrees" in td.keys() else 0
        num_invalid_done_disconnected = td.get("num_invalid_done_disconnected", torch.zeros(1))[0].item() if "num_invalid_done_disconnected" in td.keys() else 0
        
        # Store metrics in TensorDict for logging (expand to match batch dimension)
        td.set("num_invalid_tours", torch.full((batch_size,), num_invalid, dtype=torch.long, device=rewards.device))
        td.set("num_step_limit_hits", torch.full((batch_size,), num_step_limit, dtype=torch.long, device=rewards.device))
        
        # Summary log at info level
        log.info(
            f"Batch rewards: mean={rewards.mean():.2f}, std={rewards.std():.2f} | "
            f"Invalid tours: {num_invalid}/{batch_size} | "
            f"Step limit: {num_step_limit}/{batch_size} | "
            f"Violations - Degree>2: {num_degree_violations}, Invalid DONE (deg): {num_invalid_done_degrees}, "
            f"Invalid DONE (disconn): {num_invalid_done_disconnected}"
        )
        
        return rewards

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for the episode (instance method wrapper).
        
        Args:
            td: TensorDict containing final state
            actions: Tensor of actions taken (not used for this environment)
            
        Returns:
            Reward tensor
        """
        return self._get_reward_static(td, self.deletion_penalty_factor)
    
    def _make_spec(self, generator: CustomTSPGenerator):
        """
        Make the observation and action specs from the parameters.
        
        Defines the structure and constraints of observations and actions.
        """
        num_loc = generator.num_loc
        num_add_actions = num_loc * (num_loc - 1) // 2
        max_delete_actions = num_loc
        total_actions = num_add_actions + max_delete_actions + 1  # +1 for DONE
        
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(num_loc, 2),
                dtype=torch.float32,
            ),
            adjacency=Unbounded(
                shape=(num_loc, num_loc),
                dtype=torch.bool,
            ),
            degrees=Bounded(
                low=0,
                high=2,
                shape=(num_loc,),
                dtype=torch.long,
            ),
            num_edges=Bounded(
                low=0,
                high=num_loc,
                shape=(1,),
                dtype=torch.long,
            ),
            current_step=Bounded(
                low=0,
                high=self.max_steps,
                shape=(1,),
                dtype=torch.long,
            ),
            num_deletions=Unbounded(
                shape=(1,),
                dtype=torch.long,
            ),
            action_mask=Unbounded(
                shape=(total_actions,),
                dtype=torch.bool,
            ),
            shape=(),
        )
        
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=total_actions,
        )
        
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)
        
        log.info(
            f"Specs created: total_actions={total_actions}, "
            f"num_add={num_add_actions}, max_delete={max_delete_actions}"
        )
    
    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        """
        Render the current state (to be implemented in visualization module).
        """
        raise NotImplementedError(
            "Rendering will be implemented in the visualization module (Step 5)"
        )
