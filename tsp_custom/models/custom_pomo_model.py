"""
Custom POMO Model for TSP with Edge Selection/Deletion

This model extends RL4CO's REINFORCE to work with our custom environment and policy.
It implements POMO-style training with:
- Shared baseline (mean of rewards across starts)
- Multi-start rollouts during validation/test
- Sequential action construction
- Custom policy integration

Architecture inspired by:
- Kwon et al. (2020) POMO: http://arxiv.org/abs/2010.16011
- Kool et al. (2019) Attention Model

Author: Step 8 implementation
Date: November 22, 2025
"""

import torch
import torch.nn as nn
from typing import Any, Optional
from tensordict import TensorDict

# Try to import from rl4co, fallback to basic Lightning if not available
try:
    from rl4co.models.rl import REINFORCE
    from rl4co.envs.common.base import RL4COEnvBase
    from rl4co.utils.ops import unbatchify
    USE_RL4CO = True
except ImportError:
    import lightning as L
    from lightning.pytorch import LightningModule as REINFORCE
    USE_RL4CO = False
    print("Warning: rl4co not found, using basic Lightning implementation")

import logging

log = logging.getLogger(__name__)


class CustomPOMOModel(REINFORCE if USE_RL4CO else nn.Module):
    """
    Custom POMO model for training with CustomTSPEnv and CustomPOMOPolicy.
    
    This model follows the POMO training paradigm:
    - During training: use multiple starts, shared baseline
    - During validation/test: evaluate multiple starts, report best
    
    The key difference from standard POMO is that we use a sequential
    environment where the agent selects or deletes edges one at a time,
    rather than the autoregressive node selection in standard TSP.
    
    Args:
        env: CustomTSPEnv instance
        policy: CustomPOMOPolicy instance
        baseline: Baseline type ('shared', 'rollout', etc.). POMO uses 'shared'
        num_starts: Number of multi-start rollouts (default: None, will use all starts)
        **kwargs: Additional arguments for REINFORCE base class
    """
    
    def __init__(
        self,
        env: Any,
        policy: nn.Module,
        baseline: str = 'shared',
        num_starts: int = None,
        **kwargs
    ):
        super().__init__(env=env, policy=policy, baseline=baseline, **kwargs)
        
        self.num_starts = num_starts
        
        # POMO only supports shared baseline
        if baseline != 'shared':
            log.warning(f"POMO requires shared baseline, but got '{baseline}'. Setting to 'shared'.")
            self.baseline_type = 'shared'
        
        log.info(f"Initialized CustomPOMOModel with {self.policy.__class__.__name__}")
        log.info(f"  Baseline: {baseline}")
        log.info(f"  Num starts: {num_starts if num_starts else 'auto'}")
    
    def shared_step(
        self,
        batch: Any,
        batch_idx: int,
        phase: str,
        dataloader_idx: Optional[int] = None
    ):
        """
        Shared step for train/val/test phases with POMO multi-start.
        
        POMO Strategy:
        - For each problem, run num_starts different rollouts
        - First start always uses greedy heuristic (ensures valid baseline)
        - Remaining starts use policy sampling with different random seeds
        - During val/test: report best solution per problem
        - During training: use all rollouts for learning (shared baseline)
        
        Args:
            batch: Batch of problem instances (locs)
            batch_idx: Batch index
            phase: 'train', 'val', or 'test'
            dataloader_idx: Dataloader index (for multiple dataloaders)
            
        Returns:
            Dictionary with loss and metrics
        """
        from tsp_custom.envs.utils import greedy_nearest_neighbor_action
        
        # Determine number of starts
        num_starts = self.num_starts if self.num_starts else 10  # Default 10 starts
        
        # Reset environment with original batch
        td_original = self.env.reset(batch)
        orig_batch_size = td_original.batch_size[0]
        
        # Replicate each problem instance num_starts times
        # Shape: (orig_batch_size * num_starts, ...)
        td_expanded = self._expand_for_multi_start(td_original, num_starts)
        expanded_batch_size = orig_batch_size * num_starts
        
        # Track which indices use greedy (first of each num_starts group)
        greedy_indices = torch.arange(0, expanded_batch_size, num_starts, device=td_expanded.device)
        
        # log.info(f"POMO multi-start: {orig_batch_size} problems × {num_starts} starts = {expanded_batch_size} rollouts")
        # log.info(f"Greedy indices (first of each group): {greedy_indices[:5].tolist()}... (total {len(greedy_indices)})")
        
        # Execute rollout
        all_log_probs = []
        step_count = 0
        max_steps = self.env.generator.num_loc  # 10 steps for 10 nodes (just enough to complete tour)
        
        while not td_expanded["done"].all() and step_count < max_steps:
            # Get policy output
            out = self.policy(
                td_expanded,
                phase=phase,
                decode_type='sampling'  # Always sample, we'll override greedy starts
            )
            
            action = out["action"].clone()  # Clone to avoid in-place modification issues
            log_prob = out['log_prob'].clone() if phase == 'train' else out['log_prob']
            
            # Override first start of each group with greedy heuristic
            for idx in greedy_indices:
                if not td_expanded["done"][idx]:
                    teacher_action_idx = greedy_nearest_neighbor_action(
                        td_expanded["locs"][idx],
                        td_expanded["adjacency"][idx],
                        td_expanded["degrees"][idx],
                        self.env.generator.num_loc
                    )
                    action[idx] = teacher_action_idx
                    
                    # Recompute log prob for greedy action
                    # Only if it's within the logits range (not DONE action)
                    if phase == 'train':
                        logits = out.get('logits')
                        if logits is not None and teacher_action_idx < logits.shape[-1]:
                            log_probs_all = torch.nn.functional.log_softmax(logits[idx], dim=-1)
                            log_prob[idx] = log_probs_all[teacher_action_idx]
                        # For DONE action or out of range, keep the original log_prob
            
            # Store log probabilities for REINFORCE
            if phase == 'train':
                all_log_probs.append(log_prob)
            
            # Step environment
            td_expanded.set("action", action)
            td_expanded = self.env.step(td_expanded)["next"]
            
            step_count += 1
        
        # If step limit reached, mark remaining episodes as done and compute rewards
        if step_count >= max_steps:
            not_done = ~td_expanded["done"]
            if not_done.any():
                # Mark as done
                td_expanded["done"] = torch.ones_like(td_expanded["done"])
                # Mark that they hit step limit
                hit_step_limit = td_expanded.get("hit_step_limit", torch.zeros_like(td_expanded["done"]))
                hit_step_limit[not_done] = True
                td_expanded["hit_step_limit"] = hit_step_limit
                # Compute rewards for all episodes
                rewards = self.env._get_reward_static(td_expanded)
                td_expanded["reward"] = rewards.unsqueeze(-1)
        
        # Get rewards from all starts: (orig_batch_size * num_starts,)
        all_rewards = td_expanded["reward"]
        
        # Check validity of greedy rollouts specifically
        greedy_rewards = all_rewards[greedy_indices]
        greedy_invalid = (greedy_rewards <= -99.0).sum().item()  # -100 penalty for invalid tours
        if greedy_invalid > 0:
            log.warning(f"WARNING: {greedy_invalid}/{len(greedy_indices)} greedy rollouts are INVALID! This should not happen.")
        
        # Reshape to (orig_batch_size, num_starts)
        rewards_per_start = all_rewards.view(orig_batch_size, num_starts)
        
        # For val/test: select best reward per problem
        if phase in ['val', 'test']:
            best_rewards, best_indices = rewards_per_start.max(dim=1)  # (orig_batch_size,)
            rewards = best_rewards
        else:
            # For training: use all starts (flatten back)
            rewards = all_rewards
        
        # Extract metrics
        num_invalid_tours = td_expanded.get("num_invalid_tours", torch.zeros(1))[0].item()
        num_step_limit_hits = td_expanded.get("num_step_limit_hits", torch.zeros(1))[0].item()

        # Compute metrics
        out = {
            "reward": rewards,
            "tour_length": -rewards,
            "invalid_tour_rate": num_invalid_tours / expanded_batch_size if expanded_batch_size > 0 else 0.0,
            "step_limit_rate": num_step_limit_hits / expanded_batch_size if expanded_batch_size > 0 else 0.0,
        }
        
        # Compute loss for training
        if phase == 'train':
            # Stack log probabilities: (expanded_batch_size, num_steps)
            log_likelihood = torch.stack(all_log_probs, dim=1).sum(dim=1)
            
            # Split into greedy and policy rollouts
            greedy_mask = torch.zeros(expanded_batch_size, dtype=torch.bool, device=all_rewards.device)
            greedy_mask[greedy_indices] = True
            policy_mask = ~greedy_mask
            
            # Greedy rollouts: supervised learning (behavior cloning)
            # We want to maximize the probability of greedy actions regardless of reward
            if greedy_mask.sum() > 0:
                greedy_log_likelihood = log_likelihood[greedy_mask]
                # Negative because we want to maximize (minimize negative log likelihood)
                greedy_loss = -greedy_log_likelihood.mean()
            else:
                greedy_loss = torch.tensor(0.0, device=all_rewards.device)
            
            # Policy rollouts: REINFORCE with baseline
            if policy_mask.sum() > 0:
                policy_log_likelihood = log_likelihood[policy_mask]
                policy_rewards = all_rewards[policy_mask]
                
                # Per-instance baseline: mean reward across starts for each instance
                # Reshape rewards to (orig_batch_size, num_starts)
                rewards_per_instance = all_rewards.view(orig_batch_size, num_starts)
                # Compute mean for each instance (excluding greedy start at index 0)
                baseline_per_instance = rewards_per_instance[:, 1:].mean(dim=1)  # (orig_batch_size,)
                # Expand to match all policy rollouts
                # Each instance has (num_starts - 1) policy rollouts
                baseline_val = baseline_per_instance.repeat_interleave(num_starts - 1)  # (orig_batch_size * (num_starts-1),)
                
                # REINFORCE loss
                advantage = policy_rewards - baseline_val
                policy_loss = -(advantage * policy_log_likelihood).mean()
            else:
                policy_loss = torch.tensor(0.0, device=all_rewards.device)
                baseline_val = torch.tensor(0.0, device=all_rewards.device)
            
            # Combined loss: weighted sum of greedy (supervised) and policy (RL) losses
            # Give more weight to supervised learning early in training
            greedy_weight = 1.0
            policy_weight = 0.5
            loss = greedy_weight * greedy_loss + policy_weight * policy_loss
            
            out.update({
                "loss": loss,
                "greedy_loss": greedy_loss,
                "policy_loss": policy_loss,
                "baseline_val": baseline_val,
                "log_likelihood": log_likelihood.mean(),
            })
        
        # Log metrics explicitly
        if phase == 'train':
            self.log(f"{phase}/reward", rewards.mean(), prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{phase}/tour_length", (-rewards).mean(), prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{phase}/loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{phase}/greedy_loss", out["greedy_loss"], on_step=True, on_epoch=True)
            self.log(f"{phase}/policy_loss", out["policy_loss"], on_step=True, on_epoch=True)
            baseline_val = out["baseline_val"]
            baseline_val_scalar = baseline_val.mean() if baseline_val.numel() > 1 else baseline_val
            self.log(f"{phase}/baseline_val", baseline_val_scalar, on_step=True, on_epoch=True)
            self.log(f"{phase}/log_likelihood", out["log_likelihood"], on_step=True, on_epoch=True)
            self.log(f"{phase}/invalid_tour_rate", out["invalid_tour_rate"], on_step=True, on_epoch=True)
            self.log(f"{phase}/step_limit_rate", out["step_limit_rate"], on_step=True, on_epoch=True)
        else:
            self.log(f"{phase}/reward", rewards.mean(), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}/tour_length", (-rewards).mean(), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}/invalid_tour_rate", out["invalid_tour_rate"], on_step=False, on_epoch=True)
            self.log(f"{phase}/step_limit_rate", out["step_limit_rate"], on_step=False, on_epoch=True)
        
        return {"loss": out.get("loss", None)}
    
    def _expand_for_multi_start(self, td: TensorDict, num_starts: int) -> TensorDict:
        """
        Expand TensorDict to replicate each problem instance num_starts times.
        
        Args:
            td: Original TensorDict with batch_size (N,)
            num_starts: Number of starts per problem
            
        Returns:
            Expanded TensorDict with batch_size (N * num_starts,)
        """
        batch_size = td.batch_size[0]
        device = td.device
        
        # Build dict of expanded tensors
        expanded_dict = {}
        
        for key in td.keys():
            value = td[key]
            if isinstance(value, torch.Tensor):
                # Repeat each instance num_starts times
                # Shape: (N, ...) -> (N, num_starts, ...) -> (N * num_starts, ...)
                expanded = value.unsqueeze(1).repeat(1, num_starts, *([1] * (value.dim() - 1)))
                expanded = expanded.view(batch_size * num_starts, *value.shape[1:])
                expanded_dict[key] = expanded
            else:
                expanded_dict[key] = value
        
        # Create new TensorDict with correct batch size
        td_expanded = TensorDict(
            expanded_dict,
            batch_size=[batch_size * num_starts],
            device=device
        )
        
        return td_expanded
    
    def on_train_epoch_end(self):
        """
        Called at end of training epoch.
        Update delete bias in policy for curriculum learning.
        """
        super().on_train_epoch_end()
        
        # Update delete bias schedule
        self.policy.set_epoch(self.current_epoch)
        current_bias = self.policy.get_delete_bias('train')
        log.info(f"Epoch {self.current_epoch}: delete_bias = {current_bias:.4f}")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.hparams.get('lr', 1e-4),
            weight_decay=self.hparams.get('weight_decay', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
        }


# Fallback implementation if rl4co not available
if not USE_RL4CO:
    import lightning as L
    
    class CustomPOMOModel(L.LightningModule):
        """
        Fallback implementation without rl4co.
        Implements basic REINFORCE with manual training loop.
        """
        
        def __init__(
            self,
            env: Any,
            policy: nn.Module,
            baseline: str = 'shared',
            batch_size: int = 64,
            val_batch_size: int = 1024,
            test_batch_size: int = 1024,
            train_data_size: int = 1_280_000,
            val_data_size: int = 10_000,
            test_data_size: int = 10_000,
            optimizer_kwargs: dict = None,
            **kwargs
        ):
            super().__init__()
            self.save_hyperparameters(ignore=['env', 'policy'])
            
            self.env = env
            self.policy = policy
            self.baseline_type = baseline
            
            self.batch_size = batch_size
            self.val_batch_size = val_batch_size
            self.test_batch_size = test_batch_size
            
            self.train_data_size = train_data_size
            self.val_data_size = val_data_size
            self.test_data_size = test_data_size
            
            self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-4, 'weight_decay': 1e-6}
            
            log.info(f"Initialized CustomPOMOModel (fallback mode without rl4co)")
        
        def forward(self, td, **kwargs):
            """Forward pass through policy."""
            return self.policy(td, **kwargs)
        
        def training_step(self, batch, batch_idx):
            """Training step."""
            out = self.shared_step(batch, batch_idx, 'train')
            return out['loss']
        
        def validation_step(self, batch, batch_idx):
            """Validation step."""
            return self.shared_step(batch, batch_idx, 'val')
        
        def test_step(self, batch, batch_idx):
            """Test step."""
            return self.shared_step(batch, batch_idx, 'test')
        
        def shared_step(self, batch, batch_idx, phase):
            """Shared step implementation (same as above)."""
            td = self.env.reset(batch)
            
            all_log_probs = []
            step_count = 0
            max_steps = 2 * self.env.generator.num_loc
            
            while not td["done"].all() and step_count < max_steps:
                out = self.policy(
                    td,
                    phase=phase,
                    decode_type='sampling' if phase == 'train' else 'greedy'
                )
                
                if phase == 'train':
                    all_log_probs.append(out['log_prob'])
                
                td.set("action", out["action"])
                td = self.env.step(td)["next"]
                step_count += 1
            
            rewards = td["reward"]
            tour_lengths = -rewards
            
            # Log metrics - use on_step=True for train to see immediate progress
            on_step = (phase == 'train')
            self.log(f"{phase}/reward", rewards.mean(), prog_bar=True, on_step=on_step, on_epoch=True)
            self.log(f"{phase}/tour_length", tour_lengths.mean(), prog_bar=True, on_step=on_step, on_epoch=True)
            
            if phase == 'train':
                if len(all_log_probs) == 0:
                    log.warning("No log probs collected! Episode may have finished immediately.")
                    return {'loss': torch.tensor(0.0, device=rewards.device)}
                
                log_likelihood = torch.stack(all_log_probs, dim=1).sum(dim=1)
                
                # Check for NaN/Inf in log_likelihood
                if torch.isnan(log_likelihood).any() or torch.isinf(log_likelihood).any():
                    log.error(f"Invalid log_likelihood detected! NaN: {torch.isnan(log_likelihood).sum()}, Inf: {torch.isinf(log_likelihood).sum()}")
                    # Replace NaN/Inf with very negative value
                    log_likelihood = torch.where(
                        torch.isnan(log_likelihood) | torch.isinf(log_likelihood),
                        torch.tensor(-1e10, device=log_likelihood.device),
                        log_likelihood
                    )
                
                # Check for NaN/Inf in rewards
                if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                    log.error(f"Invalid rewards detected! NaN: {torch.isnan(rewards).sum()}, Inf: {torch.isinf(rewards).sum()}")
                    # Replace NaN/Inf with very negative reward
                    rewards = torch.where(
                        torch.isnan(rewards) | torch.isinf(rewards),
                        torch.tensor(-100.0, device=rewards.device),
                        rewards
                    )
                
                baseline_val = rewards.mean()
                advantage = rewards - baseline_val
                
                # Normalize advantages for training stability
                # Prevents extreme reward scales from causing unstable gradients
                if advantage.numel() > 1:
                    adv_mean = advantage.mean()
                    adv_std = advantage.std() + 1e-8
                    normalized_advantage = (advantage - adv_mean) / adv_std
                else:
                    normalized_advantage = advantage
                
                loss = -(normalized_advantage * log_likelihood).mean()
                
                # Final NaN check on loss
                if torch.isnan(loss) or torch.isinf(loss):
                    log.error(f"NaN/Inf loss detected! Setting to 0. normalized_advantage: {normalized_advantage[:5]}, log_likelihood: {log_likelihood[:5]}")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                
                # Log both raw and normalized advantages for debugging
                self.log(f"{phase}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
                self.log(f"{phase}/advantage_raw", advantage.mean(), on_step=True, on_epoch=True)
                self.log(f"{phase}/advantage_std", advantage.std(), on_step=True, on_epoch=True)
                self.log(f"{phase}/baseline", baseline_val, on_step=False, on_epoch=True)
                
                return {'loss': loss}
            
            return {'reward': rewards.mean()}
        
        def configure_optimizers(self):
            """Configure optimizer."""
            return torch.optim.Adam(
                self.policy.parameters(),
                **self.optimizer_kwargs
            )
        
        def train_dataloader(self):
            """Create training dataloader."""
            # Generate random TSP instances
            from torch.utils.data import DataLoader, TensorDataset
            
            locs = torch.rand(self.train_data_size, self.env.generator.num_loc, 2)
            dataset = TensorDataset(locs)
            
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True
            )
        
        def val_dataloader(self):
            """Create validation dataloader."""
            from torch.utils.data import DataLoader, TensorDataset
            
            # Use fixed seed for validation
            torch.manual_seed(1234)
            locs = torch.rand(self.val_data_size, self.env.generator.num_loc, 2)
            dataset = TensorDataset(locs)
            
            return DataLoader(
                dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True
            )
        
        def test_dataloader(self):
            """Create test dataloader."""
            from torch.utils.data import DataLoader, TensorDataset
            
            # Use different fixed seed for test
            torch.manual_seed(5678)
            locs = torch.rand(self.test_data_size, self.env.generator.num_loc, 2)
            dataset = TensorDataset(locs)
            
            return DataLoader(
                dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True
            )
        
        def on_train_epoch_end(self):
            """Update delete bias schedule."""
            self.policy.set_epoch(self.current_epoch)
            current_bias = self.policy.get_delete_bias('train')
            log.info(f"Epoch {self.current_epoch}: delete_bias = {current_bias:.4f}")
        
        def log_metrics(self, out, phase, **kwargs):
            """Log metrics."""
            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    self.log(f"{phase}/{key}", value.mean() if value.numel() > 1 else value)
            return {}


if __name__ == '__main__':
    # Test model creation
    print("Testing CustomPOMOModel...")
    
    from tsp_custom.envs import CustomTSPEnv, CustomTSPGenerator
    from tsp_custom.models import CustomPOMOPolicy
    
    # Create components
    num_loc = 20
    generator = CustomTSPGenerator(num_loc=num_loc)
    env = CustomTSPEnv(generator=generator)
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Create model
    model = CustomPOMOModel(
        env=env,
        policy=policy,
        batch_size=4,
        train_data_size=100,
        val_data_size=20,
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ CustomPOMOModel test passed!")
