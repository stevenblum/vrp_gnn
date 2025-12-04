"""
Lightning Callbacks for Visualization and Metrics Plotting (Step 9)

Callbacks to create visualizations during validation and plot training/validation
metrics. Outputs are saved to the Lightning checkpoint directory with epoch prefixes.

Author: Step 9 implementation
Date: November 22, 2025
"""


import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import PillowWriter
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from lightning.pytorch.callbacks import Callback
from tensordict import TensorDict

log = logging.getLogger(__name__)


class ValidationVisualizationCallback(Callback):
    """
    Creates animated GIFs of validation episodes during training.
    
    This callback:
    - Runs a few validation instances through the model
    - Records the action sequence (ADD/DELETE edges)
    - Creates animated GIFs showing the tour construction
    - Saves GIFs with epoch prefix to checkpoint directory
    
    Args:
        num_instances: Number of validation instances to visualize per epoch
        save_dir: Directory to save GIFs (default: checkpoint_dir)
        fps: Frames per second for GIF animation
        figsize: Figure size for plots
        every_n_epochs: Only visualize every N epochs (default: 1)
    """
    
    def __init__(
        self,
        num_instances: int = 3,
        save_dir: Optional[str] = None,
        fps: int = 2,
        figsize: tuple = (10, 10),
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.num_instances = num_instances
        self.save_dir = save_dir
        self.fps = fps
        self.figsize = figsize
        self.every_n_epochs = every_n_epochs
        
        log.info(
            f"ValidationVisualizationCallback initialized: "
            f"num_instances={num_instances}, fps={fps}, every_n_epochs={every_n_epochs}"
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        current_epoch = trainer.current_epoch
        
        # Only visualize every N epochs
        if current_epoch % self.every_n_epochs != 0:
            return
        
        log.info(f"Creating validation visualizations for epoch {current_epoch}...")
        
        # Determine save directory - use Lightning logs directory
        if self.save_dir is None:
            save_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        else:
            save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get validation dataset
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            log.warning("No validation dataloader available, skipping visualization")
            return
        
        # Get a batch from validation set
        batch = next(iter(val_dataloader))
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        # Move to device
        device = pl_module.device
        if isinstance(batch, TensorDict):
            batch = batch.to(device)
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        
        # Get locs
        if isinstance(batch, TensorDict):
            locs = batch["locs"] if "locs" in batch.keys() else batch
        else:
            locs = batch
        
        # Select subset of instances
        num_to_viz = min(self.num_instances, locs.shape[0])
        locs_subset = locs[:num_to_viz]
        
        # Run episodes and record actions
        for inst_idx in range(num_to_viz):
            try:
                self._create_episode_gif(
                    pl_module,
                    locs_subset[inst_idx:inst_idx+1],
                    save_dir,
                    current_epoch,
                    inst_idx
                )
            except Exception as e:
                log.error(f"Failed to create GIF for instance {inst_idx}: {e}")
        
        log.info(f"Saved {num_to_viz} validation GIFs to {save_dir}")
    
    def _create_episode_gif(
        self,
        pl_module,
        locs: torch.Tensor,
        save_dir: Path,
        epoch: int,
        instance_idx: int
    ):
        """Create GIF for a single validation instance."""
        # Reset environment
        env = pl_module.env
        policy = pl_module.policy
        
        # Create initial TensorDict - ensure on correct device
        if isinstance(locs, torch.Tensor) and locs.dim() == 2:
            locs = locs.unsqueeze(0)  # Add batch dimension
        
        # Ensure locs is on the correct device
        device = pl_module.device
        locs = locs.to(device)
        
        batch = TensorDict({"locs": locs}, batch_size=[1], device=device)
        td = env.reset(batch)
        
        # Record actions
        action_sequence = []
        step = 0
        max_steps = 2 * env.generator.num_loc
        
        # Run episode with greedy decoding
        while not td["done"].all() and step < max_steps:
            # Get action from policy (greedy)
            out = policy(td, phase='val', decode_type='greedy')
            
            # Decode action
            action_idx = out["action"]
            adjacency = td["adjacency"][0]
            num_loc = locs.shape[1]
            
            # Decode action to (type, i, j)
            from tsp_custom.envs.utils import decode_action
            action_type, node_i, node_j = decode_action(action_idx, adjacency.unsqueeze(0), num_loc)
            
            # Record action
            action_sequence.append((
                step,
                action_type[0].item(),
                node_i[0].item(),
                node_j[0].item()
            ))
            
            # Step environment
            td.set("action", action_idx)
            td = env.step(td)["next"]
            
            step += 1
        
        # Create GIF
        output_path = save_dir / f"epoch{epoch:03d}_val_instance{instance_idx:02d}.gif"
        self._make_gif(locs[0], action_sequence, output_path)
        
        # Get final reward
        reward = td["reward"][0].item()
        tour_length = -reward
        
        log.debug(
            f"  Instance {instance_idx}: {len(action_sequence)} steps, "
            f"tour_length={tour_length:.4f}"
        )
    
    def _make_gif(
        self,
        locs: torch.Tensor,
        action_sequence: List[tuple],
        output_path: Path
    ):
        """Create animated GIF from action sequence."""
        n = len(locs)
        # Ensure locs is on CPU before converting to numpy
        locs_np = locs.detach().cpu().numpy()
        
        # Track state
        adjacency = np.zeros((n, n), dtype=int)
        all_deleted = []  # Track deleted edges for visualization
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Save as GIF
        writer = PillowWriter(fps=self.fps)
        with writer.saving(fig, str(output_path), dpi=100):
            # Initial frame
            self._create_frame(ax, locs_np, adjacency, all_deleted, "Initial State", 0, 0)
            writer.grab_frame()
            
            # Action frames
            for idx, (step, action_type, i, j) in enumerate(action_sequence):
                ax.clear()
                
                if action_type == 0:  # ADD
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
                    action_text = f"ADD ({i}, {j})"
                elif action_type == 1:  # DELETE
                    adjacency[i, j] = 0
                    adjacency[j, i] = 0
                    all_deleted.append((i, j))
                    action_text = f"DELETE ({i}, {j})"
                else:  # DONE
                    action_text = "DONE"
                
                num_edges = adjacency.sum() // 2
                title = f"Step {step+1}: {action_text} | Edges: {num_edges}/{n}"
                self._create_frame(ax, locs_np, adjacency, all_deleted, title, num_edges, len(all_deleted))
                
                # Grab frame - if this is the last action, grab 5 times for longer display
                is_last = (idx == len(action_sequence) - 1)
                num_repeats = 5 if is_last else 1
                for _ in range(num_repeats):
                    writer.grab_frame()
        
        plt.close(fig)
    
    def _create_frame(
        self,
        ax,
        locs_np: np.ndarray,
        adjacency: np.ndarray,
        deleted_edges: List[tuple],
        title: str,
        num_edges: int,
        num_deletions: int
    ) -> None:
        """Create a single frame of the animation."""
        n = len(locs_np)
        
        # Plot deleted edges (thin dotted red)
        for i, j in deleted_edges:
            x = [locs_np[i, 0], locs_np[j, 0]]
            y = [locs_np[i, 1], locs_np[j, 1]]
            ax.plot(x, y, 'r:', alpha=0.4, linewidth=1, zorder=1)
        
        # Plot selected edges (thick green)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    x = [locs_np[i, 0], locs_np[j, 0]]
                    y = [locs_np[i, 1], locs_np[j, 1]]
                    ax.plot(x, y, 'g-', linewidth=2.5, zorder=2)
        
        # Plot nodes
        ax.scatter(locs_np[:, 0], locs_np[:, 1],
                  c='blue', s=150, zorder=3, edgecolors='black', linewidths=1.5)
        
        # Add node labels
        for i in range(n):
            ax.annotate(str(i), (locs_np[i, 0], locs_np[i, 1]),
                       ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold', zorder=4)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='green', label=f'Selected edges ({num_edges})'),
            mpatches.Patch(color='red', alpha=0.4, label=f'Deleted edges ({num_deletions})'),
            mpatches.Patch(color='blue', label=f'Nodes ({n})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()


class TrainingVisualizationCallback(Callback):
    """
    Creates animated GIFs of training episodes.
    
    Similar to ValidationVisualizationCallback but captures training rollouts.
    Shows how the policy constructs tours during training (with POMO multi-start).
    
    Args:
        num_instances: Number of training instances to visualize per epoch
        save_dir: Directory to save GIFs (default: checkpoint_dir)
        fps: Frames per second for GIF animation
        figsize: Figure size for plots
        every_n_epochs: Only visualize every N epochs (default: 1)
    """
    
    def __init__(
        self,
        num_instances: int = 5,
        save_dir: Optional[str] = None,
        fps: int = 2,
        figsize: tuple = (10, 10),
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.num_instances = num_instances
        self.save_dir = save_dir
        self.fps = fps
        self.figsize = figsize
        self.every_n_epochs = every_n_epochs
        self.last_batch_locs = None
        
        log.info(
            f"TrainingVisualizationCallback initialized: "
            f"num_instances={num_instances}, fps={fps}, every_n_epochs={every_n_epochs}"
        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Capture the last training batch locs for visualization."""
        if batch_idx == 0:  # Only capture first batch
            if isinstance(batch, TensorDict):
                self.last_batch_locs = batch["locs"][:self.num_instances].detach().cpu()
            else:
                self.last_batch_locs = batch[:self.num_instances].detach().cpu()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of training epoch."""
        current_epoch = trainer.current_epoch
        
        # Only visualize every N epochs
        if current_epoch % self.every_n_epochs != 0:
            return
        
        if self.last_batch_locs is None:
            log.warning("No training batch captured, skipping visualization")
            return
        
        log.info(f"Creating training visualizations for epoch {current_epoch}...")
        
        # Determine save directory - use Lightning logs directory
        if self.save_dir is None:
            save_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        else:
            save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move captured locs to device
        device = pl_module.device
        locs_subset = self.last_batch_locs.to(device)
        
        # Run episodes and record actions
        num_to_viz = min(self.num_instances, locs_subset.shape[0])
        for inst_idx in range(num_to_viz):
            try:
                self._create_episode_gif(
                    pl_module,
                    locs_subset[inst_idx:inst_idx+1],
                    save_dir,
                    current_epoch,
                    inst_idx
                )
            except Exception as e:
                log.error(f"Failed to create training GIF for instance {inst_idx}: {e}")
        
        log.info(f"Saved {num_to_viz} training GIFs to {save_dir}")
    
    def _create_episode_gif(
        self,
        pl_module,
        locs: torch.Tensor,
        save_dir: Path,
        epoch: int,
        instance_idx: int
    ):
        """Create GIF for a single training instance (reuses validation logic)."""
        # Reset environment
        env = pl_module.env
        policy = pl_module.policy
        
        # Create initial TensorDict
        if isinstance(locs, torch.Tensor) and locs.dim() == 2:
            locs = locs.unsqueeze(0)  # Add batch dimension
        
        device = pl_module.device
        locs = locs.to(device)
        
        batch = TensorDict({"locs": locs}, batch_size=[1], device=device)
        td = env.reset(batch)
        
        # Record actions
        action_sequence = []
        step = 0
        max_steps = 2 * env.generator.num_loc
        
        # Run episode with sampling (like training)
        while not td["done"].all() and step < max_steps:
            # Get action from policy (sampling for training realism)
            out = policy(td, phase='val', decode_type='sampling')
            
            # Decode action
            action_idx = out["action"]
            adjacency = td["adjacency"][0]
            num_loc = locs.shape[1]
            
            # Decode action to (type, i, j)
            from tsp_custom.envs.utils import decode_action
            action_type, node_i, node_j = decode_action(action_idx, adjacency.unsqueeze(0), num_loc)
            
            # Record action
            action_sequence.append((
                step,
                action_type[0].item(),
                node_i[0].item(),
                node_j[0].item()
            ))
            
            # Step environment
            td.set("action", action_idx)
            td = env.step(td)["next"]
            
            step += 1
        
        # Create GIF - note different filename prefix
        output_path = save_dir / f"epoch{epoch:03d}_train_instance{instance_idx:02d}.gif"
        self._make_gif(locs[0], action_sequence, output_path)
        
        # Get final reward
        reward = td["reward"][0].item()
        tour_length = -reward
        
        log.debug(
            f"  Training instance {instance_idx}: {len(action_sequence)} steps, "
            f"tour_length={tour_length:.4f}"
        )
    
    def _make_gif(self, locs, action_sequence, output_path):
        """Create animated GIF from action sequence (reuses validation logic)."""
        n = len(locs)
        locs_np = locs.detach().cpu().numpy()
        
        # Track state
        adjacency = np.zeros((n, n), dtype=int)
        all_deleted = []
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Save as GIF
        writer = PillowWriter(fps=self.fps)
        with writer.saving(fig, str(output_path), dpi=100):
            # Initial frame
            self._create_frame(ax, locs_np, adjacency, all_deleted, "Initial State (Training)", 0, 0)
            writer.grab_frame()
            
            # Action frames
            for idx, (step, action_type, i, j) in enumerate(action_sequence):
                ax.clear()
                
                if action_type == 0:  # ADD
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
                    action_text = f"ADD ({i}, {j})"
                elif action_type == 1:  # DELETE
                    adjacency[i, j] = 0
                    adjacency[j, i] = 0
                    all_deleted.append((i, j))
                    action_text = f"DELETE ({i}, {j})"
                else:  # DONE
                    action_text = "DONE"
                
                num_edges = adjacency.sum() // 2
                title = f"Step {step+1}: {action_text} | Edges: {num_edges}/{n}"
                self._create_frame(ax, locs_np, adjacency, all_deleted, title, num_edges, len(all_deleted))
                
                # Grab frame
                is_last = (idx == len(action_sequence) - 1)
                num_repeats = 5 if is_last else 1
                for _ in range(num_repeats):
                    writer.grab_frame()
        
        plt.close(fig)
    
    def _create_frame(self, ax, locs_np, adjacency, deleted_edges, title, num_edges, num_deletions):
        """Create a single frame of the animation (reuses validation logic)."""
        n = len(locs_np)
        
        # Plot deleted edges (thin dotted red)
        for i, j in deleted_edges:
            x = [locs_np[i, 0], locs_np[j, 0]]
            y = [locs_np[i, 1], locs_np[j, 1]]
            ax.plot(x, y, 'r:', alpha=0.4, linewidth=1, zorder=1)
        
        # Plot selected edges (thick green)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    x = [locs_np[i, 0], locs_np[j, 0]]
                    y = [locs_np[i, 1], locs_np[j, 1]]
                    ax.plot(x, y, 'g-', linewidth=2.5, zorder=2)
        
        # Plot nodes
        ax.scatter(locs_np[:, 0], locs_np[:, 1],
                  c='blue', s=150, zorder=3, edgecolors='black', linewidths=1.5)
        
        # Add node labels
        for i in range(n):
            ax.annotate(str(i), (locs_np[i, 0], locs_np[i, 1]),
                       ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold', zorder=4)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='green', label=f'Selected edges ({num_edges})'),
            mpatches.Patch(color='red', alpha=0.4, label=f'Deleted edges ({num_deletions})'),
            mpatches.Patch(color='blue', label=f'Nodes ({n})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()


class MetricsPlotCallback(Callback):
    """
    Creates plots of training and validation metrics.
    
    This callback:
    - Tracks training and validation rewards/tour lengths
    - Creates line plots showing progress over epochs
    - Saves plots with epoch prefix to checkpoint directory
    
    Args:
        save_dir: Directory to save plots (default: checkpoint_dir)
        plot_every_n_epochs: Create plots every N epochs
        metrics_to_plot: List of metric names to plot
    """
    
    def __init__(
        self,
        save_dir: Optional[str] = None,
        plot_every_n_epochs: int = 1,
        metrics_to_plot: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        
        if metrics_to_plot is None:
            self.metrics_to_plot = ['reward', 'tour_length']
        else:
            self.metrics_to_plot = metrics_to_plot
        
        # Storage for metrics
        self.train_metrics = {m: [] for m in self.metrics_to_plot}
        self.val_metrics = {m: [] for m in self.metrics_to_plot}
        self.epochs = []
        
        log.info(
            f"MetricsPlotCallback initialized: "
            f"metrics={self.metrics_to_plot}, plot_every_n_epochs={plot_every_n_epochs}"
        )
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Record training metrics at epoch end."""
        current_epoch = trainer.current_epoch
        
        # Get metrics from trainer
        for metric_name in self.metrics_to_plot:
            train_key = f"train/{metric_name}_epoch"
            if train_key in trainer.callback_metrics:
                value = trainer.callback_metrics[train_key].item()
                self.train_metrics[metric_name].append(value)
            else:
                # Try without _epoch suffix
                train_key = f"train/{metric_name}"
                if train_key in trainer.callback_metrics:
                    value = trainer.callback_metrics[train_key].item()
                    self.train_metrics[metric_name].append(value)
                else:
                    self.train_metrics[metric_name].append(None)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Record validation metrics and create plots."""
        current_epoch = trainer.current_epoch
        self.epochs.append(current_epoch)
        
        # Get validation metrics
        for metric_name in self.metrics_to_plot:
            val_key = f"val/{metric_name}"
            if val_key in trainer.callback_metrics:
                value = trainer.callback_metrics[val_key].item()
                self.val_metrics[metric_name].append(value)
            else:
                self.val_metrics[metric_name].append(None)
        
        # Create plots every N epochs
        if current_epoch % self.plot_every_n_epochs == 0:
            self._create_plots(trainer, current_epoch)
    
    def _create_plots(self, trainer, epoch: int):
        """Create and save metric plots."""
        # Determine save directory - use Lightning logs directory
        if self.save_dir is None:
            save_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        else:
            save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a plot for each metric
        for metric_name in self.metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            train_values = self.train_metrics[metric_name]
            val_values = self.val_metrics[metric_name]
            
            # Filter out None values
            train_epochs = [e for e, v in zip(self.epochs, train_values) if v is not None]
            train_clean = [v for v in train_values if v is not None]
            
            val_epochs = [e for e, v in zip(self.epochs, val_values) if v is not None]
            val_clean = [v for v in val_values if v is not None]
            
            # Plot lines
            if train_clean:
                ax.plot(train_epochs, train_clean, 'b-o', label='Training', linewidth=2, markersize=4)
            if val_clean:
                ax.plot(val_epochs, val_clean, 'r-s', label='Validation', linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric_name.replace("_", " ").title()} vs Epoch', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            output_path = save_dir / f"epoch{epoch:03d}_metrics_{metric_name}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            log.debug(f"  Saved {metric_name} plot to {output_path}")
        
        log.info(f"Saved metric plots for epoch {epoch} to {save_dir}")


class CombinedMetricsPlotCallback(Callback):
    """
    Creates a combined plot of all key metrics in subplots.
    
    Args:
        save_dir: Directory to save plots
        plot_every_n_epochs: Create plots every N epochs
    """
    
    def __init__(
        self,
        save_dir: Optional[str] = None,
        plot_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        
        # Storage
        self.metrics = {
            'train_reward': [],
            'val_reward': [],
            'train_tour_length': [],
            'val_tour_length': [],
            'train_loss': [],
        }
        self.epochs = []
        
        log.info(f"CombinedMetricsPlotCallback initialized: plot_every_n_epochs={plot_every_n_epochs}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Record training metrics."""
        for key in ['train_reward', 'train_tour_length', 'train_loss']:
            metric_key = f"{key.replace('_', '/')}_epoch"
            if metric_key in trainer.callback_metrics:
                self.metrics[key].append(trainer.callback_metrics[metric_key].item())
            else:
                metric_key = key.replace('_', '/')
                if metric_key in trainer.callback_metrics:
                    self.metrics[key].append(trainer.callback_metrics[metric_key].item())
                else:
                    self.metrics[key].append(None)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Record validation metrics and create plots."""
        current_epoch = trainer.current_epoch
        self.epochs.append(current_epoch)
        
        for key in ['val_reward', 'val_tour_length']:
            metric_key = key.replace('_', '/')
            if metric_key in trainer.callback_metrics:
                self.metrics[key].append(trainer.callback_metrics[metric_key].item())
            else:
                self.metrics[key].append(None)
        
        if current_epoch % self.plot_every_n_epochs == 0:
            self._create_combined_plot(trainer, current_epoch)
    
    def _create_combined_plot(self, trainer, epoch: int):
        """Create combined metrics plot."""
        # Determine save directory - use Lightning logs directory
        if self.save_dir is None:
            save_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        else:
            save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Plot 1: Rewards
        ax = axes[0, 0]
        self._plot_metric(ax, 'train_reward', 'val_reward', 'Reward', 'Higher is Better')
        
        # Plot 2: Tour Length
        ax = axes[0, 1]
        self._plot_metric(ax, 'train_tour_length', 'val_tour_length', 'Tour Length', 'Lower is Better')
        
        # Plot 3: Loss
        ax = axes[1, 0]
        train_loss = [v for v in self.metrics['train_loss'] if v is not None]
        if train_loss:
            ax.plot(self.epochs[:len(train_loss)], train_loss, 'b-o', label='Training', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        # Get latest metrics
        latest_train_reward = self.metrics['train_reward'][-1] if self.metrics['train_reward'] else None
        latest_val_reward = self.metrics['val_reward'][-1] if self.metrics['val_reward'] else None
        latest_train_tour = self.metrics['train_tour_length'][-1] if self.metrics['train_tour_length'] else None
        latest_val_tour = self.metrics['val_tour_length'][-1] if self.metrics['val_tour_length'] else None
        
        # Format metrics with fallback for None values
        train_reward_str = f"{latest_train_reward:.4f}" if latest_train_reward is not None else "N/A"
        val_reward_str = f"{latest_val_reward:.4f}" if latest_val_reward is not None else "N/A"
        train_tour_str = f"{latest_train_tour:.4f}" if latest_train_tour is not None else "N/A"
        val_tour_str = f"{latest_val_tour:.4f}" if latest_val_tour is not None else "N/A"
        
        summary_text = f"""
        Epoch {epoch} Summary
        {'='*40}
        
        Training:
          Reward: {train_reward_str}
          Tour Length: {train_tour_str}
        
        Validation:
          Reward: {val_reward_str}
          Tour Length: {val_tour_str}
        
        Total Epochs: {epoch + 1}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        output_path = save_dir / f"epoch{epoch:03d}_combined_metrics.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        log.info(f"Saved combined metrics plot to {output_path}")
    
    def _plot_metric(self, ax, train_key: str, val_key: str, title: str, subtitle: str):
        """Helper to plot a single metric."""
        train_values = [v for v in self.metrics[train_key] if v is not None]
        val_values = [v for v in self.metrics[val_key] if v is not None]
        
        if train_values:
            ax.plot(self.epochs[:len(train_values)], train_values, 'b-o', 
                   label='Training', linewidth=2, markersize=4)
        if val_values:
            ax.plot(self.epochs[:len(val_values)], val_values, 'r-s',
                   label='Validation', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
