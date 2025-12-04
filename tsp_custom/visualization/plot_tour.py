"""
Visualization functions for CustomTSPEnv

Functions to visualize TSP instances, solutions, and create animated GIFs
showing the edge selection and deletion process.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def plot_tsp_instance(
    locs: torch.Tensor,
    adjacency: Optional[torch.Tensor] = None,
    deleted_edges: Optional[List[Tuple[int, int]]] = None,
    title: str = "TSP Instance",
    ax: Optional[plt.Axes] = None,
    show_node_labels: bool = True,
    node_size: int = 100,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot a TSP instance with selected and deleted edges.
    
    Args:
        locs: Node locations (N, 2)
        adjacency: Adjacency matrix (N, N), optional
        deleted_edges: List of deleted edge tuples (i, j)
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        show_node_labels: Whether to show node indices
        node_size: Size of node markers
        figsize: Figure size if creating new figure
        
    Returns:
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    locs_np = locs.cpu().numpy()
    n = len(locs_np)
    
    # Plot deleted edges first (faded red)
    if deleted_edges:
        for i, j in deleted_edges:
            x = [locs_np[i, 0], locs_np[j, 0]]
            y = [locs_np[i, 1], locs_np[j, 1]]
            ax.plot(x, y, 'r--', alpha=0.3, linewidth=1, zorder=1)
    
    # Plot selected edges (green)
    if adjacency is not None:
        adj_np = adjacency.cpu().numpy()
        for i in range(n):
            for j in range(i + 1, n):
                if adj_np[i, j] > 0:
                    x = [locs_np[i, 0], locs_np[j, 0]]
                    y = [locs_np[i, 1], locs_np[j, 1]]
                    ax.plot(x, y, 'g-', linewidth=2, zorder=2)
    
    # Plot nodes
    ax.scatter(locs_np[:, 0], locs_np[:, 1], 
              c='blue', s=node_size, zorder=3, edgecolors='black', linewidths=1)
    
    # Add node labels
    if show_node_labels:
        for i in range(n):
            ax.annotate(str(i), (locs_np[i, 0], locs_np[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='black', zorder=4)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label='Selected edges'),
        mpatches.Patch(color='red', alpha=0.3, label='Deleted edges'),
        mpatches.Patch(color='blue', label='Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def create_episode_gif(
    locs: torch.Tensor,
    action_sequence: List[Tuple[int, int, int, int]],  # (step, action_type, i, j)
    output_path: str,
    fps: int = 2,
    figsize: Tuple[int, int] = (10, 10),
    show_metrics: bool = True
) -> None:
    """
    Create an animated GIF showing the episode progression.
    
    Args:
        locs: Node locations (N, 2)
        action_sequence: List of (step, action_type, node_i, node_j) tuples
            action_type: 0=ADD, 1=DELETE, 2=DONE
        output_path: Path to save GIF
        fps: Frames per second
        figsize: Figure size
        show_metrics: Whether to show step counter and metrics
    """
    n = len(locs)
    locs_np = locs.cpu().numpy()
    
    # Track state through episode
    adjacency = np.zeros((n, n), dtype=int)
    deleted_edges = []
    all_deleted = []  # Track all deletions for visualization
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    def update_frame(frame_idx):
        ax.clear()
        
        if frame_idx == 0:
            # Initial state (no edges)
            step = 0
            action_text = "Initial State"
            num_edges = 0
            num_deletions = 0
        else:
            # Apply action from previous frame
            step, action_type, i, j = action_sequence[frame_idx - 1]
            
            if action_type == 0:  # ADD
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                action_text = f"Step {step}: ADD edge ({i}, {j})"
            elif action_type == 1:  # DELETE
                adjacency[i, j] = 0
                adjacency[j, i] = 0
                all_deleted.append((i, j))
                action_text = f"Step {step}: DELETE edge ({i}, {j})"
            else:  # DONE
                action_text = f"Step {step}: DONE"
            
            num_edges = np.sum(adjacency) // 2
            num_deletions = len(all_deleted)
        
        # Plot deleted edges (faded red)
        for i, j in all_deleted:
            x = [locs_np[i, 0], locs_np[j, 0]]
            y = [locs_np[i, 1], locs_np[j, 1]]
            ax.plot(x, y, 'r--', alpha=0.3, linewidth=1, zorder=1)
        
        # Plot selected edges (green)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    x = [locs_np[i, 0], locs_np[j, 0]]
                    y = [locs_np[i, 1], locs_np[j, 1]]
                    ax.plot(x, y, 'g-', linewidth=2.5, zorder=2)
        
        # Plot nodes with degree-based coloring
        degrees = adjacency.sum(axis=1)
        colors = ['lightblue' if d < 2 else 'green' if d == 2 else 'red' for d in degrees]
        ax.scatter(locs_np[:, 0], locs_np[:, 1], 
                  c=colors, s=150, zorder=3, edgecolors='black', linewidths=2)
        
        # Add node labels
        for i in range(n):
            ax.annotate(f'{i}\n[{int(degrees[i])}]', 
                       (locs_np[i, 0], locs_np[i, 1]),
                       ha='center', va='center',
                       fontsize=9, fontweight='bold', color='black', zorder=4)
        
        # Title and metrics
        title = f"TSP Episode: N={n} nodes"
        if show_metrics:
            title += f"\n{action_text}\nEdges: {num_edges}, Deletions: {num_deletions}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='green', label='Selected edges'),
            mpatches.Patch(color='red', alpha=0.3, label='Deleted edges'),
            mpatches.Patch(color='lightblue', label='Degree < 2'),
            mpatches.Patch(color='green', label='Degree = 2'),
            mpatches.Patch(color='red', label='Degree > 2')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Create animation
    n_frames = len(action_sequence) + 1  # +1 for initial state
    anim = FuncAnimation(fig, update_frame, frames=n_frames, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"✓ Saved episode animation: {output_path}")


def visualize_state(
    td: dict,
    batch_idx: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a TensorDict state from the environment.
    
    Args:
        td: TensorDict state from environment
        batch_idx: Which batch element to visualize
        title: Custom title (auto-generated if None)
        save_path: Path to save figure (optional)
        
    Returns:
        Figure object
    """
    locs = td['locs'][batch_idx]
    adjacency = td['adjacency'][batch_idx]
    degrees = td['degrees'][batch_idx]
    current_step = td['current_step'][batch_idx].item()
    num_edges = td['num_edges'][batch_idx].item()
    num_deletions = td['num_deletions'][batch_idx].item()
    done = td['done'][batch_idx].item()
    
    if title is None:
        title = f"Step {current_step}: {num_edges} edges, {num_deletions} deletions"
        if done:
            title += " [DONE]"
    
    fig = plot_tsp_instance(locs, adjacency, title=title)
    
    # Add degree information
    ax = fig.axes[0]
    degree_text = f"Degrees: {degrees.tolist()}"
    ax.text(0.02, 0.98, degree_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=8, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")
    
    return fig


def create_test_episode_gif(
    env,
    num_loc: int = 10,
    output_dir: str = "visualization/test_outputs",
    filename: str = "test_episode.gif",
    max_steps: int = 30
) -> str:
    """
    Create a test episode GIF by taking random valid actions.
    
    Args:
        env: CustomTSPEnv instance
        num_loc: Number of nodes
        output_dir: Directory to save GIF
        filename: Output filename
        max_steps: Maximum steps to take
        
    Returns:
        Path to saved GIF
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    
    # Reset environment
    td = env.reset(batch_size=1)
    locs = td['locs'][0]
    
    # Track actions
    action_sequence = []
    
    # Take random valid actions
    for step in range(max_steps):
        if td['done'][0].item():
            break
        
        # Get valid actions
        mask = td['action_mask'][0]
        valid_actions = mask.nonzero(as_tuple=True)[0]
        
        if len(valid_actions) == 0:
            break
        
        # Pick random valid action
        action_idx = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        
        # Decode action
        from envs.utils import decode_action
        action_type, node_i, node_j = decode_action(
            torch.tensor([action_idx]), 
            td['adjacency'], 
            num_loc
        )
        
        action_sequence.append((
            step,
            action_type[0].item(),
            node_i[0].item(),
            node_j[0].item()
        ))
        
        # Execute action
        td['action'] = torch.tensor([action_idx], dtype=torch.long)
        td = env.step(td)['next']
    
    # Create GIF
    create_episode_gif(locs, action_sequence, output_path, fps=2)
    
    return output_path
