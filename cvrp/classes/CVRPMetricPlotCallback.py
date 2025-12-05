# MetricPlotCallback.py
import os
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


class CVRPMetricPlotCallback(Callback):
    """
    Plot training and validation metrics with matplotlib after each epoch
    and save them as PNGs in the current version folder (trainer.log_dir).

    Optionally overlays a baseline as a horizontal red line:
        - Concorde TSP baseline from pl_module.val_concorde_costs, or
        - CVRP baseline from pl_module.val_cvrp_costs
    
    Also creates a second plot showing individual validation sample gaps over epochs.
    """

    def __init__(
        self,
        train_metric_key: str = "train/reward",
        val_metric_key: str = "val/reward",
        filename_prefix: str = "train_val_metrics",
    ):
        super().__init__()
        self.train_metric_key = train_metric_key
        self.val_metric_key = val_metric_key
        self.filename_prefix = filename_prefix

        self.epochs = []
        self.train_hist = []
        self.val_hist = []
        self.individual_gaps = []  # List of lists: one per epoch, each with per-sample gaps

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Avoid duplicate plots when using multiple GPUs
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics  # dict[str, Tensor]
        epoch = trainer.current_epoch

        # Only log when both keys are present
        if self.train_metric_key not in metrics or self.val_metric_key not in metrics:
            print(f"[CVRPMetricPlotCallback] Missing keys at epoch {epoch}: "
                  f"train_key_present={self.train_metric_key in metrics}, "
                  f"val_key_present={self.val_metric_key in metrics}")
            return

        train_val = metrics[self.train_metric_key]
        val_val = metrics[self.val_metric_key]

        # Convert to Python floats
        if isinstance(train_val, torch.Tensor):
            train_val = train_val.item()
        if isinstance(val_val, torch.Tensor):
            val_val = val_val.item()

        print(f"[CVRPMetricPlotCallback] Epoch {epoch} "
              f"train={train_val:.4f} val={val_val:.4f}")

        self.epochs.append(epoch)
        self.train_hist.append(train_val)
        self.val_hist.append(val_val)

        # ---- Collect individual validation sample gaps ----
        individual_gaps_this_epoch = []
        if hasattr(pl_module, "val_cvrp_costs") and hasattr(pl_module, "val_rewards"):
            # Get per-sample costs and rewards
            baseline_costs = pl_module.val_cvrp_costs  # tensor of baseline costs
            model_rewards = pl_module.val_rewards  # tensor of model rewards (negative costs)
            
            # Convert rewards to costs
            model_costs = -model_rewards
            
            # Calculate per-sample gap: (model_cost - baseline_cost) / baseline_cost * 100
            per_sample_gaps = ((model_costs - baseline_costs) / baseline_costs * 100).cpu().tolist()
            individual_gaps_this_epoch = per_sample_gaps
        elif hasattr(pl_module, "val_concorde_costs") and hasattr(pl_module, "val_rewards"):
            # TSP case
            baseline_costs = pl_module.val_concorde_costs
            model_rewards = pl_module.val_rewards
            model_costs = -model_rewards
            per_sample_gaps = ((model_costs - baseline_costs) / baseline_costs * 100).cpu().tolist()
            individual_gaps_this_epoch = per_sample_gaps
        
        self.individual_gaps.append(individual_gaps_this_epoch)

        out_dir = trainer.log_dir  # e.g. lightning_logs/version_0
        os.makedirs(out_dir, exist_ok=True)

        # ---- PLOT 1: Overall metrics ----
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.epochs, self.train_hist, marker="o", label=self.train_metric_key)
        ax.plot(self.epochs, self.val_hist, marker="o", label=self.val_metric_key)

        # ---- Baseline overlay (Concorde for TSP or OR-Tools for CVRP) ----
        baseline_cost = None
        baseline_reward = None
        baseline_label = None

        if hasattr(pl_module, "val_concorde_costs"):
            # TSP: mean optimal cost over the validation set (Concorde)
            baseline_cost = pl_module.val_concorde_costs.mean().item()
            baseline_label = "Concorde baseline"
        elif hasattr(pl_module, "val_cvrp_costs"):
            # CVRP: mean baseline cost over the validation set (e.g. OR-Tools)
            baseline_cost = pl_module.val_cvrp_costs.mean().item()
            baseline_label = "CVRP baseline"

        if baseline_cost is not None:
            # assuming reward = -cost
            baseline_reward = -baseline_cost

            ax.axhline(
                baseline_reward,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"{baseline_label} (reward={baseline_reward:.3f})",
            )
            ax.set_title(
                f"Train vs Val metrics | {baseline_label} cost ≈ {baseline_cost:.3f}"
            )
        else:
            ax.set_title("Train vs Val metrics")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.grid(True, linestyle="--", alpha=0.8)
        ax.legend()

        fname = os.path.join(
            out_dir,
            f"{self.filename_prefix}.png",
        )
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

        # ---- PLOT 2: Individual validation sample gaps ----
        if self.individual_gaps and len(self.individual_gaps[-1]) > 0:
            num_samples = len(self.individual_gaps[0])
            
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            
            # Plot one line per validation sample
            for sample_idx in range(num_samples):
                # Extract gap for this sample across all epochs
                sample_gaps = [epoch_gaps[sample_idx] for epoch_gaps in self.individual_gaps if sample_idx < len(epoch_gaps)]
                sample_epochs = self.epochs[:len(sample_gaps)]
                
                ax2.plot(sample_epochs, sample_gaps, marker='o', markersize=3, 
                        alpha=0.7, linewidth=1.5, label=f'Sample {sample_idx}')
            
            # Add 0% gap reference line
            ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Gap to Baseline (%)")
            ax2.set_title("Individual Validation Sample Gaps")
            ax2.grid(True, linestyle="--", alpha=0.3)
            
            # Only show legend if not too many samples
            if num_samples <= 20:
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            fname2 = os.path.join(
                out_dir,
                f"{self.filename_prefix}_individual_gaps.png",
            )
            fig2.savefig(fname2, bbox_inches="tight")
            plt.close(fig2)
