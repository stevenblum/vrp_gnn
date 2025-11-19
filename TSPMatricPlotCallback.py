# MetricPlotCallback.py
import os
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


class PlotMetricCallback(Callback):
    """
    Plot training and validation metrics with matplotlib after each epoch
    and save them as PNGs in the current version folder (trainer.log_dir).
    Optionally overlays a Concorde baseline (from pl_module.val_concorde_costs)
    as a horizontal red line.
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

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Avoid duplicate plots when using multiple GPUs
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics  # dict[str, Tensor]
        epoch = trainer.current_epoch

        # Only log when both keys are present
        if self.train_metric_key not in metrics or self.val_metric_key not in metrics:
            return

        train_val = metrics[self.train_metric_key]
        val_val = metrics[self.val_metric_key]

        # Convert to Python floats
        if isinstance(train_val, torch.Tensor):
            train_val = train_val.item()
        if isinstance(val_val, torch.Tensor):
            val_val = val_val.item()

        self.epochs.append(epoch)
        self.train_hist.append(train_val)
        self.val_hist.append(val_val)

        out_dir = trainer.log_dir  # e.g. lightning_logs/version_0
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.epochs, self.train_hist, marker="o", label=self.train_metric_key)
        ax.plot(self.epochs, self.val_hist, marker="o", label=self.val_metric_key)

        # ---- Concorde baseline overlay (if available) ----
        baseline_cost = None
        baseline_reward = None
        if hasattr(pl_module, "val_concorde_costs"):
            # mean optimal cost over the validation set
            baseline_cost = pl_module.val_concorde_costs.mean().item()
            # assuming reward = -cost
            baseline_reward = -baseline_cost

            ax.axhline(
                baseline_reward,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Concorde baseline (reward={baseline_reward:.3f})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")

        if baseline_cost is not None:
            ax.set_title(
                f"Train vs Val metrics | Concorde cost ≈ {baseline_cost:.3f}"
            )
        else:
            ax.set_title("Train vs Val metrics")

        ax.grid(True, linestyle="--", alpha=0.8)
        ax.legend()

        fname = os.path.join(
            out_dir,
            f"{self.filename_prefix}.png",
        )
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
