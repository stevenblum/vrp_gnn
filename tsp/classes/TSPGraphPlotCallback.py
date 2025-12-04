# PlotTSPCallback.py
import os
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class TSPGraphPlotCallback(Callback):
    def __init__(
        self,
        env,
        num_examples: int = 5,
        subdir: str = "val_plots",
        decode_type: str = "greedy",
    ):
        """
        env: RL4CO TSPEnv
        num_examples: how many validation instances to plot (from the first val batch)
        """
        super().__init__()
        self.env = env
        self.num_examples = num_examples
        self.subdir = subdir
        self.decode_type = decode_type
        self.out_dir = None  # will be set once we know trainer.log_dir

    def _ensure_out_dir(self, trainer):
        if self.out_dir is None:
            # e.g. lightning_logs/version_0/val_plots
            self.out_dir = os.path.join(trainer.log_dir, self.subdir)
            os.makedirs(self.out_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # only one process writes files (important if you ever go multi-GPU)
        if not trainer.is_global_zero:
            return

        self._ensure_out_dir(trainer)

        # Grab a single batch from the validation dataloader
        val_loader = pl_module.val_dataloader()
        try:
            batch = next(iter(val_loader))  # this batch has only 'locs' etc.
        except StopIteration:
            pl_module.print("[PlotTSPCallback] Empty val_dataloader, skipping plots.")
            return

        device = pl_module.device
        batch = batch.to(device)

        # Convert raw dataset batch -> full environment state
        td = pl_module.env.reset(batch)  # or self.env.reset(batch)

        pl_module.eval()
        with torch.no_grad():
            out = pl_module.policy(
                td.clone(),
                phase="test",
                decode_type=self.decode_type,
                return_actions=True,
            )
            actions = out["actions"].cpu()  # [B, N]
            rewards = out["reward"].cpu()   # [B]

        # Check if Concorde baseline is available
        has_baseline = (
            hasattr(pl_module, "val_concorde_tours")
            and hasattr(pl_module, "val_concorde_costs")
        )

        if has_baseline:
            baseline_tours = pl_module.val_concorde_tours.cpu()   # [num_val, N]
            baseline_costs = pl_module.val_concorde_costs.cpu()   # [num_val]
        else:
            baseline_tours = None
            baseline_costs = None

        # How many examples to plot
        B = td.batch_size[0] if hasattr(td, "batch_size") else td["locs"].shape[0]
        max_examples = min(self.num_examples, B)
        if has_baseline:
            max_examples = min(
                max_examples,
                baseline_tours.shape[0],
                baseline_costs.shape[0],
            )

        for i in range(max_examples):
            td_i = td[i].cpu()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            # ----- Draw Concorde baseline first (underneath) -----
            if has_baseline:
                opt_tour = baseline_tours[i]        # [N]
                opt_cost = baseline_costs[i].item()

                # Extract node coordinates from td_i
                locs = td_i["locs"].numpy()         # [N, 2]
                coords = locs[opt_tour]             # reorder by tour
                coords = torch.from_numpy(coords).numpy()
                # close the tour
                coords = coords.tolist()
                coords.append(coords[0])
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]

                # Thin blue dashed line
                ax.plot(xs, ys, linestyle="--", linewidth=1.0, color="blue", alpha=0.9)

            # ----- Draw POMO model tour on top -----
            model_cost = -rewards[i].item()
            self.env.render(td_i, actions[i], ax=ax)

            if has_baseline:
                gap_pct = (
                    100.0 * (model_cost - opt_cost) / opt_cost
                    if opt_cost > 0
                    else float("nan")
                )
                ax.set_title(
                    f"Epoch {trainer.current_epoch} | ex {i} | "
                    f"model = {model_cost:.3f}, baseline = {opt_cost:.3f}, "
                    f"gap = {gap_pct:.2f}%"
                )
            else:
                ax.set_title(
                    f"Epoch {trainer.current_epoch} | ex {i} | cost = {model_cost:.3f}"
                )

            fname = os.path.join(
                self.out_dir,
                f"epoch{trainer.current_epoch:03d}_example{i}.png",
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
