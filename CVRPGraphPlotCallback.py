# PlotCVRPCallback.py
import os
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class CVRPGraphPlotCallback(Callback):
    def __init__(
        self,
        env,
        num_examples: int = 5,
        subdir: str = "val_plots_cvrp",
        decode_type: str = "greedy",
    ):
        """
        env: RL4CO CVRPEnv
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
            # e.g. lightning_logs/version_0/val_plots_cvrp
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
            batch = next(iter(val_loader))
        except StopIteration:
            pl_module.print("[CVRPPlotCallback] Empty val_dataloader, skipping plots.")
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
            actions = out["actions"].cpu()  # [B, T] (CVRP tour with depot returns)
            rewards = out["reward"].cpu()   # [B]

        # Check if CVRP baseline is available
        has_baseline = (
            hasattr(pl_module, "val_cvrp_routes")
            and hasattr(pl_module, "val_cvrp_costs")
        )

        if has_baseline:
            # routes: list length = num_val, each element is a list of np.ndarrays
            baseline_routes = pl_module.val_cvrp_routes
            baseline_costs = pl_module.val_cvrp_costs.cpu()  # [num_val]
        else:
            baseline_routes = None
            baseline_costs = None

        # How many examples to plot
        B = td.batch_size[0] if hasattr(td, "batch_size") else td["locs"].shape[0]
        max_examples = min(self.num_examples, B)
        if has_baseline:
            max_examples = min(
                max_examples,
                len(baseline_routes),
                baseline_costs.shape[0],
            )

        for i in range(max_examples):
            td_i = td[i].cpu()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            # ----- Draw baseline (OR-Tools CVRP solution) first (underneath) -----
            if has_baseline:
                routes_i = baseline_routes[i]   # list of np.ndarray routes
                opt_cost = baseline_costs[i].item()

                locs = td_i["locs"].numpy()     # [1+N, 2], node 0 = depot

                # Each vehicle route is a path like [0, ..., 0]
                for route in routes_i:
                    if len(route) <= 1:
                        continue
                    coords = locs[route]        # [L, 2]
                    xs = coords[:, 0]
                    ys = coords[:, 1]
                    # Thin blue dashed lines for baseline routes
                    ax.plot(xs, ys, linestyle="--", linewidth=1.0, color="blue", alpha=0.9)

            # ----- Draw model (POMO / policy) solution on top -----
            model_cost = -rewards[i].item()
            # For CVRP, actions[i] is a sequence visiting depot (0) multiple times
            # env.render knows how to interpret that
            self.env.render(td_i, actions[i], ax=ax)

            if has_baseline:
                gap_pct = (
                    100.0 * (model_cost - opt_cost) / opt_cost
                    if opt_cost > 0
                    else float("nan")
                )
                ax.set_title(
                    f"CVRP | Epoch {trainer.current_epoch} | ex {i} | "
                    f"model = {model_cost:.3f}, baseline = {opt_cost:.3f}, "
                    f"gap = {gap_pct:.2f}%"
                )
            else:
                ax.set_title(
                    f"CVRP | Epoch {trainer.current_epoch} | ex {i} | cost = {model_cost:.3f}"
                )

            fname = os.path.join(
                self.out_dir,
                f"epoch{trainer.current_epoch:03d}_example{i}.png",
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
