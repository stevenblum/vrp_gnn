# PlotCVRPCallback.py
import os
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from itertools import cycle

class CVRPGraphPlotCallback(Callback):
    def __init__(
        self,
        env,
        num_examples: int = 5,
        subdir: str = "val_plots_cvrp",
        decode_type: str = "greedy",
        logger=None,  # TensorBoard logger for logging individual rewards
    ):
        super().__init__()
        self.env = env
        self.num_examples = num_examples
        self.subdir = subdir
        self.decode_type = decode_type
        self.logger = logger
        self.out_dir = None

    def _ensure_out_dir(self, trainer):
        if self.out_dir is None:
            self.out_dir = os.path.join(trainer.log_dir, self.subdir)
            os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _split_model_routes(action_seq):
        """
        action_seq: 1D tensor/list with depot=0 repeated to separate routes.
        Returns list of routes, each like [0, ..., 0].
        """
        seq = action_seq.tolist() if hasattr(action_seq, "tolist") else list(action_seq)

        routes = []
        cur = []
        for n in seq:
            if n == 0:
                if cur:
                    routes.append([0] + cur + [0])
                    cur = []
            else:
                cur.append(n)
        if cur:
            routes.append([0] + cur + [0])
        return routes

    @staticmethod
    def _plot_nodes(ax, locs, demand=None, probs=None):
        """
        locs: numpy array [1+N, 2], node 0 = depot
        demand: optional numpy array [N] for customer demand labels
        probs: optional numpy array [N] for per-customer selection probabilities
        """
        depot = locs[0]
        cust = locs[1:]
        ax.scatter(cust[:, 0], cust[:, 1], s=12)
        ax.scatter([depot[0]], [depot[1]], s=40, marker="s")
        if demand is not None:
            for idx, (x, y) in enumerate(cust):
                ax.text(x + 0.005, y + 0.005, f"{demand[idx]:.2f}", fontsize=6, color="dimgray")
        if probs is not None and probs.size > 0:
            # Draw thin vertical bars above each customer to visualize selection probability
            bar_width = 0.01
            for (x, y), p in zip(cust, probs):
                height = float(p)
                ax.add_patch(plt.Rectangle((x - bar_width / 2, y), bar_width, height, color="cornflowerblue", alpha=0.7))
                ax.text(x, y + height + 0.005, f"{p:.2f}", fontsize=6, ha="center", va="bottom", color="navy")
        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def _plot_route_without_depot(ax, coords, **plot_kwargs):
        """Plot route lines skipping depot-to-first and last-to-depot segments."""
        if coords.shape[0] < 3:
            return
        inner = coords[1:-1]
        if inner.shape[0] < 2:
            return
        ax.plot(inner[:, 0], inner[:, 1], **plot_kwargs)

    @staticmethod
    def _actions_to_probs(actions, num_customers):
        """Convert action sequence to per-customer selection probabilities."""
        counts = torch.zeros(num_customers, dtype=torch.float32)
        seq = actions.tolist() if hasattr(actions, "tolist") else list(actions)
        for n in seq:
            if n > 0:
                counts[n - 1] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self._ensure_out_dir(trainer)

        # Grab a single batch from validation loader
        val_loader = pl_module.val_dataloader()
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            pl_module.print("[CVRPPlotCallback] Empty val_dataloader, skipping.")
            return

        device = pl_module.device
        batch = batch.to(device)

        # Convert dataset batch -> environment state
        td = pl_module.env.reset(batch)

        pl_module.eval()
        with torch.no_grad():
            out = pl_module.policy(
                td.clone(),
                phase="test",
                decode_type=self.decode_type,
                return_actions=True,
            )
            actions = out["actions"].cpu()   # [B, T]
            rewards = out["reward"].cpu()    # [B]

        # Log individual validation rewards to TensorBoard
        if self.logger is not None:
            for idx in range(rewards.shape[0]):
                self.logger.experiment.add_scalar(
                    f"val_individual/reward_sample_{idx}",
                    rewards[idx].item(),
                    trainer.current_epoch
                )

        # Baseline availability
        has_baseline = (
            hasattr(pl_module, "val_cvrp_routes")
            and hasattr(pl_module, "val_cvrp_costs")
        )

        if has_baseline:
            baseline_routes = pl_module.val_cvrp_routes
            baseline_costs = pl_module.val_cvrp_costs.cpu()
        else:
            baseline_routes, baseline_costs = None, None

        B = td.batch_size[0] if hasattr(td, "batch_size") else td["locs"].shape[0]
        max_examples = min(self.num_examples, B)
        if has_baseline:
            max_examples = min(max_examples, len(baseline_routes), baseline_costs.shape[0])

        for i in range(max_examples):
            td_i = td[i].cpu()
            locs = td_i["locs"].numpy()  # [1+N, 2]
            demand = td_i.get("demand", None)
            demand_np = demand.numpy() if demand is not None else None
            num_customers = locs.shape[0] - 1
            probs = self._actions_to_probs(actions[i], num_customers).numpy()

            fig, (ax_base, ax_model) = plt.subplots(
                1, 2, figsize=(10, 5), sharex=True, sharey=True
            )

            # ---------------- BASELINE AXIS ----------------
            baseline_color_by_first = {}
            if has_baseline:
                routes_i = baseline_routes[i]  # list of np.ndarray routes
                opt_cost = baseline_costs[i].item()

                self._plot_nodes(ax_base, locs, demand=demand_np)
                ax_base.set_title(f"Baseline (OR-Tools)\ncost = {opt_cost:.3f}")

                # plot baseline routes and record colors
                for route in routes_i:
                    route = route.tolist() if hasattr(route, "tolist") else list(route)
                    if len(route) <= 2:
                        continue

                    coords = locs[route]
                    line, = ax_base.plot(
                        [], [],  # placeholder to get consistent colors
                        linestyle="--", linewidth=1.2, alpha=0.9
                    )
                    # Plot without depot edges
                    self._plot_route_without_depot(
                        ax_base, coords,
                        linestyle="--", linewidth=1.2, alpha=0.9, color=line.get_color()
                    )
                    color = line.get_color()

                    first_customer = route[1] if len(route) > 2 else None
                    if first_customer is not None and first_customer != 0:
                        baseline_color_by_first[first_customer] = color
            else:
                self._plot_nodes(ax_base, locs)
                ax_base.set_title("Baseline unavailable")

            # ---------------- MODEL AXIS ----------------
            model_cost = -rewards[i].item()
            self._plot_nodes(ax_model, locs, demand=demand_np, probs=probs)

            model_routes = self._split_model_routes(actions[i])
            # If we need fallback colors, use matplotlib cycle
            fallback_colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

            for r in model_routes:
                if len(r) <= 2:
                    continue
                first_customer = r[1] if len(r) > 2 else None

                if first_customer in baseline_color_by_first:
                    color = baseline_color_by_first[first_customer]
                else:
                    color = next(fallback_colors)

                coords = locs[r]
                self._plot_route_without_depot(
                    ax_model, coords,
                    linestyle="-", linewidth=2.0, color=color, alpha=0.95
                )

            if has_baseline:
                gap_pct = (
                    100.0 * (model_cost - opt_cost) / opt_cost
                    if opt_cost > 0 else float("nan")
                )
                ax_model.set_title(
                    f"Model ({self.decode_type})\n"
                    f"cost = {model_cost:.3f} | gap = {gap_pct:.2f}%"
                )
            else:
                ax_model.set_title(f"Model ({self.decode_type})\ncost = {model_cost:.3f}")

            fig.suptitle(f"CVRP | Epoch {trainer.current_epoch} | ex {i}", y=1.02)

            fname = os.path.join(
                self.out_dir,
                f"epoch{trainer.current_epoch:03d}_example{i}.png",
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
