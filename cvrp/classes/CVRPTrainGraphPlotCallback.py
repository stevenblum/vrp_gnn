# CVRPTrainPlotCallback.py
import os
import torch
import matplotlib.pyplot as plt
from itertools import cycle
from lightning.pytorch.callbacks import Callback
from tensordict import TensorDict


class CVRPTrainGraphPlotCallback(Callback):
    def __init__(
        self,
        env,
        num_examples: int = 5,
        subdir: str = "train_plots_cvrp",
        decode_type: str = "greedy",   # keep greedy for stable visuals
    ):
        super().__init__()
        self.env = env
        self.num_examples = num_examples
        self.subdir = subdir
        self.decode_type = decode_type
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
    def _full_locs_from_td(td_i: TensorDict):
        """
        Make a [1+N, 2] loc tensor with depot at index 0.
        RL4CO typically stores customers in 'locs' and depot separately.
        """
        locs = td_i["locs"]
        if "depot" in td_i.keys():
            depot = td_i["depot"]
            # If locs are customers only, prepend depot
            if locs.ndim == 2 and depot.ndim == 1:
                if "demand" in td_i.keys() and locs.shape[0] == td_i["demand"].shape[-1]:
                    locs = torch.cat([depot.unsqueeze(0), locs], dim=0)
        return locs

    @staticmethod
    def _plot_nodes(ax, locs, demand=None, probs=None):
        depot = locs[0]
        cust = locs[1:]
        ax.scatter(cust[:, 0], cust[:, 1], s=14)
        ax.scatter([depot[0]], [depot[1]], s=60, marker="s")
        if demand is not None:
            for idx, (x, y) in enumerate(cust):
                ax.text(x + 0.005, y + 0.005, f"{demand[idx]:.2f}", fontsize=6, color="dimgray")
        if probs is not None and probs.size > 0:
            bar_width = 0.01
            for (x, y), p in zip(cust, probs):
                height = float(p)
                ax.add_patch(plt.Rectangle((x - bar_width / 2, y), bar_width, height, color="cornflowerblue", alpha=0.7))
                ax.text(x, y + height + 0.005, f"{p:.2f}", fontsize=6, ha="center", va="bottom", color="navy")
        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def _plot_route_without_depot(ax, coords, **plot_kwargs):
        if coords.shape[0] < 3:
            return
        inner = coords[1:-1]
        if inner.shape[0] < 2:
            return
        ax.plot(inner[:, 0], inner[:, 1], **plot_kwargs)

    @staticmethod
    def _actions_to_probs(actions, num_customers):
        counts = torch.zeros(num_customers, dtype=torch.float32)
        seq = actions.tolist() if hasattr(actions, "tolist") else list(actions)
        for n in seq:
            if n > 0:
                counts[n - 1] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._ensure_out_dir(trainer)

        # Grab first training batch every epoch
        train_loader = pl_module.train_dataloader()
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            pl_module.print("[CVRPTrainPlotCallback] Empty train_dataloader, skipping.")
            return

        device = pl_module.device
        batch = batch.to(device)

        # Reset env on this batch
        td = pl_module.env.reset(batch)

        pl_module.eval()
        with torch.no_grad():
            # NOTE: calc_reward=False avoids capacity-validity asserts during plotting
            out = pl_module.policy(
                td.clone(),
                phase="test",
                decode_type=self.decode_type,
                return_actions=True,
                calc_reward=False,
            )
            actions = out["actions"].cpu()  # [B, T]

        B = actions.shape[0]
        max_examples = min(self.num_examples, B)

        for i in range(max_examples):
            td_i = td[i].cpu()
            locs_full = self._full_locs_from_td(td_i).numpy()  # [1+N, 2]
            demand = td_i.get("demand", None)
            demand_np = demand.numpy() if demand is not None else None
            num_customers = locs_full.shape[0] - 1
            probs = self._actions_to_probs(actions[i], num_customers).numpy()

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            # ---- 1) Plot routes first (each a different color) ----
            model_routes = self._split_model_routes(actions[i])
            colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

            for r in model_routes:
                if len(r) <= 2:
                    continue
                coords = locs_full[r]
                self._plot_route_without_depot(
                    ax, coords,
                    linestyle="-", linewidth=2.0,
                    color=next(colors), alpha=0.95
                )

            # ---- 2) Plot nodes, demands, and probability bars ----
            self._plot_nodes(ax, locs_full, demand=demand_np, probs=probs)

            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Train sample {i} | Epoch {trainer.current_epoch}")

            fname = os.path.join(
                self.out_dir,
                f"train_epoch{trainer.current_epoch:03d}_ex{i}.png"
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        pl_module.train()
