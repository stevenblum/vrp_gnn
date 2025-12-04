# CVRPGraphPlotCallback.py
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from itertools import cycle
from tensordict import TensorDict
from classes.CVRPLibHelpers import batchify_td


class CVRPLibGraphPlotCallback(Callback):
    def __init__(
        self,
        env,
        instance_names,          # list like ["X-n101-k25", "X-n106-k14", ...]
        sol_base_dir: str,       # path containing matching .sol files
        subdir: str = "val_plots_cvrp",
        decode_type: str = "greedy",
        logger=None,  # TensorBoard logger for logging individual rewards
        use_model_solutions: bool = True,  # Use solutions from model's validation step if available
    ):
        super().__init__()
        self.env = env
        self.instance_names = list(instance_names)
        self.sol_base_dir = sol_base_dir
        self.subdir = subdir
        self.decode_type = decode_type
        self.logger = logger
        self.use_model_solutions = use_model_solutions
        self.out_dir = None

        # Cache parsed solutions on init
        self._bks_routes = {}  # name -> list[list[int]] (0-based, depot=0 wrapped)
        self._bks_costs_raw = {}   # name -> raw CVRPLib cost (original scale, optional)
        self._load_all_solutions()
        # Fixed sampling settings for callback evaluation
        self.eval_num_samples = 10_000
        self.eval_temperature = 0.5

    def _ensure_out_dir(self, trainer):
        if self.out_dir is None:
            self.out_dir = os.path.join(trainer.log_dir, self.subdir)
            os.makedirs(self.out_dir, exist_ok=True)

    # ---------------- Solution handling ----------------

    @staticmethod
    def _parse_n_total_from_name(name: str):
        """
        For X instances like 'X-n101-k25', returns n_total=101.
        Returns None if pattern not found.
        """
        m = re.search(r"-n(\d+)-", name)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_cvrplib_sol(sol_path: str, n_total: int | None = None):
        """
        Parse CVRPLib .sol file with auto-detection of 0-based vs 1-based IDs.

        Heuristics:
          - If any 0 appears in routes -> 0-based customer IDs.
          - Else if n_total known and max_id <= n_total-1 -> 0-based.
          - Else -> 1-based.

        Returns:
            routes: list[list[int]] in RL4CO 0-based indexing, each route [0, ..., 0]
            cost_raw: float or None (original CVRPLib cost scale)
        """
        routes_raw = []
        cost_raw = None
        all_ids = []

        with open(sol_path, "r") as f:
            for line in f:
                line = line.strip()

                if line.lower().startswith("route"):
                    nums = [int(x) for x in re.findall(r"\d+", line.split(":")[-1])]
                    if nums:
                        routes_raw.append(nums)
                        all_ids.extend(nums)

                if "cost" in line.lower():
                    m = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if m:
                        cost_raw = float(m[-1])

        if not all_ids:
            return [], cost_raw

        mn, mx = min(all_ids), max(all_ids)

        sol_is_zero_based = (mn == 0) or (n_total is not None and mx <= n_total - 1)

        routes = []
        for r in routes_raw:
            if sol_is_zero_based:
                r0 = r[:]                 # already 0-based
            else:
                r0 = [x - 1 for x in r]   # 1-based -> 0-based

            # If depot appears in solution lines, remove it (will be 0 after conversion).
            r0 = [x for x in r0 if x != 0]

            # Wrap with depot
            routes.append([0] + r0 + [0])

        return routes, cost_raw

    def _load_all_solutions(self):
        """Read all .sol files matching instance_names into caches."""
        for name in self.instance_names:
            sol_path = os.path.join(self.sol_base_dir, f"{name}.sol")
            if not os.path.exists(sol_path):
                raise FileNotFoundError(f"Missing solution file: {sol_path}")

            n_total = self._parse_n_total_from_name(name)
            routes, cost_raw = self._parse_cvrplib_sol(sol_path, n_total=n_total)

            self._bks_routes[name] = routes
            self._bks_costs_raw[name] = cost_raw

    # ---------------- Plotting helpers ----------------

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
        Make a [1+N, 2] loc tensor with depot at index 0, regardless of td layout.
        RL4CO sometimes stores depot separately.
        """
        locs = td_i["locs"]
        if "depot" in td_i.keys():
            depot = td_i["depot"]
            if locs.ndim == 2 and depot.ndim == 1:
                if "demand" in td_i.keys() and locs.shape[0] == td_i["demand"].shape[-1]:
                    locs = torch.cat([depot.unsqueeze(0), locs], dim=0)
        return locs

    @staticmethod
    def _plot_nodes(ax, locs_np, demand=None, probs=None):
        """
        locs_np: numpy array [1+N, 2], node 0 = depot
        demand: optional numpy array [N]
        probs: optional numpy array [N]
        """
        depot = locs_np[0]
        cust = locs_np[1:]
        ax.scatter(cust[:, 0], cust[:, 1], s=12)
        ax.scatter([depot[0]], [depot[1]], s=40, marker="s")
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
        """Plot route lines skipping depot edges."""
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

    @staticmethod
    def _routes_length(locs_np, routes):
        """Compute total length for list of routes in CURRENT locs space."""
        total = 0.0
        for r in routes:
            if len(r) <= 1:
                continue
            coords = locs_np[r]  # (k,2)
            diffs = coords[1:] - coords[:-1]
            total += np.sqrt((diffs ** 2).sum(axis=1)).sum()
        return float(total)

    # ---------------- Lightning hook ----------------

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self._ensure_out_dir(trainer)

        val_loader = pl_module.val_dataloader()
        device = pl_module.device
        pl_module.eval()

        instance_counter = 0
        rewards_logged = []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = batch.to(device)

                # Create a temporary environment for this batch size
                # Since validation instances may have different sizes than training
                num_loc_val = batch["locs"].shape[1]  # number of customer locations
                
                # Extract capacity from batch if available, otherwise use default
                if "capacity" in batch.keys():
                    # Capacity is already in the batch from the npz file
                    capacity = batch["capacity"]
                else:
                    # Default capacity of 1.0 (since demand is already normalized)
                    capacity = torch.ones(batch.batch_size[0], device=device)
                
                from rl4co.envs.routing import CVRPEnv, CVRPGenerator
                temp_generator = CVRPGenerator(num_loc=num_loc_val, capacity=capacity.item() if capacity.numel() == 1 else 1.0)
                temp_env = CVRPEnv(temp_generator)
                
                # Reset with the temporary env to get proper state
                td = temp_env.reset(batch)

                # Make sure reward exists to avoid errors
                if "reward" not in td.keys():
                    td.set("reward", torch.zeros(td.batch_size, device=td.device))

                # Run our own sampling eval: num_samples=eval_num_samples, temperature=eval_temperature
                # Repeat the single instance num_samples times
                td_batched = batchify_td(td)
                td_batched = TensorDict.cat(
                    [td_batched.clone() for _ in range(self.eval_num_samples)], dim=0
                )
                out = pl_module.policy(
                    td_batched.clone(),
                    phase="test",
                    decode_type="sampling",
                    temperature=self.eval_temperature,
                    return_actions=True,
                    calc_reward=True,
                )
                rewards_out = out.get("reward", None)
                actions = out.get("actions", None)
                if actions is None or rewards_out is None:
                    continue
                actions = actions.cpu()
                rewards_out = rewards_out.cpu()
                # Group rewards by original batch (B=1) and take best over samples
                bs = td.batch_size[0]
                rewards_grouped = rewards_out.split(bs)
                rewards_stacked = torch.stack(rewards_grouped, dim=1)  # [num_samples, B]
                best_vals, best_idx = rewards_stacked.max(dim=0)       # [B]

                B = actions.shape[0]

                for i in range(B):
                    td_i = td[i].cpu()

                    # figure out which instance name this corresponds to
                    if "instance_name" in td_i.keys():
                        inst_name = str(td_i["instance_name"][0])
                    elif "instance_id" in td_i.keys():
                        inst_id = int(td_i["instance_id"].item())
                        inst_name = self.instance_names[inst_id]
                    else:
                        inst_name = self.instance_names[instance_counter]

                    instance_counter += 1

                    bks_routes = self._bks_routes.get(inst_name, None)

                    locs_full = self._full_locs_from_td(td_i).numpy()
                    demand = td_i.get("demand", None)
                    demand_np = demand.numpy() if demand is not None else None
                    num_customers = locs_full.shape[0] - 1
                    if demand_np is not None:
                        if demand_np.shape[-1] == num_customers + 1:
                            demand_np = demand_np[1:]
                        elif demand_np.shape[-1] != num_customers:
                            demand_np = demand_np[..., :num_customers]
                    probs = self._actions_to_probs(actions[i], num_customers).numpy()

                    # ---- DISTANCE CORRECTION ----
                    # Recompute BKS cost on normalized coords so it's comparable to model cost
                    if bks_routes is not None and len(bks_routes) > 0:
                        bks_cost_norm = self._routes_length(locs_full, bks_routes)
                    else:
                        bks_cost_norm = None

                    # Model routes + cost (also in normalized coords)
                    # Select best actions according to reward
                    act_i = actions[best_idx[i] + i * self.eval_num_samples]
                    model_routes = self._split_model_routes(act_i)
                    reward_val = best_vals[i].item()
                    model_cost = -reward_val

                    # Log individual validation reward to TensorBoard
                    # Reward is negative cost
                    if self.logger is not None:
                        self.logger.experiment.add_scalar(
                            f"val_individual/{inst_name}/reward",
                            reward_val,
                            trainer.global_step
                        )
                        # Also log the cost for easier interpretation
                        self.logger.experiment.add_scalar(
                            f"val_individual/{inst_name}/cost",
                            model_cost,
                            trainer.global_step
                        )
                        # Log gap if BKS is available
                        if bks_cost_norm is not None and bks_cost_norm > 0:
                            gap_pct = 100.0 * (model_cost - bks_cost_norm) / bks_cost_norm
                            self.logger.experiment.add_scalar(
                                f"val_individual/{inst_name}/gap_percent",
                                gap_pct,
                                trainer.global_step
                            )
                    rewards_logged.append(reward_val)

                    fig, (ax_bks, ax_model) = plt.subplots(
                        1, 2, figsize=(10, 5), sharex=True, sharey=True
                    )

                    # ----- BKS axis -----
                    bks_color_by_first = {}
                    self._plot_nodes(ax_bks, locs_full, demand=demand_np)
                    if bks_routes is not None and bks_cost_norm is not None:
                        ax_bks.set_title(f"BKS (normalized)\ncost = {bks_cost_norm:.3f}")
                        for route in bks_routes:
                            if len(route) <= 2:
                                continue
                            coords = locs_full[route]
                            line, = ax_bks.plot([], [], linestyle="--", linewidth=1.2, alpha=0.9)
                            color = line.get_color()
                            self._plot_route_without_depot(
                                ax_bks, coords,
                                linestyle="--", linewidth=1.2, alpha=0.9, color=color
                            )
                            first_customer = route[1]
                            if first_customer != 0:
                                bks_color_by_first[first_customer] = color
                    else:
                        ax_bks.set_title("BKS unavailable")

                    # ----- Model axis -----
                    self._plot_nodes(ax_model, locs_full, demand=demand_np, probs=probs)

                    fallback_colors = cycle(
                        plt.rcParams["axes.prop_cycle"].by_key()["color"]
                    )

                    for r in model_routes:
                        if len(r) <= 2:
                            continue
                        first_customer = r[1]
                        color = bks_color_by_first.get(first_customer, next(fallback_colors))
                        coords = locs_full[r]
                        self._plot_route_without_depot(
                            ax_model, coords,
                            linestyle="-", linewidth=2.0, color=color, alpha=0.95
                        )

                    if bks_cost_norm is not None and bks_cost_norm > 0:
                        gap_pct = 100.0 * (model_cost - bks_cost_norm) / bks_cost_norm
                        ax_model.set_title(
                            f"Model ({self.decode_type})\n"
                            f"cost = {model_cost:.3f} | gap = {gap_pct:.2f}%"
                        )
                    else:
                        ax_model.set_title(
                            f"Model ({self.decode_type})\ncost = {model_cost:.3f}"
                        )

                    fig.suptitle(
                        f"{inst_name} | Epoch {trainer.current_epoch}",
                        y=1.02
                    )

                    fname = os.path.join(
                        self.out_dir,
                        f"{inst_name}_epoch{trainer.current_epoch:03d}.png",
                    )
                    fig.savefig(fname, bbox_inches="tight")
                    plt.close(fig)
        
        # Log average reward across instances for this callback run
        if self.logger is not None and rewards_logged:
            avg_reward = float(np.mean(rewards_logged))
            self.logger.experiment.add_scalar("val/10k_reward", avg_reward, trainer.global_step)
