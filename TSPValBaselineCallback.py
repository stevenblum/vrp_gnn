# ValBaselineCallback.py
import os
import tempfile
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from concorde.tsp import TSPSolver


def solve_tsp_concorde(coords: np.ndarray, scale: float = 10_000.0):
    """
    coords: [N, 2] float in [0, 1]
    Returns:
        tour: np.array [N] with permutation 0..N-1
        cost: float, Euclidean tour length in original coords
    """
    assert coords.ndim == 2 and coords.shape[1] == 2

    # Scale to ints for Concorde
    scaled = (coords * scale).astype(int)
    xs, ys = scaled[:, 0], scaled[:, 1]

    # Optionally sandbox Concorde's .res files in a temp dir
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="concorde_") as tmpdir:
        os.chdir(tmpdir)
        try:
            solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
            sol = solver.solve(verbose=False)  # quiet output
        finally:
            os.chdir(old_cwd)

    tour = np.array(sol.tour, dtype=int)

    # recompute length in original coordinate space
    tour_coords = coords[tour]
    tour_coords = np.vstack([tour_coords, tour_coords[0]])  # close tour
    diffs = np.diff(tour_coords, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cost = float(seg_lengths.sum())

    return tour, cost


def concorde_tours_and_costs_for_batch(td):
    """
    td: TensorDict with key "locs": [B, N, 2]
    Returns:
        tours: LongTensor [B, N]
        costs: FloatTensor [B]
    """
    locs = td["locs"].cpu().numpy()  # [B, N, 2]
    B, N, _ = locs.shape

    all_tours = []
    all_costs = []

    for b in range(B):
        tour, cost = solve_tsp_concorde(locs[b])
        all_tours.append(tour)
        all_costs.append(cost)

    tours = torch.tensor(np.stack(all_tours, axis=0), dtype=torch.long)   # [B, N]
    costs = torch.tensor(all_costs, dtype=torch.float32)                  # [B]

    return tours, costs


class TSPValBaselineCallback(Callback):
    def __init__(
        self,
        tours_attr: str = "val_concorde_tours",
        costs_attr: str = "val_concorde_costs",
        max_batches: int | None = None,
    ):
        super().__init__()
        self.tours_attr = tours_attr
        self.costs_attr = costs_attr
        self.max_batches = max_batches
        self.has_run = False

    def on_validation_start(self, trainer, pl_module) -> None:
        # Run only once (including sanity check)
        if self.has_run:
            return
        self.has_run = True

        pl_module.print("[ConcordeBaseline] Computing Concorde tours & costs for validation set...")

        val_loader = pl_module.val_dataloader()

        all_tours = []
        all_costs = []
        num_solved = 0

        for b_idx, batch in enumerate(val_loader):
            if self.max_batches is not None and b_idx >= self.max_batches:
                break

            tours_b, costs_b = concorde_tours_and_costs_for_batch(batch)
            all_tours.append(tours_b)
            all_costs.append(costs_b)

            num_solved += batch["locs"].size(0)
            if (b_idx + 1) % 5 == 0 or b_idx == 0:
                pl_module.print(f"[ConcordeBaseline] Solved {num_solved} val instances so far...")

        if not all_tours:
            pl_module.print("[ConcordeBaseline] No validation batches, skipping.")
            return

        all_tours = torch.cat(all_tours, dim=0)   # [num_val, N]
        all_costs = torch.cat(all_costs, dim=0)   # [num_val]

        setattr(pl_module, self.tours_attr, all_tours)
        setattr(pl_module, self.costs_attr, all_costs)

        pl_module.print(
            f"[ConcordeBaseline] Done. Stored {all_tours.size(0)} Concorde tours "
            f"and costs as pl_module.{self.tours_attr} / pl_module.{self.costs_attr}."
        )