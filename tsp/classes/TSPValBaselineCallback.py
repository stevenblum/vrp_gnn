# ValBaselineCallback.py (OR-Tools version)
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_tsp_ortools(
    coords: np.ndarray,
    scale: float = 10_000.0,
    time_limit_s: float = 1.0,
    first_solution_strategy: int = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: int = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    depot: int = 0,
):
    """
    coords: [N, 2] float in [0, 1]
    OR-Tools RoutingModel uses integer arc costs, so we scale+round distances.
    Returns:
        tour: np.array [N] with permutation 0..N-1
        cost: float, Euclidean tour length in original coords
    """
    assert coords.ndim == 2 and coords.shape[1] == 2
    N = coords.shape[0]

    # Pairwise Euclidean distances (float)
    diffs = coords[:, None, :] - coords[None, :, :]
    dist_f = np.linalg.norm(diffs, axis=-1)  # [N, N]

    # Integer distance matrix for OR-Tools (routing is integral)
    dist_int = np.rint(dist_f * scale).astype(np.int64)

    # Create routing model for 1 vehicle starting/ending at depot
    manager = pywrapcp.RoutingIndexManager(N, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_int[from_node, to_node])

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = first_solution_strategy
    search_parameters.local_search_metaheuristic = local_search_metaheuristic

    # Time limit can be fractional seconds
    secs = int(time_limit_s)
    nanos = int((time_limit_s - secs) * 1e9)
    search_parameters.time_limit.seconds = secs
    search_parameters.time_limit.nanos = nanos
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        # Extremely rare for Euclidean TSP; fallback to identity tour
        tour = np.arange(N, dtype=int)
    else:
        # Extract tour
        index = routing.Start(0)
        tour_list = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tour_list.append(node)
            index = solution.Value(routing.NextVar(index))
        tour = np.array(tour_list, dtype=int)

    # Recompute true float cost in original coordinate space
    cost = float(np.sum(dist_f[tour, np.roll(tour, -1)]))
    return tour, cost


def ortools_tours_and_costs_for_batch(td, **solver_kwargs):
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
        tour, cost = solve_tsp_ortools(locs[b], **solver_kwargs)
        all_tours.append(tour)
        all_costs.append(cost)

    tours = torch.tensor(np.stack(all_tours, axis=0), dtype=torch.long)   # [B, N]
    costs = torch.tensor(all_costs, dtype=torch.float32)                  # [B]
    return tours, costs


class TSPValBaselineCallback(Callback):
    def __init__(
        self,
        tours_attr: str = "val_ortools_tours",
        costs_attr: str = "val_ortools_costs",
        max_batches: int | None = None,
        # Solver knobs
        scale: float = 10_000.0,
        time_limit_s: float = 1.0,
        first_solution_strategy: int = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        local_search_metaheuristic: int = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        depot: int = 0,
    ):
        super().__init__()
        self.tours_attr = tours_attr
        self.costs_attr = costs_attr
        self.max_batches = max_batches

        self.solver_kwargs = dict(
            scale=scale,
            time_limit_s=time_limit_s,
            first_solution_strategy=first_solution_strategy,
            local_search_metaheuristic=local_search_metaheuristic,
            depot=depot,
        )

        self.has_run = False

    def on_validation_start(self, trainer, pl_module) -> None:
        # Run only once (including sanity check)
        if self.has_run:
            return
        self.has_run = True

        pl_module.print("[ORToolsBaseline] Computing OR-Tools tours & costs for validation set...")

        val_loader = pl_module.val_dataloader()

        all_tours = []
        all_costs = []
        num_solved = 0

        for b_idx, batch in enumerate(val_loader):
            if self.max_batches is not None and b_idx >= self.max_batches:
                break

            tours_b, costs_b = ortools_tours_and_costs_for_batch(batch, **self.solver_kwargs)
            all_tours.append(tours_b)
            all_costs.append(costs_b)

            num_solved += batch["locs"].size(0)
            if (b_idx + 1) % 5 == 0 or b_idx == 0:
                pl_module.print(f"[ORToolsBaseline] Solved {num_solved} val instances so far...")

        if not all_tours:
            pl_module.print("[ORToolsBaseline] No validation batches, skipping.")
            return

        all_tours = torch.cat(all_tours, dim=0)   # [num_val, N]
        all_costs = torch.cat(all_costs, dim=0)   # [num_val]

        setattr(pl_module, self.tours_attr, all_tours)
        setattr(pl_module, self.costs_attr, all_costs)

        pl_module.print(
            f"[ORToolsBaseline] Done. Stored {all_tours.size(0)} OR-Tools tours "
            f"and costs as pl_module.{self.tours_attr} / pl_module.{self.costs_attr}."
        )
