# CVRPValBaselineCallback.py
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_cvrp_ortools(
    coords: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: float,
    num_vehicles: int | None = None,
    distance_scale: float = 1_000.0,
    log_search: bool = False,
):
    """
    Solve a single CVRP instance using OR-Tools.

    Args:
        coords: [N_nodes, 2] array of (x, y), with node 0 = depot.
        demands: [N_customers] or [N_nodes] array of demands.
                 If length is N_customers, it is assumed to be for nodes 1..N-1.
        vehicle_capacity: scalar capacity (same for all vehicles).
        num_vehicles: number of vehicles. If None, defaults to N_customers.
        distance_scale: factor to scale Euclidean distances into ints for OR-Tools.
        log_search: if True, OR-Tools prints search progress to stdout.

    Returns:
        routes: list of 1D np.ndarray, one per vehicle
                Each route is a sequence of node indices starting and ending at depot (0).
                Vehicles that aren't used may have route [0, 0].
        total_cost: float, sum of Euclidean tour lengths in original coord space.
    """
    assert coords.ndim == 2 and coords.shape[1] == 2
    N_nodes = coords.shape[0]

    # Normalize demands to length N_nodes (0 at depot)
    if demands.shape[0] == N_nodes - 1:
        demands_full = np.concatenate([[0.0], demands.astype(float)])
    elif demands.shape[0] == N_nodes:
        demands_full = demands.astype(float)
    else:
        raise ValueError(
            f"Unexpected demands length {demands.shape[0]} for N_nodes={N_nodes}"
        )

    # Number of customers (excluding depot)
    num_customers = N_nodes - 1
    if num_vehicles is None:
        # Upper bound: at most one customer per vehicle
        num_vehicles = num_customers

    # Precompute distance matrix in integer form for OR-Tools
    dist_matrix = np.zeros((N_nodes, N_nodes), dtype=int)
    for i in range(N_nodes):
        for j in range(N_nodes):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = int(dist * distance_scale)

    # OR-Tools data model
    manager = pywrapcp.RoutingIndexManager(N_nodes, num_vehicles, 0)  # depot = 0
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return dist_matrix[i, j]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Demand callback
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(demands_full[node])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # no capacity slack
        [int(vehicle_capacity)] * num_vehicles,
        True,  # start cumul at zero
        "Capacity",
    )

    # Search parameters (heuristic baseline, not exact optimal)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(5)

    # Turn on OR-Tools' internal search logging if requested
    search_params.log_search = log_search

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        raise RuntimeError("OR-Tools could not find a CVRP solution")

    # Extract routes
    routes = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        # Add final depot
        node = manager.IndexToNode(index)
        route.append(node)
        routes.append(np.array(route, dtype=int))

    # Compute continuous cost in original coordinate space
    total_cost = 0.0
    for route in routes:
        if len(route) <= 1:
            continue
        route_coords = coords[route]
        diffs = np.diff(route_coords, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total_cost += float(seg_lengths.sum())

    return routes, total_cost


def cvrp_ortools_solutions_for_batch(
    td,
    num_vehicles: int | None = None,
    log_search: bool = False,
):
    """
    td: TensorDict with keys:
        "locs": [B, 1+N, 2]
        "demand": [B, N]
        "vehicle_capacity" or "capacity": [B] or [B, 1]

    Returns:
        all_routes: list of length B, each element is a list of np.ndarray routes
        costs: FloatTensor [B]
    """
    locs = td["locs"].cpu().numpy()  # [B, 1+N, 2]
    demand = td["demand"].cpu().numpy()  # [B, N]

    if "vehicle_capacity" in td.keys():
        cap = td["vehicle_capacity"].cpu().numpy()
    elif "capacity" in td.keys():
        cap = td["capacity"].cpu().numpy()
    else:
        raise KeyError("Expected 'vehicle_capacity' or 'capacity' in TensorDict")

    B, N_nodes, _ = locs.shape

    all_routes = []
    all_costs = []

    for b in range(B):
        coords_b = locs[b]  # [1+N, 2], node 0 is depot
        demand_b = demand[b]  # [N]
        cap_b = cap[b]
        cap_scalar = float(cap_b[0] if np.ndim(cap_b) > 0 else cap_b)

        routes_b, cost_b = solve_cvrp_ortools(
            coords_b,
            demand_b,
            vehicle_capacity=cap_scalar,
            num_vehicles=num_vehicles,
            log_search=log_search,
        )
        all_routes.append(routes_b)
        all_costs.append(cost_b)

    costs = torch.tensor(all_costs, dtype=torch.float32)
    return all_routes, costs


class CVRPValBaselineCallback(Callback):
    """
    Compute OR-Tools CVRP baselines for the validation set.

    Design:
      - During sanity check:
          * OR-Tools is run on each validation batch (for logs / debugging),
            but results are NOT stored.
      - During the first real validation epoch:
          * OR-Tools is run again and the results are accumulated over batches,
            then stored on pl_module at epoch end.
      - After that, the baseline is not recomputed (has_run=True).

    Attributes set on pl_module (after first real validation):
        pl_module.<routes_attr>: list of length num_val, each entry is
            a list of np.ndarray, one route per vehicle.
        pl_module.<costs_attr>: FloatTensor [num_val] of baseline costs.
    """

    def __init__(
        self,
        routes_attr: str = "val_cvrp_routes",
        costs_attr: str = "val_cvrp_costs",
        max_batches: int | None = None,
        num_vehicles: int | None = None,
        log_search: bool = False,
    ):
        super().__init__()
        self.routes_attr = routes_attr
        self.costs_attr = costs_attr
        self.max_batches = max_batches
        self.num_vehicles = num_vehicles
        self.log_search = log_search

        # Internal state
        self.has_run = False
        self._routes_acc = []
        self._costs_acc = []
        self._seen_batches = 0

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        # If we've already computed the baseline once, nothing to do.
        if self.has_run:
            return

        # Reset accumulators at the start of each validation epoch
        self._routes_acc = []
        self._costs_acc = []
        self._seen_batches = 0

        phase = "sanity check" if trainer.sanity_checking else "validation"
        pl_module.print(
            f"[CVRPBaseline] Starting {phase} epoch; running OR-Tools on CVRP batches..."
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # If we've already computed and stored baseline, we do nothing.
        if self.has_run:
            return

        # respect optional max_batches for baseline computation
        if self.max_batches is not None and self._seen_batches >= self.max_batches:
            return

        # Run OR-Tools on this batch
        pl_module.print(
            f"[CVRPBaseline] OR-Tools solving batch {batch_idx} "
            f"(sanity_checking={trainer.sanity_checking})..."
        )

        routes_b, costs_b = cvrp_ortools_solutions_for_batch(
            batch,
            num_vehicles=self.num_vehicles,
            log_search=self.log_search,
        )

        # During sanity check, we DO NOT store the results (just run OR-Tools for logs/debugging)
        if trainer.sanity_checking:
            return

        # During the first real validation epoch, we accumulate results
        self._routes_acc.extend(routes_b)
        self._costs_acc.append(costs_b)
        self._seen_batches += 1

        # Optional progress print
        num_instances = len(self._routes_acc)
        pl_module.print(
            f"[CVRPBaseline] Accumulated OR-Tools solutions for {num_instances} CVRP val instances so far..."
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # If we've already computed baseline or we're in sanity check, do nothing
        if self.has_run or trainer.sanity_checking:
            return

        if not self._routes_acc:
            pl_module.print(
                "[CVRPBaseline] No CVRP baseline routes accumulated this epoch, skipping."
            )
            return

        all_costs = torch.cat(self._costs_acc, dim=0)  # [num_val]

        setattr(pl_module, self.routes_attr, self._routes_acc)
        setattr(pl_module, self.costs_attr, all_costs)

        pl_module.print(
            f"[CVRPBaseline] Done. Stored OR-Tools baseline routes and costs for "
            f"{len(self._routes_acc)} validation instances as "
            f"pl_module.{self.routes_attr} / pl_module.{self.costs_attr}."
        )

        # Mark that we've done it once; no need to recompute in later epochs.
        self.has_run = True
