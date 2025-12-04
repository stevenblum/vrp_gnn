import os, glob
from collections import defaultdict

import numpy as np
import torch
import vrplib
from tensordict import TensorDict

# RL4CO helper if available; otherwise we fall back to numpy save
try:
    from rl4co.data.utils import save_tensordict_to_npz
    HAVE_RL4CO_SAVE = True
except Exception:
    HAVE_RL4CO_SAVE = False


def normalize_coord(coords: torch.Tensor) -> torch.Tensor:
    """Min-max scale to [0,1] per axis."""
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    return (coords - mins) / (maxs - mins + 1e-8)


def vrp_to_tensordict(vrp_path: str) -> TensorDict:
    """Convert one CVRPLib .vrp file to an RL4CO-style TensorDict with batch dim = 1."""
    prob = vrplib.read_instance(vrp_path)

    coords = torch.tensor(prob["node_coord"], dtype=torch.float32)
    coords = normalize_coord(coords)

    depot = coords[0]          # (2,)
    locs = coords[1:]          # (n_loc, 2), customers only

    demand = torch.tensor(prob["demand"][1:], dtype=torch.float32)  # skip depot
    capacity = float(prob["capacity"])

    # --- add batch dimension so RL4CO npz loader infers batch_size=1 ---
    td = TensorDict(
        {
            "locs": (locs.unsqueeze(0)),                  # (1, n_loc, 2)
            "depot": (depot.unsqueeze(0)),                # (1, 2)
            "demand": ((demand / capacity).unsqueeze(0)), # (1, n_loc)
        },
        batch_size=[1]
    )
    return td


def save_td_npz(td: TensorDict, out_path: str):
    """Save TensorDict in the RL4CO npz convention."""
    out_path = out_path if out_path.endswith(".npz") else out_path + ".npz"

    if HAVE_RL4CO_SAVE:
        # official rl4co saver :contentReference[oaicite:1]{index=1}
        save_tensordict_to_npz(td, out_path, compress=False)
    else:
        # fallback: identical format (dict of numpy arrays) :contentReference[oaicite:2]{index=2}
        x_dict = {k: v.cpu().numpy() for k, v in td.items()}
        np.savez(out_path, **x_dict)


def main():
    # ---- INPUT: where your extracted .vrp files live ----
    # e.g. ".../cvrplib_x/X/" from your logs
    ROOT = os.path.abspath("cvrplib_x/X")

    # ---- OUTPUT: where to write npz files ----
    OUT_INST_DIR = os.path.abspath("cvrplib_x_npz/instances")
    OUT_GROUP_DIR = os.path.abspath("cvrplib_x_npz/groups")

    os.makedirs(OUT_INST_DIR, exist_ok=True)
    os.makedirs(OUT_GROUP_DIR, exist_ok=True)

    vrp_files = sorted(glob.glob(os.path.join(ROOT, "**", "X-*.vrp"), recursive=True))
    print(f"Found {len(vrp_files)} X-set .vrp files under {ROOT}")

    if len(vrp_files) == 0:
        raise RuntimeError("No X-*.vrp files found. Check ROOT path.")

    # group by total nodes (nXXX) for convenience
    groups = defaultdict(list)

    for p in vrp_files:
        base = os.path.basename(p).replace(".vrp", "")  # e.g. X-n101-k25

        # parse n_total from "X-n101-k25"
        try:
            n_token = base.split("-")[1]   # "n101"
            n_total = int(n_token[1:])    # 101
        except Exception:
            print("Skipping unrecognized filename:", base)
            continue

        td = vrp_to_tensordict(p)

        # save per-instance
        out_inst = os.path.join(OUT_INST_DIR, base + ".npz")
        save_td_npz(td, out_inst)

        groups[n_total].append(td)

    print("Saved per-instance npz files to:", OUT_INST_DIR)

    # save grouped-by-size files (each group is size 1 for X, but still useful)
    for n_total, td_list in groups.items():
        batch_td = TensorDict.stack(td_list, dim=0)  # (batch, ...)
        out_group = os.path.join(OUT_GROUP_DIR, f"X_n{n_total}.npz")
        save_td_npz(batch_td, out_group)

    print("Saved grouped npz files to:", OUT_GROUP_DIR)
    print("Groups:", {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
