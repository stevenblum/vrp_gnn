from pathlib import Path
import os
import torch
import vrplib
from tensordict import TensorDict
from torch.utils.data import DataLoader

from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from rl4co.data.utils import load_npz_to_tensordict  # fast RL4CO loader :contentReference[oaicite:1]{index=1}
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from CVRPValBaselineCallback import CVRPValBaselineCallback
from CVRPGraphPlotCallback import CVRPGraphPlotCallback
from CVRPLibGraphPlotCallback import CVRPLibGraphPlotCallback
from CVRPMetricPlotCallback import CVRPMetricPlotCallback

# ---------- helpers to load your chosen X instances ----------

def normalize_coord(coords: torch.Tensor) -> torch.Tensor:
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    return (coords - mins) / (maxs - mins + 1e-8)

def vrp_to_td(vrp_path: str) -> TensorDict:
    """Read raw .vrp -> RL4CO-style TensorDict with batch dim = 1."""
    prob = vrplib.read_instance(vrp_path)
    coords = torch.tensor(prob["node_coord"], dtype=torch.float32)
    coords = normalize_coord(coords)

    depot = coords[0]
    locs  = coords[1:]

    demand = torch.tensor(prob["demand"][1:], dtype=torch.float32)
    capacity = float(prob["capacity"])

    td = TensorDict(
        {
            "locs": locs,
            "depot": depot,
            "demand": demand / capacity,  # normalized demand expected by RL4CO CVRP
        },
        batch_size=[]
    )
    return batchify_td(td)

def batchify_td(td: TensorDict) -> TensorDict:
    """Ensure a leading batch dimension of 1 for env.reset(batch)."""
    if td["locs"].ndim == 2:   # (n_loc,2) -> (1,n_loc,2)
        td = TensorDict(
            {k: v.unsqueeze(0) for k, v in td.items()},
            batch_size=[1],
        )
    return td

def load_val_instance(path: str) -> TensorDict:
    p = Path(path)
    if p.suffix == ".npz":
        td = load_npz_to_tensordict(str(p))  # assumes first axis is batch :contentReference[oaicite:2]{index=2}
        # if you saved per-instance without batch, fix it:
        if td["locs"].ndim == 2:
            td = batchify_td(td)
        return td
    elif p.suffix == ".vrp":
        return vrp_to_td(str(p))
    else:
        raise ValueError(f"Unsupported validation file type: {p}")


# ---------- POMO subclass that swaps only validation ----------

class POMOWithXVal(POMO):
    def __init__(self, *args, val_tds=None, val_batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_val_tds = val_tds
        self._external_val_bs = val_batch_size

    def val_dataloader(self):
        if not self._external_val_tds:
            return super().val_dataloader()  # fall back to random val
        # IMPORTANT: don't stack different sizes; batch_size=1 works for any mix
        return DataLoader(
            self._external_val_tds,
            batch_size=self._external_val_bs,
            shuffle=False,
            collate_fn=lambda batch: batch[0] if self._external_val_bs == 1 else TensorDict.stack(batch, 0),
        )


def main():
    # ---------------- TRAINING (random as before) ----------------
    num_loc_train = 30
    exp_name = f"CVRP_{num_loc_train}_POMO_AttentionModel"

    generator = CVRPGenerator(
        num_loc=num_loc_train,
        loc_distribution="uniform",
        min_demand=1,
        max_demand=10,
    )
    env = CVRPEnv(generator)

    policy = AttentionModelPolicy(env_name=env.name, num_encoder_layers=6)

    # ---------------- VALIDATION (X instances you pick) ----------------
    # Point this to wherever you saved the converted files.
    # Can be .npz (preferred) or .vrp.
    VAL_BASE = Path("cvrplib_x_npz/instances")

    # Specify the instances you want by filename stem.
    # Example stems: "X-n101-k25", "X-n200-k36", ...
    VAL_INSTANCES = [
        "X-n101-k25",
        "X-n106-k14",
        "X-n110-k13",
        "X-n115-k10",
        "X-n120-k6",
        # add/remove freely
    ]

    val_paths = []
    for stem in VAL_INSTANCES:
        npz = VAL_BASE / f"{stem}.npz"
        vrp = Path("cvrplib_x/X") / f"{stem}.vrp"  # fallback if you prefer raw vrp
        if npz.exists():
            val_paths.append(str(npz))
        elif vrp.exists():
            val_paths.append(str(vrp))
        else:
            raise FileNotFoundError(f"Couldn't find {stem} as .npz or .vrp")

    val_tds = [load_val_instance(p) for p in val_paths]
    print("Validation set:", val_paths)

    model = POMOWithXVal(
        env,
        policy,
        batch_size=512,                 # training batch size
        optimizer_kwargs={"lr": 1e-4},
        train_data_size=100_000,      # random train data per epoch
        val_data_size=0,               # unused when external val loader is provided
        dataloader_num_workers=4,
        val_tds=val_tds,               # <-- your X instances
        val_batch_size=1,              # safest for mixed sizes
    )

    # ---------------- callbacks/logger/trainer ----------------
    #baseline_cb = CVRPValBaselineCallback(max_batches=2)
    #plot_graph_cb = CVRPGraphPlotCallback(env, num_examples=5)
    plot_metric_cb = CVRPMetricPlotCallback()

    plot_graph_cb = CVRPLibGraphPlotCallback(
        env,
        instance_names=VAL_INSTANCES,
        sol_base_dir="cvrplib_x/X/",  # folder with X-n101-k25.sol etc.
        decode_type="greedy",
    )

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=exp_name,
        version=None,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        filename=exp_name + "-epoch{epoch:03d}-val{val/reward:.3f}",
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    callback_list = [ckpt_cb, 
                     #baseline_cb, 
                     plot_graph_cb, 
                     plot_metric_cb]
    
    trainer = RL4COTrainer(
        max_epochs=300,
        callbacks=callback_list,
        accelerator="mps",
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(model)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
