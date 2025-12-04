from pathlib import Path
import os
import torch
import vrplib
import argparse
import yaml
from datetime import datetime
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from rl4co.data.utils import load_npz_to_tensordict  # fast RL4CO loader :contentReference[oaicite:1]{index=1}
from rl4co.envs.common.distribution_utils import Mix_Multi_Distributions
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from classes.CVRPValBaselineCallback import CVRPValBaselineCallback
from classes.CVRPGraphPlotCallback import CVRPGraphPlotCallback
from classes.CVRPLibGraphPlotCallback import CVRPLibGraphPlotCallback
from classes.CVRPMetricPlotCallback import CVRPMetricPlotCallback
from classes.CVRPSamplerCluster import SamplerCluster
from classes.CVRPSamplerMixed import SamplerMixed
from classes.CVRPTrainGraphPlotCallback import CVRPTrainGraphPlotCallback
from classes.CVRPLibHelpers import normalize_coord, vrp_to_td, batchify_td, load_val_instance

# ---------- POMO subclass that swaps only validation ----------

class POMOWithXVal(POMO):
    def __init__(
        self,
        *args,
        val_tds=None,
        val_batch_size=1,
        fixed_test_dataset=None,
        lr_reduce_factor=0.5,
        lr_reduce_patience=3,
        lr_reduce_threshold=1e-3,
        **kwargs,
    ):
        self.val_num_samples = kwargs.pop("val_num_samples", None)
        self.val_temperature = kwargs.pop("val_temperature", None)
        super().__init__(*args, **kwargs)
        # Avoid storing env/policy objects in hparams (non-picklable / already checkpointed)
        self.save_hyperparameters(ignore=["env", "policy"])
        # Clean any captured entries from prior save_hyperparameters calls
        if hasattr(self, "hparams"):
            self.hparams.pop("env", None)
            self.hparams.pop("policy", None)
        if hasattr(self, "_hparams_initial"):
            self._hparams_initial.pop("env", None)
            self._hparams_initial.pop("policy", None)
        self._external_val_tds = val_tds
        self._external_val_bs = val_batch_size
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_patience = lr_reduce_patience
        self.lr_reduce_threshold = lr_reduce_threshold
        self.best_val_actions = None  # Store best actions from validation
        self.best_val_rewards = None  # Store corresponding rewards
        self.dataloader_names = None  # Defensive default for RL4CO logging
        self._fixed_test_dataset = fixed_test_dataset
        self._patch_env_mask()

    def configure_optimizers(self):
        """Use base optimizer plus ReduceLROnPlateau on val/reward."""
        cfg = super().configure_optimizers()

        def attach_scheduler(optimizer):
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.lr_reduce_factor,
                patience=self.lr_reduce_patience,
                threshold=self.lr_reduce_threshold,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/reward",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if isinstance(cfg, dict) and "optimizer" in cfg:
            return attach_scheduler(cfg["optimizer"])
        if isinstance(cfg, (list, tuple)):
            # If only one optimizer, wrap it; otherwise leave as-is
            if len(cfg) == 1:
                return attach_scheduler(cfg[0])
            return cfg
        return attach_scheduler(cfg)

    def on_validation_epoch_start(self):
        """Log validation configuration at start of each epoch"""
        super().on_validation_epoch_start()
        n_start = self.val_num_samples if self.val_num_samples is not None else self.num_starts
        n_aug = 1 if self.val_num_samples is not None else self.num_augment
        print(f"\n{'='*60}")
        print(f"VALIDATION EPOCH {self.current_epoch} - Configuration:")
        print(f"  Policy val_decode_type: {self.policy.val_decode_type}")
        print(f"  Model num_starts (train/default): {self.num_starts}")
        print(f"  Effective val_num_samples: {n_start}")
        print(f"  Effective num_augment (val): {n_aug}")
        print(f"  Val temperature: {self.val_temperature}")
        print(f"{'='*60}\n")

    def on_validation_epoch_end(self):
        """Store the best validation solutions for callbacks to use"""
        super().on_validation_epoch_end()
        # The solutions are already computed in validation_step, we just need to cache them
        # This will be populated by validation_step
    
    def shared_step(self, batch, batch_idx: int, phase: str, dataloader_idx: int = None):
        """Override shared_step to add batched sampling for validation"""
        # External CVRPLib batches can occasionally carry the depot inside `locs`;
        # if so, strip it so env.reset doesn't double-prepend the depot.
        if phase != "train":
            batch = self._sanitize_batch(batch)
            orig_aug = self.num_augment
            # Keep augmentation shape-safe (no aug but >=1)
            self.num_augment = 1
            # Optional overrides for validation sampling
            orig_temp = getattr(self.policy, "temperature", None)
            if self.val_temperature is not None:
                self.policy.temperature = self.val_temperature
        else:
            orig_aug = None
            orig_temp = None

        # Use standard POMO for training
        if phase == "train":
            return super().shared_step(batch, batch_idx, phase, dataloader_idx)
        
        # For validation/test with large num_starts, use batched sampling
        td = self.env.reset(batch)
        td = self._align_demand_with_locs(td)
        n_aug = self.num_augment
        if phase != "train" and self.val_num_samples is not None:
            n_start = self.val_num_samples
        else:
            n_start = self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start
        
        # Apply augmentation if needed
        if n_aug > 1:
            td = self.augment(td)
        
        # Use standard POMO path but ensure it reuses the preprocessed td
        orig_reset = self.env.reset
        self.env.reset = lambda *_args, **_kwargs: td
        try:
            return super().shared_step(batch, batch_idx, phase, dataloader_idx)
        finally:
            self.env.reset = orig_reset
            if orig_aug is not None:
                self.num_augment = orig_aug
            if orig_temp is not None:
                self.policy.temperature = orig_temp

    @staticmethod
    def _sanitize_batch(batch):
        """Ensure external batches follow RL4CO format: customers in `locs`, depot separate."""
        td = batch.clone()

        # Some preprocessed CVRPLib tensors may already include the depot as locs[0].
        if "locs" in td.keys() and "demand" in td.keys():
            locs = td["locs"]
            demand = td["demand"]

            # If locs length = demand length + 1 and first loc matches depot, drop that extra depot
            if (
                locs.shape[-2] == demand.shape[-1] + 1
                and "depot" in td.keys()
                and torch.allclose(locs[..., 0, :], td["depot"], atol=1e-6)
            ):
                td = td.clone()  # avoid mutating upstream
                td.set("locs", locs[..., 1:, :])

            # If demand already contains a depot slot, drop it to keep only customer demands
            if td["demand"].shape[-1] == td["locs"].shape[-2] + 1:
                td.set("demand", td["demand"][..., 1:])

            # As a final guard, force locs and demand to have matching customer length
            locs_len = td["locs"].shape[-2]
            dem_len = td["demand"].shape[-1]
            if locs_len != dem_len:
                if locs_len == dem_len + 1:
                    td.set("locs", td["locs"][..., 1:, :])  # drop leading node (likely depot)
                elif dem_len == locs_len + 1:
                    td.set("demand", td["demand"][..., 1:])  # drop depot demand if present
                else:
                    min_len = min(locs_len, dem_len)
                    td.set("locs", td["locs"][..., :min_len, :])
                    td.set("demand", td["demand"][..., :min_len])

        return td

    def _patch_env_mask(self):
        """Monkeypatch env.get_action_mask to tolerate minor length mismatches."""
        env = self.env

        def safe_action_mask(td):
            exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"] + 1e-5
            mask_loc = td["visited"][..., 1:]

            # Align lengths defensively (some external batches include an extra depot)
            if mask_loc.shape[-1] != exceeds_cap.shape[-1]:
                min_len = min(mask_loc.shape[-1], exceeds_cap.shape[-1])
                mask_loc = mask_loc[..., :min_len]
                exceeds_cap = exceeds_cap[..., :min_len]

            mask_loc = mask_loc.to(exceeds_cap.dtype) | exceeds_cap
            mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)[
                :, None
            ]
            return ~torch.cat((mask_depot, mask_loc), -1)

        # Bind method to env instance
        env.get_action_mask = safe_action_mask

    @staticmethod
    def _align_demand_with_locs(td):
        """Ensure demand length matches number of customer nodes (locs minus depot)."""
        if "locs" not in td.keys() or "demand" not in td.keys():
            return td

        locs_len = td["locs"].shape[-2] - 1  # exclude depot at locs[:,0]
        dem_len = td["demand"].shape[-1]

        if dem_len == locs_len:
            return td

        td = td.clone()
        if dem_len > locs_len:
            td.set("demand", td["demand"][..., :locs_len])
        else:
            pad = torch.zeros(
                *td["demand"].shape[:-1],
                locs_len - dem_len,
                device=td.device,
                dtype=td["demand"].dtype,
            )
            td.set("demand", torch.cat((td["demand"], pad), dim=-1))
        return td

    def train_dataloader(self):
        """Override to add persistent_workers for efficiency"""
        dataloader = super().train_dataloader()
        # Create new dataloader with persistent_workers if workers > 0
        if dataloader.num_workers > 0:
            return DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=True,
                num_workers=dataloader.num_workers,
                persistent_workers=True,  # Keep workers alive between epochs
                pin_memory=False,         # Disabled to avoid instability with persistent_workers and reload_dataloaders
                collate_fn=dataloader.collate_fn,
                prefetch_factor=2,        # Pre-load batches
            )
        return dataloader

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

    def test_dataloader(self):
        if self._fixed_test_dataset is not None:
            return DataLoader(
                self._fixed_test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                persistent_workers=self.dataloader_num_workers > 0,
            )
        return super().test_dataloader()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CVRP Training with AttentionModel')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension for the attention model (default: 256)')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers (default: 6)')
    parser.add_argument('--num_attn_heads', type=int, default=16,
                        help='Number of attention heads (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer (default: 1e-6)')
    parser.add_argument('--num_loc_train', type=int, default=100,
                        help='Number of customer locations for training (default: 100)')
    parser.add_argument('--min_demand', type=int, default=3,
                        help='Minimum customer demand (default: 3)')
    parser.add_argument('--max_demand', type=int, default=30,
                        help='Maximum customer demand (default: 30)')
    parser.add_argument('--vehicle_capacity', type=float, default=100.0,
                        help='Override vehicle capacity; higher allows more customers per tour (default: 100.0)')
    parser.add_argument('--test_seed', type=int, default=1234,
                        help='Seed for generating fixed test set (default: 1234)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--train_data_size', type=int, default=2_000_000,
                        help='Number of training instances per epoch (default: 2,000,000)')
    parser.add_argument('--test_data_size', type=int, default=100,
                        help='Number of random test instances (default: 100)')
    parser.add_argument('--pomo_num_starts', type=int, default=None,
                        help='Number of POMO starts for validation (default: same as num_loc_train)')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of training epochs (default: 300)')
    parser.add_argument('--normalization', type=str, default='batch',
                        choices=['batch', 'layer', 'none'],
                        help='Normalization type for attention model (default: batch)')
    parser.add_argument('--limit_train_batches', type=float, default=None,
                        help='Limit fraction of training batches per epoch (default: None, use all)')
    parser.add_argument('--log_base_dir', type=str, default='lightning_logs',
                        help='Base directory for logging (default: lightning_logs)')
    parser.add_argument('--train_decode_type', type=str, default='sampling',
                        choices=['sampling', 'greedy'],
                        help='Decode type for training (default: sampling)')
    parser.add_argument('--train_temperature', type=float, default=1.0,
                        help='Sampling temperature for training (higher = more exploration) (default: 1.0)')
    parser.add_argument('--val_decode_type', type=str, default='sampling',
                        choices=['sampling', 'greedy'],
                        help='Decode type for validation (POMO will prepend multistart_) (default: sampling)')
    parser.add_argument('--val_num_samples', type=int, default=10_000,
                        help='Number of samples during validation (default: 10,000). Applies when validation uses sampling.')
    parser.add_argument('--val_temperature', type=float, default=1.0,
                        help='Sampling temperature during validation (default: 1.0)')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Run validation every N epochs (default: 1). Set higher to reduce validation overhead.')
    parser.add_argument('--checkpoint_after_epoch', type=int, default=0,
                        help='Only save checkpoints after this epoch number (default: 0, save from start)')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.7,
                        help='ReduceLROnPlateau: factor to reduce LR when plateauing (default: 0.7)')
    parser.add_argument('--lr_reduce_patience', type=int, default=3,
                        help='ReduceLROnPlateau: epochs with no improvement before reducing LR (default: 3)')
    parser.add_argument('--lr_reduce_threshold', type=float, default=1e-3,
                        help='ReduceLROnPlateau: improvement threshold to ignore (default: 1e-3)')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Precision for training (default: 16-mixed to save memory)')
    args = parser.parse_args()

    # ---------------- TRAINING (random as before) ----------------
    num_loc_train = args.num_loc_train
    num_starts = args.pomo_num_starts if args.pomo_num_starts is not None else num_loc_train
    normalization = None if args.normalization == 'none' else args.normalization
    exp_name = f"emb{args.embedding_dim}_enc{args.num_encoder_layers}_attn{args.num_attn_heads}"

    # Mixed sampler: 1/3 centered depot, 2/3 offset depot
    #                1/2 random customers, 1/2 clustered customers
    '''
    loc_sampler = SamplerMixed(
        p_center=1/3,              # 1/3 centered, 2/3 offset
        p_cluster=0.5,             # 1/2 clustered, 1/2 random
        n_clusters_list=range(3, 7),  # pick k uniformly from this list
        cluster_std=(0.02, 0.08),  # sample std uniformly from this range per instance
        offset_depot_pos=(0.0, 0.0),  # offset depot at corner
    )
    '''

    loc_sampler = Mix_Multi_Distributions()

    capacity_kwargs = {}
    if args.vehicle_capacity is not None:
        capacity_kwargs["capacity"] = args.vehicle_capacity

    generator = CVRPGenerator(
        num_loc=num_loc_train,
        loc_sampler=loc_sampler,
        min_demand=args.min_demand,
        max_demand=args.max_demand,
        **capacity_kwargs,
    )
    env = CVRPEnv(generator)
    # Fixed test dataset for reproducibility
    torch.manual_seed(args.test_seed)
    fixed_test_dataset = env.dataset(args.test_data_size, phase="test")

    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=args.embedding_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_attn_heads,
        normalization=normalization, # 'batch', 'layer', None
        train_decode_type=args.train_decode_type,
        val_decode_type=args.val_decode_type,
        test_decode_type=args.val_decode_type,  # match validation decoding
        temperature=args.train_temperature,
        use_graph_context=False, # For POMO, helps prevent overfitting to training graph size
        )
    # Ensure decode types are explicitly set on the policy (constructor may set defaults)
    policy.val_decode_type = args.val_decode_type
    policy.test_decode_type = args.val_decode_type

    # ---------------- VALIDATION (X instances you pick) ----------------
    # Point this to wherever you saved the converted files.
    # Can be .npz (preferred) or .vrp.
    VAL_BASE = Path("cvrplib_instances/cvrplib_x_npz/instances")

    # Specify the instances you want by filename stem.
    # Example stems: "X-n101-k25", "X-n200-k36", ...
    VAL_INSTANCES = [
        #"X-n101-k25",
        #"X-n106-k14",
        "X-n110-k13", # Center, Random, Q=66, MinVehicles=13
        "X-n115-k10", # Center, Random, Q=169, MinVehicles=10
        #"X-n139-k10", # Center, Random, Q=106, MinVehicles=10
        "X-n153-k22", # Center, Cluster, Q=144, MinVehicles=22
        "X-n172-k51", # Center, Cluster, Q=161, MinVehicles=51
        "X-n120-k6", # Offset, Random, Q=21, MinVehicles=6
        "X-n129-k18", # Offset, Random, Q=39, MinVehicles=18
        #"X-n148-k46", # Offset, Random, Q=18, MinVehicles=46
        #"X-n143-k7", # Offset, Random, Q=1190, MinVehicles=7
        "X-n125-k30", # Offset, Cluster, Q=188, MinVehicles=30
        "X-n134-k13", # Offset, Cluster, Q=643, MinVehicles=13

        "X-n344-k43", # Center, Random, Q=61, MinVehicles=43
        "X-n351-k40", # Center, Random, Q=436
        # "X-n322-k28", # Center, Random, Q=868, MinVehicles=28
        "X-n393-k38", # Center, Cluster, Q=78, MinVehicles=38
        "X-n420-k130", # Center, Cluster, Q=18, MinVehicles=130
        "X-n376-k94", # Offset, Random, Q=4
        "X-n384-k52", # Offset, Random, Q=564
        # "X-n359-k29", # Offset, Random, Q=68
        # "X-n384-k52", # Offset, Random, Q=23, MinVehicles=15
        "X-n317-k53", # Offset, Cluster, Q=6, MinVehicles=53
        "X-n367-k17", # Offset, Cluster, Q=218
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
        policy=policy,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        train_data_size=args.train_data_size,
        test_data_size=args.test_data_size,
        val_data_size=0,               # unused when external val loader is provided
        dataloader_num_workers=4,
        val_tds=val_tds,               # <-- your X instances
        val_batch_size=1,              # safest for mixed sizes
        val_num_samples=args.val_num_samples,
        val_temperature=args.val_temperature,
        lr_reduce_factor=args.lr_reduce_factor,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_reduce_threshold=args.lr_reduce_threshold,
        num_starts=num_starts,         # Number of POMO starts during train/validation
        fixed_test_dataset=fixed_test_dataset,
    )

    # ---------------- callbacks/logger/trainer ----------------
    #baseline_cb = CVRPValBaselineCallback(max_batches=2)
    num_starts=num_starts,  # Number of POMO starts during test/validation
    
    # Create logger first so we can pass it to callbacks
    run_id = datetime.now().strftime("%y%m%d_%H%M")
    logger = TensorBoardLogger(
        save_dir=args.log_base_dir,
        name=exp_name,
        version=run_id,
    )
    print(f"[INFO] Logging to: {logger.log_dir}")

    # Prepare all hyperparameters including argparse arguments
    def _sanitize_hparams(hparams: dict):
        """Convert any non-primitive values to strings so TensorBoard HParams accepts them."""
        primals = (int, float, str, bool, type(None))
        return {k: (v if isinstance(v, primals) else str(v)) for k, v in hparams.items()}

    hparams = {
        # Model and Policy Architecture
        "embedding_dim": args.embedding_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_attn_heads": args.num_attn_heads,
        "normalization": args.normalization,
        
        # Optimizer Configuration
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        
        # POMO Configuration
        "pomo_num_starts": args.pomo_num_starts,
        "val_num_samples": args.val_num_samples,
        "val_temperature": args.val_temperature,
        "lr_reduce_factor": args.lr_reduce_factor,
        "lr_reduce_patience": args.lr_reduce_patience,
        "lr_reduce_threshold": args.lr_reduce_threshold,
        
        # Decode Types
        "train_decode_type": args.train_decode_type,
                                "pomo_num_starts": num_starts,
        "train_temperature": args.train_temperature,
        
        # Training Configuration
        "max_epochs": args.max_epochs,
        "num_loc_train": args.num_loc_train,
        "batch_size": args.batch_size,
        "train_data_size": args.train_data_size,
        "test_data_size": args.test_data_size,
        "test_seed": args.test_seed,
        "limit_train_batches": args.limit_train_batches,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "checkpoint_after_epoch": args.checkpoint_after_epoch,
        
        # Data Generation
        "loc_sampler": str(loc_sampler),
        "min_demand": args.min_demand,
        "max_demand": args.max_demand,
        "vehicle_capacity": args.vehicle_capacity,
        
        # Dataloader Configuration
        "val_batch_size": 1,
        "dataloader_num_workers": 4,
        "val_data_size": 0,
        
        # Logging
        "log_base_dir": args.log_base_dir,

        # Validation Instances
        "num_val_instances": len(VAL_INSTANCES),
        "val_instances": VAL_INSTANCES,
    }
    
    # Log to TensorBoard (include a dummy metric so HParams tab renders)
    logger.log_hyperparams(hparams, metrics={"hp/placeholder": 0.0})
    # Some TB frontends only show HParams when add_hparams is used; sanitize to primitives first
    try:
        logger.experiment.add_hparams(_sanitize_hparams(hparams), {"hp/placeholder": 0.0})
        print("[INFO] add_hparams logged successfully")
    except Exception as e:
        print(f"[WARN] add_hparams failed: {e}")
    
    # Save YAML file with all hyperparameters
    config_path = os.path.join(logger.log_dir, "config.yaml")
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {config_path}")

    # Now create callbacks with logger
    plot_metric_cb = CVRPMetricPlotCallback()
    train_plot_graph_cb = CVRPTrainGraphPlotCallback(env, num_examples=5)

    val_plot_graph_cb = CVRPLibGraphPlotCallback(
        env,
        instance_names=VAL_INSTANCES,
        sol_base_dir="cvrplib_instances/X/",  # folder with X-n101-k25.sol etc.
        decode_type="greedy",
        logger=logger,  # Pass logger for individual reward logging
        use_model_solutions=True,  # Use validation solutions if available
    )

    # Custom checkpoint callback that only saves after specified epoch
    class DelayedCheckpoint(ModelCheckpoint):
        def __init__(self, start_epoch=0, *args, **kwargs):
            kwargs.setdefault("save_on_train_epoch_end", False)
            super().__init__(*args, **kwargs)
            self.start_epoch = start_epoch

        def on_train_epoch_end(self, trainer, pl_module):
            # Skip train-end checkpointing; we handle it after validation
            return

        def on_validation_end(self, trainer, pl_module):
            # Only start saving once the threshold epoch is reached; otherwise defer to
            # the default ModelCheckpoint logic to handle best-k tracking.
            if trainer.current_epoch >= self.start_epoch:
                super().on_validation_end(trainer, pl_module)

    ckpt_cb = DelayedCheckpoint(
        start_epoch=args.checkpoint_after_epoch,
        dirpath=f"{logger.log_dir}/checkpoints",
        filename=exp_name + "-epoch{epoch:03d}-val{val/reward:.3f}",
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    callback_list = [ckpt_cb, 
                     #baseline_cb, 
                     train_plot_graph_cb,
                     val_plot_graph_cb, 
                     plot_metric_cb]
    
    trainer = RL4COTrainer(
        max_epochs=args.max_epochs,
        callbacks=callback_list,
        accelerator="cuda",
        precision=args.precision,
        logger=logger,
        log_every_n_steps=50,
        limit_train_batches=args.limit_train_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_model_summary=True,  # Enable model summary
        num_sanity_val_steps=0,      # Skip sanity validation to start training faster
        enable_progress_bar=True,
    )

    trainer.fit(model)
    print(f"\nRunning test on {args.test_data_size} random CVRP instances...")
    test_results = trainer.test(model, ckpt_path="best", verbose=True)
    print(f"Test results on {args.test_data_size} random CVRP instances: {test_results}")
    # Manually push test metrics to TensorBoard for visibility
    if logger is not None and hasattr(logger, "experiment") and test_results:
        try:
            step = trainer.global_step if hasattr(trainer, "global_step") else 0
            for k, v in test_results[0].items():
                logger.experiment.add_scalar(f"test/{k}", v, step)
            print("[INFO] Test metrics logged to TensorBoard.")
        except Exception as e:
            print(f"[WARN] Failed to log test metrics: {e}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
