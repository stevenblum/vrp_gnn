from pathlib import Path
import os
import torch
import vrplib
import argparse
import yaml
from datetime import datetime
import random
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from types import MethodType

from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from rl4co.models import AttentionModelPolicy, AttentionModel, POMO
from rl4co.models.nn.graph.mpnn import MessagePassingEncoder
from rl4co.utils import RL4COTrainer
from rl4co.data.utils import load_npz_to_tensordict  # fast RL4CO loader :contentReference[oaicite:1]{index=1}
from rl4co.data.dataset import TensorDictDataset
from rl4co.envs.common.distribution_utils import Mix_Multi_Distributions, Mix_Distribution
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

from classes.CVRPValBaselineCallback import CVRPValBaselineCallback
from classes.CVRPGraphPlotCallback import CVRPGraphPlotCallback
from classes.CVRPLibGraphPlotCallback import CVRPLibGraphPlotCallback
from classes.CVRPMetricPlotCallback import CVRPMetricPlotCallback
from classes.CVRPSamplerCluster import SamplerCluster
from classes.CVRPSamplerMixed import SamplerMixed
from classes.CVRPTrainGraphPlotCallback import CVRPTrainGraphPlotCallback
from classes.CVRPLibHelpers import normalize_coord, vrp_to_td, batchify_td, load_val_instance
from torch.utils.data import Dataset
from classes.HParamsLoggerCallback import HParamsLoggerCallback

class ListTensorDictDataset(Dataset):
    """Simple Dataset that returns one TensorDict per index."""
    def __init__(self, tds):
        self.tds = tds

    def __len__(self):
        return len(self.tds)

    def __getitem__(self, idx):
        return self.tds[idx]

def collate_single(batch):
    # batch is a list of length = batch_size
    # with batch_size=1, batch = [td]; we just unwrap
    assert len(batch) == 1
    return batch[0]

def create_val_loader():

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

    from torch.utils.data import Dataset

    # val_tds: list[TensorDict], one per CVRPLib instance
    val_tds = [load_val_instance(p) for p in val_paths]
    print("Test (CVRPLib) set:", val_paths)

    val_dataset = ListTensorDictDataset(val_tds)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        num_workers=4,
        collate_fn=collate_single,
    )

    return val_loader, VAL_INSTANCES

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
    parser.add_argument('--train_seed', type=str, default="7",
                        help='Seed for training/generator (int or "random"). Default: 7')
    parser.add_argument('--num_train_locs', type=int, default=None,
                        help='Alias for num_loc_train (default: None)')
    parser.add_argument('--train_set_clusters', type=int, default=3,
                        help='Optional cluster count hint for training set sampler (if applicable)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name for logger subdir (default: derived from model config)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--train_data_size', type=int, default=2_000_000,
                        help='Number of training instances per epoch (default: 2,000,000)')
    parser.add_argument('--test_data_size', type=int, default=100,
                        help='Number of random test instances (default: 100)')
    parser.add_argument('--train_num_starts', type=int, default=None,
                        help='Number of POMO starts for validation (default: same as num_loc_train)')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of training epochs (default: 300)')
    parser.add_argument('--normalization', type=str, default='batch',
                        choices=['batch', 'layer', 'instance', 'none'],
                        help='Normalization type for attention model (default: batch)')
    parser.add_argument('--limit_train_batches', type=float, default=None,
                        help='Limit fraction of training batches per epoch (default: None, use all)')
    parser.add_argument('--log_base_dir', type=str, default='lightning_logs',
                        help='Base directory for logging (default: lightning_logs)')
    parser.add_argument('--train_decode_type', type=str, default='sampling',
                        choices=['sampling', 'greedy', 'multistart_sampling', 'multistart_greedy'],
                        help='Decode type for training (default: sampling; multistart_* will be normalized to sampling/greedy)')
    parser.add_argument('--train_temperature', type=float, default=1.0,
                        help='Sampling temperature for training (higher = more exploration) (default: 1.0)')
    parser.add_argument('--val_decode_type', type=str, default='sampling',
                        choices=['sampling', 'greedy', 'multistart_sampling', 'multistart_greedy','multistart_beam_search'],
                        help='Decode type for validation (POMO will prepend multistart_) (default: sampling; multistart_* allowed)')
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
    parser.add_argument('--device', type=str, default=None,
                        help='Device selection for this run: e.g. "cpu", "mps", "cuda", "cuda:0", or a numeric GPU index (0).')
    args = parser.parse_args()

    # Resolve train seed (int or "random")
    if isinstance(args.train_seed, str) and args.train_seed.lower() == "random":
        for _ in range(1):
            args.train_seed = random.randint(1, 10**9)
    else:
        args.train_seed = int(args.train_seed)
    torch.manual_seed(args.train_seed)
    random.seed(args.train_seed)
    try:
        import numpy as np
        np.random.seed(args.train_seed)
    except Exception:
        pass

    # Determine accelerator early in main so it's set before Trainer is constructed.
    # Priority: --device CLI arg > ACCELERATOR env var > auto-detect via torch
    acc = None
    # 1) CLI device argument
    if args.device:
        dev = str(args.device)
        # numeric index -> treat as CUDA index
        if dev.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = dev
            acc = "cuda"
            # expose that we selected a single device via env
        elif dev.startswith("cuda:"):
            idx = dev.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            acc = "cuda"
        elif dev in ("cuda", "cpu", "mps"):
            acc = dev
        else:
            # fallback: treat as accelerator name
            acc = dev
    # 2) environment override
    if acc is None:
        acc = os.environ.get("ACCELERATOR")

    # 3) auto-detect
    if not acc:
        if torch.cuda.is_available():
            acc = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            acc = "mps"
        else:
            acc = "cpu"

    print(f"Using accelerator: {acc}")

    # !!! Experiment Name, Pytorch Lightning Will Then Give the Version A Name !!!!
    exp_name = args.run_name if args.run_name else f"emb{args.embedding_dim}_enc{args.num_encoder_layers}_attn{args.num_attn_heads}"

    val_loader, VAL_INSTANCES = create_val_loader()

    #loc_sampler = Mix_Multi_Distributions()
    loc_sampler = Mix_Distribution(n_cluster=args.train_set_clusters)

    generator = CVRPGenerator(
        num_loc=args.num_loc_train,
        loc_sampler=loc_sampler,
    )
    env = CVRPEnv(generator)
    '''
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=args.embedding_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_attn_heads,
        normalization=args.normalization, # 'batch', 'layer', None
        train_decode_type=args.train_decode_type,
        val_decode_type=args.val_decode_type,
        val_num_samples=args.val_num_samples,
        temperature=args.train_temperature,
        use_graph_context=False, # For POMO, helps prevent overfitting to training graph size
        )
    
    
    model = POMO(
        env,
        policy=policy,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        train_data_size=args.train_data_size,
        num_starts=args.train_num_starts,         # Number of POMO starts during train/validation
        policy_kwargs={
            "val_decode_type": args.val_decode_type,
            "val_num_samples": args.val_num_samples,
            "val_temperature": args.val_temperature,
        }
    )
    

    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        train_data_size=args.train_data_size,
        policy_kwargs={
            "val_decode_type": args.val_decode_type,
            "val_num_samples": args.val_num_samples,
            "val_temperature": args.val_temperature,
        }
    )
    '''
    '''
    mpnn_encoder = MessagePassingEncoder(
        env_name=env.name,
        embed_dim=args.embedding_dim//2,
        num_nodes=args.num_loc_train,
        num_layers=args.num_encoder_layers//3,
    )
    '''
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=args.embedding_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_attn_heads,
        normalization=args.normalization,
        #train_decode_type=args.train_decode_type,
        #val_decode_type=args.val_decode_type,
        #test_decode_type=args.val_decode_type,
        #temperature=args.train_temperature,
        )
    

    model = POMO(
        env,
        policy=policy,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        train_data_size=args.train_data_size,
        )

    print("CHECKING POLICY SETUP:")
    print(f"  policy.train_decode_type: {model.policy.train_decode_type}")
    print(f"  policy.temperature: {model.policy.temperature}")
    print(f"  policy.val_decode_type: {model.policy.val_decode_type}")
    #print(f"  policy.val_num_samples: {model.policy.val_num_samples}")
    #print(f"  policy.val_temperature: {model.policy.val_temperature}")
    print(f"  policy.test_decode_type: {model.policy.test_decode_type}")

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
        "train_num_starts": args.train_num_starts,
        "val_num_samples": args.val_num_samples,
        "val_temperature": args.val_temperature,
        "lr_reduce_factor": args.lr_reduce_factor,
        "lr_reduce_patience": args.lr_reduce_patience,
        "lr_reduce_threshold": args.lr_reduce_threshold,
        
        # Decode Types
        "train_decode_type": args.train_decode_type,
        "train_num_starts": args.train_num_starts,
        "train_temperature": args.train_temperature,
        
        # Training Configuration
        "max_epochs": args.max_epochs,
        "num_loc_train": args.num_loc_train,
        "batch_size": args.batch_size,
        "train_data_size": args.train_data_size,
        "test_data_size": args.test_data_size,
        "test_seed": args.test_seed,
        "train_seed": args.train_seed,
        "train_set_clusters": args.train_set_clusters,
        "limit_train_batches": args.limit_train_batches,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "checkpoint_after_epoch": args.checkpoint_after_epoch,
        
        # Data Generation
        #"loc_sampler": str(loc_sampler),
        "min_demand": args.min_demand,
        "max_demand": args.max_demand,
        "vehicle_capacity": args.vehicle_capacity,
        
        # Dataloader Configuration
        "val_batch_size": 1,
        "dataloader_num_workers": 4,
        "val_data_size": 0,
        
        # Logging
        "log_base_dir": args.log_base_dir,

        # Test Instances (CVRPLib)
        "num_test_instances": len(VAL_INSTANCES),
        "test_instances": VAL_INSTANCES,
        # Device used for this run (CLI arg or detected)
        "device": (args.device if args.device is not None else acc),
    }
    
    # Save YAML file with all hyperparameters
    config_path = os.path.join(logger.log_dir, "config.yaml")
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {config_path}")
    

    hparams_log= {
            "exp_embedding_dim": args.embedding_dim,
            "exp_num_encoder_layers": args.num_encoder_layers,
            "exp_num_attn_heads": args.num_attn_heads,
            "exp_weight_decay": args.weight_decay,
            "exp_vehicle_capacity": args.vehicle_capacity,
            "exp_num_loc_train": args.num_loc_train,
            "exp_train_set_clusters": args.train_set_clusters,
        "device": (args.device if args.device is not None else acc),
        }
    
    print("\n\n\nHYPERPARAMS TO LOG:", hparams_log,"\n\n")
    # Log hyperparameters with a starter hp_metric so HParams tab is populated
    def _sanitize_hparams_log(d):
        primals = (int, float, str, bool, type(None))
        return {k: (v if isinstance(v, primals) else str(v)) for k, v in d.items()}
    logger.log_hyperparams(_sanitize_hparams_log(hparams_log), metrics={"hp_metric": 0.0})

    # Now create callbacks with logger
    plot_metric_cb = CVRPMetricPlotCallback()
    train_plot_graph_cb = CVRPTrainGraphPlotCallback(env, num_examples=5)

    '''    val_plot_graph_cb = CVRPLibGraphPlotCallback(
        env,
        instance_names=VAL_INSTANCES,
        sol_base_dir="cvrplib_instances/X/",  # folder with X-n101-k25.sol etc.
        decode_type="greedy",
        logger=logger,  # Pass logger for individual reward logging
        use_model_solutions=True,  # Use validation solutions if available
    )
    '''
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val/reward",
        mode="max",
        filename="best-{epoch:02d}-{val/reward:.4f}",
        save_top_k=5,
        save_last=True,
        verbose=True,
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
    )
    
    callback_list = [checkpoint_cb, 
                     #rich_model_cb,
                     #baseline_cb,
                     train_plot_graph_cb,
                     HParamsLoggerCallback(),
                     #val_plot_graph_cb, 
                     #plot_metric_cb,
                     #avg_reward_cb,
                     ]
    
    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        callbacks=callback_list,
        accelerator=acc,
        # If we're not on CUDA, disable mixed-precision/autocast defaults that target CUDA
        precision=(args.precision if acc == "cuda" else (
            "32" if ("16" in str(args.precision).lower() or "bf16" in str(args.precision).lower()) else args.precision
        )),
        logger=logger,
        log_every_n_steps=50,
        limit_train_batches=args.limit_train_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    # Notify if precision was adjusted for the chosen accelerator
    chosen_prec = trainer_kwargs.get("precision")
    if chosen_prec != args.precision:
        print(f"Adjusted precision from {args.precision} to {chosen_prec} because accelerator={acc}")

    # If using a hardware accelerator, request a single device
    if acc != "cpu":
        trainer_kwargs["devices"] = 1

    trainer = RL4COTrainer(**trainer_kwargs)

    print(VAL_INSTANCES)
    model.dataloader_names = VAL_INSTANCES
    
    trainer.fit(model, val_dataloaders=val_loader)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
