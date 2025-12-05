from pathlib import Path
import os
import torch
import vrplib
import argparse
import yaml
from datetime import datetime
import random
import math
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
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR

#from classes.CVRPValBaselineCallback import CVRPValBaselineCallback
#from classes.CVRPGraphPlotCallback import CVRPGraphPlotCallback
#from classes.CVRPLibGraphPlotCallback import CVRPLibGraphPlotCallback
from classes.CVRPMetricPlotCallback import CVRPMetricPlotCallback
#from classes.CVRPSamplerCluster import SamplerCluster
#from classes.CVRPSamplerMixed import SamplerMixed
from classes.CVRPTrainGraphPlotCallback import CVRPTrainGraphPlotCallback
from classes.CVRPLibValSamplerCallback import CVRPLibValSamplerCallback
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
    val_tds = []
    for idx, p in enumerate(val_paths):
        td = load_val_instance(p)
        td.set("instance_id", torch.tensor([idx], dtype=torch.long))
        td.set("instance_name", VAL_INSTANCES[idx])  # keep human-readable name alongside id
        val_tds.append(td)
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
    parser = argparse.ArgumentParser(description='CVRP Training Arguments')

    # ADMINISTRATIVE ###########################
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name, used as wandb project name and local log path')
    parser.add_argument('--combo_name', type=str, default=None,
                        help='Defines which factor combination from the experiment is running')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Datetime string to identify each run, last two are millis to make them unique.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='[combo_name]_[run_id]')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='./Logs/[exp_name]')
    parser.add_argument('--device_name', type=str, default=None,
                            help='Device selection for this run: e.g. "cpu", "mps", "cuda", "cuda:0").')
    parser.add_argument('--device_num', type=int, default=None,
                            help='Device number that is passed to the Pytorch Lightning trainer.')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Precision for training (default: 16-mixed to save memory)')

    # GENERAL EXECUTION ##########################
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of training epochs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--limit_train_batches', type=float, default=None,
                        help='Limit fraction of training batches per epoch (default: None, use all)')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Run validation every N epochs (default: 1). Set higher to reduce validation overhead.')
    parser.add_argument('--checkpoint_after_epoch', type=int, default=0,
                        help='Only save checkpoints after this epoch number (default: 0, save from start)')

    # MODEL AND POLICY ARCHITECTURE #################
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension for the attention model (default: 256)')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers (default: 6)')
    parser.add_argument('--num_attn_heads', type=int, default=16,
                        help='Number of attention heads (default: 16)')
    parser.add_argument('--normalization', type=str, default='batch',
                        choices=['batch', 'layer', 'instance', 'none'],
                        help='Normalization type for attention model (default: batch)')
    
    # OPTIMIZER #######################
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer (default: 1e-6)')
    parser.add_argument('--optimizer', choices=['Adam','RMSprop'], default='Adam',
                        help='Optimizer to use for this run (default: Adam). Options: Adam, RMSprop')
    
    # TRAINING DATA GENERATION #################
    parser.add_argument('--train_data_size', type=int, default=100_000,
                        help='Number of training instances per epoch (default: 100,000)')
    parser.add_argument('--train_num_locs', type=int, default=100,
                        help='Number of customer locations for training (default: 100)') ###################################### Num Locations
    parser.add_argument('--train_vehicle_capacity', type=float, default=100.0,
                        help='Override vehicle capacity; higher allows more customers per tour (default: 100.0)')
    parser.add_argument('--train_set_clusters', type=int, default=3,
                        help='Optional cluster count hint for training set sampler (if applicable)') ########################## Clusters
    parser.add_argument('--train_seed', type=str, default="7",
                        help='Seed for training/generator (int or "random"). Default: 7')
        
    # VALIDATION ###########################
    parser.add_argument('--val_num_samples', type=int, default=10_000,
                        help='Number of samples during validation (default: 10,000). Applies when validation uses sampling.')
    parser.add_argument('--val_temperature', type=float, default=1.0,
                        help='Sampling temperature during validation (default: 1.0)')
    
    args = parser.parse_args()

    # Resolve train seed (int or "random")
    if isinstance(args.train_seed, str) and args.train_seed.lower() == "random":
            args.train_seed = random.randint(1, 10**9)

    ##### ALL ARGUMENTS SHOULD BE FIXED BELOW THIS POINT #####################
        
    val_loader, VAL_INSTANCES = create_val_loader()

    #loc_sampler = Mix_Multi_Distributions()
    loc_sampler = Mix_Distribution(n_cluster=args.train_set_clusters)

    # CVRP Generator: https://github.com/ai4co/rl4co/blob/2def49fa2aea18ca66cb3625d8dbc34a14f63bf6/rl4co/envs/routing/cvrp/generator.py#L33
    # Distributions: https://github.com/ai4co/rl4co/blob/2def49fa2aea18ca66cb3625d8dbc34a14f63bf6/rl4co/envs/common/utils.py#L34
    generator = CVRPGenerator(
        num_loc=args.train_num_locs,
        capacity=args.train_vehicle_capacity,
        min_demand = 3,
        max_demand = 20,
        demand_distribution = "exponential",
        demand_rate = 5.0,
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

    # OneCycleLR needs an explicit step count; derive it from dataset size and epochs.
    steps_per_epoch = math.ceil(args.train_data_size / args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs

    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=args.embedding_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_attn_heads,
        normalization=args.normalization,
        )
    
    model = POMO(
        env,
        policy=policy,
        batch_size=args.batch_size,
        val_batch_size=1,
        optimizer=args.optimizer,
        optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        lr_scheduler="OneCycleLR",
        lr_scheduler_kwargs = {"max_lr": args.learning_rate, "total_steps": total_steps},
        train_data_size=args.train_data_size,
        )

    logger = WandbLogger(
        project=args.exp_name,
        name=args.run_name,
        save_dir=args.log_dir,
        mode="online",
        resume="never"
    )
    logger.experiment.define_metric("train/reward", summary="max")
    logger.experiment.define_metric("val/reward", summary="max")
    logger.experiment.define_metric("val/10k-reward", summary="max")
    logger.experiment.define_metric("val/10k-reward-small", summary="max")
    logger.experiment.define_metric("val/10k-reward-large", summary="max")

    # Prepare all hyperparameters including argparse arguments
    def _sanitize_hparams(hparams: dict):
        """Convert any non-primitive values to strings so TensorBoard HParams accepts them."""
        primals = (int, float, str, bool, type(None))
        return {k: (v if isinstance(v, primals) else str(v)) for k, v in hparams.items()}

    args_dict = vars(args)
    
    def _sanitize_args_dict(d):
        primals = (int, float, str, bool, type(None))
        return {k: (v if isinstance(v, primals) else str(v)) for k, v in d.items()}
    
    logger.log_hyperparams(_sanitize_args_dict(args_dict))
    
    # Save YAML file with all hyperparameters
    config_path = os.path.join(args.log_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {config_path}")
    
    hparams_dict = {
            "exp_embedding_dim": args.embedding_dim,
            "exp_num_encoder_layers": args.num_encoder_layers,
            "exp_num_attn_heads": args.num_attn_heads,
            "exp_weight_decay": args.weight_decay,
            "exp_train_vehicle_capacity": args.train_vehicle_capacity,
            "exp_train_num_locs": args.train_num_locs,
            "exp_train_set_clusters": args.train_set_clusters,
            "exp_optimizer": args.optimizer,
        }
    
    print("\n\n\nHYPERPARAMS TO LOG:", hparams_dict,"\n\n")
    # Log hyperparameters with a starter hp_metric so HParams tab is populated
    
    #hparam_metrics = {"train_reward": 0.0,"val_reward":0.0,"val_10k-reward":0.0, "best-val-10k-reward":0.0, "best-val-10k-step":0.0}
    logger.log_hyperparams(hparams_dict) #, metrics=hparam_metrics)

    # Now create callbacks with logger
    #plot_metric_cb = CVRPMetricPlotCallback()
    train_plot_graph_cb = CVRPTrainGraphPlotCallback(env, num_examples=5)

    '''
    val_plot_graph_cb = CVRPLibGraphPlotCallback(
        env,
        instance_names=VAL_INSTANCES,
        sol_base_dir="cvrplib_instances/X/",  # folder with X-n101-k25.sol etc.
        decode_type="greedy",
        logger=logger,  # Pass logger for individual reward logging
        use_model_solutions=True,  # Use validation solutions if available
    )
    '''
    val_sample_cb = CVRPLibValSamplerCallback(args.val_num_samples,args.val_temperature)
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val/10k-reward",
        mode="max",
        filename="best-{epoch:02d}",
        save_top_k=5,
        save_last=True,
        verbose=True,
        dirpath=os.path.join(args.log_dir, args.run_name, "checkpoints"),
    )
    early_stop_cb = EarlyStopping(
        monitor="val/10k-reward",
        mode="max",
        patience=8,
        verbose=True,
        strict=False,
    )
    
    callback_list = [ 
                     #rich_model_cb,
                     #baseline_cb,
                     train_plot_graph_cb,
                     #HParamsLoggerCallback(),
                     #val_plot_graph_cb, 
                     val_sample_cb,
                     early_stop_cb,
                     checkpoint_cb,
                     #plot_metric_cb,
                     #avg_reward_cb,
                     ]

    def resolve_accelerator_and_devices(device_name, device_num):
        """Translate CLI device hints into Lightning accelerator/devices."""
        if device_name:
            name = device_name.lower()
            if name == "cpu":
                return "cpu", 1
            if name == "mps":
                return "mps", 1
            if name.startswith("cuda") or name == "gpu":
                # e.g., "cuda:1" -> [1]; fallback to device_num or single GPU
                if ":" in device_name:
                    try:
                        idx = int(device_name.split(":")[1])
                        return "gpu", [idx]
                    except ValueError:
                        pass
                if device_num and device_num > 0:
                    return "gpu", device_num
                return "gpu", 1
        # No explicit device name: guard against invalid 0 value
        if device_num == 0:
            if torch.cuda.is_available():
                return "gpu", [0]
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps", 1
            return "cpu", 1
        return "auto", device_num

    accelerator, devices = resolve_accelerator_and_devices(args.device_name, args.device_num)
    print(f"Trainer hardware: accelerator={accelerator}, devices={devices}")

    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        callbacks=callback_list,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        logger=logger,
        log_every_n_steps=50,
        limit_train_batches=args.limit_train_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=0,
    )

    trainer = RL4COTrainer(**trainer_kwargs)

    print(VAL_INSTANCES)
    model.dataloader_names = VAL_INSTANCES
    
    trainer.fit(model, val_dataloaders=val_loader)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
