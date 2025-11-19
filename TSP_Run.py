# simple.py
import torch
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from lightning.pytorch.callbacks import EarlyStopping
from TSPCustomInitEmbedding import CustomTSPInitEmbedding
from TSPGraphPlotCallback import PlotTSPCallback
from TSPMatricPlotCallback import PlotMetricCallback
from TSPValBaselineCallback import ValBaselineCallback
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    accelerator = "mps"
    print("!!! Using MPS accelerator - performance may be suboptimal !!!")
else:
    device = torch.device("cpu")
    accelerator = "cpu"
    print("!!! Using CPU - performance may be suboptimal !!!")

time.sleep(10)

# Instantiate generator and environment
generator = TSPGenerator(num_loc=50, loc_distribution="uniform")
env = TSPEnv(generator)

embed_dim = 512
'''
policy = AttentionModelPolicy(
    env_name=env.name, 
    num_encoder_layers=6,
    embed_dim=embed_dim,)
'''
policy = AttentionModelPolicy(
    env_name="tsp",
    num_encoder_layers=6,
    embed_dim=embed_dim,
    init_embedding=CustomTSPInitEmbedding(embed_dim, k_neighbors=5),  
)

# RL model – put dataloader workers here
model = POMO(
    env,
    policy,
    batch_size=64,
    optimizer_kwargs={"lr": 1e-4},
    train_data_size=10_000,   # or whatever you like
    val_data_size=128,
)

plot_graphs_cb = PlotTSPCallback(
    env,
    num_examples=5,
    subdir="val_plots",
    decode_type="greedy",
)

plot_metric_cb = PlotMetricCallback(
    train_metric_key="train/reward",
    val_metric_key="val/reward",
    filename_prefix="reward_curves",
)

val_baseline_cb = ValBaselineCallback(
    tours_attr="val_concorde_tours",
    costs_attr="val_concorde_costs",
    max_batches=None,  # or something like 4 to limit Concorde runtime
)

early_stop_cb = EarlyStopping(
    monitor="val/reward",   # metric name
    mode="max",             # we want to maximize reward
    patience=15,             # epochs with no improvement
    min_delta=0.0,          # minimum change to qualify as "improvement"
    verbose=True,
)

# Trainer
trainer = RL4COTrainer(
    max_epochs=300,
    accelerator="gpu",
    precision="16-mixed",
    callbacks=[plot_graphs_cb,plot_metric_cb, early_stop_cb,val_baseline_cb],
)

trainer.fit(model,
            #ckpt_path="lightning_logs/version_10/checkpoints/epoch=29-step=46890.ckpt"
            )
