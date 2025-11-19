# simple.py
import torch
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from lightning.pytorch.callbacks import EarlyStopping
from CustomTSPInitEmbedding import CustomTSPInitEmbedding
from PlotTSPCallback import PlotTSPCallback
from PlotMetricCallback import PlotMetricCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate generator and environment
generator = TSPGenerator(num_loc=50, loc_distribution="uniform")
env = TSPEnv(generator)

embed_dim = 256
policy = AttentionModelPolicy(
    env_name=env.name, 
    num_encoder_layers=6,
    embed_dim=embed_dim,)

'''
policy = AttentionModelPolicy(
    env_name="tsp",
    embed_dim=embed_dim,
    init_embedding=CustomTSPInitEmbedding(embed_dim, k_neighbors=5),
    num_encoder_layers=6
)
'''

# RL model – put dataloader workers here
model = POMO(
    env,
    policy,
    batch_size=64,
    optimizer_kwargs={"lr": 1e-4},
)

# Fixed plots
td_plot = env.reset(batch_size=[5]).to(device)
plot_graphs_cb = PlotTSPCallback(env, td_plot, subdir="val_plots")

plot_metric_cb = PlotMetricCallback(
    train_metric_key="train/reward",
    val_metric_key="val/reward",
    filename_prefix="reward_curves",
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
    callbacks=[plot_graphs_cb,plot_metric_cb, early_stop_cb],
)

trainer.fit(model,
            #ckpt_path="lightning_logs/version_10/checkpoints/epoch=29-step=46890.ckpt"
            )
