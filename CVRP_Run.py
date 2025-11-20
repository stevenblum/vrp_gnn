from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from CVRPValBaselineCallback import CVRPValBaselineCallback
from CVRPGraphPlotCallback import CVRPGraphPlotCallback
from CVRPMetricPlotCallback import CVRPMetricPlotCallback

# Instantiate generator and environment (random CVRP)
generator = CVRPGenerator(
    num_loc=30,            # number of customers (depot added automatically)
    loc_distribution="uniform",
    min_demand=1,
    max_demand=10,
    # capacity=None -> RL4CO will pick a capacity based on num_loc by default
)
env = CVRPEnv(generator)

# Create policy and RL model (same as your TSP setup)
policy = AttentionModelPolicy(env_name=env.name, num_encoder_layers=6)
model = POMO(
    env, 
    policy, batch_size=64, optimizer_kwargs={"lr": 1e-4},
    train_data_size=1_000_000,   # or whatever you like
    val_data_size=128,
)

# Add CVRP baseline callback
baseline_cb = CVRPValBaselineCallback(max_batches=2) 
plot_graph_cb = CVRPGraphPlotCallback(env, num_examples=5)
plot_metric_cb = CVRPMetricPlotCallback()

trainer = RL4COTrainer(max_epochs=300,
                        callbacks=[baseline_cb, plot_graph_cb, plot_metric_cb],)

trainer.fit(model)