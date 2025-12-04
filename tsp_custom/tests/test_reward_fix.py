"""
Test to verify rewards are properly computed
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from tensordict import TensorDict
from tsp_custom.envs import CustomTSPEnv, CustomTSPGenerator
from tsp_custom.models import CustomPOMOPolicy

print("="*80)
print("Testing Reward Computation")
print("="*80)

# Create environment
generator = CustomTSPGenerator(num_loc=10)
env = CustomTSPEnv(generator=generator)

# Create a batch of 4 instances
batch_size = 4
locs = torch.rand(batch_size, 10, 2)
td_init = TensorDict({"locs": locs}, batch_size=[batch_size])

# Reset environment
td = env.reset(td_init)
print(f"\n✓ Environment reset: batch_size={batch_size}, num_loc=10")

# Create policy
policy = CustomPOMOPolicy(
    num_loc=10,
    embed_dim=64,
    num_heads=4,
    num_encoder_layers=2,
    delete_bias_start=-5.0,
    delete_bias_end=0.0,
    delete_bias_warmup_epochs=100,
)
print(f"✓ Policy created")

# Run a few steps
print("\n--- Running Episode ---")
step = 0
max_steps = 20
while not td["done"].all() and step < max_steps:
    # Policy forward
    out = policy(td, phase='train', decode_type='sampling')
    
    # Set action
    td.set("action", out["action"])
    
    # Step environment
    td = env.step(td)["next"]
    
    # Check if we have any rewards yet
    rewards = td["reward"]
    done_mask = td["done"]
    
    if done_mask.any():
        done_indices = done_mask.nonzero(as_tuple=True)[0]
        for idx in done_indices:
            if rewards[idx].item() != 0.0:
                print(f"  Step {step}: Episode {idx} finished with reward = {rewards[idx].item():.4f}")
    
    step += 1

print(f"\n--- Episode Complete (or max steps reached) ---")
print(f"Final step: {step}")
print(f"Episodes done: {td['done'].sum().item()} / {batch_size}")

# Check final rewards
final_rewards = td["reward"].squeeze()
print(f"\nFinal Rewards:")
for i in range(batch_size):
    status = "DONE" if td["done"][i] else "NOT DONE"
    print(f"  Episode {i}: {final_rewards[i].item():8.4f} ({status})")

# Statistics
non_zero_rewards = (final_rewards != 0.0).sum().item()
print(f"\n{'='*80}")
print(f"Summary:")
print(f"  Non-zero rewards: {non_zero_rewards} / {batch_size}")
print(f"  Mean reward (done episodes): {final_rewards[td['done']].mean().item():.4f}")
print(f"  Std reward (done episodes): {final_rewards[td['done']].std().item():.4f}")

if non_zero_rewards > 0:
    print(f"\n✓ SUCCESS: Rewards are being computed!")
else:
    print(f"\n✗ FAILURE: All rewards are still 0.0!")
    print(f"  This indicates the reward computation is not being called properly.")

print("="*80)
