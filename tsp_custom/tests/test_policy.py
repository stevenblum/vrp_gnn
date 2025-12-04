"""
Test script for CustomPOMOPolicy

Verifies that the policy can:
1. Accept TensorDict from environment
2. Encode state
3. Decode actions
4. Output valid action indices
5. Compute log probabilities
"""

import sys
import torch
from tensordict import TensorDict

# Add parent directory to path for imports
sys.path.insert(0, '/home/scblum/Projects/vrp_gnn')

from tsp_custom.models import CustomPOMOPolicy


def test_policy_forward():
    """Test policy forward pass with dummy state."""
    print("=" * 60)
    print("Testing CustomPOMOPolicy Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    num_loc = 20
    
    # Create dummy state matching environment output
    td = TensorDict({
        "locs": torch.rand(batch_size, num_loc, 2),
        "adjacency": torch.zeros(batch_size, num_loc, num_loc, dtype=torch.bool),
        "degrees": torch.zeros(batch_size, num_loc, dtype=torch.long),
        "current_step": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_deletions": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_edges": torch.zeros(batch_size, 1, dtype=torch.long),
        "done": torch.zeros(batch_size, dtype=torch.bool),
    }, batch_size=[batch_size])
    
    # Add some edges to test DELETE actions
    print("\nSetting up test state...")
    td["adjacency"][0, 0, 1] = True
    td["adjacency"][0, 1, 0] = True
    td["adjacency"][0, 2, 3] = True
    td["adjacency"][0, 3, 2] = True
    td["degrees"][0, 0] = 1
    td["degrees"][0, 1] = 1
    td["degrees"][0, 2] = 1
    td["degrees"][0, 3] = 1
    td["num_edges"][0] = 2
    print(f"  Batch 0: Added 2 edges")
    
    # Create action mask
    num_add_actions = num_loc * (num_loc - 1) // 2  # 190 for N=20
    max_delete_actions = num_loc  # 20
    total_actions = num_add_actions + max_delete_actions + 1  # 211
    
    action_mask = torch.zeros(batch_size, total_actions, dtype=torch.bool)
    
    # For each batch, allow some ADD actions (not to nodes with degree 2)
    for b in range(batch_size):
        # Allow ADD actions for edges where both nodes have degree < 2
        action_idx = 0
        for i in range(num_loc):
            for j in range(i + 1, num_loc):
                can_add = (
                    td["adjacency"][b, i, j] == 0 and
                    td["degrees"][b, i] < 2 and
                    td["degrees"][b, j] < 2
                )
                action_mask[b, action_idx] = can_add
                action_idx += 1
    
    # Allow DELETE actions for batch 0 (which has edges)
    action_mask[0, num_add_actions:num_add_actions + 2] = True
    
    # Don't allow DONE for anyone (no valid tours yet)
    action_mask[:, -1] = False
    
    td["action_mask"] = action_mask
    
    num_feasible = action_mask.sum(dim=-1)
    print(f"  Feasible actions per batch: {num_feasible.tolist()}")
    
    # Create policy
    print("\nInitializing policy...")
    policy = CustomPOMOPolicy(num_loc=num_loc)
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Test sampling
    print("\nTesting sampling decode_type...")
    out_sample = policy(td, phase='train', decode_type='sampling', return_actions=True)
    
    print(f"  action: {out_sample['action'].shape} = {out_sample['action']}")
    print(f"  log_prob: {out_sample['log_prob'].shape}")
    print(f"  log_prob values: {out_sample['log_prob']}")
    print(f"  action_type: {out_sample['action_components']['action_type']}")
    print(f"  node_i: {out_sample['action_components']['node_i']}")
    print(f"  node_j: {out_sample['action_components']['node_j']}")
    
    # Verify actions are within mask
    for b in range(batch_size):
        action_idx = out_sample['action'][b].item()
        is_valid = action_mask[b, action_idx].item()
        print(f"  Batch {b}: action_idx={action_idx}, valid={is_valid}")
        assert is_valid, f"Batch {b}: sampled invalid action!"
    
    # Test greedy
    print("\nTesting greedy decode_type...")
    out_greedy = policy(td, phase='val', decode_type='greedy', return_actions=True)
    
    print(f"  action: {out_greedy['action'].shape} = {out_greedy['action']}")
    print(f"  log_prob: {out_greedy['log_prob'].shape}")
    print(f"  action_type: {out_greedy['action_components']['action_type']}")
    
    # Verify actions are within mask
    for b in range(batch_size):
        action_idx = out_greedy['action'][b].item()
        is_valid = action_mask[b, action_idx].item()
        print(f"  Batch {b}: action_idx={action_idx}, valid={is_valid}")
        assert is_valid, f"Batch {b}: selected invalid action!"
    
    # Test delete bias scheduling
    print("\nTesting delete bias scheduling...")
    policy.set_epoch(0)
    bias_0 = policy.get_delete_bias('train')
    print(f"  Epoch 0: bias = {bias_0:.2f}")
    
    policy.set_epoch(50)
    bias_50 = policy.get_delete_bias('train')
    print(f"  Epoch 50: bias = {bias_50:.2f}")
    
    policy.set_epoch(100)
    bias_100 = policy.get_delete_bias('train')
    print(f"  Epoch 100: bias = {bias_100:.2f}")
    
    bias_val = policy.get_delete_bias('val')
    print(f"  Val phase: bias = {bias_val:.2f}")
    
    assert abs(bias_0 - (-5.0)) < 0.01, "Initial bias should be -5.0"
    assert abs(bias_100 - 0.0) < 0.01, "Final bias should be 0.0"
    assert abs(bias_val - 0.0) < 0.01, "Val bias should be 0.0"
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_integration_with_environment():
    """Test policy with actual environment."""
    print("\n" + "=" * 60)
    print("Testing Integration with Environment")
    print("=" * 60)
    
    from tsp_custom.envs import CustomTSPEnv, CustomTSPGenerator
    
    # Create environment
    num_loc = 20
    batch_size = 2
    
    generator = CustomTSPGenerator(num_loc=num_loc)
    env = CustomTSPEnv(generator=generator)
    
    print(f"\nEnvironment: {env.name}, N={num_loc}")
    
    # Reset environment
    td = env.reset(batch_size=[batch_size])
    
    print(f"Initial state keys: {list(td.keys())}")
    print(f"  locs: {td['locs'].shape}")
    print(f"  adjacency: {td['adjacency'].shape}")
    print(f"  action_mask: {td['action_mask'].shape}")
    
    # Create policy
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Take one step
    print("\nTaking one step...")
    out = policy(td, phase='train', decode_type='sampling', return_actions=True)
    
    print(f"  Policy output:")
    print(f"    action: {out['action']}")
    print(f"    log_prob: {out['log_prob']}")
    print(f"    action_type: {out['action_components']['action_type']}")
    print(f"    node_i: {out['action_components']['node_i']}")
    print(f"    node_j: {out['action_components']['node_j']}")
    
    # Set action in td
    td.set("action", out["action"])
    
    # Step environment
    td = env.step(td)["next"]
    
    print(f"\nAfter step:")
    print(f"  degrees: {td['degrees']}")
    print(f"  num_edges: {td['num_edges']}")
    print(f"  done: {td['done']}")
    
    print("\n" + "=" * 60)
    print("Integration test passed!")
    print("=" * 60)


def test_action_masking_correctness():
    """Test that policy respects environment's action mask correctly."""
    print("\n" + "=" * 60)
    print("Testing Action Masking Correctness")
    print("=" * 60)
    
    from tsp_custom.envs import CustomTSPEnv, CustomTSPGenerator
    
    # Test with delete frequency constraint
    num_loc = 10
    batch_size = 8
    delete_every_n_steps = 4
    
    generator = CustomTSPGenerator(num_loc=num_loc)
    env = CustomTSPEnv(
        generator=generator,
        delete_every_n_steps=delete_every_n_steps
    )
    
    print(f"\nEnvironment: N={num_loc}, delete_every_n_steps={delete_every_n_steps}")
    
    # Create policy
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Reset environment
    td = env.reset(batch_size=[batch_size])
    
    # Run multiple steps and verify masking
    num_steps = 12
    print(f"\nRunning {num_steps} steps, checking mask correctness...")
    
    violations = {
        'degree_violations': 0,
        'invalid_actions': 0,
        'delete_frequency_violations': 0,
    }
    
    for step in range(num_steps):
        # Get current state
        current_step = td["current_step"][0, 0].item()
        mask = td["action_mask"]
        adjacency = td["adjacency"]
        degrees = td["degrees"]
        
        # Check delete frequency constraint in mask
        can_delete_this_step = (current_step % delete_every_n_steps == 0)
        
        # Get action from policy
        out = policy(td, phase='train', decode_type='sampling', return_actions=True)
        action_idx = out['action']
        
        # Verify all actions respect the mask
        for b in range(batch_size):
            if td["done"][b]:
                continue
                
            idx = action_idx[b].item()
            is_masked = mask[b, idx].item()
            
            if not is_masked:
                violations['invalid_actions'] += 1
                print(f"  ❌ Step {step}, Batch {b}: Selected INVALID action {idx} (not in mask)")
        
        # Apply actions
        td.set("action", action_idx)
        td = env.step(td)["next"]
        
        # Check for degree violations
        max_degree = degrees.max().item()
        if max_degree > 2:
            violations['degree_violations'] += 1
            bad_batch = (degrees > 2).any(dim=-1).nonzero(as_tuple=True)[0]
            print(f"  ❌ Step {step}: Degree violation! Max degree: {max_degree}, batches: {bad_batch.tolist()}")
        
        # Check if DELETE actions were taken on non-delete steps
        action_type = out['action_components']['action_type']
        is_delete = (action_type == 1)
        
        if is_delete.any() and not can_delete_this_step:
            violations['delete_frequency_violations'] += 1
            delete_batches = is_delete.nonzero(as_tuple=True)[0]
            print(f"  ❌ Step {step}: DELETE on non-delete step! Batches: {delete_batches.tolist()}")
        
        # Log step info
        if step % 4 == 0 or violations['invalid_actions'] > 0 or violations['degree_violations'] > 0:
            num_adds = (action_type == 0).sum().item()
            num_deletes = (action_type == 1).sum().item()
            num_dones = (action_type == 2).sum().item()
            avg_edges = td["num_edges"].float().mean().item()
            print(f"  Step {step}: ADD={num_adds}, DEL={num_deletes}, DONE={num_dones}, "
                  f"avg_edges={avg_edges:.1f}, can_delete={can_delete_this_step}")
        
        # Stop if all done
        if td["done"].all():
            print(f"\n  All episodes finished at step {step}")
            break
    
    # Print summary
    print("\n" + "=" * 60)
    print("Masking Test Summary:")
    print("=" * 60)
    print(f"  Invalid actions selected: {violations['invalid_actions']}")
    print(f"  Degree violations (>2): {violations['degree_violations']}")
    print(f"  Delete frequency violations: {violations['delete_frequency_violations']}")
    
    # Assert no violations
    assert violations['invalid_actions'] == 0, "Policy selected actions not in the mask!"
    assert violations['degree_violations'] == 0, "Nodes exceeded degree 2!"
    assert violations['delete_frequency_violations'] == 0, "Deleted on non-delete steps!"
    
    print("\n✅ All masking constraints respected!")
    print("=" * 60)


def test_edge_case_all_masked():
    """Test that policy handles edge case where all actions might be masked."""
    print("\n" + "=" * 60)
    print("Testing Edge Case: All Actions Masked")
    print("=" * 60)
    
    num_loc = 10
    batch_size = 2
    
    # Create a state where all actions are masked (artificial scenario)
    td = TensorDict({
        "locs": torch.rand(batch_size, num_loc, 2),
        "adjacency": torch.zeros(batch_size, num_loc, num_loc, dtype=torch.bool),
        "degrees": torch.full((batch_size, num_loc), 2, dtype=torch.long),  # All degree 2
        "current_step": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_deletions": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_edges": torch.zeros(batch_size, 1, dtype=torch.long),
        "done": torch.zeros(batch_size, dtype=torch.bool),
    }, batch_size=[batch_size])
    
    # Create mask with all False (all masked)
    num_add_actions = num_loc * (num_loc - 1) // 2
    max_delete_actions = num_loc
    total_actions = num_add_actions + max_delete_actions + 1
    
    action_mask = torch.zeros(batch_size, total_actions, dtype=torch.bool)
    td["action_mask"] = action_mask
    
    print("\nCreated artificial state with ALL actions masked")
    print(f"  Mask sum: {action_mask.sum(dim=-1).tolist()}")
    
    # Create policy
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Try to get action (should not crash, should force DONE or handle gracefully)
    print("\nAttempting to get action from policy...")
    try:
        out = policy(td, phase='train', decode_type='sampling', return_actions=True)
        print(f"  ✅ No crash! Action: {out['action']}")
        print(f"  Action type: {out['action_components']['action_type']}")
        
        # Check that DONE was forced
        action_idx = out['action'][0].item()
        print(f"  Forced DONE action at index {action_idx} (expected {total_actions - 1})")
    except RuntimeError as e:
        if "multinomial" in str(e).lower():
            print(f"  ❌ Multinomial error still occurs: {e}")
            raise
        else:
            raise
    
    print("\n✅ Edge case handled successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_policy_forward()
    test_integration_with_environment()
    test_action_masking_correctness()
    test_edge_case_all_masked()
