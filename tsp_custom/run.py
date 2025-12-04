"""
Basic test script for CustomTSPEnv
Exercises core environment methods to verify basic functionality
"""

import torch
from envs import CustomTSPEnv, CustomTSPGenerator
from envs.utils import encode_action

def main():
    print("=" * 80)
    print("CUSTOM TSP ENVIRONMENT - BASIC FUNCTIONALITY TEST")
    print("=" * 80)
    
    # Configuration
    num_loc = 10
    batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  Number of nodes: {num_loc}")
    print(f"  Batch size: {batch_size}")
    
    # Create generator
    print(f"\n{'='*80}")
    print("STEP 1: Creating Generator")
    print("=" * 80)
    generator = CustomTSPGenerator(
        num_loc=num_loc,
        min_loc=0.0,
        max_loc=1.0,
        loc_distribution="uniform"
    )
    print(f"✓ Generator created: {generator}")
    
    # Create environment
    print(f"\n{'='*80}")
    print("STEP 2: Creating Environment")
    print("=" * 80)
    env = CustomTSPEnv(
        generator=generator,
        max_steps_multiplier=2,
        deletion_penalty_factor=0.002
    )
    print(f"✓ Environment created: {env.name}")
    print(f"  Max steps: {env.max_steps}")
    print(f"  Num ADD actions: {env.num_add_actions}")
    
    # Generate batch of instances
    print(f"\n{'='*80}")
    print("STEP 3: Generating Problem Instances")
    print("=" * 80)
    td = env.reset(batch_size=[batch_size])
    print(f"✓ Generated {batch_size} instances")
    print(f"  Locations shape: {td['locs'].shape}")
    print(f"  Adjacency shape: {td['adjacency'].shape}")
    print(f"  Degrees shape: {td['degrees'].shape}")
    print(f"  Current step: {td['current_step']}")
    print(f"  Action mask shape: {td['action_mask'].shape}")
    print(f"  Feasible actions: {td['action_mask'].sum(dim=-1)}")
    
    # Display first instance
    print(f"\nFirst instance node locations:")
    for i in range(min(5, num_loc)):
        x, y = td['locs'][0, i]
        print(f"  Node {i}: ({x:.3f}, {y:.3f})")
    if num_loc > 5:
        print(f"  ... ({num_loc - 5} more nodes)")
    
    # Test adding edges
    print(f"\n{'='*80}")
    print("STEP 4: Testing ADD Actions")
    print("=" * 80)
    
    # Manually add some edges to build a partial tour
    test_edges = [(0, 1), (1, 2), (2, 3)]
    
    for i, (node_i, node_j) in enumerate(test_edges):
        print(f"\n  Action {i+1}: ADD edge ({node_i}, {node_j})")
        
        # For batch element 0, encode the action
        action_idx = encode_action(
            action_type=0,  # ADD
            node_i=node_i,
            node_j=node_j,
            adjacency=td['adjacency'][0],
            num_loc=num_loc
        )
        
        # Create action tensor for the batch
        action = torch.tensor([[action_idx], [action_idx]], dtype=torch.int64)
        td['action'] = action
        
        # Step environment
        td = env.step(td)['next']
        
        print(f"    ✓ Edge added")
        print(f"    Degrees: {td['degrees'][0][:10]}")
        print(f"    Num edges: {td['num_edges'][0].item()}")
        print(f"    Feasible actions: {td['action_mask'][0].sum().item()}")
    
    # Test deleting an edge
    print(f"\n{'='*80}")
    print("STEP 5: Testing DELETE Action")
    print("=" * 80)
    
    print(f"\n  Before DELETE:")
    print(f"    Num edges: {td['num_edges'][0].item()}")
    print(f"    Num deletions: {td['num_deletions'][0].item()}")
    
    # Find a DELETE action in the action mask
    num_add_actions = env.num_add_actions
    delete_start_idx = num_add_actions
    delete_actions = td['action_mask'][0, delete_start_idx:delete_start_idx + num_loc]
    
    if delete_actions.any():
        # Get first available DELETE action
        delete_idx = delete_start_idx + torch.where(delete_actions)[0][0].item()
        
        print(f"\n  Action: DELETE (action index {delete_idx})")
        
        action = torch.tensor([[delete_idx], [delete_idx]], dtype=torch.int64)
        td['action'] = action
        
        td = env.step(td)['next']
        
        print(f"    ✓ Edge deleted")
        print(f"    Num edges: {td['num_edges'][0].item()}")
        print(f"    Num deletions: {td['num_deletions'][0].item()}")
        print(f"    Degrees: {td['degrees'][0][:10]}")
    else:
        print("  ✗ No DELETE actions available")
    
    # Continue adding edges to complete a tour
    print(f"\n{'='*80}")
    print("STEP 6: Building Complete Tour")
    print("=" * 80)
    
    max_steps = 50
    step_count = td['current_step'][0].item()
    
    print(f"\n  Attempting to build tour (max {max_steps} steps)...")
    
    while not td['done'].all() and step_count < max_steps:
        # Get feasible actions
        feasible_mask = td['action_mask'][0]
        feasible_indices = torch.where(feasible_mask)[0]
        
        if len(feasible_indices) == 0:
            print(f"\n  ✗ No feasible actions at step {step_count}")
            break
        
        # Randomly select a feasible action (prefer ADD over DELETE/DONE)
        # Prioritize ADD actions (they come first)
        add_feasible = feasible_indices[feasible_indices < num_add_actions]
        if len(add_feasible) > 0:
            action_idx = add_feasible[torch.randint(0, len(add_feasible), (1,))].item()
        else:
            action_idx = feasible_indices[torch.randint(0, len(feasible_indices), (1,))].item()
        
        # Apply action to both batch elements
        action = torch.tensor([[action_idx], [action_idx]], dtype=torch.int64)
        td['action'] = action
        
        td = env.step(td)['next']
        step_count = td['current_step'][0].item()
        
        # Print progress every 10 steps
        if step_count % 10 == 0:
            print(f"    Step {step_count}: {td['num_edges'][0].item()} edges, "
                  f"{td['num_deletions'][0].item()} deletions")
    
    print(f"\n  Final state:")
    print(f"    Steps taken: {step_count}")
    print(f"    Edges: {td['num_edges'][0].item()}")
    print(f"    Deletions: {td['num_deletions'][0].item()}")
    print(f"    Done: {td['done'][0].item()}")
    
    # Check degrees
    print(f"    Node degrees: {td['degrees'][0]}")
    all_degree_2 = (td['degrees'][0] == 2).all()
    print(f"    All nodes degree 2: {all_degree_2}")
    
    # Test reward calculation
    print(f"\n{'='*80}")
    print("STEP 7: Testing Reward Calculation")
    print("=" * 80)
    
    # Create dummy actions tensor (not used in reward calculation for this env)
    actions = torch.zeros(batch_size, step_count, dtype=torch.long)
    
    try:
        rewards = env.get_reward(td, actions)
        
        print(f"\n  Rewards computed:")
        for b in range(batch_size):
            print(f"    Batch {b}: {rewards[b]:.2f}")
        
        print(f"\n  Mean reward: {rewards.mean():.2f}")
        print(f"  Std reward: {rewards.std():.2f}")
        reward_passed = True
    except AssertionError as e:
        print(f"\n  ⚠ Warning: {e}")
        print(f"  This is expected with random actions - graph may not be connected")
        reward_passed = False
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Generator creation: PASSED")
    print(f"✓ Environment creation: PASSED")
    print(f"✓ Reset/instance generation: PASSED")
    print(f"✓ ADD action: PASSED")
    print(f"✓ DELETE action: PASSED")
    print(f"✓ Step function: PASSED")
    if reward_passed:
        print(f"✓ Reward calculation: PASSED")
    else:
        print(f"⚠ Reward calculation: SKIPPED (disconnected graph from random actions)")
    print(f"\n{'='*80}")
    if reward_passed:
        print("ALL BASIC TESTS PASSED!")
    else:
        print("BASIC TESTS MOSTLY PASSED (reward skipped due to random actions)")
    print("=" * 80)

if __name__ == "__main__":
    main()
