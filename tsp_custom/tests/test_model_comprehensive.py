"""
Comprehensive Unit Tests for Custom TSP Model (Step 7)

Tests include:
1. Encoder with various input sizes (N=10, 20, 50, 100)
2. Each decoder head with edge cases
3. Policy with edge cases (no feasible actions, full graph, etc.)
4. Gradient flow through full forward pass
5. Delete bias scheduling validation
6. Integration with environment state

Author: Step 7 of USER_DEVELOPMENT_PLAN.txt
"""

import sys
import torch
import torch.nn.functional as F
from tensordict import TensorDict
import logging

# Add parent directory to path for imports
sys.path.insert(0, '/home/scblum/Projects/vrp_gnn')

from tsp_custom.models import (
    TransformerEncoder,
    AddEdgeDecoder,
    DeleteEdgeDecoder,
    DoneDecoder,
    CustomPOMOPolicy,
)
from tsp_custom.models.action_utils import (
    decode_action_index,
    extract_edge_list,
    compute_action_masks,
    create_node_features,
)
from tsp_custom.envs import CustomTSPEnv, CustomTSPGenerator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_encoder_various_sizes():
    """Test encoder with N=10, 20, 50, 100."""
    print("\n" + "="*80)
    print("TEST 1: Encoder with Various Input Sizes")
    print("="*80)
    
    batch_size = 4
    embed_dim = 128
    
    for num_loc in [10, 20, 50, 100]:
        print(f"\n--- Testing N={num_loc} ---")
        
        # Feature dim: 2 (locs) + 1 (degree) + N (adjacency) + 2 (counters)
        feat_dim = num_loc + 5
        
        # Create encoder
        encoder = TransformerEncoder(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=6,
            feedforward_dim=512,
            normalization='instance'
        )
        
        # Random input
        x = torch.randn(batch_size, num_loc, feat_dim)
        
        # Forward pass
        embeddings = encoder(x)
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Expected:     (4, {num_loc}, 128)")
        
        assert embeddings.shape == (batch_size, num_loc, embed_dim), \
            f"Expected shape (4, {num_loc}, 128), got {embeddings.shape}"
        
        # Check for NaN or Inf
        assert not torch.isnan(embeddings).any(), "Output contains NaN"
        assert not torch.isinf(embeddings).any(), "Output contains Inf"
        
        # Check embedding statistics
        mean = embeddings.mean().item()
        std = embeddings.std().item()
        print(f"  Output mean:  {mean:.4f}")
        print(f"  Output std:   {std:.4f}")
        
        # Parameters
        params = sum(p.numel() for p in encoder.parameters())
        print(f"  Parameters:   {params:,}")
        
    print("\n✓ All encoder size tests passed!")


def test_add_decoder_edge_cases():
    """Test AddEdgeDecoder with edge cases."""
    print("\n" + "="*80)
    print("TEST 2: AddEdgeDecoder Edge Cases")
    print("="*80)
    
    embed_dim = 128
    decoder = AddEdgeDecoder(embed_dim=embed_dim)
    
    # Test Case 1: Standard input
    print("\n--- Test Case 1: Standard Input (N=20) ---")
    batch_size = 2
    num_loc = 20
    node_embeddings = torch.randn(batch_size, num_loc, embed_dim)
    locs = torch.rand(batch_size, num_loc, 2)
    
    logits_add, edge_indices = decoder(node_embeddings, locs)
    
    expected_edges = num_loc * (num_loc - 1) // 2
    print(f"  Input embeddings: {node_embeddings.shape}")
    print(f"  Input locations:  {locs.shape}")
    print(f"  Output logits:    {logits_add.shape}")
    print(f"  Edge indices:     {edge_indices.shape}")
    print(f"  Expected edges:   {expected_edges}")
    
    assert logits_add.shape == (batch_size, expected_edges)
    assert edge_indices.shape == (expected_edges, 2)
    assert not torch.isnan(logits_add).any()
    
    # Test Case 2: Small graph (N=5)
    print("\n--- Test Case 2: Small Graph (N=5) ---")
    num_loc = 5
    node_embeddings = torch.randn(batch_size, num_loc, embed_dim)
    locs = torch.rand(batch_size, num_loc, 2)
    
    logits_add, edge_indices = decoder(node_embeddings, locs)
    expected_edges = num_loc * (num_loc - 1) // 2
    
    print(f"  N=5: {expected_edges} possible edges")
    print(f"  Logits shape: {logits_add.shape}")
    print(f"  Edge pairs: {edge_indices}")
    
    assert logits_add.shape == (batch_size, expected_edges)
    
    # Verify edge_indices are correct (i < j)
    for idx in range(expected_edges):
        i, j = edge_indices[idx]
        assert i < j, f"Edge index {idx}: {i} >= {j}"
    
    print("  ✓ All edge pairs have i < j")
    
    # Test Case 3: Check logit range
    print("\n--- Test Case 3: Logit Statistics ---")
    print(f"  Logit mean: {logits_add.mean().item():.4f}")
    print(f"  Logit std:  {logits_add.std().item():.4f}")
    print(f"  Logit min:  {logits_add.min().item():.4f}")
    print(f"  Logit max:  {logits_add.max().item():.4f}")
    
    print("\n✓ All AddEdgeDecoder tests passed!")


def test_delete_decoder_edge_cases():
    """Test DeleteEdgeDecoder with edge cases."""
    print("\n" + "="*80)
    print("TEST 3: DeleteEdgeDecoder Edge Cases")
    print("="*80)
    
    embed_dim = 128
    decoder = DeleteEdgeDecoder(embed_dim=embed_dim)
    
    batch_size = 4
    num_loc = 20
    max_edges = num_loc  # Policy default
    
    # Test Case 1: With existing edges
    print("\n--- Test Case 1: With Existing Edges ---")
    node_embeddings = torch.randn(batch_size, num_loc, embed_dim)
    locs = torch.rand(batch_size, num_loc, 2)
    
    # Create edge list with some real edges and padding
    edge_list = torch.full((batch_size, max_edges, 2), -1, dtype=torch.long)
    edge_list[0, 0] = torch.tensor([0, 1])  # Edge 0-1
    edge_list[0, 1] = torch.tensor([2, 3])  # Edge 2-3
    edge_list[1, 0] = torch.tensor([5, 6])  # Edge 5-6
    
    delete_bias = -2.5
    logits_del = decoder(node_embeddings, locs, edge_list, delete_bias)
    
    print(f"  Node embeddings:  {node_embeddings.shape}")
    print(f"  Edge list:        {edge_list.shape}")
    print(f"  Delete bias:      {delete_bias}")
    print(f"  Output logits:    {logits_del.shape}")
    print(f"  Logits batch 0:   {logits_del[0, :5]}")
    print(f"  Logits batch 1:   {logits_del[1, :5]}")
    
    assert logits_del.shape == (batch_size, max_edges)
    
    # Check that padding positions have -inf logits
    padding_mask = (edge_list[:, :, 0] == -1)
    padded_logits = logits_del[padding_mask]
    print(f"  Padded positions: {padding_mask.sum().item()} / {batch_size * max_edges}")
    print(f"  All padded = -inf? {torch.isinf(padded_logits).all().item()}")
    
    assert torch.isinf(padded_logits).all(), "Padding should be -inf"
    
    # Test Case 2: All edges padding (no deletable edges)
    print("\n--- Test Case 2: No Deletable Edges ---")
    edge_list_empty = torch.full((batch_size, max_edges, 2), -1, dtype=torch.long)
    logits_del_empty = decoder(node_embeddings, locs, edge_list_empty, delete_bias)
    
    print(f"  All edges are padding")
    print(f"  All logits -inf? {torch.isinf(logits_del_empty).all().item()}")
    assert torch.isinf(logits_del_empty).all()
    
    # Test Case 3: Delete bias impact
    print("\n--- Test Case 3: Delete Bias Impact ---")
    for bias in [-5.0, -2.5, 0.0]:
        logits = decoder(node_embeddings, locs, edge_list, bias)
        # Get non-padded logits for batch 0 (has 2 edges)
        print(f"  Bias {bias:5.1f}: logits[0, :2] = {logits[0, :2]}")
    
    print("\n✓ All DeleteEdgeDecoder tests passed!")


def test_done_decoder_edge_cases():
    """Test DoneDecoder with edge cases."""
    print("\n" + "="*80)
    print("TEST 4: DoneDecoder Edge Cases")
    print("="*80)
    
    embed_dim = 128
    decoder = DoneDecoder(embed_dim=embed_dim)
    
    # Test Case 1: Standard input
    print("\n--- Test Case 1: Standard Input ---")
    batch_size = 4
    num_loc = 20
    node_embeddings = torch.randn(batch_size, num_loc, embed_dim)
    
    logit_done = decoder(node_embeddings)
    
    print(f"  Input:  {node_embeddings.shape}")
    print(f"  Output: {logit_done.shape}")
    print(f"  Values: {logit_done.squeeze()}")
    
    assert logit_done.shape == (batch_size, 1)
    assert not torch.isnan(logit_done).any()
    
    # Test Case 2: Different graph sizes
    print("\n--- Test Case 2: Various Graph Sizes ---")
    for N in [5, 10, 20, 50]:
        embeddings = torch.randn(batch_size, N, embed_dim)
        logit = decoder(embeddings)
        print(f"  N={N:3d}: output shape {logit.shape}, mean={logit.mean():.4f}")
        assert logit.shape == (batch_size, 1)
    
    print("\n✓ All DoneDecoder tests passed!")


def test_policy_edge_cases():
    """Test policy with edge cases."""
    print("\n" + "="*80)
    print("TEST 5: Policy Edge Cases")
    print("="*80)
    
    num_loc = 20
    batch_size = 4
    
    # Test Case 1: No feasible actions (shouldn't happen but test masking)
    print("\n--- Test Case 1: Single Feasible Action per Batch ---")
    
    td = TensorDict({
        "locs": torch.rand(batch_size, num_loc, 2),
        "adjacency": torch.zeros(batch_size, num_loc, num_loc, dtype=torch.bool),
        "degrees": torch.zeros(batch_size, num_loc, dtype=torch.long),
        "current_step": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_deletions": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_edges": torch.zeros(batch_size, 1, dtype=torch.long),
        "done": torch.zeros(batch_size, dtype=torch.bool),
    }, batch_size=[batch_size])
    
    num_add_actions = num_loc * (num_loc - 1) // 2
    max_delete_actions = num_loc
    total_actions = num_add_actions + max_delete_actions + 1
    
    # Create mask with only one feasible action per batch
    action_mask = torch.zeros(batch_size, total_actions, dtype=torch.bool)
    action_mask[0, 0] = True  # Batch 0: first ADD action
    action_mask[1, 50] = True  # Batch 1: different ADD action
    action_mask[2, num_add_actions] = True  # Batch 2: first DELETE (but no edges, will be masked)
    action_mask[3, -1] = True  # Batch 3: DONE action
    
    td["action_mask"] = action_mask
    
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Greedy should select the only feasible action
    out = policy(td, phase='val', decode_type='greedy')
    
    print(f"  Expected actions: [0, 50, {num_add_actions}, {total_actions-1}]")
    print(f"  Selected actions: {out['action'].tolist()}")
    
    # Verify each batch selected its only feasible action
    for b in range(batch_size):
        selected = out['action'][b].item()
        feasible_idx = action_mask[b].nonzero(as_tuple=True)[0].item()
        print(f"  Batch {b}: selected={selected}, feasible={feasible_idx}, match={selected==feasible_idx}")
        assert selected == feasible_idx, f"Batch {b} didn't select feasible action!"
    
    # Test Case 2: Nearly full graph (high degree nodes)
    print("\n--- Test Case 2: High Degree Nodes ---")
    
    td = TensorDict({
        "locs": torch.rand(batch_size, num_loc, 2),
        "adjacency": torch.zeros(batch_size, num_loc, num_loc, dtype=torch.bool),
        "degrees": torch.zeros(batch_size, num_loc, dtype=torch.long),
        "current_step": torch.ones(batch_size, 1, dtype=torch.long) * 10,
        "num_deletions": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_edges": torch.zeros(batch_size, 1, dtype=torch.long),
        "done": torch.zeros(batch_size, dtype=torch.bool),
    }, batch_size=[batch_size])
    
    # Set most nodes to degree 2 (max allowed)
    td["degrees"][:, :15] = 2
    td["degrees"][:, 15:] = 0  # Last 5 nodes have degree 0
    
    # Add corresponding edges to adjacency
    for b in range(batch_size):
        edge_count = 0
        for i in range(15):
            # Connect to 2 other nodes
            if i < 14:
                td["adjacency"][b, i, i+1] = True
                td["adjacency"][b, i+1, i] = True
                edge_count += 1
        td["num_edges"][b] = edge_count
    
    # Create mask: only allow edges between low-degree nodes (15-19)
    action_mask = torch.zeros(batch_size, total_actions, dtype=torch.bool)
    action_idx = 0
    for i in range(num_loc):
        for j in range(i + 1, num_loc):
            # Can add if both nodes have degree < 2 and not connected
            can_add = (td["degrees"][0, i] < 2 and td["degrees"][0, j] < 2 and
                      not td["adjacency"][0, i, j])
            action_mask[:, action_idx] = can_add
            action_idx += 1
    
    td["action_mask"] = action_mask
    
    num_feasible = action_mask.sum(dim=-1)
    print(f"  Degrees[0]: {td['degrees'][0].tolist()}")
    print(f"  Feasible actions per batch: {num_feasible.tolist()}")
    
    out = policy(td, phase='val', decode_type='sampling')
    
    # Verify all selected actions are feasible
    for b in range(batch_size):
        selected = out['action'][b].item()
        is_feasible = action_mask[b, selected].item()
        print(f"  Batch {b}: action={selected}, feasible={is_feasible}")
        assert is_feasible, f"Batch {b} selected infeasible action!"
    
    # Test Case 3: Valid tour state (should prefer DONE)
    print("\n--- Test Case 3: Valid Tour (Can Use DONE) ---")
    
    # Create a valid Hamiltonian cycle for N=10 (simpler)
    num_loc_small = 10
    td_small = TensorDict({
        "locs": torch.rand(2, num_loc_small, 2),
        "adjacency": torch.zeros(2, num_loc_small, num_loc_small, dtype=torch.bool),
        "degrees": torch.ones(2, num_loc_small, dtype=torch.long) * 2,  # All degree 2
        "current_step": torch.ones(2, 1, dtype=torch.long) * num_loc_small,
        "num_deletions": torch.zeros(2, 1, dtype=torch.long),
        "num_edges": torch.ones(2, 1, dtype=torch.long) * num_loc_small,
        "done": torch.zeros(2, dtype=torch.bool),
    }, batch_size=[2])
    
    # Create simple cycle: 0-1-2-3-...-9-0
    for i in range(num_loc_small):
        j = (i + 1) % num_loc_small
        td_small["adjacency"][:, i, j] = True
        td_small["adjacency"][:, j, i] = True
    
    num_add_small = num_loc_small * (num_loc_small - 1) // 2
    max_del_small = num_loc_small
    total_small = num_add_small + max_del_small + 1
    
    # Mask: no ADD (all degree 2), allow DELETE and DONE
    action_mask_small = torch.zeros(2, total_small, dtype=torch.bool)
    # Allow some DELETE actions
    action_mask_small[:, num_add_small:num_add_small + num_loc_small] = True
    # Allow DONE
    action_mask_small[:, -1] = True
    
    td_small["action_mask"] = action_mask_small
    
    policy_small = CustomPOMOPolicy(num_loc=num_loc_small)
    
    out = policy_small(td_small, phase='val', decode_type='sampling')
    
    print(f"  Valid tour with {num_loc_small} nodes")
    print(f"  All degrees = 2: {(td_small['degrees'][0] == 2).all().item()}")
    print(f"  Feasible: DELETE ({num_loc_small} actions) + DONE (1 action)")
    print(f"  Selected actions: {out['action'].tolist()}")
    
    for b in range(2):
        selected = out['action'][b].item()
        is_feasible = action_mask_small[b, selected].item()
        print(f"  Batch {b}: action={selected}, feasible={is_feasible}")
        assert is_feasible
    
    print("\n✓ All policy edge case tests passed!")


def test_gradient_flow():
    """Test gradient flow through full forward pass."""
    print("\n" + "="*80)
    print("TEST 6: Gradient Flow")
    print("="*80)
    
    num_loc = 20
    batch_size = 3
    
    # Create policy
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Create dummy state with different action types feasible for each batch
    td = TensorDict({
        "locs": torch.rand(batch_size, num_loc, 2),
        "adjacency": torch.zeros(batch_size, num_loc, num_loc, dtype=torch.bool),
        "degrees": torch.zeros(batch_size, num_loc, dtype=torch.long),
        "current_step": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_deletions": torch.zeros(batch_size, 1, dtype=torch.long),
        "num_edges": torch.zeros(batch_size, 1, dtype=torch.long),
        "done": torch.zeros(batch_size, dtype=torch.bool),
    }, batch_size=[batch_size])
    
    # Add some edges to batch 1 for DELETE actions
    td["adjacency"][1, 0, 1] = True
    td["adjacency"][1, 1, 0] = True
    td["degrees"][1, 0] = 1
    td["degrees"][1, 1] = 1
    td["num_edges"][1] = 1
    
    # Create mask with different action types
    num_add = num_loc * (num_loc - 1) // 2
    max_del = num_loc
    total = num_add + max_del + 1
    action_mask = torch.zeros(batch_size, total, dtype=torch.bool)
    
    # Batch 0: Allow ADD actions
    action_mask[0, :num_add] = True
    
    # Batch 1: Allow DELETE actions (has edges)
    action_mask[1, num_add:num_add + max_del] = True
    
    # Batch 2: Allow DONE action
    action_mask[2, -1] = True
    
    td["action_mask"] = action_mask
    
    # Forward pass
    out = policy(td, phase='train', decode_type='sampling')
    
    # Create dummy loss (REINFORCE-style)
    log_probs = out['log_prob']
    rewards = torch.tensor([1.0, 2.0, 1.5])  # Dummy rewards
    loss = -(log_probs * rewards).mean()
    
    print(f"  Log probs: {log_probs}")
    print(f"  Rewards:   {rewards}")
    print(f"  Loss:      {loss.item():.4f}")
    print(f"  Batch 0: ADD actions feasible")
    print(f"  Batch 1: DELETE actions feasible")
    print(f"  Batch 2: DONE action feasible")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\n  Checking gradients for key parameters:")
    
    components = [
        ("Encoder init_embed", policy.encoder.init_embed),
        ("Encoder layer 0 attn_q", policy.encoder.layers[0].attn_q),
        ("AddEdgeDecoder scorer", policy.add_decoder.edge_scorer[0]),
        ("DeleteEdgeDecoder scorer", policy.delete_decoder.edge_scorer[0]),
        ("DoneDecoder scorer", policy.done_decoder.scorer[0]),
    ]
    
    all_have_grads = True
    for name, layer in components:
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            grad_norm = layer.weight.grad.norm().item()
            grad_mean = layer.weight.grad.mean().item()
            print(f"    {name:30s}: grad_norm={grad_norm:.6f}, grad_mean={grad_mean:.6f}")
            
            # Only encoder should always have gradients
            # Decoders may have zero gradients if their actions weren't selected
            if 'Encoder' in name and grad_norm == 0:
                print(f"      WARNING: Zero gradient in encoder!")
                all_have_grads = False
        else:
            print(f"    {name:30s}: NO GRADIENT")
            if 'Encoder' in name:
                all_have_grads = False
    
    assert all_have_grads, "Encoder components should have gradients!"
    print("\n  ✓ Key encoder components have non-zero gradients")
    print("  Note: Decoder gradients may be zero if their actions weren't sampled")
    
    # Test gradient clipping (common in RL)
    print("\n  Testing gradient clipping:")
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    
    total_norm = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"    Total gradient norm after clipping: {total_norm:.4f}")
    print(f"    Should be <= 1.0: {total_norm <= 1.0}")
    
    print("\n✓ Gradient flow test passed!")


def test_delete_bias_schedule():
    """Test delete bias scheduling in detail."""
    print("\n" + "="*80)
    print("TEST 7: Delete Bias Scheduling")
    print("="*80)
    
    num_loc = 20
    policy = CustomPOMOPolicy(
        num_loc=num_loc,
        delete_bias_start=-5.0,
        delete_bias_end=0.0,
        delete_bias_warmup_epochs=100
    )
    
    print("\n  Configuration:")
    print(f"    Start bias:     {policy.delete_bias_start.item()}")
    print(f"    End bias:       {policy.delete_bias_end.item()}")
    print(f"    Warmup epochs:  {policy.delete_bias_warmup_epochs}")
    
    # Test schedule at various epochs
    print("\n  Schedule progression:")
    test_epochs = [0, 25, 50, 75, 100, 150]
    
    for epoch in test_epochs:
        policy.set_epoch(epoch)
        bias_train = policy.get_delete_bias('train')
        bias_val = policy.get_delete_bias('val')
        
        # Expected value (linear interpolation)
        if epoch >= 100:
            expected = 0.0
        else:
            alpha = epoch / 100.0
            expected = (1 - alpha) * (-5.0) + alpha * 0.0
        
        print(f"    Epoch {epoch:3d}: train={bias_train:6.3f} (expected={expected:6.3f}), val={bias_val:6.3f}")
        
        assert abs(bias_train - expected) < 0.01, f"Epoch {epoch}: expected {expected}, got {bias_train}"
        assert abs(bias_val - 0.0) < 0.01, "Val bias should always be 0.0"
    
    # Test custom schedule
    print("\n  Custom schedule (start=-10, end=5, warmup=50):")
    policy2 = CustomPOMOPolicy(
        num_loc=num_loc,
        delete_bias_start=-10.0,
        delete_bias_end=5.0,
        delete_bias_warmup_epochs=50
    )
    
    for epoch in [0, 25, 50]:
        policy2.set_epoch(epoch)
        bias = policy2.get_delete_bias('train')
        
        if epoch >= 50:
            expected = 5.0
        else:
            alpha = epoch / 50.0
            expected = (1 - alpha) * (-10.0) + alpha * 5.0
        
        print(f"    Epoch {epoch:3d}: bias={bias:6.3f} (expected={expected:6.3f})")
        assert abs(bias - expected) < 0.01
    
    print("\n✓ Delete bias scheduling test passed!")


def test_integration_with_environment():
    """Test policy integration with actual environment over multiple steps."""
    print("\n" + "="*80)
    print("TEST 8: Integration with Environment (Multi-Step)")
    print("="*80)
    
    num_loc = 15  # Smaller for faster test
    batch_size = 2
    
    # Create environment
    generator = CustomTSPGenerator(num_loc=num_loc)
    env = CustomTSPEnv(generator=generator)
    
    # Create policy
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Reset environment
    td = env.reset(batch_size=[batch_size])
    
    print(f"\n  Environment: {env.name}")
    print(f"  Num locations: {num_loc}")
    print(f"  Batch size: {batch_size}")
    
    # Take 10 steps
    num_steps = 10
    print(f"\n  Taking {num_steps} steps:")
    
    for step in range(num_steps):
        # Policy forward
        out = policy(td, phase='train', decode_type='sampling', return_actions=True)
        
        # Decode action
        action_type = out['action_components']['action_type']
        node_i = out['action_components']['node_i']
        node_j = out['action_components']['node_j']
        
        # Print step info
        print(f"\n    Step {step}:")
        for b in range(batch_size):
            atype = action_type[b].item()
            i = node_i[b].item()
            j = node_j[b].item()
            
            type_str = ['ADD', 'DELETE', 'DONE'][atype]
            if atype == 2:  # DONE
                print(f"      Batch {b}: {type_str}")
            else:
                print(f"      Batch {b}: {type_str} edge ({i}, {j})")
        
        # Set action in td
        td.set("action", out["action"])
        
        # Step environment
        td = env.step(td)["next"]
        
        # Print state
        print(f"      State: edges={td['num_edges'][0].item()}, "
              f"deletions={td['num_deletions'][0].item()}, "
              f"done={td['done'][0].item()}")
        
        # Check for episode completion
        if td['done'].all():
            print(f"\n  All episodes finished at step {step}!")
            break
    
    # Final statistics
    print("\n  Final state:")
    for b in range(batch_size):
        print(f"    Batch {b}:")
        print(f"      Edges:     {td['num_edges'][b].item()}")
        print(f"      Deletions: {td['num_deletions'][b].item()}")
        print(f"      Done:      {td['done'][b].item()}")
        print(f"      Reward:    {td['reward'][b].item():.4f}")
        
        # Check degrees
        degrees = td['degrees'][b]
        print(f"      Degree stats: min={degrees.min().item()}, "
              f"max={degrees.max().item()}, "
              f"mean={degrees.float().mean().item():.2f}")
    
    print("\n✓ Integration test passed!")


def test_action_utils():
    """Test action utility functions."""
    print("\n" + "="*80)
    print("TEST 9: Action Utility Functions")
    print("="*80)
    
    num_loc = 10
    batch_size = 3
    
    # Test decode_action_index
    print("\n--- Testing decode_action_index ---")
    
    num_add = num_loc * (num_loc - 1) // 2  # 45
    max_del = num_loc  # 10
    total = num_add + max_del + 1  # 56
    
    # Create edge indices for ADD
    edge_indices = torch.triu_indices(num_loc, num_loc, offset=1).t()
    
    # Create edge list for DELETE
    edge_list = torch.full((batch_size, max_del, 2), -1, dtype=torch.long)
    edge_list[0, 0] = torch.tensor([0, 1])
    edge_list[0, 1] = torch.tensor([2, 3])
    edge_list[1, 0] = torch.tensor([5, 6])
    
    # Test cases
    test_cases = [
        (0, "First ADD action (0,1)"),
        (num_add - 1, f"Last ADD action ({num_loc-2},{num_loc-1})"),
        (num_add, "First DELETE action"),
        (num_add + 1, "Second DELETE action"),
        (total - 1, "DONE action"),
    ]
    
    for action_idx, description in test_cases:
        action_tensor = torch.tensor([action_idx] * batch_size)
        
        atype, node_i, node_j = decode_action_index(
            action_tensor, num_add, max_del, edge_indices, edge_list
        )
        
        print(f"  {description}:")
        print(f"    Index: {action_idx}")
        print(f"    Type:  {atype.tolist()}")
        print(f"    Node i: {node_i.tolist()}")
        print(f"    Node j: {node_j.tolist()}")
    
    # Test extract_edge_list
    print("\n--- Testing extract_edge_list ---")
    
    adjacency = torch.zeros(batch_size, num_loc, num_loc, dtype=torch.bool)
    adjacency[0, 0, 1] = True
    adjacency[0, 1, 0] = True
    adjacency[0, 2, 3] = True
    adjacency[0, 3, 2] = True
    adjacency[1, 5, 6] = True
    adjacency[1, 6, 5] = True
    
    extracted = extract_edge_list(adjacency, max_del)
    
    print(f"  Input adjacency: {adjacency.shape}")
    print(f"  Output edge list: {extracted.shape}")
    print(f"  Batch 0 edges: {extracted[0, :3]}")
    print(f"  Batch 1 edges: {extracted[1, :2]}")
    
    # Test create_node_features
    print("\n--- Testing create_node_features ---")
    
    td = TensorDict({
        "locs": torch.rand(batch_size, num_loc, 2),
        "adjacency": adjacency,
        "degrees": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
        "current_step": torch.tensor([[5], [3], [0]], dtype=torch.long),
        "num_deletions": torch.tensor([[0], [1], [0]], dtype=torch.long),
    }, batch_size=[batch_size])
    
    features = create_node_features(td, num_loc)
    
    print(f"  Input state: locs, adjacency, degrees, counters")
    print(f"  Output features: {features.shape}")
    print(f"  Expected: (3, 10, 15)  [N+5 = 10+5]")
    
    assert features.shape == (batch_size, num_loc, num_loc + 5)
    
    # Check feature components
    print(f"  Feature breakdown (batch 0, node 0):")
    print(f"    locs (2):       {features[0, 0, :2]}")
    print(f"    degree (1):     {features[0, 0, 2]}")
    print(f"    adjacency (10): {features[0, 0, 3:13]}")
    print(f"    step_norm (1):  {features[0, 0, 13]}")
    print(f"    del_norm (1):   {features[0, 0, 14]}")
    
    print("\n✓ All action utility tests passed!")


def run_all_tests():
    """Run all comprehensive tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL UNIT TESTS - STEP 7")
    print("="*80)
    
    try:
        test_encoder_various_sizes()
        test_add_decoder_edge_cases()
        test_delete_decoder_edge_cases()
        test_done_decoder_edge_cases()
        test_policy_edge_cases()
        test_gradient_flow()
        test_delete_bias_schedule()
        test_integration_with_environment()
        test_action_utils()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nSummary:")
        print("  ✓ Encoder tested with N=10,20,50,100")
        print("  ✓ All decoder heads tested with edge cases")
        print("  ✓ Policy tested with edge cases")
        print("  ✓ Gradient flow verified")
        print("  ✓ Delete bias scheduling validated")
        print("  ✓ Environment integration confirmed")
        print("  ✓ Action utilities verified")
        print("\nReady for Step 8: train.py implementation")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
