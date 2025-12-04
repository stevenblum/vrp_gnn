"""
Unit tests for CustomTSPEnv

Tests cover:
- Environment initialization and reset
- ADD action state updates
- DELETE action state updates  
- DONE action and termination
- Action masking correctness
- Reward calculation
- Validity checking
- Edge cases and boundary conditions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import unittest
from tensordict import TensorDict
from pathlib import Path

from envs.custom_tsp_env import CustomTSPEnv
from envs.generator import CustomTSPGenerator
from envs.utils import encode_action, compute_tour_length, is_graph_connected
from visualization.plot_tour import (
    plot_tsp_instance, 
    create_episode_gif, 
    visualize_state,
    create_test_episode_gif
)


def encode_add_action(node_i, node_j, num_loc):
    """Helper to encode ADD action without needing adjacency"""
    # For ADD actions, we can compute directly
    num_add_actions = num_loc * (num_loc - 1) // 2
    idx = 0
    for ii in range(min(node_i, node_j)):
        idx += (num_loc - ii - 1)
    idx += (max(node_i, node_j) - min(node_i, node_j) - 1)
    return idx


class TestCustomTSPEnvInitialization(unittest.TestCase):
    """Test environment initialization and reset"""
    
    def setUp(self):
        """Create environment for testing"""
        self.num_loc = 10
        self.batch_size = 2
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
    
    def test_env_creation(self):
        """Test that environment can be created"""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.name, "custom_tsp")
    
    def test_reset_creates_valid_state(self):
        """Test that reset creates a valid initial state"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Check all required keys are present
        required_keys = ['locs', 'adjacency', 'degrees', 'num_edges', 
                        'current_step', 'num_deletions', 'done', 'action_mask']
        for key in required_keys:
            self.assertIn(key, td.keys())
        
        # Check shapes
        self.assertEqual(td['locs'].shape, (self.batch_size, self.num_loc, 2))
        self.assertEqual(td['adjacency'].shape, (self.batch_size, self.num_loc, self.num_loc))
        self.assertEqual(td['degrees'].shape, (self.batch_size, self.num_loc))
        self.assertEqual(td['num_edges'].shape, (self.batch_size, 1))
        self.assertEqual(td['current_step'].shape, (self.batch_size, 1))
        self.assertEqual(td['num_deletions'].shape, (self.batch_size, 1))
    
    def test_reset_initializes_to_zero(self):
        """Test that reset initializes state to zeros"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # All adjacency should be zero (no edges)
        self.assertTrue(torch.all(td['adjacency'] == 0))
        
        # All degrees should be zero
        self.assertTrue(torch.all(td['degrees'] == 0))
        
        # Counters should be zero
        self.assertTrue(torch.all(td['num_edges'] == 0))
        self.assertTrue(torch.all(td['current_step'] == 0))
        self.assertTrue(torch.all(td['num_deletions'] == 0))
        
        # Done should be False
        self.assertTrue(torch.all(~td['done']))
    
    def test_initial_action_mask(self):
        """Test that initial action mask allows ADD actions only"""
        td = self.env.reset(batch_size=self.batch_size)
        mask = td['action_mask']
        
        # Calculate number of possible ADD actions: N*(N-1)/2
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        
        # First num_add_actions should be True (ADD allowed)
        for b in range(self.batch_size):
            # ADD actions should be unmasked
            self.assertTrue(torch.all(mask[b, :num_add_actions]))
            
            # DELETE and DONE should be masked (no edges to delete, tour incomplete)
            self.assertTrue(torch.all(~mask[b, num_add_actions:]))


class TestCustomTSPEnvAddAction(unittest.TestCase):
    """Test ADD action functionality"""
    
    def setUp(self):
        """Create environment for testing"""
        self.num_loc = 10
        self.batch_size = 2
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
        self.td = self.env.reset(batch_size=self.batch_size)
    
    def test_add_single_edge(self):
        """Test adding a single edge"""
        # Add edge (0, 1)
        action = encode_add_action(0, 1, self.num_loc)  # ADD, node 0, node 1
        action_tensor = torch.tensor([action, action], dtype=torch.long)
        
        td = self.td.clone()
        td['action'] = action_tensor
        td_next = self.env.step(td)['next']
        
        # Check adjacency matrix updated
        self.assertEqual(td_next['adjacency'][0, 0, 1].item(), 1)
        self.assertEqual(td_next['adjacency'][0, 1, 0].item(), 1)
        
        # Check degrees updated
        self.assertEqual(td_next['degrees'][0, 0].item(), 1)
        self.assertEqual(td_next['degrees'][0, 1].item(), 1)
        
        # Check num_edges updated
        self.assertEqual(td_next['num_edges'][0].item(), 1)
        
        # Check step counter incremented
        self.assertEqual(td_next['current_step'][0].item(), 1)
    
    def test_add_multiple_edges(self):
        """Test adding multiple edges sequentially"""
        td = self.td.clone()
        
        # Add edge (0, 1)
        action1 = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action1, action1], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Add edge (1, 2)
        action2 = encode_add_action(1, 2, self.num_loc)
        td['action'] = torch.tensor([action2, action2], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Add edge (2, 3)
        action3 = encode_add_action(2, 3, self.num_loc)
        td['action'] = torch.tensor([action3, action3], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Check all edges present
        self.assertEqual(td['adjacency'][0, 0, 1].item(), 1)
        self.assertEqual(td['adjacency'][0, 1, 2].item(), 1)
        self.assertEqual(td['adjacency'][0, 2, 3].item(), 1)
        
        # Check degrees
        self.assertEqual(td['degrees'][0, 0].item(), 1)  # degree 1
        self.assertEqual(td['degrees'][0, 1].item(), 2)  # degree 2
        self.assertEqual(td['degrees'][0, 2].item(), 2)  # degree 2
        self.assertEqual(td['degrees'][0, 3].item(), 1)  # degree 1
        
        # Check num_edges
        self.assertEqual(td['num_edges'][0].item(), 3)
        
        # Check step counter
        self.assertEqual(td['current_step'][0].item(), 3)
    
    def test_add_edge_degree_2_nodes(self):
        """Test adding edges to create degree 2 nodes"""
        td = self.td.clone()
        
        # Add edges to node 0: (0,1) and (0,2)
        action1 = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action1, action1], dtype=torch.long)
        td = self.env.step(td)['next']
        
        action2 = encode_add_action(0, 2, self.num_loc)
        td['action'] = torch.tensor([action2, action2], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Node 0 should now have degree 2
        self.assertEqual(td['degrees'][0, 0].item(), 2)
        
        # Action mask should prevent adding more edges to node 0
        mask = td['action_mask'][0]
        
        # Try to encode an action that would connect to node 0
        # Since node 0 has degree 2, these should be masked
        action_03 = encode_add_action(0, 3, self.num_loc)
        self.assertFalse(mask[action_03].item(), 
                        "Action to add edge (0,3) should be masked")


class TestCustomTSPEnvDeleteAction(unittest.TestCase):
    """Test DELETE action functionality"""
    
    def setUp(self):
        """Create environment with some edges added"""
        self.num_loc = 10
        self.batch_size = 2
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
        self.td = self.env.reset(batch_size=self.batch_size)
        
        # Add a few edges to create initial state
        # Edge (0, 1)
        action1 = encode_add_action(0, 1, self.num_loc)
        self.td['action'] = torch.tensor([action1, action1], dtype=torch.long)
        self.td = self.env.step(self.td)['next']
        
        # Edge (1, 2)
        action2 = encode_add_action(1, 2, self.num_loc)
        self.td['action'] = torch.tensor([action2, action2], dtype=torch.long)
        self.td = self.env.step(self.td)['next']
        
        # Edge (2, 3)
        action3 = encode_add_action(2, 3, self.num_loc)
        self.td['action'] = torch.tensor([action3, action3], dtype=torch.long)
        self.td = self.env.step(self.td)['next']
    
    def test_delete_single_edge(self):
        """Test deleting a single edge"""
        td = self.td.clone()
        
        # Verify edge (0,1) exists
        self.assertEqual(td['adjacency'][0, 0, 1].item(), 1)
        
        # Delete edge (0, 1) - DELETE actions start after ADD actions
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        delete_action = num_add_actions  # First DELETE action
        td['action'] = torch.tensor([delete_action, delete_action], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Check edge removed
        self.assertEqual(td['adjacency'][0, 0, 1].item(), 0)
        self.assertEqual(td['adjacency'][0, 1, 0].item(), 0)
        
        # Check degrees decremented
        self.assertEqual(td['degrees'][0, 0].item(), 0)
        self.assertEqual(td['degrees'][0, 1].item(), 1)  # Still connected to 2
        
        # Check num_edges decremented
        self.assertEqual(td['num_edges'][0].item(), 2)
        
        # Check num_deletions incremented
        self.assertEqual(td['num_deletions'][0].item(), 1)
    
    def test_delete_and_readd_edge(self):
        """Test deleting an edge and adding it back"""
        td = self.td.clone()
        
        # Delete edge (1, 2)
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        delete_action = num_add_actions + 1  # Second edge in list
        td['action'] = torch.tensor([delete_action, delete_action], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Verify deleted
        self.assertEqual(td['adjacency'][0, 1, 2].item(), 0)
        self.assertEqual(td['num_edges'][0].item(), 2)
        
        # Re-add edge (1, 2)
        action = encode_add_action(1, 2, self.num_loc)
        td['action'] = torch.tensor([action, action], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Verify re-added
        self.assertEqual(td['adjacency'][0, 1, 2].item(), 1)
        self.assertEqual(td['num_edges'][0].item(), 3)
        
        # Deletion counter should still be 1
        self.assertEqual(td['num_deletions'][0].item(), 1)


class TestCustomTSPEnvDoneAction(unittest.TestCase):
    """Test DONE action and episode termination"""
    
    def setUp(self):
        """Create environment"""
        self.num_loc = 5  # Smaller for easier valid tour construction
        self.batch_size = 1
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
    
    def test_done_with_valid_tour(self):
        """Test DONE action with a complete valid tour"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Create a valid tour: 0-1-2-3-4-0
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        
        for i, j in edges:
            action = encode_add_action(i, j, self.num_loc)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = self.env.step(td)['next']
        
        # Verify all degrees are 2
        self.assertTrue(torch.all(td['degrees'][0] == 2))
        
        # Verify graph is connected
        self.assertTrue(is_graph_connected(td['adjacency'][0]))
        
        # DONE action should be available in mask
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        done_action_idx = num_add_actions + self.num_loc  # After ADD and DELETE
        mask = td['action_mask'][0]
        self.assertTrue(mask[done_action_idx].item(), 
                       "DONE action should be unmasked for valid tour")
        
        # Execute DONE action
        td['action'] = torch.tensor([done_action_idx], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Episode should be done
        self.assertTrue(td['done'][0].item())
    
    def test_done_masked_with_incomplete_tour(self):
        """Test that DONE is masked when tour is incomplete"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Add only some edges (incomplete tour)
        action1 = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action1], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # DONE should be masked
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        done_action_idx = num_add_actions + self.num_loc
        mask = td['action_mask'][0]
        self.assertFalse(mask[done_action_idx].item(),
                        "DONE action should be masked for incomplete tour")
    
    def test_step_limit_termination(self):
        """Test that episode terminates at step limit"""
        td = self.env.reset(batch_size=self.batch_size)
        max_steps = 2 * self.num_loc
        
        # Take random valid actions until step limit
        for step in range(max_steps):
            mask = td['action_mask'][0]
            valid_actions = mask.nonzero(as_tuple=True)[0]
            
            if len(valid_actions) == 0:
                break
                
            action = valid_actions[0]
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = self.env.step(td)['next']
            
            if td['done'][0].item():
                break
        
        # Should hit step limit
        self.assertTrue(td['done'][0].item())
        self.assertTrue(td['hit_step_limit'][0].item())


class TestCustomTSPEnvActionMasking(unittest.TestCase):
    """Test action masking logic"""
    
    def setUp(self):
        """Create environment"""
        self.num_loc = 10
        self.batch_size = 2
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
        self.td = self.env.reset(batch_size=self.batch_size)
    
    def test_mask_existing_edge(self):
        """Test that existing edges are masked for ADD"""
        td = self.td.clone()
        
        # Add edge (0, 1)
        action = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action, action], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Try to add same edge again - should be masked
        mask = td['action_mask'][0]
        self.assertFalse(mask[action].item(),
                        "Existing edge should be masked for ADD")
    
    def test_mask_degree_2_nodes(self):
        """Test that nodes with degree 2 cannot have more edges"""
        td = self.td.clone()
        
        # Add two edges to node 0
        action1 = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action1, action1], dtype=torch.long)
        td = self.env.step(td)['next']
        
        action2 = encode_add_action(0, 2, self.num_loc)
        td['action'] = torch.tensor([action2, action2], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Node 0 has degree 2, all edges to it should be masked
        mask = td['action_mask'][0]
        
        # Check various edges to node 0 are masked
        for other_node in range(3, self.num_loc):
            action = encode_add_action(0, other_node, self.num_loc)
            self.assertFalse(mask[action].item(),
                           f"Edge (0,{other_node}) should be masked")
    
    def test_delete_only_existing_edges(self):
        """Test that DELETE actions only available for existing edges"""
        td = self.td.clone()
        
        # Initially, no DELETE actions should be available
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        mask = td['action_mask'][0]
        
        # All DELETE actions should be masked (no edges exist)
        delete_region = mask[num_add_actions:num_add_actions + self.num_loc]
        self.assertTrue(torch.all(~delete_region),
                       "No DELETE actions should be available initially")
        
        # Add an edge
        action = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action, action], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Now DELETE action for this edge should be available
        mask = td['action_mask'][0]
        # First DELETE action corresponds to first edge added
        first_delete_idx = num_add_actions
        self.assertTrue(mask[first_delete_idx].item(),
                       "DELETE action should be available for existing edge")


class TestCustomTSPEnvRewardCalculation(unittest.TestCase):
    """Test reward calculation"""
    
    def setUp(self):
        """Create environment"""
        self.num_loc = 5
        self.batch_size = 1
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
    
    def test_reward_for_valid_tour(self):
        """Test reward calculation for valid tour"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Create a valid tour: 0-1-2-3-4-0
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        
        for i, j in edges:
            action = encode_add_action(i, j, self.num_loc)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = self.env.step(td)['next']
        
        # Get reward (using get_reward method)
        actions = torch.zeros(1, 5, dtype=torch.long)  # Dummy actions
        reward = self.env.get_reward(td, actions)
        
        # Reward should be negative tour length (no deletions)
        tour_length = compute_tour_length(td['locs'][0], td['adjacency'][0])
        expected_reward = -tour_length
        
        self.assertAlmostEqual(reward[0].item(), expected_reward, places=5)
    
    def test_reward_with_deletions(self):
        """Test reward includes deletion penalty"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Add edges including one to delete
        action1 = encode_add_action(0, 1, self.num_loc)
        td['action'] = torch.tensor([action1], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Delete the edge
        num_add_actions = self.num_loc * (self.num_loc - 1) // 2
        delete_action = num_add_actions
        td['action'] = torch.tensor([delete_action], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Verify deletion counter incremented
        self.assertEqual(td['num_deletions'][0].item(), 1)
        
        # Complete the valid tour
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for i, j in edges:
            action = encode_add_action(i, j, self.num_loc)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = self.env.step(td)['next']
        
        # Get reward
        actions = torch.zeros(1, 10, dtype=torch.long)
        reward = self.env.get_reward(td, actions)
        
        # Should include deletion penalty
        self.assertLess(reward[0].item(), 0)  # Reward is negative


class TestCustomTSPEnvEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Create environment"""
        self.num_loc = 10
        self.batch_size = 2
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
    
    def test_batch_independence(self):
        """Test that batch elements are independent"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Apply different actions to different batch elements
        action1 = encode_add_action(0, 1, self.num_loc)
        action2 = encode_add_action(2, 3, self.num_loc)
        
        td['action'] = torch.tensor([action1, action2], dtype=torch.long)
        td = self.env.step(td)['next']
        
        # Check that batch 0 has edge (0,1)
        self.assertEqual(td['adjacency'][0, 0, 1].item(), 1)
        self.assertEqual(td['adjacency'][0, 2, 3].item(), 0)
        
        # Check that batch 1 has edge (2,3)
        self.assertEqual(td['adjacency'][1, 2, 3].item(), 1)
        self.assertEqual(td['adjacency'][1, 0, 1].item(), 0)
    
    def test_small_problem_size(self):
        """Test with very small problem (N=3)"""
        small_gen = CustomTSPGenerator(num_loc=3)
        small_env = CustomTSPEnv(generator=small_gen)
        
        td = small_env.reset(batch_size=1)
        
        # Should be able to create valid tour with 3 nodes
        edges = [(0, 1), (1, 2), (2, 0)]
        for i, j in edges:
            action = encode_add_action(i, j, 3)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = small_env.step(td)['next']
        
        # Should be valid
        self.assertTrue(torch.all(td['degrees'][0] == 2))
        self.assertTrue(is_graph_connected(td['adjacency'][0]))
    
    def test_action_mask_consistency(self):
        """Test that action mask is consistent with state"""
        td = self.env.reset(batch_size=1)
        
        # Take several random valid actions
        for _ in range(10):
            mask = td['action_mask'][0]
            valid_actions = mask.nonzero(as_tuple=True)[0]
            
            if len(valid_actions) == 0:
                break
            
            # Pick first valid action
            action = valid_actions[0]
            td['action'] = torch.tensor([action], dtype=torch.long)
            td_prev = td.clone()
            td = self.env.step(td)['next']
            
            # Verify state changes are consistent with action
            # (degrees should only increase/decrease by 1 or 2)
            degree_change = torch.abs(td['degrees'] - td_prev['degrees']).sum()
            self.assertLessEqual(degree_change.item(), 4)  # Max 2 nodes, +/- 2 each


class TestCustomTSPEnvWarnings(unittest.TestCase):
    """Test warning conditions programmed into the environment"""
    
    def setUp(self):
        """Create environment"""
        self.num_loc = 10
        self.batch_size = 1
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
    
    def test_warning_degree_3_on_add(self):
        """Test that warning is issued when ADD action creates degree > 2"""
        import logging
        import io
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('envs.custom_tsp_env')
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        
        try:
            td = self.env.reset(batch_size=self.batch_size)
            
            # Add two edges to node 0: (0,1) and (0,2) - degree becomes 2
            action1 = encode_add_action(0, 1, self.num_loc)
            td['action'] = torch.tensor([action1], dtype=torch.long)
            td = self.env.step(td)['next']
            
            action2 = encode_add_action(0, 2, self.num_loc)
            td['action'] = torch.tensor([action2], dtype=torch.long)
            td = self.env.step(td)['next']
            
            # Verify node 0 has degree 2
            self.assertEqual(td['degrees'][0, 0].item(), 2)
            
            # Now manually force an action that would add a third edge to node 0
            # This should trigger the warning (but action masking should prevent this in practice)
            # We'll bypass masking by directly modifying the action
            action3 = encode_add_action(0, 3, self.num_loc)
            td['action'] = torch.tensor([action3], dtype=torch.long)
            td = self.env.step(td)['next']
            
            # Check if warning was logged
            log_contents = log_capture.getvalue()
            self.assertIn("degree", log_contents.lower(), 
                         "Warning about degree > 2 should be logged")
            self.assertIn("should be prevented by action masking", log_contents.lower(),
                         "Warning should mention action masking")
            
        finally:
            # Restore logger state
            logger.removeHandler(handler)
            logger.setLevel(original_level)
    
    def test_warning_done_with_invalid_degrees(self):
        """Test that warning is issued when DONE called with invalid degrees"""
        import logging
        import io
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('envs.custom_tsp_env')
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        
        try:
            td = self.env.reset(batch_size=self.batch_size)
            
            # Add only a few edges (incomplete tour)
            action1 = encode_add_action(0, 1, self.num_loc)
            td['action'] = torch.tensor([action1], dtype=torch.long)
            td = self.env.step(td)['next']
            
            action2 = encode_add_action(1, 2, self.num_loc)
            td['action'] = torch.tensor([action2], dtype=torch.long)
            td = self.env.step(td)['next']
            
            # Now force DONE action (bypassing mask)
            num_add_actions = self.num_loc * (self.num_loc - 1) // 2
            done_action = num_add_actions + self.num_loc  # DONE action index
            td['action'] = torch.tensor([done_action], dtype=torch.long)
            td = self.env.step(td)['next']
            
            # Check if warning was logged
            log_contents = log_capture.getvalue()
            self.assertIn("done called with invalid degrees", log_contents.lower(),
                         "Warning about invalid degrees should be logged")
            self.assertIn("should be prevented by action masking", log_contents.lower(),
                         "Warning should mention action masking")
            
        finally:
            # Restore logger state
            logger.removeHandler(handler)
            logger.setLevel(original_level)
    
    def test_warning_done_with_disconnected_graph(self):
        """Test that warning is issued when DONE called with disconnected graph"""
        import logging
        import io
        
        # Use smaller problem for easier disconnected graph creation
        small_gen = CustomTSPGenerator(num_loc=6)
        small_env = CustomTSPEnv(generator=small_gen)
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('envs.custom_tsp_env')
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        
        try:
            td = small_env.reset(batch_size=1)
            
            # Create two separate cycles (disconnected)
            # Cycle 1: 0-1-2-0
            edges_cycle1 = [(0, 1), (1, 2), (2, 0)]
            for i, j in edges_cycle1:
                action = encode_add_action(i, j, 6)
                td['action'] = torch.tensor([action], dtype=torch.long)
                td = small_env.step(td)['next']
            
            # Cycle 2: 3-4-5-3
            edges_cycle2 = [(3, 4), (4, 5), (5, 3)]
            for i, j in edges_cycle2:
                action = encode_add_action(i, j, 6)
                td['action'] = torch.tensor([action], dtype=torch.long)
                td = small_env.step(td)['next']
            
            # Now all nodes have degree 2, but graph is disconnected
            self.assertTrue(torch.all(td['degrees'][0] == 2))
            self.assertFalse(is_graph_connected(td['adjacency'][0]))
            
            # Force DONE action
            num_add_actions = 6 * 5 // 2
            done_action = num_add_actions + 6  # DONE action index
            td['action'] = torch.tensor([done_action], dtype=torch.long)
            td = small_env.step(td)['next']
            
            # Check if warning was logged
            log_contents = log_capture.getvalue()
            self.assertIn("disconnected", log_contents.lower(),
                         "Warning about disconnected graph should be logged")
            self.assertIn("should be prevented by action masking", log_contents.lower(),
                         "Warning should mention action masking")
            
        finally:
            # Restore logger state
            logger.removeHandler(handler)
            logger.setLevel(original_level)


class TestCustomTSPEnvVisualization(unittest.TestCase):
    """Test visualization functions"""
    
    def setUp(self):
        """Create environment and output directory"""
        self.num_loc = 10
        self.batch_size = 1
        self.generator = CustomTSPGenerator(num_loc=self.num_loc)
        self.env = CustomTSPEnv(generator=self.generator)
        self.output_dir = "tests/test_outputs"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def test_plot_tsp_instance(self):
        """Test static TSP instance plotting"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Add a few edges
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            action = encode_add_action(i, j, self.num_loc)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = self.env.step(td)['next']
        
        # Create plot
        fig = plot_tsp_instance(
            td['locs'][0], 
            td['adjacency'][0],
            title="Test TSP Plot"
        )
        
        # Save and verify
        output_path = f"{self.output_dir}/test_plot.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        self.assertTrue(Path(output_path).exists())
        print(f"  ✓ Created test plot: {output_path}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_visualize_state(self):
        """Test state visualization from TensorDict"""
        td = self.env.reset(batch_size=self.batch_size)
        
        # Add some edges
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            action = encode_add_action(i, j, self.num_loc)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = self.env.step(td)['next']
        
        # Visualize state
        output_path = f"{self.output_dir}/test_state.png"
        fig = visualize_state(td, batch_idx=0, save_path=output_path)
        
        self.assertTrue(Path(output_path).exists())
        print(f"  ✓ Created state visualization: {output_path}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_create_episode_gif(self):
        """Test episode GIF creation"""
        td = self.env.reset(batch_size=self.batch_size)
        locs = td['locs'][0]
        
        # Create action sequence
        action_sequence = []
        edges_to_add = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        
        for step, (i, j) in enumerate(edges_to_add):
            action_sequence.append((step, 0, i, j))  # 0 = ADD
        
        # Add a DELETE action
        action_sequence.append((5, 1, 1, 2))  # DELETE edge (1,2)
        
        # Re-add it
        action_sequence.append((6, 0, 1, 2))  # ADD edge (1,2) back
        
        # Create GIF
        output_path = f"{self.output_dir}/test_episode.gif"
        create_episode_gif(
            locs, 
            action_sequence, 
            output_path,
            fps=2,
            show_metrics=True
        )
        
        self.assertTrue(Path(output_path).exists())
        print(f"  ✓ Created episode GIF: {output_path}")
    
    def test_create_random_episode_gif(self):
        """Test GIF creation with random valid actions"""
        output_path = create_test_episode_gif(
            self.env,
            num_loc=self.num_loc,
            output_dir=self.output_dir,
            filename="test_random_episode.gif",
            max_steps=20
        )
        
        self.assertTrue(Path(output_path).exists())
        print(f"  ✓ Created random episode GIF: {output_path}")
    
    def test_visualize_complete_tour(self):
        """Test visualization of a complete valid tour"""
        # Use smaller problem for complete tour
        small_gen = CustomTSPGenerator(num_loc=5)
        small_env = CustomTSPEnv(generator=small_gen)
        
        td = small_env.reset(batch_size=1)
        locs = td['locs'][0]
        
        # Create valid tour: 0-1-2-3-4-0
        action_sequence = []
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        
        for step, (i, j) in enumerate(edges):
            action_sequence.append((step, 0, i, j))
            action = encode_add_action(i, j, 5)
            td['action'] = torch.tensor([action], dtype=torch.long)
            td = small_env.step(td)['next']
        
        # Add DONE action
        action_sequence.append((5, 2, -1, -1))
        
        # Create GIF
        output_path = f"{self.output_dir}/test_complete_tour.gif"
        create_episode_gif(
            locs,
            action_sequence,
            output_path,
            fps=1,
            show_metrics=True
        )
        
        self.assertTrue(Path(output_path).exists())
        print(f"  ✓ Created complete tour GIF: {output_path}")
    
    def test_visualize_with_deletions(self):
        """Test visualization showing deletion process"""
        td = self.env.reset(batch_size=self.batch_size)
        locs = td['locs'][0]
        
        action_sequence = []
        step = 0
        
        # Add some edges
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
            action_sequence.append((step, 0, i, j))
            step += 1
        
        # Delete some edges
        for i, j in [(1, 2), (3, 4)]:
            action_sequence.append((step, 1, i, j))
            step += 1
        
        # Add different edges
        for i, j in [(1, 5), (3, 0)]:
            action_sequence.append((step, 0, i, j))
            step += 1
        
        # Create GIF
        output_path = f"{self.output_dir}/test_deletions.gif"
        create_episode_gif(
            locs,
            action_sequence,
            output_path,
            fps=2,
            show_metrics=True
        )
        
        self.assertTrue(Path(output_path).exists())
        print(f"  ✓ Created deletion demonstration GIF: {output_path}")


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvAddAction))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvDeleteAction))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvDoneAction))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvActionMasking))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvRewardCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvWarnings))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomTSPEnvVisualization))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("="*80)
    print("CUSTOM TSP ENVIRONMENT - UNIT TESTS")
    print("="*80)
    print()
    
    result = run_all_tests()
    
    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    sys.exit(0 if result.wasSuccessful() else 1)
