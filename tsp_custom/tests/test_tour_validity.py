"""
Unit tests for tour validity checking.

Tests check_tour_validity function with various valid and invalid tour configurations.
"""

import torch
import pytest
from tsp_custom.envs.utils import check_tour_validity


def test_valid_tour_simple_triangle():
    """Test a valid 3-node tour: 0-1-2-0"""
    adjacency = torch.zeros(3, 3, dtype=torch.bool)
    # Connect 0-1, 1-2, 2-0
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[2, 0] = adjacency[0, 2] = True
    
    assert check_tour_validity(adjacency, 3), "Valid 3-node triangle tour should be valid"
    
    # Verify degrees
    degrees = adjacency.sum(dim=0)
    assert (degrees == 2).all(), "All nodes should have degree 2"


def test_valid_tour_square():
    """Test a valid 4-node tour: 0-1-2-3-0"""
    adjacency = torch.zeros(4, 4, dtype=torch.bool)
    # Connect 0-1, 1-2, 2-3, 3-0
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[2, 3] = adjacency[3, 2] = True
    adjacency[3, 0] = adjacency[0, 3] = True
    
    assert check_tour_validity(adjacency, 4), "Valid 4-node square tour should be valid"


def test_valid_tour_10_nodes():
    """Test a valid 10-node tour"""
    adjacency = torch.zeros(10, 10, dtype=torch.bool)
    # Create a simple cycle: 0-1-2-3-4-5-6-7-8-9-0
    for i in range(10):
        j = (i + 1) % 10
        adjacency[i, j] = adjacency[j, i] = True
    
    assert check_tour_validity(adjacency, 10), "Valid 10-node tour should be valid"


def test_invalid_small_loop_missing_nodes():
    """Test invalid: only 3 nodes form a loop in a 10-node problem"""
    adjacency = torch.zeros(10, 10, dtype=torch.bool)
    # Only connect nodes 0, 1, 2 in a loop
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[2, 0] = adjacency[0, 2] = True
    # Nodes 3-9 are not connected (degree 0)
    
    assert not check_tour_validity(adjacency, 10), "3-node loop in 10-node problem should be INVALID"
    
    # Verify why it's invalid
    degrees = adjacency.sum(dim=0)
    assert degrees[0] == 2 and degrees[1] == 2 and degrees[2] == 2, "Loop nodes have degree 2"
    assert degrees[3] == 0, "Node 3 should have degree 0 (not in tour)"
    assert not (degrees == 2).all(), "Not all nodes have degree 2"


def test_invalid_two_separate_loops():
    """Test invalid: two disconnected loops"""
    adjacency = torch.zeros(6, 6, dtype=torch.bool)
    # Loop 1: 0-1-2-0
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[2, 0] = adjacency[0, 2] = True
    # Loop 2: 3-4-5-3
    adjacency[3, 4] = adjacency[4, 3] = True
    adjacency[4, 5] = adjacency[5, 4] = True
    adjacency[5, 3] = adjacency[3, 5] = True
    
    assert not check_tour_validity(adjacency, 6), "Two separate loops should be INVALID"
    
    # Verify why it's invalid
    degrees = adjacency.sum(dim=0)
    assert (degrees == 2).all(), "All nodes have degree 2"
    # But graph is disconnected


def test_invalid_incomplete_path():
    """Test invalid: incomplete path (not a cycle)"""
    adjacency = torch.zeros(5, 5, dtype=torch.bool)
    # Create a path: 0-1-2-3-4 (no connection back to 0)
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[2, 3] = adjacency[3, 2] = True
    adjacency[3, 4] = adjacency[4, 3] = True
    
    assert not check_tour_validity(adjacency, 5), "Incomplete path should be INVALID"
    
    # Verify degrees
    degrees = adjacency.sum(dim=0)
    assert degrees[0] == 1 and degrees[4] == 1, "End nodes should have degree 1"
    assert not (degrees == 2).all(), "Not all nodes have degree 2"


def test_invalid_degree_3_node():
    """Test invalid: node with degree > 2"""
    adjacency = torch.zeros(4, 4, dtype=torch.bool)
    # Node 0 connects to all others (degree 3)
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[0, 2] = adjacency[2, 0] = True
    adjacency[0, 3] = adjacency[3, 0] = True
    # Close the loop: 1-2-3-1
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[2, 3] = adjacency[3, 2] = True
    adjacency[3, 1] = adjacency[1, 3] = True
    
    assert not check_tour_validity(adjacency, 4), "Node with degree > 2 should be INVALID"
    
    # Verify degrees
    degrees = adjacency.sum(dim=0)
    assert degrees[0] == 3, "Node 0 should have degree 3"


def test_invalid_empty_graph():
    """Test invalid: no edges at all"""
    adjacency = torch.zeros(5, 5, dtype=torch.bool)
    
    assert not check_tour_validity(adjacency, 5), "Empty graph should be INVALID"
    
    # Verify degrees
    degrees = adjacency.sum(dim=0)
    assert (degrees == 0).all(), "All nodes should have degree 0"


def test_invalid_partial_tour():
    """Test invalid: only 7 out of 10 nodes in tour"""
    adjacency = torch.zeros(10, 10, dtype=torch.bool)
    # Create cycle with only nodes 0-6
    for i in range(7):
        j = (i + 1) % 7
        adjacency[i, j] = adjacency[j, i] = True
    # Nodes 7, 8, 9 not connected
    
    assert not check_tour_validity(adjacency, 10), "Partial tour (7/10 nodes) should be INVALID"
    
    # Verify degrees
    degrees = adjacency.sum(dim=0)
    assert degrees[7] == 0 and degrees[8] == 0 and degrees[9] == 0, "Unused nodes should have degree 0"


def test_valid_tour_alternative_order():
    """Test valid tour with non-sequential node ordering"""
    adjacency = torch.zeros(5, 5, dtype=torch.bool)
    # Create tour: 0-2-4-1-3-0 (non-sequential)
    adjacency[0, 2] = adjacency[2, 0] = True
    adjacency[2, 4] = adjacency[4, 2] = True
    adjacency[4, 1] = adjacency[1, 4] = True
    adjacency[1, 3] = adjacency[3, 1] = True
    adjacency[3, 0] = adjacency[0, 3] = True
    
    assert check_tour_validity(adjacency, 5), "Valid tour with non-sequential ordering should be valid"


if __name__ == "__main__":
    # Run tests
    print("Running tour validity tests...")
    
    test_valid_tour_simple_triangle()
    print("✓ Valid 3-node triangle")
    
    test_valid_tour_square()
    print("✓ Valid 4-node square")
    
    test_valid_tour_10_nodes()
    print("✓ Valid 10-node tour")
    
    test_invalid_small_loop_missing_nodes()
    print("✓ Invalid: 3-node loop in 10-node problem")
    
    test_invalid_two_separate_loops()
    print("✓ Invalid: two disconnected loops")
    
    test_invalid_incomplete_path()
    print("✓ Invalid: incomplete path")
    
    test_invalid_degree_3_node()
    print("✓ Invalid: node with degree > 2")
    
    test_invalid_empty_graph()
    print("✓ Invalid: empty graph")
    
    test_invalid_partial_tour()
    print("✓ Invalid: partial tour (7/10 nodes)")
    
    test_valid_tour_alternative_order()
    print("✓ Valid tour with non-sequential ordering")
    
    print("\n✅ All tests passed!")
