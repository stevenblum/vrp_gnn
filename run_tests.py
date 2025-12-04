#!/usr/bin/env python3
"""Simple test runner for tour validity tests."""

import sys
sys.path.insert(0, '/home/scblum/Projects/vrp_gnn')

from tsp_custom.tests.test_tour_validity import *

if __name__ == "__main__":
    print("=" * 80)
    print("Running Tour Validity Tests")
    print("=" * 80)
    
    # Run all test functions
    test_functions = [
        ("Valid Tour: Simple Triangle (3 nodes)", test_valid_tour_simple_triangle),
        ("Valid Tour: Square (4 nodes)", test_valid_tour_square),
        ("Valid Tour: 10 nodes cycle", test_valid_tour_10_nodes),
        ("Invalid Tour: 3-node loop in 10-node problem (KEY TEST)", test_invalid_small_loop_missing_nodes),
        ("Invalid Tour: Two separate loops", test_invalid_two_separate_loops),
        ("Invalid Tour: Incomplete path (not a cycle)", test_invalid_incomplete_path),
        ("Invalid Tour: Node with degree > 2", test_invalid_degree_3_node),
        ("Invalid Tour: Empty graph", test_invalid_empty_graph),
        ("Invalid Tour: Partial tour (7/10 nodes)", test_invalid_partial_tour),
        ("Valid Tour: Alternative ordering", test_valid_tour_alternative_order),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {name}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)
