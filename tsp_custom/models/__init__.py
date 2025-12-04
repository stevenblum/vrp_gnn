"""
Models package for Custom TSP Environment

Contains:
- TransformerEncoder: 6-layer transformer for encoding node features
- AddEdgeDecoder, DeleteEdgeDecoder, DoneDecoder: Specialized decoder heads
- CustomPOMOPolicy: Main policy network combining encoder + 3 decoders
- CustomPOMOModel: Lightning module for REINFORCE training
- Action utilities: Helper functions for action encoding/decoding and masking
"""

from .encoder import TransformerEncoder, TransformerEncoderLayer
from .decoders import AddEdgeDecoder, DeleteEdgeDecoder, DoneDecoder
from .custom_pomo_policy import CustomPOMOPolicy
from .custom_pomo_model import CustomPOMOModel
from .action_utils import (
    decode_action_index,
    extract_edge_list,
    compute_action_masks,
    create_node_features
)

__all__ = [
    # Encoder
    "TransformerEncoder",
    "TransformerEncoderLayer",
    # Decoders
    "AddEdgeDecoder",
    "DeleteEdgeDecoder",
    "DoneDecoder",
    # Policy
    "CustomPOMOPolicy",
    # Model
    "CustomPOMOModel",
    # Utilities
    "decode_action_index",
    "extract_edge_list",
    "compute_action_masks",
    "create_node_features",
]
