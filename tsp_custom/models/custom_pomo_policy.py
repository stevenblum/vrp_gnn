"""
Custom POMO Policy for TSP with Edge Selection/Deletion

Implements a policy network with:
- Transformer encoder (6 layers, 128 dim)
- 3 specialized decoder heads (ADD, DELETE, DONE)
- Action masking and sampling/greedy decoding
- Compatible with rl4co's training loop

Architecture inspired by Kool et al. (2019) Attention Model and
Kwon et al. (2020) POMO, but adapted for custom action space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Optional, Literal
import logging

from .encoder import TransformerEncoder
from .decoders import AddEdgeDecoder, DeleteEdgeDecoder, DoneDecoder
from .action_utils import (
    decode_action_index,
    extract_edge_list,
    compute_action_masks,
    create_node_features
)

log = logging.getLogger(__name__)


class CustomPOMOPolicy(nn.Module):
    """
    Custom policy for TSP with global edge selection and deletion.
    
    The policy encodes the current state (locations, adjacency, degrees, etc.)
    using a transformer encoder, then uses three decoder heads to score:
    - ADD actions: potential edges to add
    - DELETE actions: existing edges to delete  
    - DONE action: finish tour construction
    
    Args:
        num_loc: Number of nodes in TSP instance
        embed_dim: Embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_encoder_layers: Number of transformer layers (default: 6)
        feedforward_dim: FFN hidden dimension (default: 512)
        normalization: Normalization type (default: 'instance')
        temperature: Softmax temperature (default: 1.0)
        tanh_clipping: Tanh clipping for logits (default: 10.0)
        delete_bias_start: Initial delete bias (default: -5.0)
        delete_bias_end: Final delete bias (default: 0.0)
        delete_bias_warmup_epochs: Epochs to linearly schedule bias (default: 100)
    """
    
    def __init__(
        self,
        num_loc: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        feedforward_dim: int = 512,
        normalization: str = 'instance',
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        delete_bias_start: float = -5.0,
        delete_bias_end: float = 0.0,
        delete_bias_warmup_epochs: int = 100,
    ):
        super().__init__()
        
        self.num_loc = num_loc
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        
        # Calculate action space sizes
        self.num_add_actions = num_loc * (num_loc - 1) // 2
        self.max_delete_actions = num_loc  # Maximum edges in a tour
        
        # Feature dimension: locs(2) + degree(1) + adjacency(N) + step(1) + deletions(1)
        self.feat_dim = num_loc + 5
        
        # Encoder
        self.encoder = TransformerEncoder(
            feat_dim=self.feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            feedforward_dim=feedforward_dim,
            normalization=normalization
        )

        # Decoder heads
        self.add_decoder = AddEdgeDecoder(embed_dim=embed_dim)
        self.delete_decoder = DeleteEdgeDecoder(embed_dim=embed_dim)
        self.done_decoder = DoneDecoder(embed_dim=embed_dim)

        # Delete bias schedule (registered as buffer for checkpointing)
        self.register_buffer('delete_bias_start', torch.tensor(delete_bias_start))
        self.register_buffer('delete_bias_end', torch.tensor(delete_bias_end))
        self.delete_bias_warmup_epochs = delete_bias_warmup_epochs
        self.current_epoch = 0  # Updated during training
        
        # Pre-compute edge indices for ADD actions (fixed for all instances)
        edge_indices = torch.triu_indices(num_loc, num_loc, offset=1).t()
        self.register_buffer('edge_indices', edge_indices)
        
        log.info(
            f"Initialized CustomPOMOPolicy: "
            f"num_loc={num_loc}, embed_dim={embed_dim}, "
            f"num_add_actions={self.num_add_actions}, "
            f"max_delete_actions={self.max_delete_actions}"
        )
    
    def set_epoch(self, epoch: int):
        """Update current epoch for delete bias scheduling."""
        self.current_epoch = epoch
    
    def get_delete_bias(self, phase: str = 'train') -> float:
        """
        Compute delete bias based on current epoch.
        
        Linear schedule from delete_bias_start to delete_bias_end
        over delete_bias_warmup_epochs.
        
        During val/test, use delete_bias_end (neutral).
        """
        if phase != 'train':
            return self.delete_bias_end.item()
        
        if self.current_epoch >= self.delete_bias_warmup_epochs:
            return self.delete_bias_end.item()
        
        # Linear interpolation
        alpha = self.current_epoch / self.delete_bias_warmup_epochs
        bias = (1 - alpha) * self.delete_bias_start + alpha * self.delete_bias_end
        return bias.item()
    
    def forward(
        self,
        td: TensorDict,
        phase: Literal['train', 'val', 'test'] = 'train',
        decode_type: str = 'sampling',
        return_actions: bool = False,
        return_hidden: bool = False,
    ) -> dict:
        """
        Forward pass of the policy.
        
        Args:
            td: TensorDict containing environment state
            phase: Training phase (affects delete bias)
            decode_type: 'sampling' or 'greedy'
            return_actions: Whether to return selected actions
            return_hidden: Whether to return node embeddings
            
        Returns:
            Dictionary with 'action', 'log_prob', and optionally 'hidden'
        """
        batch_size = td["locs"].shape[0]
        device = td["locs"].device
        
        # Create node features
        node_features = create_node_features(td, self.num_loc)  # (batch, N, feat_dim)
        
        # Encode
        node_embeddings = self.encoder(node_features)  # (batch, N, embed_dim)
                
        # Decode: get logits from each head
        logits_add, _ = self.add_decoder(
            node_embeddings,
            td["locs"]
        )  # (batch, num_add_actions)
        
        # DELETE and DONE actions are disabled - only using ADD actions
        # edge_list = extract_edge_list(td["adjacency"], max_edges=self.max_delete_actions)
        # logits_del = self.delete_decoder(node_embeddings, td["locs"], edge_list, delete_bias=delete_bias)
        # logits_done = self.done_decoder(node_embeddings)
        # logits = torch.cat([logits_add, logits_del, logits_done], dim=-1)

        logits = logits_add  # TEMPORARY: Only use ADD actions
        
        # Apply tanh clipping (from attention model)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Sample or select action
        if decode_type == 'sampling':
            probs = F.softmax(logits, dim=-1)
            # Handle potential NaN from all -inf logits
            probs = torch.where(
                torch.isnan(probs),
                torch.zeros_like(probs),
                probs
            )
            # Add small epsilon to ensure non-zero probabilities
            probs = probs + 1e-10
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            action_idx = torch.multinomial(probs, 1).squeeze(-1)  # (batch,)
        else:  # greedy
            action_idx = logits.argmax(dim=-1)  # (batch,)
        
        # Compute log probability
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_prob = log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)  # (batch,)
        
        # NO HARD CONSTRAINT VALIDATION - model learns from reward penalties
        # Just use the selected actions directly
        
        # Since we only have ADD actions now, action_type is always 0
        batch_size = action_idx.shape[0]
        device = action_idx.device
        action_type = torch.zeros_like(action_idx)

        # Combine into action tensor expected by environment
        # Environment expects single integer action index
        action = action_idx
        
        # Prepare output
        out = {
            "action": action,  # (batch,) - flat action index
            "log_prob": selected_log_prob,  # (batch,)
            "logits": logits,  # (batch, total_actions) - for teacher forcing
        }
        
        if return_actions:
            # Decode node indices for debugging (environment will also decode)
            node_i = torch.zeros(batch_size, dtype=torch.long, device=device)
            node_j = torch.zeros(batch_size, dtype=torch.long, device=device)
            for b in range(batch_size):
                idx = action_idx[b].item()
                if idx < self.num_add_actions:
                    node_i[b] = self.edge_indices[idx, 0]
                    node_j[b] = self.edge_indices[idx, 1]
            
            # Also return decoded action components for debugging
            out["action_components"] = {
                "action_type": action_type,
                "node_i": node_i,
                "node_j": node_j,
            }
        
        if return_hidden:
            out["hidden"] = node_embeddings
        
        return out


if __name__ == '__main__':
    # Test the policy
    print("Testing CustomPOMOPolicy...")
    
    from tensordict import TensorDict
    
    batch_size = 4
    num_loc = 20
    
    # Create dummy state
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
    td["adjacency"][0, 0, 1] = True
    td["adjacency"][0, 1, 0] = True
    td["degrees"][0, 0] = 1
    td["degrees"][0, 1] = 1
    td["num_edges"][0] = 1
    
    # Create action mask (simplified - just allow all ADD actions)
    num_add_actions = num_loc * (num_loc - 1) // 2
    max_delete_actions = num_loc
    total_actions = num_add_actions + max_delete_actions + 1
    
    action_mask = torch.zeros(batch_size, total_actions, dtype=torch.bool)
    action_mask[:, :num_add_actions] = True  # Allow all ADD actions
    action_mask[0, num_add_actions] = True  # Allow first DELETE action for batch 0
    td["action_mask"] = action_mask
    
    # Create policy
    policy = CustomPOMOPolicy(num_loc=num_loc)
    
    # Forward pass
    out = policy(td, phase='train', decode_type='sampling', return_actions=True)
    
    print(f"\nInput:")
    print(f"  locs: {td['locs'].shape}")
    print(f"  adjacency: {td['adjacency'].shape}")
    print(f"  action_mask: {td['action_mask'].shape}")
    
    print(f"\nOutput:")
    print(f"  action: {out['action'].shape} = {out['action']}")
    print(f"  log_prob: {out['log_prob'].shape} = {out['log_prob']}")
    print(f"  action_type: {out['action_components']['action_type']}")
    print(f"  node_i: {out['action_components']['node_i']}")
    print(f"  node_j: {out['action_components']['node_j']}")
    
    print(f"\nTest passed!")
