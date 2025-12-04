"""
Transformer Encoder for Custom TSP Environment

This encoder converts node features into embeddings that are then used by the
3-head decoder to predict ADD/DELETE/DONE actions. Architecture inspired by
rl4co's AttentionModelEncoder but simplified for this specific task.

Reference: Kool et al. (2019) Attention Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head self-attention and feed-forward network."""
    
    def __init__(self, embed_dim=128, num_heads=8, feedforward_dim=512, normalization='instance'):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Multi-head attention
        self.attn_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        
        # Normalization
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm1d(embed_dim)
            self.norm2 = nn.BatchNorm1d(embed_dim)
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm1d(embed_dim, affine=True)
            self.norm2 = nn.InstanceNorm1d(embed_dim, affine=True)
        elif normalization == 'layer':
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        self.normalization = normalization
        
    def forward(self, h):
        """
        Args:
            h: Node embeddings (batch, N, embed_dim)
            
        Returns:
            Updated node embeddings (batch, N, embed_dim)
        """
        batch, N, embed_dim = h.shape
        
        # Multi-head self-attention with residual
        h_attn = self._multi_head_attention(h)
        h = h + h_attn
        
        # Normalize (batch_norm/instance_norm expect (batch, features, seq))
        if self.normalization in ['batch', 'instance']:
            h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        else:  # layer norm
            h = self.norm1(h)
        
        # Feed-forward with residual
        h_ff = self.ff(h)
        h = h + h_ff
        
        # Normalize
        if self.normalization in ['batch', 'instance']:
            h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        else:
            h = self.norm2(h)
        
        return h
    
    def _multi_head_attention(self, h):
        """Multi-head self-attention."""
        batch, N, embed_dim = h.shape
        
        # Compute Q, K, V
        Q = self.attn_q(h)  # (batch, N, embed_dim)
        K = self.attn_k(h)  # (batch, N, embed_dim)
        V = self.attn_v(h)  # (batch, N, embed_dim)
        
        # Reshape for multi-head: (batch, N, num_heads, head_dim) -> (batch, num_heads, N, head_dim)
        Q = Q.view(batch, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, num_heads, N, N)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, N, head_dim)
        
        # Concatenate heads: (batch, num_heads, N, head_dim) -> (batch, N, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, N, embed_dim)
        
        # Output projection
        return self.attn_out(attn_output)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for custom TSP environment.
    
    Encodes node features into embeddings used by the decoder heads.
    
    Args:
        feat_dim: Dimension of input node features
        embed_dim: Dimension of node embeddings (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        feedforward_dim: Hidden dimension of FFN (default: 512)
        normalization: Type of normalization ('instance', 'batch', 'layer')
    """
    
    def __init__(
        self,
        feat_dim=20,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        feedforward_dim=512,
        normalization='instance'
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initial embedding projection
        self.init_embed = nn.Linear(feat_dim, embed_dim)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                normalization=normalization
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, node_features):
        """
        Encode node features into embeddings.
        
        Args:
            node_features: Input node features (batch, N, feat_dim)
            
        Returns:
            node_embeddings: Encoded node embeddings (batch, N, embed_dim)
        """
        # Initial projection
        h = self.init_embed(node_features)  # (batch, N, embed_dim)
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)
        
        return h


if __name__ == '__main__':
    # Test the encoder
    batch_size = 4
    num_nodes = 20
    feat_dim = 10
    
    # Create dummy input
    node_features = torch.randn(batch_size, num_nodes, feat_dim)
    
    # Create encoder
    encoder = TransformerEncoder(
        feat_dim=feat_dim,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        feedforward_dim=512,
        normalization='instance'
    )
    
    # Forward pass
    embeddings = encoder(node_features)
    
    print(f"Input shape: {node_features.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: (batch={batch_size}, N={num_nodes}, embed_dim=128)")
    print(f"Test passed: {embeddings.shape == (batch_size, num_nodes, 128)}")
