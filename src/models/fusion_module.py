"""
Fusion module: Cross-modal attention and transformer-based fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention block for fusing different modalities.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal attention.
        
        Args:
            query: Query tensor (batch_size, embedding_dim)
            key: Key tensor (batch_size, embedding_dim)
            value: Value tensor (batch_size, embedding_dim)
            
        Returns:
            Attended features (batch_size, embedding_dim)
        """
        batch_size = query.shape[0]
        residual = query
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.view(batch_size, self.embedding_dim)
        
        # Output projection and residual
        output = self.out_proj(attn_output)
        output = self.layer_norm(output + residual)
        
        return output


class FusionModule(nn.Module):
    """
    Fusion module: Combines video, audio, and lip-sync embeddings using attention.
    """
    
    def __init__(
        self,
        video_embedding_dim: int = 512,
        audio_embedding_dim: int = 512,
        lip_embedding_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        """
        Initialize fusion module.
        
        Args:
            video_embedding_dim: Dimension of video embeddings
            audio_embedding_dim: Dimension of audio embeddings
            lip_embedding_dim: Dimension of lip-sync embeddings
            fusion_dim: Dimension after fusion
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            use_cross_attention: Use cross-modal attention (True) or simple concatenation (False)
        """
        super().__init__()
        
        # Project all embeddings to same dimension
        self.video_proj = nn.Linear(video_embedding_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_embedding_dim, fusion_dim)
        self.lip_proj = nn.Linear(lip_embedding_dim, fusion_dim)
        
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            # Cross-modal attention blocks
            self.video_audio_attn = CrossModalAttention(fusion_dim, num_heads, dropout)
            self.audio_video_attn = CrossModalAttention(fusion_dim, num_heads, dropout)
            self.lip_video_attn = CrossModalAttention(fusion_dim, num_heads, dropout)
            
            # Transformer decoder for final fusion
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=False
            )
            self.fusion_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            
            # Learnable query for fusion
            self.fusion_query = nn.Parameter(torch.randn(1, fusion_dim))
        else:
            # Simple concatenation + MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim)
            )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.fusion_dim = fusion_dim
    
    def forward(
        self,
        video_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        lip_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-modal embeddings.
        
        Args:
            video_embedding: Video embedding (batch_size, video_embedding_dim)
            audio_embedding: Audio embedding (batch_size, audio_embedding_dim)
            lip_embedding: Lip-sync embedding (batch_size, lip_embedding_dim)
            
        Returns:
            Fused embedding (batch_size, fusion_dim)
        """
        # Project to common dimension
        video_feat = self.video_proj(video_embedding)
        audio_feat = self.audio_proj(audio_embedding)
        lip_feat = self.lip_proj(lip_embedding)
        
        if self.use_cross_attention:
            # Cross-modal attention
            video_attended = self.video_audio_attn(video_feat, audio_feat, audio_feat)
            audio_attended = self.audio_video_attn(audio_feat, video_feat, video_feat)
            lip_attended = self.lip_video_attn(lip_feat, video_feat, video_feat)
            
            # Stack attended features
            attended_features = torch.stack([video_attended, audio_attended, lip_attended], dim=0)
            # Shape: (3, batch_size, fusion_dim)
            
            # Transformer decoder fusion
            fusion_query = self.fusion_query.unsqueeze(1).repeat(1, video_feat.shape[0], 1)
            # Shape: (1, batch_size, fusion_dim)
            
            fused = self.fusion_transformer(fusion_query, attended_features)
            fused = fused.squeeze(0)  # (batch_size, fusion_dim)
        else:
            # Simple concatenation
            concatenated = torch.cat([video_feat, audio_feat, lip_feat], dim=1)
            fused = self.fusion_mlp(concatenated)
        
        # Final normalization
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

