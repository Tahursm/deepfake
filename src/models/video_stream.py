"""
Video stream model: CNN backbone + Transformer for temporal modeling.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VideoStream(nn.Module):
    """
    Video stream: CNN backbone + Transformer encoder for temporal modeling.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 512,
        num_frames: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize video stream model.
        
        Args:
            backbone: CNN backbone ('resnet50', 'efficientnet_b0')
            pretrained: Use pretrained weights
            embedding_dim: Dimension of frame embeddings
            num_frames: Maximum number of frames per clip
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer layers
            dropout: Dropout rate
            freeze_backbone: Freeze CNN backbone weights
        """
        super().__init__()
        
        # CNN backbone for per-frame feature extraction
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection to embedding dimension
        self.projection = nn.Linear(backbone_dim, embedding_dim)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=num_frames, dropout=dropout)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.embedding_dim = embedding_dim
        self.num_frames = num_frames
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            frames: Input frames (batch_size, num_frames, C, H, W)
            
        Returns:
            Video embedding (batch_size, embedding_dim)
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # Extract features for each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i, :, :, :]
            # Backbone expects (batch, C, H, W)
            feat = self.backbone(frame)
            # Flatten spatial dimensions
            feat = feat.view(batch_size, -1)
            frame_features.append(feat)
        
        # Stack: (num_frames, batch_size, feat_dim)
        frame_features = torch.stack(frame_features, dim=0)
        
        # Project to embedding dimension
        frame_embeddings = self.projection(frame_features)  # (num_frames, batch_size, embedding_dim)
        
        # Add positional encoding
        frame_embeddings = self.pos_encoder(frame_embeddings)
        
        # Transformer encoding
        # Transformer expects (seq_len, batch, features)
        encoded = self.transformer(frame_embeddings)  # (num_frames, batch_size, embedding_dim)
        
        # Global average pooling over temporal dimension
        video_embedding = encoded.mean(dim=0)  # (batch_size, embedding_dim)
        
        # Final projection and normalization
        video_embedding = self.output_proj(video_embedding)
        video_embedding = self.layer_norm(video_embedding)
        
        return video_embedding


class VideoStreamLightweight(nn.Module):
    """
    Lightweight video stream using 3D CNN instead of transformer (faster, less memory).
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 512,
        num_frames: int = 64
    ):
        super().__init__()
        
        # Use 2D CNN + temporal pooling
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Temporal modeling with 1D convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(backbone_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, C, H, W = frames.shape
        
        # Extract features
        frame_features = []
        for i in range(num_frames):
            feat = self.backbone(frames[:, i])
            feat = feat.view(batch_size, -1)
            frame_features.append(feat)
        
        # Stack: (batch_size, feat_dim, num_frames)
        frame_features = torch.stack(frame_features, dim=2)
        
        # Temporal convolution
        temporal_feat = self.temporal_conv(frame_features)  # (batch_size, embedding_dim, 1)
        temporal_feat = temporal_feat.squeeze(2)  # (batch_size, embedding_dim)
        
        # Final projection
        video_embedding = self.projection(temporal_feat)
        
        return video_embedding

