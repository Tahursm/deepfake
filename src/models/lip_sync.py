"""
Lip-sync module: Analyzes lip motion and correlates with audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LipSyncModule(nn.Module):
    """
    Lip-sync module: Extracts lip motion features and computes correlation with audio.
    """
    
    def __init__(
        self,
        input_size: int = 64,  # Mouth region size (64x64)
        embedding_dim: int = 256,
        num_frames: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize lip-sync module.
        
        Args:
            input_size: Size of mouth region input (assumed square)
            embedding_dim: Dimension of lip-sync embedding
            num_frames: Maximum number of frames
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # CNN for mouth region feature extraction
        self.mouth_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Project CNN features
        cnn_feat_dim = 128
        self.cnn_proj = nn.Linear(cnn_feat_dim, embedding_dim)
        
        # Temporal transformer for lip motion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(num_frames, embedding_dim))
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, mouth_frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mouth_frames: Mouth region frames (batch_size, num_frames, C, H, W)
            
        Returns:
            Lip-sync embedding (batch_size, embedding_dim)
        """
        batch_size, num_frames, C, H, W = mouth_frames.shape
        
        # Extract features for each frame
        frame_features = []
        for i in range(num_frames):
            frame = mouth_frames[:, i, :, :, :]
            feat = self.mouth_cnn(frame)
            feat = feat.view(batch_size, -1)
            frame_features.append(feat)
        
        # Stack: (num_frames, batch_size, feat_dim)
        frame_features = torch.stack(frame_features, dim=0)
        
        # Project to embedding dimension
        lip_embeddings = self.cnn_proj(frame_features)  # (num_frames, batch_size, embedding_dim)
        
        # Add positional encoding
        seq_len = lip_embeddings.shape[0]
        lip_embeddings = lip_embeddings + self.pos_encoder[:seq_len].unsqueeze(1)
        
        # Temporal transformer
        encoded = self.temporal_transformer(lip_embeddings)  # (num_frames, batch_size, embedding_dim)
        
        # Global average pooling
        lip_embedding = encoded.mean(dim=0)  # (batch_size, embedding_dim)
        
        # Final projection
        lip_embedding = self.output_proj(lip_embedding)
        lip_embedding = self.layer_norm(lip_embedding)
        
        return lip_embedding
    
    def compute_sync_score(
        self,
        lip_embedding: torch.Tensor,
        audio_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute lip-audio synchronization score.
        
        Args:
            lip_embedding: Lip motion embedding (batch_size, embedding_dim)
            audio_embedding: Audio embedding (batch_size, embedding_dim)
            
        Returns:
            Sync score (batch_size, 1) - higher means more synchronized
        """
        # Cosine similarity
        lip_norm = F.normalize(lip_embedding, p=2, dim=1)
        audio_norm = F.normalize(audio_embedding, p=2, dim=1)
        sync_score = (lip_norm * audio_norm).sum(dim=1, keepdim=True)
        
        return sync_score

