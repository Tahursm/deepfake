"""
Audio stream model: CNN + Transformer for spectrogram processing.
"""

import torch
import torch.nn as nn
from typing import Optional


class AudioStream(nn.Module):
    """
    Audio stream: CNN + Transformer for Mel-spectrogram processing.
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_transformer: bool = True
    ):
        """
        Initialize audio stream model.
        
        Args:
            n_mels: Number of Mel frequency bins
            embedding_dim: Dimension of audio embeddings
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer layers
            dropout: Dropout rate
            use_transformer: Use transformer (True) or just CNN (False)
        """
        super().__init__()
        
        # CNN feature extractor for spectrogram
        self.cnn = nn.Sequential(
            # Input: (batch, 1, n_mels, time_frames)
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Pool frequency dimension
        )
        
        # Project CNN features to embedding dimension
        self.cnn_proj = nn.Linear(256, embedding_dim)
        
        self.use_transformer = use_transformer
        
        if use_transformer:
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
            self.pos_encoder = nn.Parameter(torch.randn(1000, embedding_dim))
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            spectrogram: Mel-spectrogram (batch_size, n_mels, time_frames)
                        or (batch_size, 1, n_mels, time_frames)
            
        Returns:
            Audio embedding (batch_size, embedding_dim)
        """
        # Ensure input has channel dimension
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)  # (batch, 1, n_mels, time)
        
        batch_size = spectrogram.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn(spectrogram)  # (batch, 256, 1, time)
        cnn_features = cnn_features.squeeze(2)  # (batch, 256, time)
        cnn_features = cnn_features.permute(0, 2, 1)  # (batch, time, 256)
        
        # Project to embedding dimension
        audio_embeddings = self.cnn_proj(cnn_features)  # (batch, time, embedding_dim)
        
        if self.use_transformer:
            # Transformer expects (seq_len, batch, features)
            audio_embeddings = audio_embeddings.permute(1, 0, 2)  # (time, batch, embedding_dim)
            
            # Add positional encoding
            seq_len = audio_embeddings.shape[0]
            audio_embeddings = audio_embeddings + self.pos_encoder[:seq_len].unsqueeze(1)
            
            # Transformer encoding
            encoded = self.transformer(audio_embeddings)  # (time, batch, embedding_dim)
            
            # Global average pooling
            audio_embedding = encoded.mean(dim=0)  # (batch, embedding_dim)
        else:
            # Just use mean pooling
            audio_embedding = audio_embeddings.mean(dim=1)  # (batch, embedding_dim)
        
        # Final projection and normalization
        audio_embedding = self.output_proj(audio_embedding)
        audio_embedding = self.layer_norm(audio_embedding)
        
        return audio_embedding


class AudioStreamPretrained(nn.Module):
    """
    Audio stream using pretrained audio embeddings (Wav2Vec, HuBERT, etc.).
    This is a placeholder - would need to integrate with transformers library.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = 'wav2vec2-base'
    ):
        super().__init__()
        # Placeholder - would integrate with transformers library
        # from transformers import Wav2Vec2Model
        # self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        # self.projection = nn.Linear(self.backbone.config.hidden_size, embedding_dim)
        
        # Fallback to simple CNN if pretrained not available
        self.audio_stream = AudioStream(embedding_dim=embedding_dim, use_transformer=False)
        self.embedding_dim = embedding_dim
    
    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_waveform: Raw audio waveform (batch_size, samples)
        """
        # Would use pretrained model here
        # For now, use fallback
        return self.audio_stream(audio_waveform)

