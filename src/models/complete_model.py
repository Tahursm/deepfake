"""
Complete multi-stream deepfake detection model.
Combines video stream, audio stream, lip-sync module, and fusion.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .video_stream import VideoStream
from .audio_stream import AudioStream
from .lip_sync import LipSyncModule
from .fusion_module import FusionModule


class DeepfakeDetectionModel(nn.Module):
    """
    Complete multi-stream deepfake detection model.
    """
    
    def __init__(
        self,
        # Video stream config
        video_backbone: str = 'resnet50',
        video_embedding_dim: int = 512,
        num_frames: int = 64,
        # Audio stream config
        n_mels: int = 128,
        audio_embedding_dim: int = 512,
        # Lip-sync config
        lip_embedding_dim: int = 256,
        # Fusion config
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        # Classification
        num_classes: int = 2,  # Real vs Fake
        use_auxiliary_loss: bool = True
    ):
        """
        Initialize complete model.
        
        Args:
            video_backbone: CNN backbone for video ('resnet50', 'efficientnet_b0')
            video_embedding_dim: Dimension of video embeddings
            num_frames: Number of frames per clip
            n_mels: Number of Mel frequency bins
            audio_embedding_dim: Dimension of audio embeddings
            lip_embedding_dim: Dimension of lip-sync embeddings
            fusion_dim: Dimension after fusion
            num_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
            dropout: Dropout rate
            use_cross_attention: Use cross-modal attention
            num_classes: Number of output classes (2 for binary)
            use_auxiliary_loss: Use auxiliary lip-sync loss
        """
        super().__init__()
        
        # Video stream
        self.video_stream = VideoStream(
            backbone=video_backbone,
            pretrained=True,
            embedding_dim=video_embedding_dim,
            num_frames=num_frames,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # Audio stream
        self.audio_stream = AudioStream(
            n_mels=n_mels,
            embedding_dim=audio_embedding_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            use_transformer=True
        )
        
        # Lip-sync module
        self.lip_sync_module = LipSyncModule(
            input_size=64,
            embedding_dim=lip_embedding_dim,
            num_frames=num_frames,
            num_heads=num_heads // 2,
            num_layers=2,
            dropout=dropout
        )
        
        # Fusion module
        self.fusion_module = FusionModule(
            video_embedding_dim=video_embedding_dim,
            audio_embedding_dim=audio_embedding_dim,
            lip_embedding_dim=lip_embedding_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
            use_cross_attention=use_cross_attention
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Projection layer for audio embedding to match lip embedding dimension (for sync score)
        if audio_embedding_dim != lip_embedding_dim:
            self.audio_to_lip_proj = nn.Linear(audio_embedding_dim, lip_embedding_dim)
        else:
            self.audio_to_lip_proj = nn.Identity()
        
        self.use_auxiliary_loss = use_auxiliary_loss
        self.num_classes = num_classes
        self.audio_embedding_dim = audio_embedding_dim
        self.lip_embedding_dim = lip_embedding_dim
    
    def forward(
        self,
        frames: torch.Tensor,
        spectrogram: torch.Tensor,
        mouth_frames: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: Video frames (batch_size, num_frames, C, H, W)
            spectrogram: Mel-spectrogram (batch_size, n_mels, time_frames) or (batch_size, 1, n_mels, time_frames)
            mouth_frames: Mouth region frames (batch_size, num_frames, C, H, W)
            return_embeddings: Return intermediate embeddings
            
        Returns:
            Dictionary with:
                - 'logits': Classification logits (batch_size, num_classes)
                - 'probs': Class probabilities (batch_size, num_classes)
                - 'video_embedding': Video embedding (if return_embeddings)
                - 'audio_embedding': Audio embedding (if return_embeddings)
                - 'lip_embedding': Lip embedding (if return_embeddings)
                - 'fused_embedding': Fused embedding (if return_embeddings)
        """
        # Extract embeddings from each stream
        video_embedding = self.video_stream(frames)
        audio_embedding = self.audio_stream(spectrogram)
        lip_embedding = self.lip_sync_module(mouth_frames)
        
        # Fuse embeddings
        fused_embedding = self.fusion_module(
            video_embedding,
            audio_embedding,
            lip_embedding
        )
        
        # Classification
        logits = self.classifier(fused_embedding)
        probs = torch.softmax(logits, dim=1)
        
        output = {
            'logits': logits,
            'probs': probs
        }
        
        if return_embeddings:
            output.update({
                'video_embedding': video_embedding,
                'audio_embedding': audio_embedding,
                'lip_embedding': lip_embedding,
                'fused_embedding': fused_embedding
            })
        
        return output
    
    def compute_auxiliary_loss(
        self,
        lip_embedding: torch.Tensor,
        audio_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary lip-sync loss.
        
        Args:
            lip_embedding: Lip motion embedding
            audio_embedding: Audio embedding
            
        Returns:
            Lip-sync loss (scalar)
        """
        if not self.use_auxiliary_loss:
            return torch.tensor(0.0, device=lip_embedding.device)
        
        # Project audio to lip embedding dimension for sync score
        audio_embedding_proj = self.audio_to_lip_proj(audio_embedding)
        sync_score = self.lip_sync_module.compute_sync_score(
            lip_embedding,
            audio_embedding_proj
        )
        
        # Encourage high sync score (this is a simplified loss)
        # In practice, you might want to use contrastive loss or correlation loss
        sync_loss = -sync_score.mean()  # Negative because we want to maximize sync
        
        return sync_loss

