"""
Explainability modules: Grad-CAM for visual explanations and audio saliency.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Grad-CAM for visualizing important regions in video frames.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Model to explain
            target_layer: Target layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate class activation map.
        
        Args:
            input_tensor: Input tensor (batch_size, num_frames, C, H, W)
            class_idx: Class index to generate CAM for (None for predicted class)
            
        Returns:
            CAM heatmap (num_frames, H, W)
        """
        self.model.eval()
        
        # Forward pass - need to provide all inputs
        # Create dummy inputs for missing modalities
        batch_size, num_frames = input_tensor.shape[:2]
        dummy_spectrogram = torch.zeros(batch_size, 1, 128, 100, device=input_tensor.device)
        dummy_mouths = torch.zeros(batch_size, num_frames, 3, 64, 64, device=input_tensor.device)
        
        output = self.model(
            frames=input_tensor,
            spectrogram=dummy_spectrogram,
            mouth_frames=dummy_mouths
        )
        
        if class_idx is None:
            class_idx = torch.argmax(output['logits'], dim=1)[0].item()
        
        # Backward pass
        self.model.zero_grad()
        output['logits'][0, class_idx].backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # (num_frames, channels, H, W)
        activations = self.activations[0]  # (num_frames, channels, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (num_frames, channels, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=False)  # (num_frames, H, W)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        for i in range(cam.shape[0]):
            cam[i] = (cam[i] - cam[i].min()) / (cam[i].max() - cam[i].min() + 1e-8)
        
        return cam
    
    def overlay_cam(
        self,
        frames: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay CAM on frames.
        
        Args:
            frames: Original frames (num_frames, H, W, C)
            cam: CAM heatmap (num_frames, H, W)
            alpha: Blending factor
            
        Returns:
            Overlaid frames (num_frames, H, W, C)
        """
        overlaid = frames.copy()
        
        for i in range(len(frames)):
            # Resize CAM to match frame size
            cam_resized = cv2.resize(cam[i], (frames.shape[2], frames.shape[1]))
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlaid[i] = (alpha * heatmap + (1 - alpha) * frames[i]).astype(np.uint8)
        
        return overlaid


class AudioSaliency:
    """
    Audio saliency maps using input gradients.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize audio saliency.
        
        Args:
            model: Model to explain
        """
        self.model = model
    
    def generate_saliency(
        self,
        spectrogram: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate saliency map for spectrogram.
        
        Args:
            spectrogram: Input spectrogram (batch_size, 1, n_mels, time_frames)
            class_idx: Class index to generate saliency for
        
        Returns:
            Saliency map (n_mels, time_frames)
        """
        self.model.eval()
        
        # Enable gradient computation for input
        spectrogram = spectrogram.clone().detach().requires_grad_(True)
        
        # Forward pass through audio stream
        audio_output = self.model.audio_stream(spectrogram)
        
        if class_idx is None:
            # Create dummy inputs for full model forward to get class
            batch_size = spectrogram.shape[0]
            num_frames = 64
            dummy_frames = torch.zeros(batch_size, num_frames, 3, 224, 224, device=spectrogram.device)
            dummy_mouths = torch.zeros(batch_size, num_frames, 3, 64, 64, device=spectrogram.device)
            
            with torch.no_grad():
                full_output = self.model(
                    frames=dummy_frames,
                    spectrogram=spectrogram.detach(),
                    mouth_frames=dummy_mouths
                )
                class_idx = torch.argmax(full_output['logits'], dim=1)[0].item()
        
        # Backward pass on audio stream output
        self.model.zero_grad()
        audio_output.sum().backward()
        
        # Get gradients
        saliency = spectrogram.grad.abs()
        saliency = saliency[0, 0].cpu().numpy()  # (n_mels, time_frames)
        
        # Normalize
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        else:
            saliency = np.zeros_like(saliency)
        
        return saliency
    
    def visualize_saliency(
        self,
        spectrogram: np.ndarray,
        saliency: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """
        Visualize saliency map overlaid on spectrogram.
        
        Args:
            spectrogram: Original spectrogram (n_mels, time_frames)
            saliency: Saliency map (n_mels, time_frames)
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original spectrogram
        axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Original Mel-Spectrogram')
        axes[0].set_xlabel('Time Frames')
        axes[0].set_ylabel('Mel Frequency Bins')
        
        # Saliency map
        im = axes[1].imshow(saliency, aspect='auto', origin='lower', cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Mel Frequency Bins')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class IntegratedGradients:
    """
    Integrated gradients for audio saliency (more robust than simple gradients).
    """
    
    def __init__(self, model: torch.nn.Module, steps: int = 50):
        """
        Initialize integrated gradients.
        
        Args:
            model: Model to explain
            steps: Number of integration steps
        """
        self.model = model
        self.steps = steps
    
    def generate_ig(
        self,
        spectrogram: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate integrated gradients.
        
        Args:
            spectrogram: Input spectrogram
            baseline: Baseline (zero or mean)
            class_idx: Class index
        
        Returns:
            Integrated gradients (n_mels, time_frames)
        """
        if baseline is None:
            baseline = torch.zeros_like(spectrogram)
        
        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, self.steps)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (spectrogram - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model.audio_stream(interpolated)
            
            # Backward pass
            self.model.zero_grad()
            if class_idx is None:
                output.sum().backward()
            else:
                output[0, class_idx].backward()
            
            gradients.append(interpolated.grad)
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients
        ig = (spectrogram - baseline) * avg_gradients
        ig = ig[0, 0].abs().cpu().numpy()
        
        # Normalize
        ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
        
        return ig

