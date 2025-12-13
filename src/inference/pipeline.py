"""
Inference pipeline for deepfake detection.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2

from ..models.complete_model import DeepfakeDetectionModel
from ..preprocessing.face_extractor import FaceExtractor
from ..preprocessing.audio_extract import AudioExtractor
from ..utils.explainability import GradCAM, AudioSaliency


class DeepfakeInferencePipeline:
    """
    Complete inference pipeline for deepfake detection.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        num_frames: int = 64,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on
            num_frames: Number of frames to process
            frame_size: Size of face frames
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        model_config = config.get('model', {})
        self.model = DeepfakeDetectionModel(
            video_backbone=model_config.get('video_backbone', 'resnet50'),
            video_embedding_dim=model_config.get('video_embedding_dim', 512),
            num_frames=num_frames,
            n_mels=model_config.get('n_mels', 128),
            audio_embedding_dim=model_config.get('audio_embedding_dim', 512),
            fusion_dim=model_config.get('fusion_dim', 512)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize extractors
        self.face_extractor = FaceExtractor(face_size=frame_size)
        self.audio_extractor = AudioExtractor()
        
        # Initialize explainability
        # Get target layer for Grad-CAM (e.g., last conv layer of video backbone)
        target_layer = None
        for module in self.model.video_stream.backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer:
            self.gradcam = GradCAM(self.model, target_layer)
        else:
            self.gradcam = None
        
        self.audio_saliency = AudioSaliency(self.model)
    
    def preprocess_video(self, video_path: str) -> Dict[str, np.ndarray]:
        """
        Preprocess video for inference.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with preprocessed data
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Extract faces
        face_sequence = self.face_extractor.extract_face_sequence(
            frames,
            fps=25.0,
            max_frames=self.num_frames
        )
        
        if len(face_sequence) == 0:
            raise ValueError(f"No faces extracted from {video_path}")
        
        # Extract mouth regions
        mouth_sequence = []
        for frame in frames[:len(face_sequence)]:
            mouth = self.face_extractor.extract_mouth_region(frame)
            if mouth is None:
                mouth = np.zeros((64, 64, 3), dtype=np.uint8)
            mouth_sequence.append(mouth)
        
        # Extract audio
        audio = self.audio_extractor.extract_audio_from_video(video_path)
        
        # Synchronize audio
        num_frames_actual = len(face_sequence)
        audio_segment = self.audio_extractor.synchronize_audio_with_frames(
            audio,
            num_frames_actual,
            fps=25.0
        )
        
        # Compute spectrogram
        spectrogram = self.audio_extractor.compute_mel_spectrogram(audio_segment)
        
        # Pad or truncate
        face_sequence = self._pad_or_truncate(face_sequence, self.num_frames)
        mouth_sequence = self._pad_or_truncate(mouth_sequence, self.num_frames)
        
        return {
            'faces': np.stack(face_sequence, axis=0),
            'mouths': np.stack(mouth_sequence, axis=0),
            'spectrogram': spectrogram
        }
    
    def _pad_or_truncate(self, sequence: list, target_len: int) -> list:
        """Pad or truncate sequence to target length."""
        if len(sequence) > target_len:
            indices = np.linspace(0, len(sequence) - 1, target_len, dtype=int)
            return [sequence[i] for i in indices]
        elif len(sequence) < target_len:
            padding = [sequence[-1]] * (target_len - len(sequence))
            return sequence + padding
        return sequence
    
    def predict(
        self,
        video_path: str,
        return_explanations: bool = False
    ) -> Dict:
        """
        Predict if video is deepfake.
        
        Args:
            video_path: Path to video file
            return_explanations: Return explainability visualizations
        
        Returns:
            Prediction results dictionary
        """
        # Preprocess
        data = self.preprocess_video(video_path)
        
        # Convert to tensors
        faces = torch.from_numpy(data['faces']).float().permute(0, 3, 1, 2) / 255.0
        mouths = torch.from_numpy(data['mouths']).float().permute(0, 3, 1, 2) / 255.0
        spectrogram = torch.from_numpy(data['spectrogram']).float().unsqueeze(0)
        
        # Normalize spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        
        # Add batch dimension
        faces = faces.unsqueeze(0).to(self.device)
        mouths = mouths.unsqueeze(0).to(self.device)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(
                frames=faces,
                spectrogram=spectrogram,
                mouth_frames=mouths,
                return_embeddings=return_explanations
            )
        
        # Get prediction
        probs = output['probs'][0].cpu().numpy()
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        result = {
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': confidence,
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            }
        }
        
        # Generate explanations if requested
        if return_explanations and self.gradcam:
            try:
                # Grad-CAM
                cam = self.gradcam.generate_cam(faces)
                overlaid = self.gradcam.overlay_cam(data['faces'], cam)
                result['gradcam_frames'] = overlaid
                
                # Audio saliency
                saliency = self.audio_saliency.generate_saliency(spectrogram)
                result['audio_saliency'] = saliency
            except Exception as e:
                print(f"Error generating explanations: {e}")
                result['explanations_error'] = str(e)
        
        return result

