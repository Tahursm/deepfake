"""
Dataset class for loading video clips with synchronized audio and face frames.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import cv2
import pickle
from typing import List, Tuple, Optional, Dict
import json

from ..preprocessing.face_extractor import FaceExtractor
from ..preprocessing.audio_extract import AudioExtractor
from ..preprocessing.landmarks import LandmarkExtractor


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection with synchronized video and audio.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 64,
        frame_size: Tuple[int, int] = (224, 224),
        mouth_size: Tuple[int, int] = (64, 64),
        fps: float = 25.0,
        clip_duration: float = 3.0,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing dataset
            split: Dataset split ('train', 'val', 'test')
            num_frames: Number of frames per clip
            frame_size: Size of face frames (height, width)
            mouth_size: Size of mouth region (height, width)
            fps: Frames per second
            clip_duration: Duration of clip in seconds
            cache_dir: Directory to cache preprocessed features
            use_cache: Use cached features if available
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mouth_size = mouth_size
        self.fps = fps
        self.clip_duration = clip_duration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.augment = augment
        
        # Initialize extractors
        self.face_extractor = FaceExtractor(face_size=frame_size)
        self.audio_extractor = AudioExtractor()
        self.landmark_extractor = LandmarkExtractor()
        
        # Load dataset metadata
        self.samples = self._load_metadata()
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata (video paths and labels)."""
        metadata_file = self.data_dir / f'{self.split}_metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            # If metadata file doesn't exist, create it from directory structure
            # This assumes structure: data_dir/split/{real|fake}/videos/
            samples = []
            split_dir = self.data_dir / self.split
            
            if split_dir.exists():
                for label_dir in ['real', 'fake']:
                    label = 0 if label_dir == 'real' else 1
                    label_path = split_dir / label_dir
                    
                    if label_path.exists():
                        for video_file in label_path.glob('*.mp4'):
                            samples.append({
                                'video_path': str(video_file),
                                'label': label
                            })
            
            # Save metadata for future use
            if self.cache_dir:
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(samples, f, indent=2)
            
            return samples
    
    def _get_cache_path(self, video_path: str) -> Optional[Path]:
        """Get cache path for a video."""
        if not self.cache_dir:
            return None
        
        video_name = Path(video_path).stem
        cache_path = self.cache_dir / f'{video_name}_{self.split}.pkl'
        return cache_path
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load preprocessed data from cache."""
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save preprocessed data to cache."""
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_video(self, video_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load video frames and audio."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Extract audio
        audio = self.audio_extractor.extract_audio_from_video(video_path)
        
        return frames, audio
    
    def _preprocess_sample(self, video_path: str) -> Dict:
        """Preprocess a video sample."""
        # Check cache
        cache_path = self._get_cache_path(video_path)
        if self.use_cache and cache_path:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data
        
        # Load video
        frames, audio = self._load_video(video_path)
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Extract faces
        face_sequence = self.face_extractor.extract_face_sequence(
            frames,
            fps=self.fps,
            max_frames=self.num_frames
        )
        
        if len(face_sequence) == 0:
            raise ValueError(f"No faces extracted from {video_path}")
        
        # Extract mouth regions
        mouth_sequence = []
        for frame in frames[:len(face_sequence)]:
            mouth = self.face_extractor.extract_mouth_region(frame)
            if mouth is None:
                # Use placeholder if mouth extraction fails
                mouth = np.zeros((*self.mouth_size, 3), dtype=np.uint8)
            mouth_sequence.append(mouth)
        
        # Synchronize audio with frames
        num_frames_actual = len(face_sequence)
        audio_segment = self.audio_extractor.synchronize_audio_with_frames(
            audio,
            num_frames_actual,
            fps=self.fps
        )
        
        # Compute spectrogram
        spectrogram = self.audio_extractor.compute_mel_spectrogram(audio_segment)
        
        # Pad or truncate to fixed size
        face_sequence = self._pad_or_truncate_frames(face_sequence, self.num_frames)
        mouth_sequence = self._pad_or_truncate_frames(mouth_sequence, self.num_frames)
        spectrogram = self._pad_or_truncate_spectrogram(spectrogram, self.num_frames)
        
        # Convert to numpy arrays
        face_array = np.stack(face_sequence, axis=0)
        mouth_array = np.stack(mouth_sequence, axis=0)
        
        data = {
            'faces': face_array,
            'mouths': mouth_array,
            'spectrogram': spectrogram
        }
        
        # Save to cache
        if cache_path:
            self._save_to_cache(cache_path, data)
        
        return data
    
    def _pad_or_truncate_frames(self, frames: List[np.ndarray], target_len: int) -> List[np.ndarray]:
        """Pad or truncate frame sequence to target length."""
        if len(frames) > target_len:
            # Sample evenly
            indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
            return [frames[i] for i in indices]
        elif len(frames) < target_len:
            # Pad with last frame
            padding = [frames[-1]] * (target_len - len(frames))
            return frames + padding
        return frames
    
    def _pad_or_truncate_spectrogram(self, spec: np.ndarray, target_frames: int) -> np.ndarray:
        """Pad or truncate spectrogram to match frame count."""
        # Approximate time frames based on hop length
        # This is a simplified approach - adjust based on your audio extractor settings
        current_time_frames = spec.shape[1]
        target_time_frames = int(target_frames * self.audio_extractor.hop_length / self.audio_extractor.sample_rate * self.fps)
        
        if current_time_frames > target_time_frames:
            # Truncate
            spec = spec[:, :target_time_frames]
        elif current_time_frames < target_time_frames:
            # Pad
            padding = np.zeros((spec.shape[0], target_time_frames - current_time_frames))
            spec = np.concatenate([spec, padding], axis=1)
        
        return spec
    
    def _augment(self, faces: np.ndarray, mouths: np.ndarray, spectrogram: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        if not self.augment:
            return faces, mouths, spectrogram
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            faces = np.flip(faces, axis=2).copy()  # .copy() to avoid negative strides
            mouths = np.flip(mouths, axis=2).copy()  # .copy() to avoid negative strides
        
        # Color jitter (simple version)
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            faces = np.clip(faces * brightness, 0, 255).astype(np.uint8)
            mouths = np.clip(mouths * brightness, 0, 255).astype(np.uint8)
        
        # Gaussian blur (occasionally)
        if np.random.rand() > 0.8:
            kernel_size = 3
            for i in range(len(faces)):
                faces[i] = cv2.GaussianBlur(faces[i], (kernel_size, kernel_size), 0)
        
        return faces, mouths, spectrogram
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']
        
        # Preprocess
        data = self._preprocess_sample(video_path)
        
        faces = data['faces']
        mouths = data['mouths']
        spectrogram = data['spectrogram']
        
        # Augment
        faces, mouths, spectrogram = self._augment(faces, mouths, spectrogram)
        
        # Convert to tensors
        # Faces: (num_frames, H, W, C) -> (num_frames, C, H, W)
        faces = torch.from_numpy(faces).float().permute(0, 3, 1, 2) / 255.0
        mouths = torch.from_numpy(mouths).float().permute(0, 3, 1, 2) / 255.0
        spectrogram = torch.from_numpy(spectrogram).float().unsqueeze(0)  # Add channel dim
        
        # Normalize spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'frames': faces,
            'mouth_frames': mouths,
            'spectrogram': spectrogram,
            'label': label_tensor
        }

