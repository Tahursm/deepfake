"""
Audio extraction and spectrogram generation module.
Extracts audio from video and converts to Mel-spectrogram features.
"""

import librosa
import numpy as np
from typing import Tuple, Optional
import soundfile as sf
from pathlib import Path
import cv2
import subprocess
import io


class AudioExtractor:
    """Extract audio and generate spectrogram features."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 256,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ):
        """
        Initialize audio extractor.
        
        Args:
            sample_rate: Target sample rate for audio (Hz)
            n_mels: Number of Mel filter banks
            hop_length: Hop length for STFT (samples)
            n_fft: FFT window size
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz), None for Nyquist
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
    
    def extract_audio_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save extracted audio
            
        Returns:
            Audio waveform as numpy array
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        # Try using ffmpeg via subprocess first (most reliable)
        try:
            # Use ffmpeg to extract audio and pipe to stdout as WAV
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-f', 'wav',  # WAV format
                '-'  # Output to stdout
            ]
            
            process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Load from bytes using soundfile
            audio, sr = sf.read(io.BytesIO(process.stdout))
            audio = audio.astype(np.float32)
            
            # Ensure correct sample rate
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback to librosa (requires ffmpeg backend)
            try:
                audio, sr = librosa.load(str(video_path), sr=self.sample_rate, mono=True)
            except Exception as librosa_error:
                raise ValueError(
                    f"Error extracting audio from {video_path}. "
                    f"FFmpeg error: {str(e)}. "
                    f"Librosa error: {str(librosa_error)}. "
                    f"Please ensure ffmpeg is installed and in PATH."
                )
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        if output_path:
            sf.write(output_path, audio, self.sample_rate)
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Normalized audio
        """
        if audio.max() > 0 or audio.min() < 0:
            max_val = max(abs(audio.max()), abs(audio.min()))
            if max_val > 0:
                audio = audio / max_val
        return audio
    
    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute Mel-spectrogram from audio waveform.
        
        Args:
            audio: Audio waveform
            duration: Optional duration to extract (seconds). If None, uses full audio.
            
        Returns:
            Mel-spectrogram (n_mels, time_frames)
        """
        if duration:
            max_samples = int(duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        
        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def compute_log_mel_spectrogram(
        self,
        audio: np.ndarray,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute log Mel-spectrogram (same as compute_mel_spectrogram but explicit naming).
        
        Args:
            audio: Audio waveform
            duration: Optional duration to extract (seconds)
            
        Returns:
            Log Mel-spectrogram (n_mels, time_frames)
        """
        return self.compute_mel_spectrogram(audio, duration)
    
    def extract_audio_segment(
        self,
        audio: np.ndarray,
        start_time: float,
        duration: float
    ) -> np.ndarray:
        """
        Extract a segment of audio.
        
        Args:
            audio: Full audio waveform
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio segment
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        
        if end_sample > len(audio):
            segment = np.pad(
                audio[start_sample:],
                (0, end_sample - len(audio)),
                mode='constant'
            )
        else:
            segment = audio[start_sample:end_sample]
        
        return segment
    
    def augment_audio(
        self,
        audio: np.ndarray,
        pitch_shift: Optional[float] = None,
        time_stretch: Optional[float] = None,
        noise_level: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply audio augmentation.
        
        Args:
            audio: Input audio waveform
            pitch_shift: Pitch shift in semitones (e.g., -2 to 2)
            time_stretch: Time stretch factor (e.g., 0.9 to 1.1)
            noise_level: Additive noise level (0.0 to 1.0)
            
        Returns:
            Augmented audio
        """
        augmented = audio.copy()
        
        # Pitch shift
        if pitch_shift is not None:
            augmented = librosa.effects.pitch_shift(
                augmented,
                sr=self.sample_rate,
                n_steps=pitch_shift
            )
        
        # Time stretch
        if time_stretch is not None:
            augmented = librosa.effects.time_stretch(augmented, rate=time_stretch)
        
        # Add noise
        if noise_level is not None and noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(augmented), len(augmented))
            augmented = augmented + noise
        
        return self.normalize_audio(augmented)
    
    def synchronize_audio_with_frames(
        self,
        audio: np.ndarray,
        num_frames: int,
        fps: float = 25.0
    ) -> np.ndarray:
        """
        Synchronize audio segment with video frames.
        
        Args:
            audio: Full audio waveform
            num_frames: Number of video frames
            fps: Frames per second
            
        Returns:
            Audio segment synchronized with frames
        """
        duration = num_frames / fps
        return self.extract_audio_segment(audio, 0.0, duration)

