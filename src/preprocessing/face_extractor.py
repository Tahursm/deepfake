"""
Face extraction and alignment module using MediaPipe or Dlib.
Extracts and aligns faces from video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import mediapipe as mp
from pathlib import Path


class FaceExtractor:
    """Extract and align faces from video frames."""
    
    def __init__(
        self,
        face_size: Tuple[int, int] = (224, 224),
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.5
    ):
        """
        Initialize face extractor.
        
        Args:
            face_size: Target size for extracted faces (height, width)
            use_mediapipe: Use MediaPipe (True) or Dlib (False)
            min_detection_confidence: Minimum confidence for face detection
        """
        self.face_size = face_size
        self.use_mediapipe = use_mediapipe
        
        if use_mediapipe:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=min_detection_confidence
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence
            )
        else:
            # Dlib initialization would go here
            raise NotImplementedError("Dlib implementation not yet available")
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face bounding box in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Bounding box (x, y, width, height) or None if no face detected
        """
        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = 0.2
                x = max(0, int(x - padding * width))
                y = max(0, int(y - padding * height))
                width = min(w - x, int(width * (1 + 2 * padding)))
                height = min(h - y, int(height * (1 + 2 * padding)))
                
                return (x, y, width, height)
        
        return None
    
    def extract_face(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Extract and align face from frame.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Optional bounding box. If None, will detect automatically.
            
        Returns:
            Extracted and aligned face image or None
        """
        if bbox is None:
            bbox = self.detect_face(frame)
            if bbox is None:
                return None
        
        x, y, w, h = bbox
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, self.face_size, interpolation=cv2.INTER_LINEAR)
        
        return face_resized
    
    def extract_face_sequence(
        self,
        frames: List[np.ndarray],
        fps: float = 25.0,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract face sequence from video frames.
        
        Args:
            frames: List of video frames
            fps: Frames per second (for sampling)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted face images
        """
        face_sequence = []
        
        # Sample frames if needed
        if max_frames and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Track face bbox across frames for stability
        prev_bbox = None
        
        for frame in frames:
            # Try to use previous bbox first (for stability)
            face = self.extract_face(frame, prev_bbox)
            
            if face is None:
                # If extraction failed, try detecting again
                bbox = self.detect_face(frame)
                if bbox:
                    face = self.extract_face(frame, bbox)
                    prev_bbox = bbox
                else:
                    # If still no face, skip this frame or use previous face
                    if face_sequence:
                        face_sequence.append(face_sequence[-1])
                    continue
            
            face_sequence.append(face)
            if prev_bbox is None:
                prev_bbox = self.detect_face(frame)
        
        return face_sequence
    
    def extract_mouth_region(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Extract mouth/lip region from face.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Extracted mouth region or None
        """
        if face_bbox is None:
            face_bbox = self.detect_face(frame)
            if face_bbox is None:
                return None
        
        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # MediaPipe lip landmarks (indices 61, 146, 91, 181, 84, 17, 314, 405, 320, 308, 324, 318)
                # Use a subset for mouth region
                lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 308, 324, 318]
                
                lip_points = []
                for idx in lip_indices:
                    if idx < len(landmarks.landmark):
                        point = landmarks.landmark[idx]
                        lip_points.append([int(point.x * w), int(point.y * h)])
                
                if lip_points:
                    lip_points = np.array(lip_points)
                    x, y = lip_points[:, 0].min(), lip_points[:, 1].min()
                    x2, y2 = lip_points[:, 0].max(), lip_points[:, 1].max()
                    
                    # Add padding
                    padding = 0.3
                    x = max(0, int(x - padding * (x2 - x)))
                    y = max(0, int(y - padding * (y2 - y)))
                    x2 = min(w, int(x2 + padding * (x2 - x)))
                    y2 = min(h, int(y2 + padding * (y2 - y)))
                    
                    mouth_region = frame[y:y2, x:x2]
                    if mouth_region.size > 0:
                        mouth_resized = cv2.resize(mouth_region, (64, 64), interpolation=cv2.INTER_LINEAR)
                        return mouth_resized
        
        return None
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

