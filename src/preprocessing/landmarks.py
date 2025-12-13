"""
Facial landmark extraction module.
Extracts 2D facial landmarks for lip-sync analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import mediapipe as mp


class LandmarkExtractor:
    """Extract facial landmarks for lip-sync analysis."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize landmark extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe lip landmark indices (outer and inner lips)
        self.lip_landmark_indices = [
            # Outer lip
            61, 146, 91, 181, 84, 17, 314, 405, 320, 308, 324, 318,
            # Inner lip
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324
        ]
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Landmarks as numpy array (N, 2) or None if no face detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Extract all landmarks or just lip landmarks
            landmark_points = []
            for landmark in landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                landmark_points.append([x, y])
            
            return np.array(landmark_points)
        
        return None
    
    def extract_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract only lip region landmarks.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Lip landmarks as numpy array (N, 2) or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            lip_points = []
            for idx in self.lip_landmark_indices:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    lip_points.append([point.x * w, point.y * h])
            
            if lip_points:
                return np.array(lip_points)
        
        return None
    
    def extract_lip_sequence(
        self,
        frames: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """
        Extract lip landmarks from a sequence of frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of lip landmark arrays (one per frame)
        """
        lip_sequence = []
        
        for frame in frames:
            lip_landmarks = self.extract_lip_landmarks(frame)
            lip_sequence.append(lip_landmarks)
        
        return lip_sequence
    
    def compute_lip_motion(
        self,
        lip_sequence: List[Optional[np.ndarray]]
    ) -> Optional[np.ndarray]:
        """
        Compute lip motion features from landmark sequence.
        
        Args:
            lip_sequence: List of lip landmark arrays
            
        Returns:
            Motion features (num_frames, feature_dim) or None
        """
        valid_landmarks = [l for l in lip_sequence if l is not None]
        
        if len(valid_landmarks) < 2:
            return None
        
        # Compute motion as differences between consecutive frames
        motion_features = []
        for i in range(1, len(valid_landmarks)):
            motion = valid_landmarks[i] - valid_landmarks[i-1]
            # Flatten to feature vector
            motion_features.append(motion.flatten())
        
        # Pad first frame with zeros
        if motion_features:
            motion_features.insert(0, np.zeros_like(motion_features[0]))
        
        return np.array(motion_features)
    
    def get_lip_bbox(self, lip_landmarks: np.ndarray, padding: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Get bounding box for lip region from landmarks.
        
        Args:
            lip_landmarks: Lip landmark points (N, 2)
            padding: Padding factor
            
        Returns:
            Bounding box (x, y, width, height)
        """
        x_min, y_min = lip_landmarks.min(axis=0)
        x_max, y_max = lip_landmarks.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        x = int(x_min - padding * width)
        y = int(y_min - padding * height)
        width = int(width * (1 + 2 * padding))
        height = int(height * (1 + 2 * padding))
        
        return (x, y, width, height)
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

