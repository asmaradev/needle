"""
Utility functions for NEEDLE Liveness Detection System
"""

import cv2
import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
import time
from typing import List, Tuple, Dict, Optional

class EyeMovementAnalyzer:
    """Utility class for analyzing eye movement patterns"""
    
    def __init__(self):
        self.movement_history = []
        self.blink_history = []
        self.pupil_history = []
        
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        Guard against degenerate geometry (C≈0) to avoid divide-by-zero warnings.
        """
        if len(eye_landmarks) < 6:
            return 0.0

        # Vertical eye landmarks
        A = euclidean(eye_landmarks[1], eye_landmarks[5])
        B = euclidean(eye_landmarks[2], eye_landmarks[4])

        # Horizontal eye landmark (denominator)
        C = euclidean(eye_landmarks[0], eye_landmarks[3])
        eps = 1e-6
        if C <= eps:
            # Degenerate case (e.g., closed eye or detection glitch)
            return 0.0

        # Eye aspect ratio with numpy errstate to silence transient warnings
        import numpy as _np
        with _np.errstate(divide='ignore', invalid='ignore'):
            ear = (A + B) / (2.0 * C)
        return float(ear)
    
    def detect_blink(self, ear: float, threshold: float = 0.25) -> bool:
        """Detect blink based on Eye Aspect Ratio"""
        return ear < threshold
    
    def calculate_movement_velocity(self, current_pos: np.ndarray, 
                                 previous_pos: np.ndarray, dt: float) -> float:
        """Calculate movement velocity between two positions"""
        if dt == 0:
            return 0.0
        distance = euclidean(current_pos, previous_pos)
        return distance / dt
    
    def detect_saccade(self, velocity: float, threshold: float = 2.0) -> bool:
        """Detect saccadic eye movement"""
        return velocity > threshold
    
    def detect_microsaccade(self, velocity: float, 
                          min_amp: float = 0.5, max_amp: float = 2.0) -> bool:
        """Detect microsaccadic eye movement"""
        return min_amp <= velocity <= max_amp
    
    def calculate_pupil_variation(self, pupil_sizes: List[float]) -> float:
        """Calculate pupil size variation coefficient"""
        if len(pupil_sizes) < 2:
            return 0.0
        return np.std(pupil_sizes) / np.mean(pupil_sizes) if np.mean(pupil_sizes) > 0 else 0.0
    
    def smooth_movement(self, current_pos: np.ndarray, 
                       previous_pos: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Apply exponential smoothing to movement data"""
        return alpha * current_pos + (1 - alpha) * previous_pos
    
    def analyze_temporal_consistency(self, movements: List[np.ndarray], 
                                   window_size: int = 10) -> float:
        """Analyze temporal consistency of eye movements"""
        if len(movements) < window_size:
            return 0.0
            
        recent_movements = movements[-window_size:]
        velocities = []
        
        for i in range(1, len(recent_movements)):
            velocity = euclidean(recent_movements[i], recent_movements[i-1])
            velocities.append(velocity)
        
        if not velocities:
            return 0.0
            
        # Calculate consistency score based on velocity variation
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        
        if mean_velocity == 0:
            return 0.0
            
        consistency = 1.0 - min(std_velocity / mean_velocity, 1.0)
        return consistency

class PerformanceMetrics:
    """Class for tracking performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frame_times = []
        self.detection_times = []
        self.processing_times = []
        self.accuracy_scores = []
        self.start_time = time.time()
    
    def add_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
    
    def add_detection_time(self, detection_time: float):
        self.detection_times.append(detection_time)
    
    def add_processing_time(self, processing_time: float):
        self.processing_times.append(processing_time)
    
    def add_accuracy_score(self, score: float):
        self.accuracy_scores.append(score)
    
    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return len(self.frame_times) / (time.time() - self.start_time)
    
    def get_average_detection_time(self) -> float:
        return np.mean(self.detection_times) if self.detection_times else 0.0
    
    def get_average_processing_time(self) -> float:
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_average_accuracy(self) -> float:
        return np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0
    
    def get_metrics_dict(self) -> Dict[str, float]:
        return {
            'fps': self.get_fps(),
            'avg_detection_time': self.get_average_detection_time(),
            'avg_processing_time': self.get_average_processing_time(),
            'avg_accuracy': self.get_average_accuracy(),
            'total_frames': len(self.frame_times)
        }

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better eye detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    @staticmethod
    def extract_eye_region(frame: np.ndarray, eye_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract eye region from frame"""
        if frame.size == 0:
            return np.array([])
            
        x, y, w, h = eye_rect
        
        # Validate eye rectangle bounds
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return np.array([])
        
        # Add padding around eye region
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        # Ensure we have valid coordinates
        if x_start >= x_end or y_start >= y_end:
            return np.array([])
        
        eye_region = frame[y_start:y_end, x_start:x_end]
        
        # Ensure the extracted region is not empty
        if eye_region.size == 0:
            return np.array([])
            
        return eye_region
    
    @staticmethod
    def detect_pupil(eye_region: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect pupil in eye region using HoughCircles"""
        try:
            # Multiple validation checks
            if eye_region is None or eye_region.size == 0:
                return None
            
            # Ensure the image is grayscale and not empty
            if len(eye_region.shape) == 3:
                eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Check if the image is valid for HoughCircles
            if eye_region.size == 0 or eye_region.dtype != np.uint8:
                return None
            
            # Check minimum dimensions
            if eye_region.shape[0] < 10 or eye_region.shape[1] < 10:
                return None
            
            # Additional check to ensure the image is not empty and is contiguous
            if not eye_region.data.contiguous:
                eye_region = np.ascontiguousarray(eye_region)
                
            # Final validation before HoughCircles
            if eye_region.size == 0 or len(eye_region.shape) != 2:
                return None
                
            # Apply additional preprocessing for pupil detection
            blurred = cv2.GaussianBlur(eye_region, (5, 5), 0)
            
            # Detect circles (pupils)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=25
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    # Return the first detected circle (x, y, radius)
                    return tuple(circles[0])
            
            return None
            
        except Exception as e:
            # If any error occurs, return None instead of crashing
            print(f"Error in detect_pupil: {e}")
            return None
    
    @staticmethod
    def draw_eye_landmarks(frame: np.ndarray, landmarks: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw eye landmarks on frame"""
        result = frame.copy()
        for point in landmarks:
            cv2.circle(result, tuple(point.astype(int)), 2, color, -1)
        return result
    
    @staticmethod
    def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate difference between two frames"""
        if frame1.shape != frame2.shape:
            return 0.0
        
        diff = cv2.absdiff(frame1, frame2)
        return np.mean(diff) / 255.0

def create_visualization_window(title: str, width: int, height: int):
    """Create a named window for visualization"""
    cv2.namedWindow(title, cv2.WINDOW_RESIZABLE)
    cv2.resizeWindow(title, width, height)

def draw_liveness_score(frame: np.ndarray, score: float, 
                       position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw liveness score on frame"""
    result = frame.copy()
    
    # Determine color based on score
    if score >= 0.7:
        color = (0, 255, 0)  # Green for live
        status = "LIVE"
    elif score >= 0.4:
        color = (0, 255, 255)  # Yellow for uncertain
        status = "UNCERTAIN"
    else:
        color = (0, 0, 255)  # Red for not live
        status = "NOT LIVE"
    
    # Draw score text
    text = f"NEEDLE Score: {score:.3f} ({status})"
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2, cv2.LINE_AA)
    
    # Draw score bar
    bar_x, bar_y = position[0], position[1] + 40
    bar_width, bar_height = 200, 20
    
    # Background bar
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (128, 128, 128), -1)
    
    # Score bar
    score_width = int(bar_width * score)
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), 
                  color, -1)
    
    return result
