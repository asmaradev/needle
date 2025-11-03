"""
MediaPipe-based Eye Detection and Tracking for NEEDLE Liveness Detection
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from utils import ImageProcessor, PerformanceMetrics
from liveness_analyzer import LivenessDetector
from config import Config

class MediaPipeEyeDetector:
    """
    MediaPipe-based eye detection and tracking implementation
    """
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.MP_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MP_TRACKING_CONFIDENCE
        )
        
        # Initialize face detection (backup)
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=Config.MP_FACE_DETECTION_CONFIDENCE
        )
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.liveness_detector = LivenessDetector()
        self.performance_metrics = PerformanceMetrics()
        
        # Eye landmark indices
        self.left_eye_indices = Config.LEFT_EYE_LANDMARKS
        self.right_eye_indices = Config.RIGHT_EYE_LANDMARKS
        
        # Iris landmarks (MediaPipe specific)
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]
        
        # Tracking state
        self.last_landmarks = None
        self.tracking_quality = 0.0
        
    def detect_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks using MediaPipe Face Mesh
        
        Returns:
            Array of normalized landmarks or None if no face detected
        """
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        landmarks = None
        if results.multi_face_landmarks:
            # Get the first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            h, w = frame.shape[:2]
            landmarks = np.array([
                [landmark.x * w, landmark.y * h, landmark.z]
                for landmark in face_landmarks.landmark
            ])
            
            self.tracking_quality = 1.0
        else:
            self.tracking_quality *= 0.9  # Decay tracking quality
        
        detection_time = time.time() - start_time
        self.performance_metrics.add_detection_time(detection_time)
        
        return landmarks
    
    def extract_eye_landmarks(self, face_landmarks: np.ndarray, eye_type: str = 'both') -> Dict[str, np.ndarray]:
        """
        Extract eye landmarks from face landmarks
        
        Args:
            face_landmarks: Full face landmark array
            eye_type: 'left', 'right', or 'both'
            
        Returns:
            Dictionary containing eye landmarks
        """
        eye_landmarks = {}
        
        if eye_type in ['left', 'both']:
            left_eye_points = face_landmarks[self.left_eye_indices]
            eye_landmarks['left'] = left_eye_points[:, :2]  # Remove z coordinate
        
        if eye_type in ['right', 'both']:
            right_eye_points = face_landmarks[self.right_eye_indices]
            eye_landmarks['right'] = right_eye_points[:, :2]  # Remove z coordinate
        
        return eye_landmarks
    
    def extract_iris_landmarks(self, face_landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract iris landmarks for pupil detection
        """
        iris_landmarks = {}
        
        # Left iris
        if len(face_landmarks) > max(self.left_iris_indices):
            left_iris_points = face_landmarks[self.left_iris_indices]
            iris_landmarks['left'] = left_iris_points[:, :2]
        
        # Right iris
        if len(face_landmarks) > max(self.right_iris_indices):
            right_iris_points = face_landmarks[self.right_iris_indices]
            iris_landmarks['right'] = right_iris_points[:, :2]
        
        return iris_landmarks
    
    def calculate_pupil_info(self, iris_landmarks: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Calculate pupil center and radius from iris landmarks
        """
        if len(iris_landmarks) < 4:
            return None
        
        # Calculate center as mean of iris points
        center_x = np.mean(iris_landmarks[:, 0])
        center_y = np.mean(iris_landmarks[:, 1])
        
        # Calculate radius as average distance from center
        distances = np.sqrt(np.sum((iris_landmarks - [center_x, center_y])**2, axis=1))
        radius = np.mean(distances)
        
        return (center_x, center_y, radius)
    
    def get_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio for blink detection
        """
        if len(eye_landmarks) < 6:
            return 0.0
        
        # For MediaPipe landmarks, we need to identify the correct points
        # This is a simplified version - you might need to adjust indices
        try:
            # Vertical distances
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Horizontal distance
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C) if C > 0 else 0.0
            return ear
        except:
            return 0.0
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return liveness analysis
        """
        start_time = time.time()
        
        # Detect face landmarks
        face_landmarks = self.detect_face_landmarks(frame)
        
        results = {
            'face_detected': face_landmarks is not None,
            'liveness_results': [],
            'face_landmarks': face_landmarks,
            'tracking_quality': self.tracking_quality
        }
        
        if face_landmarks is not None:
            # Extract eye landmarks
            eye_landmarks_dict = self.extract_eye_landmarks(face_landmarks)
            iris_landmarks_dict = self.extract_iris_landmarks(face_landmarks)
            
            # Process each eye
            for eye_side in ['left', 'right']:
                if eye_side in eye_landmarks_dict:
                    eye_landmarks = eye_landmarks_dict[eye_side]
                    
                    # Get iris/pupil information
                    pupil_info = None
                    if eye_side in iris_landmarks_dict:
                        iris_landmarks = iris_landmarks_dict[eye_side]
                        pupil_info = self.calculate_pupil_info(iris_landmarks)
                    
                    # Analyze liveness
                    if len(eye_landmarks) > 0:
                        liveness_result = self.liveness_detector.detect_liveness(
                            eye_landmarks, pupil_info
                        )
                        
                        # Add additional information
                        liveness_result['eye_side'] = eye_side
                        liveness_result['landmarks'] = eye_landmarks
                        liveness_result['iris_landmarks'] = iris_landmarks_dict.get(eye_side)
                        liveness_result['pupil_info'] = pupil_info
                        liveness_result['ear'] = self.get_eye_aspect_ratio(eye_landmarks)
                        
                        results['liveness_results'].append(liveness_result)
        
        # Record performance metrics
        processing_time = time.time() - start_time
        self.performance_metrics.add_processing_time(processing_time)
        self.performance_metrics.add_frame_time(time.time())
        
        # Add performance metrics to results
        results['performance'] = self.performance_metrics.get_metrics_dict()
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection and liveness results on frame
        """
        output_frame = frame.copy()
        
        # Draw face landmarks if available
        if results['face_landmarks'] is not None:
            face_landmarks = results['face_landmarks']
            
            # Draw face mesh (simplified)
            for landmark in face_landmarks[::10]:  # Draw every 10th landmark to avoid clutter
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(output_frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw eye-specific results
        for liveness_result in results['liveness_results']:
            eye_side = liveness_result['eye_side']
            landmarks = liveness_result['landmarks']
            iris_landmarks = liveness_result.get('iris_landmarks')
            pupil_info = liveness_result.get('pupil_info')
            
            # Determine color based on liveness
            color = (0, 255, 0) if liveness_result['is_live'] else (0, 0, 255)
            
            # Draw eye landmarks
            if len(landmarks) > 0:
                # Draw eye contour
                eye_points = landmarks.astype(np.int32)
                cv2.polylines(output_frame, [eye_points], True, color, 2)
                
                # Draw individual landmarks
                for point in eye_points:
                    cv2.circle(output_frame, tuple(point), 2, color, -1)
            
            # Draw iris landmarks
            if iris_landmarks is not None and len(iris_landmarks) > 0:
                iris_points = iris_landmarks.astype(np.int32)
                cv2.polylines(output_frame, [iris_points], True, (255, 255, 0), 1)
            
            # Draw pupil
            if pupil_info is not None:
                px, py, pr = pupil_info
                cv2.circle(output_frame, (int(px), int(py)), int(pr), (255, 255, 0), 2)
                cv2.circle(output_frame, (int(px), int(py)), 2, (255, 255, 0), -1)
            
            # Draw liveness score
            if len(landmarks) > 0:
                # Position text near the eye
                text_x = int(np.mean(landmarks[:, 0]))
                text_y = int(np.min(landmarks[:, 1])) - 10
                
                score_text = f"{eye_side.upper()}: {liveness_result['needle_score']:.3f}"
                cv2.putText(output_frame, score_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw EAR value
                ear_text = f"EAR: {liveness_result['ear']:.3f}"
                cv2.putText(output_frame, ear_text, (text_x, text_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw overall liveness score
        if results['liveness_results']:
            avg_score = np.mean([r['smoothed_score'] for r in results['liveness_results']])
            from utils import draw_liveness_score
            output_frame = draw_liveness_score(output_frame, avg_score)
        
        # Draw performance info
        perf = results['performance']
        perf_text = f"MediaPipe - FPS: {perf['fps']:.1f} | Proc: {perf['avg_processing_time']*1000:.1f}ms"
        cv2.putText(output_frame, perf_text, (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw tracking quality
        quality_text = f"Tracking Quality: {self.tracking_quality:.2f}"
        cv2.putText(output_frame, quality_text, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def draw_detailed_analysis(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detailed component analysis
        - Auto-positions the panel to avoid overlapping the user's face when possible.
        """
        output_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Create analysis panel
        panel_width = 300
        panel_height = 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        if results['liveness_results']:
            # Get average component scores
            all_components = {}
            for result in results['liveness_results']:
                components = result.get('components', {})
                for comp_name, score in components.items():
                    if comp_name not in all_components:
                        all_components[comp_name] = []
                    all_components[comp_name].append(float(score))
            
            # Average the components
            avg_components = {name: float(np.mean(scores)) for name, scores in all_components.items()}
            
            # Draw component bars
            y_offset = 20
            # Preferred order to keep layout stable
            preferred_order = [
                'blink_pattern', 'saccade_pattern', 'microsaccade_pattern',
                'pupil_variation', 'temporal_consistency', 'movement_naturalness'
            ]
            items = [(name, avg_components[name]) for name in preferred_order if name in avg_components]
            for name, val in avg_components.items():
                if name not in preferred_order:
                    items.append((name, val))
            
            for comp_name, score in items:
                # Component name
                cv2.putText(panel, comp_name.replace('_', ' ').title(), 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Score bar
                bar_width = int(200 * max(0.0, min(1.0, score)))
                color = (0, 255, 0) if score > 0.6 else (0, 255, 255) if score > 0.3 else (0, 0, 255)
                cv2.rectangle(panel, (10, y_offset + 5), (10 + bar_width, y_offset + 15), color, -1)
                cv2.rectangle(panel, (10, y_offset + 5), (210, y_offset + 15), (128, 128, 128), 1)
                
                # Score value
                cv2.putText(panel, f"{score:.3f}", (220, y_offset + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                y_offset += 25
        
        # Estimate face rectangle from landmarks (if present) to avoid overlap
        face_rect = None
        if results.get('face_landmarks') is not None and len(results['face_landmarks']) > 0:
            pts = np.array(results['face_landmarks'], dtype=np.int32)
            x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
            x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
            face_rect = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
        
        # Fixed position: bottom-left corner
        margin = 12
        panel_x = margin
        panel_y = max(0, h - panel_height - margin)
        
        # Add semi-transparent background
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 5), 
                     (panel_x + panel_width + 5, panel_y + panel_height + 5), 
                     (0, 0, 0), -1)
        output_frame = cv2.addWeighted(output_frame, 0.75, overlay, 0.25, 0)
        
        # Add panel (bounds clamped)
        px1, py1 = max(0, panel_x), max(0, panel_y)
        px2, py2 = min(w, panel_x + panel_width), min(h, panel_y + panel_height)
        panel_cropped = panel[0:(py2 - py1), 0:(px2 - px1)]
        output_frame[py1:py2, px1:px2] = panel_cropped
        
        return output_frame
    
    def reset(self):
        """Reset detector state"""
        self.liveness_detector.reset()
        self.performance_metrics.reset()
        self.last_landmarks = None
        self.tracking_quality = 0.0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.get_metrics_dict()
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
