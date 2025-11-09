"""
OpenCV-based Eye Detection and Tracking for NEEDLE Liveness Detection
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from utils import ImageProcessor, PerformanceMetrics
from liveness_analyzer import LivenessDetector
from config import Config

class OpenCVEyeDetector:
    """
    OpenCV-based eye detection and tracking implementation
    """
    
    def __init__(self):
        # Initialize cascades
        self.face_cascade = cv2.CascadeClassifier(Config.FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(Config.EYE_CASCADE_PATH)
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.liveness_detector = LivenessDetector()
        self.performance_metrics = PerformanceMetrics()
        
        # Tracking state
        self.last_face_rect = None
        self.last_eye_rects = []
        self.tracking_quality = 0.0
        
        # Initialize trackers
        self.face_tracker = None
        self.eye_trackers = []
        
    def detect_face_and_eyes(self, frame: np.ndarray) -> Tuple[Optional[Tuple], List[Tuple]]:
        """
        Detect face and eyes in frame using OpenCV cascades
        
        Returns:
            Tuple of (face_rect, eye_rects)
        """
        start_time = time.time()
        
        # Preprocess frame
        gray = self.image_processor.preprocess_frame(frame)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_rect = None
        eye_rects = []
        
        if len(faces) > 0:
            # Use the largest face
            face_rect = tuple(max(faces, key=lambda rect: rect[2] * rect[3]))
            x, y, w, h = face_rect
            
            # Extract face region for eye detection
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes within face region
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                maxSize=(80, 80)
            )
            
            # Convert eye coordinates to global frame coordinates
            for (ex, ey, ew, eh) in eyes:
                global_eye_rect = (x + ex, y + ey, ew, eh)
                eye_rects.append(global_eye_rect)
        
        detection_time = time.time() - start_time
        self.performance_metrics.add_detection_time(detection_time)
        
        return face_rect, eye_rects
    
    def track_eyes(self, frame: np.ndarray) -> Tuple[Optional[Tuple], List[Tuple]]:
        """
        Track eyes using optical flow or re-detection
        """
        if self.last_face_rect is None:
            return self.detect_face_and_eyes(frame)
        
        # Try to track existing face and eyes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple tracking using template matching for demonstration
        # In production, you might want to use more sophisticated tracking
        face_rect, eye_rects = self.detect_face_and_eyes(frame)
        
        # Update tracking state
        if face_rect is not None:
            self.last_face_rect = face_rect
            self.last_eye_rects = eye_rects
            self.tracking_quality = 1.0
        else:
            self.tracking_quality *= 0.9  # Decay tracking quality
            
        return face_rect, eye_rects
    
    def extract_eye_landmarks(self, frame: np.ndarray, eye_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract eye landmarks from eye region using contour detection
        """
        x, y, w, h = eye_rect
        
        # Extract eye region
        eye_region = self.image_processor.extract_eye_region(frame, eye_rect)
        
        if eye_region.size == 0:
            return np.array([])
        
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            eye_gray = eye_region
        
        # Apply threshold to find eye contours
        _, thresh = cv2.threshold(eye_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: create approximate eye landmarks based on eye rectangle
            landmarks = self._create_approximate_landmarks(eye_rect)
            return landmarks
        
        # Find the largest contour (likely the eye)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle of the contour
        rect = cv2.boundingRect(largest_contour)
        
        # Create eye landmarks based on the contour
        landmarks = self._extract_landmarks_from_contour(largest_contour, eye_rect)
        
        return landmarks
    
    def _create_approximate_landmarks(self, eye_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create approximate eye landmarks when contour detection fails
        """
        x, y, w, h = eye_rect
        
        # Create 6 key points for eye aspect ratio calculation
        landmarks = np.array([
            [x, y + h//2],           # Left corner
            [x + w//4, y + h//4],    # Upper left
            [x + w//2, y],           # Top center
            [x + 3*w//4, y + h//4],  # Upper right
            [x + w, y + h//2],       # Right corner
            [x + w//2, y + h]        # Bottom center
        ])
        
        return landmarks
    
    def _extract_landmarks_from_contour(self, contour: np.ndarray, 
                                      eye_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract landmarks from eye contour
        """
        x_offset, y_offset, _, _ = eye_rect
        
        # Get extreme points of the contour
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        
        # Convert to global coordinates
        landmarks = np.array([
            [leftmost[0] + x_offset, leftmost[1] + y_offset],
            [topmost[0] + x_offset, topmost[1] + y_offset],
            [rightmost[0] + x_offset, rightmost[1] + y_offset],
            [bottommost[0] + x_offset, bottommost[1] + y_offset]
        ])
        
        # Add intermediate points for better EAR calculation
        center_x = (leftmost[0] + rightmost[0]) // 2 + x_offset
        upper_y = (topmost[1] + leftmost[1]) // 2 + y_offset
        lower_y = (bottommost[1] + leftmost[1]) // 2 + y_offset
        
        additional_landmarks = np.array([
            [center_x, upper_y],
            [center_x, lower_y]
        ])
        
        landmarks = np.vstack([landmarks, additional_landmarks])
        
        return landmarks
    
    def detect_pupil(self, frame: np.ndarray, eye_rect: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int]]:
        """
        Detect pupil in eye region
        """
        eye_region = self.image_processor.extract_eye_region(frame, eye_rect)
        pupil_info = self.image_processor.detect_pupil(eye_region)
        
        if pupil_info is not None:
            # Convert to global coordinates
            x_offset, y_offset, _, _ = eye_rect
            global_pupil = (
                pupil_info[0] + x_offset,
                pupil_info[1] + y_offset,
                pupil_info[2]
            )
            return global_pupil
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return liveness analysis
        """
        start_time = time.time()
        
        # Detect or track eyes
        face_rect, eye_rects = self.track_eyes(frame)
        
        results = {
            'face_detected': face_rect is not None,
            'eyes_detected': len(eye_rects),
            'liveness_results': [],
            'face_rect': face_rect,
            'eye_rects': eye_rects,
            'tracking_quality': self.tracking_quality
        }
        
        # Process each detected eye
        for i, eye_rect in enumerate(eye_rects):
            # Extract eye landmarks
            eye_landmarks = self.extract_eye_landmarks(frame, eye_rect)
            
            # Detect pupil (pixel units)
            pupil_info_draw = self.detect_pupil(frame, eye_rect)

            # Normalize pupil radius by eye width for liveness
            liveness_pupil_info = None
            if pupil_info_draw is not None:
                x_e, y_e, w_e, h_e = eye_rect
                eye_width = float(max(w_e, 1))
                liveness_pupil_info = (
                    float(pupil_info_draw[0]),
                    float(pupil_info_draw[1]),
                    float(pupil_info_draw[2]) / eye_width
                )
            
            # Analyze liveness
            if len(eye_landmarks) > 0:
                liveness_result = self.liveness_detector.detect_liveness(
                    eye_landmarks, liveness_pupil_info
                )
                liveness_result['eye_index'] = i
                liveness_result['eye_rect'] = eye_rect
                liveness_result['landmarks'] = eye_landmarks
                liveness_result['pupil_info'] = liveness_pupil_info
                liveness_result['pupil_info_draw'] = pupil_info_draw
                
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
        
        # Draw face rectangle
        if results['face_rect'] is not None:
            x, y, w, h = results['face_rect']
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(output_frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw eye rectangles and liveness results
        for liveness_result in results['liveness_results']:
            eye_rect = liveness_result['eye_rect']
            landmarks = liveness_result['landmarks']
            pupil_info = liveness_result['pupil_info']
            
            x, y, w, h = eye_rect
            
            # Draw eye rectangle
            color = (0, 255, 0) if liveness_result['is_live'] else (0, 0, 255)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw landmarks
            if len(landmarks) > 0:
                output_frame = self.image_processor.draw_eye_landmarks(
                    output_frame, landmarks, color
                )
            
            # Draw pupil
            if pupil_info is not None:
                px, py, pr = pupil_info
                cv2.circle(output_frame, (int(px), int(py)), int(pr), (255, 255, 0), 2)
            
            # Draw liveness score
            score_text = f"NEEDLE: {liveness_result['needle_score']:.3f}"
            cv2.putText(output_frame, score_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw overall liveness score if available
        if results['liveness_results']:
            avg_score = np.mean([r['smoothed_score'] for r in results['liveness_results']])
            from utils import draw_liveness_score
            output_frame = draw_liveness_score(output_frame, avg_score)
        
        # Draw performance info
        perf = results['performance']
        perf_text = f"FPS: {perf['fps']:.1f} | Proc: {perf['avg_processing_time']*1000:.1f}ms"
        cv2.putText(output_frame, perf_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
        
    def draw_detailed_analysis(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detailed component analysis as six progress bars (similar to MediaPipe overlay)
        - Auto-positions to avoid overlapping the user's face when possible.
        """
        output_frame = frame.copy()
        h, w = frame.shape[:2]

        # Analysis panel dimensions
        panel_width = 300
        panel_height = 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        if results.get('liveness_results'):
            # Aggregate components across eyes by averaging
            all_components: Dict[str, list] = {}
            for result in results['liveness_results']:
                components = result.get('components', {})
                for comp_name, score in components.items():
                    all_components.setdefault(comp_name, []).append(float(score))
            avg_components = {name: float(np.mean(scores)) for name, scores in all_components.items()}

            # Draw component bars
            y_offset = 20
            # Keep a consistent order when possible
            preferred_order = [
                'blink_pattern', 'saccade_pattern', 'microsaccade_pattern',
                'pupil_variation', 'temporal_consistency', 'movement_naturalness'
            ]
            items = [(name, avg_components[name]) for name in preferred_order if name in avg_components]
            # Append any remaining components not in preferred list
            for name, val in avg_components.items():
                if name not in preferred_order:
                    items.append((name, val))

            for comp_name, score in items:
                # Component name
                cv2.putText(panel, comp_name.replace('_', ' ').title(),
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Score bar
                bar_width = int(200 * max(0.0, min(1.0, score)))
                # Green if above threshold 0.6, yellow if >0.3, else red
                color = (0, 255, 0) if score > 0.6 else ((0, 255, 255) if score > 0.3 else (0, 0, 255))
                cv2.rectangle(panel, (10, y_offset + 5), (10 + bar_width, y_offset + 15), color, -1)
                cv2.rectangle(panel, (10, y_offset + 5), (210, y_offset + 15), (128, 128, 128), 1)

                # Score value
                cv2.putText(panel, f"{score:.3f}", (220, y_offset + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                y_offset += 25

        # Determine face rectangle (if available) to avoid overlapping
        face_rect = None
        if results.get('face_rect') is not None:
            x, y, fw, fh = results['face_rect']
            face_rect = (max(0, x), max(0, y), min(w, x + fw), min(h, y + fh))

        # Fixed position: bottom-left corner
        margin = 12
        panel_x = margin
        panel_y = max(0, h - panel_height - margin)

        # Semi-transparent background behind the panel area
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 5),
                      (panel_x + panel_width + 5, panel_y + panel_height + 5),
                      (0, 0, 0), -1)
        output_frame = cv2.addWeighted(output_frame, 0.75, overlay, 0.25, 0)

        # Paste panel
        # Clamp to image bounds just in case
        px1, py1 = max(0, panel_x), max(0, panel_y)
        px2, py2 = min(w, panel_x + panel_width), min(h, panel_y + panel_height)
        panel_cropped = panel[0:(py2 - py1), 0:(px2 - px1)]
        output_frame[py1:py2, px1:px2] = panel_cropped

        return output_frame
        
    def reset(self):
        """Reset detector state"""
        self.liveness_detector.reset()
        self.performance_metrics.reset()
        self.last_face_rect = None
        self.last_eye_rects = []
        self.tracking_quality = 0.0
        
        # Reset trackers
        self.face_tracker = None
        self.eye_trackers = []
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.get_metrics_dict()
