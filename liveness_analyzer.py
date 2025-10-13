"""
NEEDLE (Natural Eye-movement Evaluation for Detecting Live Entities) Algorithm
Core liveness detection based on micro eye-movement patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time
from scipy import signal
from scipy.stats import entropy
from utils import EyeMovementAnalyzer
from config import Config

class NEEDLEAnalyzer:
    """
    NEEDLE Algorithm Implementation
    Natural Eye-movement Evaluation for Detecting Live Entities
    
    This novel algorithm combines multiple micro eye-movement patterns
    to create a comprehensive liveness score.
    """
    
    def __init__(self, window_size: int = Config.NEEDLE_WINDOW_SIZE):
        self.window_size = window_size
        self.eye_analyzer = EyeMovementAnalyzer()
        
        # Data buffers for temporal analysis
        self.blink_buffer = deque(maxlen=window_size)
        self.movement_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        self.pupil_buffer = deque(maxlen=window_size)
        self.ear_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)
        
        # State tracking
        self.last_position = None
        self.last_timestamp = None
        self.blink_state = False
        self.blink_start_time = None
        self.blink_durations = []
        
        # NEEDLE components
        self.needle_components = {
            'blink_pattern': 0.0,
            'saccade_pattern': 0.0,
            'microsaccade_pattern': 0.0,
            'pupil_variation': 0.0,
            'temporal_consistency': 0.0,
            'movement_naturalness': 0.0
        }
        
    def update(self, eye_landmarks: np.ndarray, pupil_info: Optional[Tuple] = None) -> float:
        """
        Update NEEDLE analyzer with new eye data and return liveness score
        
        Args:
            eye_landmarks: Array of eye landmark coordinates
            pupil_info: Optional tuple of (x, y, radius) for pupil
            
        Returns:
            NEEDLE liveness score (0.0 to 1.0)
        """
        current_time = time.time()
        
        if len(eye_landmarks) == 0:
            return 0.0
        
        # Calculate eye center
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Calculate Eye Aspect Ratio (EAR)
        ear = self.eye_analyzer.calculate_eye_aspect_ratio(eye_landmarks)
        
        # Store data in buffers
        self.ear_buffer.append(ear)
        self.movement_buffer.append(eye_center)
        self.timestamp_buffer.append(current_time)
        
        if pupil_info:
            pupil_size = pupil_info[2]  # radius
            self.pupil_buffer.append(pupil_size)
        
        # Calculate velocity if we have previous position
        velocity = 0.0
        if self.last_position is not None and self.last_timestamp is not None:
            dt = current_time - self.last_timestamp
            if dt > 0:
                velocity = self.eye_analyzer.calculate_movement_velocity(
                    eye_center, self.last_position, dt
                )
        
        self.velocity_buffer.append(velocity)
        
        # Update state
        self.last_position = eye_center
        self.last_timestamp = current_time
        
        # Calculate NEEDLE components
        self._calculate_blink_pattern()
        self._calculate_saccade_pattern()
        self._calculate_microsaccade_pattern()
        self._calculate_pupil_variation()
        self._calculate_temporal_consistency()
        self._calculate_movement_naturalness()
        
        # Calculate final NEEDLE score
        needle_score = self._calculate_needle_score()
        
        return needle_score
    
    def _calculate_blink_pattern(self):
        """Calculate blink pattern component of NEEDLE score"""
        if len(self.ear_buffer) < 5:
            self.needle_components['blink_pattern'] = 0.0
            return
        
        ear_values = list(self.ear_buffer)
        
        # Detect blinks
        blinks = [self.eye_analyzer.detect_blink(ear) for ear in ear_values]
        self.blink_buffer.extend(blinks)
        
        # Analyze blink patterns
        blink_frequency = self._calculate_blink_frequency()
        blink_regularity = self._calculate_blink_regularity()
        blink_duration_score = self._calculate_blink_duration_score()
        
        # Combine blink metrics
        blink_score = (blink_frequency * 0.4 + 
                      blink_regularity * 0.3 + 
                      blink_duration_score * 0.3)
        
        self.needle_components['blink_pattern'] = np.clip(blink_score, 0.0, 1.0)
    
    def _calculate_saccade_pattern(self):
        """Calculate saccade pattern component of NEEDLE score"""
        if len(self.velocity_buffer) < 5:
            self.needle_components['saccade_pattern'] = 0.0
            return
        
        velocities = list(self.velocity_buffer)
        
        # Detect saccades
        saccades = [self.eye_analyzer.detect_saccade(v) for v in velocities]
        saccade_count = sum(saccades)
        
        # Calculate saccade frequency (should be natural, not too high or low)
        saccade_frequency = saccade_count / len(velocities)
        
        # Optimal saccade frequency is around 0.1-0.3
        if 0.1 <= saccade_frequency <= 0.3:
            saccade_score = 1.0
        elif saccade_frequency < 0.1:
            saccade_score = saccade_frequency / 0.1
        else:
            saccade_score = max(0.0, 1.0 - (saccade_frequency - 0.3) / 0.2)
        
        self.needle_components['saccade_pattern'] = saccade_score
    
    def _calculate_microsaccade_pattern(self):
        """Calculate microsaccade pattern component of NEEDLE score"""
        if len(self.velocity_buffer) < 10:
            self.needle_components['microsaccade_pattern'] = 0.0
            return
        
        velocities = list(self.velocity_buffer)
        
        # Detect microsaccades
        microsaccades = [self.eye_analyzer.detect_microsaccade(v) for v in velocities]
        microsaccade_count = sum(microsaccades)
        
        # Calculate microsaccade frequency
        microsaccade_frequency = microsaccade_count / len(velocities)
        
        # Optimal microsaccade frequency is around 0.05-0.15
        if 0.05 <= microsaccade_frequency <= 0.15:
            microsaccade_score = 1.0
        elif microsaccade_frequency < 0.05:
            microsaccade_score = microsaccade_frequency / 0.05
        else:
            microsaccade_score = max(0.0, 1.0 - (microsaccade_frequency - 0.15) / 0.1)
        
        self.needle_components['microsaccade_pattern'] = microsaccade_score
    
    def _calculate_pupil_variation(self):
        """Calculate pupil variation component of NEEDLE score"""
        if len(self.pupil_buffer) < 5:
            self.needle_components['pupil_variation'] = 0.0
            return
        
        pupil_sizes = list(self.pupil_buffer)
        variation = self.eye_analyzer.calculate_pupil_variation(pupil_sizes)
        
        # Natural pupil variation should be between 0.05 and 0.25
        if 0.05 <= variation <= 0.25:
            pupil_score = 1.0
        elif variation < 0.05:
            pupil_score = variation / 0.05
        else:
            pupil_score = max(0.0, 1.0 - (variation - 0.25) / 0.25)
        
        self.needle_components['pupil_variation'] = pupil_score
    
    def _calculate_temporal_consistency(self):
        """Calculate temporal consistency component of NEEDLE score"""
        if len(self.movement_buffer) < 10:
            self.needle_components['temporal_consistency'] = 0.0
            return
        
        movements = list(self.movement_buffer)
        consistency = self.eye_analyzer.analyze_temporal_consistency(movements)
        
        # Good consistency should be between 0.3 and 0.8 (not too rigid, not too chaotic)
        if 0.3 <= consistency <= 0.8:
            consistency_score = 1.0
        elif consistency < 0.3:
            consistency_score = consistency / 0.3
        else:
            consistency_score = max(0.0, 1.0 - (consistency - 0.8) / 0.2)
        
        self.needle_components['temporal_consistency'] = consistency_score
    
    def _calculate_movement_naturalness(self):
        """Calculate movement naturalness using entropy and spectral analysis"""
        if len(self.velocity_buffer) < 20:
            self.needle_components['movement_naturalness'] = 0.0
            return
        
        velocities = np.array(list(self.velocity_buffer))
        
        # Calculate entropy of velocity distribution
        hist, _ = np.histogram(velocities, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        velocity_entropy = entropy(hist)
        
        # Calculate spectral entropy
        if len(velocities) > 10:
            freqs, psd = signal.periodogram(velocities)
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]
            spectral_entropy = entropy(psd_norm)
        else:
            spectral_entropy = 0.0
        
        # Combine entropy measures
        naturalness_score = (velocity_entropy * 0.6 + spectral_entropy * 0.4) / 3.0
        naturalness_score = np.clip(naturalness_score, 0.0, 1.0)
        
        self.needle_components['movement_naturalness'] = naturalness_score
    
    def _calculate_blink_frequency(self) -> float:
        """Calculate blink frequency score"""
        if len(self.blink_buffer) < 10:
            return 0.0
        
        blinks = list(self.blink_buffer)
        blink_count = sum(blinks)
        
        # Calculate frequency (blinks per second)
        time_span = len(blinks) / 30.0  # Assuming 30 FPS
        frequency = blink_count / time_span if time_span > 0 else 0.0
        
        # Normal blink frequency is 0.1-0.8 Hz
        if Config.BLINK_FREQUENCY_MIN <= frequency <= Config.BLINK_FREQUENCY_MAX:
            return 1.0
        elif frequency < Config.BLINK_FREQUENCY_MIN:
            return frequency / Config.BLINK_FREQUENCY_MIN
        else:
            return max(0.0, 1.0 - (frequency - Config.BLINK_FREQUENCY_MAX) / 0.5)
    
    def _calculate_blink_regularity(self) -> float:
        """Calculate blink regularity score"""
        if len(self.blink_durations) < 3:
            return 0.5  # Neutral score
        
        durations = np.array(self.blink_durations[-10:])  # Last 10 blinks
        
        # Calculate coefficient of variation
        if np.mean(durations) > 0:
            cv = np.std(durations) / np.mean(durations)
            # Good regularity has CV between 0.2 and 0.6
            if 0.2 <= cv <= 0.6:
                return 1.0
            elif cv < 0.2:
                return cv / 0.2
            else:
                return max(0.0, 1.0 - (cv - 0.6) / 0.4)
        
        return 0.0
    
    def _calculate_blink_duration_score(self) -> float:
        """Calculate blink duration score"""
        if len(self.blink_durations) == 0:
            return 0.0
        
        recent_durations = self.blink_durations[-5:]  # Last 5 blinks
        avg_duration = np.mean(recent_durations)
        
        # Normal blink duration is 3-15 frames (0.1-0.5 seconds at 30fps)
        if Config.MIN_BLINK_DURATION <= avg_duration <= Config.MAX_BLINK_DURATION:
            return 1.0
        elif avg_duration < Config.MIN_BLINK_DURATION:
            return avg_duration / Config.MIN_BLINK_DURATION
        else:
            return max(0.0, 1.0 - (avg_duration - Config.MAX_BLINK_DURATION) / 10.0)
    
    def _calculate_needle_score(self) -> float:
        """Calculate final NEEDLE liveness score"""
        total_score = 0.0
        
        for component, weight in Config.NEEDLE_WEIGHTS.items():
            component_score = self.needle_components.get(component, 0.0)
            total_score += component_score * weight
        
        return np.clip(total_score, 0.0, 1.0)
    
    def get_component_scores(self) -> Dict[str, float]:
        """Get individual component scores for debugging"""
        return self.needle_components.copy()
    
    def reset(self):
        """Reset analyzer state"""
        self.blink_buffer.clear()
        self.movement_buffer.clear()
        self.velocity_buffer.clear()
        self.pupil_buffer.clear()
        self.ear_buffer.clear()
        self.timestamp_buffer.clear()
        
        self.last_position = None
        self.last_timestamp = None
        self.blink_state = False
        self.blink_start_time = None
        self.blink_durations.clear()
        
        for key in self.needle_components:
            self.needle_components[key] = 0.0

class LivenessDetector:
    """
    Main liveness detector that combines NEEDLE algorithm with traditional methods
    """
    
    def __init__(self):
        self.needle_analyzer = NEEDLEAnalyzer()
        self.frame_count = 0
        self.liveness_history = deque(maxlen=30)
        
    def detect_liveness(self, eye_landmarks: np.ndarray, 
                       pupil_info: Optional[Tuple] = None) -> Dict[str, float]:
        """
        Detect liveness using NEEDLE algorithm
        
        Returns:
            Dictionary containing liveness scores and component analysis
        """
        self.frame_count += 1
        
        # Get NEEDLE score
        needle_score = self.needle_analyzer.update(eye_landmarks, pupil_info)
        self.liveness_history.append(needle_score)
        
        # Calculate smoothed score
        smoothed_score = np.mean(list(self.liveness_history)) if self.liveness_history else 0.0
        
        # Determine liveness status
        is_live = smoothed_score >= Config.NEEDLE_THRESHOLD
        confidence = smoothed_score
        
        # Get component scores for analysis
        components = self.needle_analyzer.get_component_scores()
        
        return {
            'needle_score': needle_score,
            'smoothed_score': smoothed_score,
            'is_live': is_live,
            'confidence': confidence,
            'components': components,
            'frame_count': self.frame_count
        }
    
    def reset(self):
        """Reset detector state"""
        self.needle_analyzer.reset()
        self.frame_count = 0
        self.liveness_history.clear()
