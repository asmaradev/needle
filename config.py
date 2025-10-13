"""
Configuration file for NEEDLE Liveness Detection System
Natural Eye-movement Evaluation for Detecting Live Entities
"""

import cv2

class Config:
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # Eye detection parameters
    EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # NEEDLE Algorithm Parameters
    NEEDLE_WINDOW_SIZE = 30  # frames to analyze for liveness
    NEEDLE_THRESHOLD = 0.6   # liveness threshold (0-1)
    
    # Micro-movement detection parameters
    BLINK_THRESHOLD = 0.25
    SACCADE_THRESHOLD = 2.0
    PUPIL_VARIATION_THRESHOLD = 0.15
    MICROSACCADE_MIN_AMPLITUDE = 0.5
    MICROSACCADE_MAX_AMPLITUDE = 2.0
    
    # Temporal analysis parameters
    MIN_BLINK_DURATION = 3    # frames
    MAX_BLINK_DURATION = 15   # frames
    BLINK_FREQUENCY_MIN = 0.1 # blinks per second
    BLINK_FREQUENCY_MAX = 0.8 # blinks per second
    
    # Movement smoothing
    MOVEMENT_SMOOTHING_FACTOR = 0.3
    
    # Performance benchmarking
    BENCHMARK_DURATION = 60   # seconds
    METRICS_UPDATE_INTERVAL = 1.0  # seconds
    
    # GUI settings
    WINDOW_TITLE = "NEEDLE Liveness Detection System"
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 600
    
    # MediaPipe settings
    MP_FACE_DETECTION_CONFIDENCE = 0.7
    MP_FACE_MESH_CONFIDENCE = 0.5
    MP_TRACKING_CONFIDENCE = 0.5
    
    # Eye landmark indices for MediaPipe
    LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # NEEDLE scoring weights
    NEEDLE_WEIGHTS = {
        'blink_pattern': 0.25,
        'saccade_pattern': 0.20,
        'microsaccade_pattern': 0.15,
        'pupil_variation': 0.15,
        'temporal_consistency': 0.15,
        'movement_naturalness': 0.10
    }
