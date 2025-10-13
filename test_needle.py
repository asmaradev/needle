"""
Test script for NEEDLE Liveness Detection System
This script tests basic functionality without GUI
"""

import cv2
import numpy as np
import time
import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from config import Config
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from utils import EyeMovementAnalyzer, PerformanceMetrics
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        from liveness_analyzer import NEEDLEAnalyzer, LivenessDetector
        print("✓ NEEDLE analyzer imported successfully")
    except ImportError as e:
        print(f"✗ NEEDLE analyzer import failed: {e}")
        return False
    
    try:
        from opencv_detector import OpenCVEyeDetector
        print("✓ OpenCV detector imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV detector import failed: {e}")
        return False
    
    try:
        from mediapipe_detector import MediaPipeEyeDetector
        print("✓ MediaPipe detector imported successfully")
    except ImportError as e:
        print(f"✗ MediaPipe detector import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera functionality"""
    print("\nTesting camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Cannot read from camera")
            cap.release()
            return False
        
        print(f"✓ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
    
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_opencv_detector():
    """Test OpenCV detector basic functionality"""
    print("\nTesting OpenCV detector...")
    
    try:
        from opencv_detector import OpenCVEyeDetector
        
        detector = OpenCVEyeDetector()
        print("✓ OpenCV detector initialized")
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test processing (should not crash)
        results = detector.process_frame(test_frame)
        print("✓ OpenCV detector processing works")
        
        # Check results structure
        expected_keys = ['face_detected', 'eyes_detected', 'liveness_results', 'performance']
        for key in expected_keys:
            if key not in results:
                print(f"✗ Missing key in results: {key}")
                return False
        
        print("✓ OpenCV detector results structure correct")
        return True
    
    except Exception as e:
        print(f"✗ OpenCV detector test failed: {e}")
        traceback.print_exc()
        return False

def test_mediapipe_detector():
    """Test MediaPipe detector basic functionality"""
    print("\nTesting MediaPipe detector...")
    
    try:
        from mediapipe_detector import MediaPipeEyeDetector
        
        detector = MediaPipeEyeDetector()
        print("✓ MediaPipe detector initialized")
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test processing (should not crash)
        results = detector.process_frame(test_frame)
        print("✓ MediaPipe detector processing works")
        
        # Check results structure
        expected_keys = ['face_detected', 'liveness_results', 'performance']
        for key in expected_keys:
            if key not in results:
                print(f"✗ Missing key in results: {key}")
                return False
        
        print("✓ MediaPipe detector results structure correct")
        return True
    
    except Exception as e:
        print(f"✗ MediaPipe detector test failed: {e}")
        traceback.print_exc()
        return False

def test_needle_algorithm():
    """Test NEEDLE algorithm basic functionality"""
    print("\nTesting NEEDLE algorithm...")
    
    try:
        from liveness_analyzer import NEEDLEAnalyzer
        
        analyzer = NEEDLEAnalyzer()
        print("✓ NEEDLE analyzer initialized")
        
        # Create test eye landmarks
        test_landmarks = np.array([
            [100, 150], [110, 145], [120, 150], [130, 155], [120, 160], [110, 155]
        ])
        
        # Test analysis
        score = analyzer.update(test_landmarks)
        print(f"✓ NEEDLE analysis works - Score: {score:.3f}")
        
        # Test component scores
        components = analyzer.get_component_scores()
        print(f"✓ Component scores available: {len(components)} components")
        
        return True
    
    except Exception as e:
        print(f"✗ NEEDLE algorithm test failed: {e}")
        traceback.print_exc()
        return False

def test_live_detection():
    """Test live detection with camera (if available)"""
    print("\nTesting live detection (5 seconds)...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠ Camera not available, skipping live test")
            return True
        
        from opencv_detector import OpenCVEyeDetector
        detector = OpenCVEyeDetector()
        
        print("✓ Starting live detection test...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:  # Test for 5 seconds
            ret, frame = cap.read()
            if not ret:
                continue
            
            results = detector.process_frame(frame)
            frame_count += 1
            
            # Display basic info
            if frame_count % 30 == 0:  # Every 30 frames
                fps = results['performance'].get('fps', 0)
                print(f"  Frame {frame_count}, FPS: {fps:.1f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        print(f"✓ Live detection test completed - {frame_count} frames, {avg_fps:.1f} FPS")
        
        return True
    
    except Exception as e:
        print(f"✗ Live detection test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("NEEDLE Liveness Detection System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Camera Test", test_camera),
        ("OpenCV Detector Test", test_opencv_detector),
        ("MediaPipe Detector Test", test_mediapipe_detector),
        ("NEEDLE Algorithm Test", test_needle_algorithm),
        ("Live Detection Test", test_live_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NEEDLE system is ready to use.")
        print("\nTo start the GUI application, run:")
        print("python main.py")
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed correctly:")
        print("pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
