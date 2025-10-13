"""
NEEDLE Liveness Detection - Simple Demo
A lightweight demo without GUI for quick testing and demonstration
"""

import cv2
import numpy as np
import time
import argparse
from opencv_detector import OpenCVEyeDetector
from mediapipe_detector import MediaPipeEyeDetector

def run_demo(detector_type="opencv", duration=30, show_video=True):
    """
    Run a simple demo of the NEEDLE liveness detection system
    
    Args:
        detector_type: "opencv" or "mediapipe"
        duration: Demo duration in seconds
        show_video: Whether to show video window
    """
    
    print(f"NEEDLE Liveness Detection Demo")
    print(f"Detector: {detector_type.upper()}")
    print(f"Duration: {duration} seconds")
    print("=" * 50)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize detector
    if detector_type.lower() == "opencv":
        detector = OpenCVEyeDetector()
        print("✓ OpenCV detector initialized")
    else:
        detector = MediaPipeEyeDetector()
        print("✓ MediaPipe detector initialized")
    
    # Demo variables
    start_time = time.time()
    frame_count = 0
    total_liveness_score = 0
    liveness_detections = 0
    
    print("\nStarting detection...")
    print("Press 'q' to quit early, 'r' to reset detector")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Check if demo duration exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break
            
            # Process frame
            results = detector.process_frame(frame)
            frame_count += 1
            
            # Extract liveness information
            face_detected = results.get('face_detected', False)
            liveness_results = results.get('liveness_results', [])
            performance = results.get('performance', {})
            
            # Calculate average liveness score
            if liveness_results:
                avg_score = np.mean([r['smoothed_score'] for r in liveness_results])
                total_liveness_score += avg_score
                liveness_detections += 1
                is_live = avg_score >= 0.6
            else:
                avg_score = 0.0
                is_live = False
            
            # Draw results on frame
            if detector_type.lower() == "opencv":
                output_frame = detector.draw_results(frame, results)
            else:
                output_frame = detector.draw_results(frame, results)
            
            # Add demo information
            demo_info = [
                f"Demo Time: {elapsed_time:.1f}/{duration}s",
                f"Frames: {frame_count}",
                f"FPS: {performance.get('fps', 0):.1f}",
                f"Face: {'YES' if face_detected else 'NO'}",
                f"Eyes: {len(liveness_results)}",
                f"Live: {'YES' if is_live else 'NO'}",
                f"Score: {avg_score:.3f}"
            ]
            
            # Draw demo info
            y_offset = 10
            for info in demo_info:
                cv2.putText(output_frame, info, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Show video if requested
            if show_video:
                cv2.imshow('NEEDLE Demo', output_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nDemo stopped by user")
                    break
                elif key == ord('r'):
                    print("\nResetting detector...")
                    detector.reset()
            
            # Print periodic updates
            if frame_count % 30 == 0:  # Every 30 frames (roughly 1 second)
                fps = performance.get('fps', 0)
                proc_time = performance.get('avg_processing_time', 0) * 1000
                print(f"Frame {frame_count:4d} | FPS: {fps:5.1f} | "
                      f"Proc: {proc_time:5.1f}ms | Score: {avg_score:.3f} | "
                      f"Live: {'YES' if is_live else 'NO '}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 50)
        print("DEMO SUMMARY")
        print("=" * 50)
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_liveness = total_liveness_score / liveness_detections if liveness_detections > 0 else 0
        
        print(f"Detector Type: {detector_type.upper()}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Liveness Detections: {liveness_detections}")
        print(f"Average Liveness Score: {avg_liveness:.3f}")
        
        # Get final performance metrics
        final_metrics = detector.get_performance_metrics()
        print(f"Final Processing Time: {final_metrics.get('avg_processing_time', 0)*1000:.2f}ms")
        
        # Performance assessment
        print("\nPERFORMANCE ASSESSMENT:")
        if avg_fps >= 20:
            print("✓ Excellent real-time performance")
        elif avg_fps >= 10:
            print("✓ Good real-time performance")
        elif avg_fps >= 5:
            print("⚠ Acceptable performance")
        else:
            print("✗ Poor performance - consider optimization")
        
        if liveness_detections > 0:
            if avg_liveness >= 0.7:
                print("✓ High confidence liveness detection")
            elif avg_liveness >= 0.4:
                print("⚠ Moderate confidence liveness detection")
            else:
                print("✗ Low confidence liveness detection")
        else:
            print("⚠ No liveness detections - ensure face is visible")

def benchmark_detectors(duration=30):
    """
    Benchmark both detectors side by side
    """
    print("NEEDLE Detector Benchmark")
    print("=" * 50)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detectors = {
        'OpenCV': OpenCVEyeDetector(),
        'MediaPipe': MediaPipeEyeDetector()
    }
    
    results = {}
    
    for name, detector in detectors.items():
        print(f"\nTesting {name} detector...")
        detector.reset()
        
        start_time = time.time()
        frame_count = 0
        total_score = 0
        detections = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            result = detector.process_frame(frame)
            frame_count += 1
            
            # Collect liveness scores
            if result['liveness_results']:
                avg_score = np.mean([r['smoothed_score'] for r in result['liveness_results']])
                total_score += avg_score
                detections += 1
        
        elapsed = time.time() - start_time
        metrics = detector.get_performance_metrics()
        
        results[name] = {
            'fps': frame_count / elapsed,
            'processing_time': metrics.get('avg_processing_time', 0) * 1000,
            'avg_liveness_score': total_score / detections if detections > 0 else 0,
            'total_frames': frame_count,
            'detections': detections
        }
        
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {results[name]['fps']:.2f}")
        print(f"  Processing time: {results[name]['processing_time']:.2f}ms")
        print(f"  Liveness detections: {detections}")
        print(f"  Average score: {results[name]['avg_liveness_score']:.3f}")
    
    cap.release()
    
    # Compare results
    print("\n" + "=" * 50)
    print("BENCHMARK COMPARISON")
    print("=" * 50)
    
    opencv_result = results['OpenCV']
    mediapipe_result = results['MediaPipe']
    
    print(f"Speed (FPS):")
    print(f"  OpenCV: {opencv_result['fps']:.2f}")
    print(f"  MediaPipe: {mediapipe_result['fps']:.2f}")
    print(f"  Winner: {'OpenCV' if opencv_result['fps'] > mediapipe_result['fps'] else 'MediaPipe'}")
    
    print(f"\nProcessing Time (ms):")
    print(f"  OpenCV: {opencv_result['processing_time']:.2f}")
    print(f"  MediaPipe: {mediapipe_result['processing_time']:.2f}")
    print(f"  Winner: {'OpenCV' if opencv_result['processing_time'] < mediapipe_result['processing_time'] else 'MediaPipe'}")
    
    print(f"\nAccuracy (Avg Score):")
    print(f"  OpenCV: {opencv_result['avg_liveness_score']:.3f}")
    print(f"  MediaPipe: {mediapipe_result['avg_liveness_score']:.3f}")
    print(f"  Winner: {'OpenCV' if opencv_result['avg_liveness_score'] > mediapipe_result['avg_liveness_score'] else 'MediaPipe'}")

def main():
    parser = argparse.ArgumentParser(description='NEEDLE Liveness Detection Demo')
    parser.add_argument('--detector', choices=['opencv', 'mediapipe'], default='opencv',
                       help='Detector type to use (default: opencv)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Demo duration in seconds (default: 30)')
    parser.add_argument('--no-video', action='store_true',
                       help='Run without video display')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison of both detectors')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_detectors(args.duration)
    else:
        run_demo(args.detector, args.duration, not args.no_video)

if __name__ == "__main__":
    main()
