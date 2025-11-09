"""
NEEDLE Liveness Detection System
Natural Eye-movement Evaluation for Detecting Live Entities

Main application with GUI for benchmarking MediaPipe vs OpenCV
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from collections import deque

from opencv_detector import OpenCVEyeDetector
from mediapipe_detector import MediaPipeEyeDetector
from config import Config

class NEEDLELivenessApp:
    """
    Main application class for NEEDLE Liveness Detection System
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry("1200x800")
        
        # Initialize detectors
        self.opencv_detector = OpenCVEyeDetector()
        self.mediapipe_detector = MediaPipeEyeDetector()
        
        # Application state
        self.is_running = False
        self.current_detector = "opencv"  # "opencv" or "mediapipe"
        self.camera = None
        self.benchmark_data = []
        self.start_time = None

        # Component analysis state
        self.component_names = [
            'blink_pattern', 'saccade_pattern', 'microsaccade_pattern',
            'pupil_variation', 'temporal_consistency', 'movement_naturalness'
        ]
        # Maintain separate histories for each detector so we can compare them side-by-side
        self.component_history_by_detector = {
            'opencv': {name: deque(maxlen=300) for name in self.component_names},
            'mediapipe': {name: deque(maxlen=300) for name in self.component_names}
        }
        self.last_plot_update = 0.0
        self.plot_update_interval = 0.2  # seconds
        # Plot styling
        self.analysis_title_fontsize = 9
        self.analysis_label_fontsize = 8
        self.analysis_tick_fontsize = 7
        self.analysis_legend_fontsize = 7
        self.analysis_title_pad = 6
        self.analysis_label_pad = 2
        
        # Performance tracking
        self.performance_history = {
            'opencv': {'fps': [], 'accuracy': [], 'processing_time': []},
            'mediapipe': {'fps': [], 'accuracy': [], 'processing_time': []}
        }
        
        self.setup_gui()
        self.setup_camera()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Setup control panel
        self.setup_control_panel()
        
        # Setup content area
        self.setup_content_area()
        
    def setup_control_panel(self):
        """Setup control panel with buttons and options"""
        # Detector selection
        detector_frame = ttk.LabelFrame(self.control_frame, text="Detector Selection")
        detector_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.detector_var = tk.StringVar(value="opencv")
        ttk.Radiobutton(detector_frame, text="OpenCV", variable=self.detector_var, 
                       value="opencv", command=self.switch_detector).pack(anchor=tk.W)
        ttk.Radiobutton(detector_frame, text="MediaPipe", variable=self.detector_var, 
                       value="mediapipe", command=self.switch_detector).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.LabelFrame(self.control_frame, text="Controls")
        button_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                      command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        self.benchmark_button = ttk.Button(button_frame, text="Start Benchmark", 
                                          command=self.start_benchmark)
        self.benchmark_button.pack(side=tk.LEFT, padx=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(self.control_frame, text="Settings")
        settings_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="NEEDLE Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=Config.NEEDLE_THRESHOLD)
        threshold_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="Window Size:").pack(anchor=tk.W)
        self.window_size_var = tk.IntVar(value=Config.NEEDLE_WINDOW_SIZE)
        window_size_scale = ttk.Scale(settings_frame, from_=10, to=60, 
                                     variable=self.window_size_var, orient=tk.HORIZONTAL)
        window_size_scale.pack(fill=tk.X)
        
    def setup_content_area(self):
        """Setup content area with video display and metrics"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Video tab
        self.video_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.video_frame, text="Live Detection")
        
        # Video display
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Metrics panel
        self.metrics_frame = ttk.LabelFrame(self.video_frame, text="Real-time Metrics")
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.setup_metrics_panel()
        
        # Benchmark tab
        self.benchmark_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.benchmark_frame, text="Benchmark Results")
        
        self.setup_benchmark_panel()
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Component Analysis")
        
        self.setup_analysis_panel()
        
    def setup_metrics_panel(self):
        """Setup real-time metrics display"""
        # Current metrics
        self.fps_label = ttk.Label(self.metrics_frame, text="FPS: 0.0")
        self.fps_label.pack(anchor=tk.W, pady=2)
        
        self.processing_time_label = ttk.Label(self.metrics_frame, text="Processing Time: 0.0ms")
        self.processing_time_label.pack(anchor=tk.W, pady=2)
        
        self.needle_score_label = ttk.Label(self.metrics_frame, text="NEEDLE Score: 0.000")
        self.needle_score_label.pack(anchor=tk.W, pady=2)
        
        self.liveness_status_label = ttk.Label(self.metrics_frame, text="Status: Unknown")
        self.liveness_status_label.pack(anchor=tk.W, pady=2)
        
        # Component scores
        ttk.Separator(self.metrics_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.metrics_frame, text="NEEDLE Components:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.component_labels = {}
        components = ['blink_pattern', 'saccade_pattern', 'microsaccade_pattern', 
                     'pupil_variation', 'temporal_consistency', 'movement_naturalness']
        
        for component in components:
            label = ttk.Label(self.metrics_frame, text=f"{component.replace('_', ' ').title()}: 0.000")
            label.pack(anchor=tk.W, pady=1)
            self.component_labels[component] = label
        
        # Export button
        ttk.Separator(self.metrics_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(self.metrics_frame, text="Export Data", 
                  command=self.export_data).pack(pady=5)
        
    def setup_benchmark_panel(self):
        """Setup benchmark results panel"""
        # Benchmark controls
        benchmark_controls = ttk.Frame(self.benchmark_frame)
        benchmark_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(benchmark_controls, text="Benchmark Duration (seconds):").pack(side=tk.LEFT)
        self.benchmark_duration_var = tk.IntVar(value=Config.BENCHMARK_DURATION)
        duration_spinbox = ttk.Spinbox(benchmark_controls, from_=10, to=300, 
                                      textvariable=self.benchmark_duration_var, width=10)
        duration_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.benchmark_progress = ttk.Progressbar(benchmark_controls, mode='determinate')
        self.benchmark_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Results display
        self.benchmark_text = tk.Text(self.benchmark_frame, height=15, width=80)
        self.benchmark_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for text
        scrollbar = ttk.Scrollbar(self.benchmark_frame, orient=tk.VERTICAL, command=self.benchmark_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.benchmark_text.config(yscrollcommand=scrollbar.set)
        
    def setup_analysis_panel(self):
        """Setup component analysis panel with plots"""
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        # self.fig.suptitle('NEEDLE Component Analysis')
        
        # Map axes to components and set titles
        self.axis_component_map = {}
        titles = [
            'Blink Pattern', 'Saccade Pattern', 'Microsaccade Pattern',
            'Pupil Variation', 'Temporal Consistency', 'Movement Naturalness'
        ]
        for ax, comp_name, title in zip(self.axes.flat, self.component_names, titles):
            self.axis_component_map[ax] = comp_name
            ax.set_title(title, fontsize=self.analysis_title_fontsize, pad=self.analysis_title_pad)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel('Frame', fontsize=self.analysis_label_fontsize, labelpad=self.analysis_label_pad)
            ax.set_ylabel('Score', fontsize=self.analysis_label_fontsize, labelpad=self.analysis_label_pad)
            ax.tick_params(axis='both', labelsize=self.analysis_tick_fontsize)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.analysis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.update_analysis_plots()
        
    def setup_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, Config.FPS)
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to initialize camera: {str(e)}")
    
    def switch_detector(self):
        """Switch between OpenCV and MediaPipe detectors"""
        self.current_detector = self.detector_var.get()
        
        # Reset current detector
        if self.current_detector == "opencv":
            self.opencv_detector.reset()
        else:
            self.mediapipe_detector.reset()
    
    def start_detection(self):
        """Start live detection"""
        if not self.camera or not self.camera.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop live detection"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Process frame with both detectors so we can compare components side-by-side
            try:
                results_opencv = self.opencv_detector.process_frame(frame)
            except Exception:
                results_opencv = {'liveness_results': [], 'performance': {}}
            try:
                results_mediapipe = self.mediapipe_detector.process_frame(frame)
            except Exception:
                results_mediapipe = {'liveness_results': [], 'performance': {}}

            # Choose which output to render based on current detector
            if self.current_detector == "opencv":
                output_frame = self.opencv_detector.draw_results(frame, results_opencv)
                # Add detailed analysis for OpenCV (six progress bars)
                output_frame = self.opencv_detector.draw_detailed_analysis(output_frame, results_opencv)
            else:
                output_frame = self.mediapipe_detector.draw_results(frame, results_mediapipe)
                # Add detailed analysis for MediaPipe
                output_frame = self.mediapipe_detector.draw_detailed_analysis(output_frame, results_mediapipe)
            
            # Append component histories for both detectors
            self._append_component_history('opencv', results_opencv)
            self._append_component_history('mediapipe', results_mediapipe)
            
            # Update GUI metrics using the currently selected detector's results
            results_for_gui = results_opencv if self.current_detector == "opencv" else results_mediapipe
            self.update_gui_metrics(results_for_gui)
            self.display_frame(output_frame)
            
            # Small delay to prevent overwhelming the GUI
            time.sleep(0.01)
    
    def _append_component_history(self, detector_name, results):
        """Append averaged component scores into per-detector history buffers."""
        try:
            if not results or not results.get('liveness_results'):
                return
            # Aggregate across all detected faces (if multiple) by averaging per component
            all_components = {}
            for r in results['liveness_results']:
                comps = r.get('components', {})
                for comp_name, score in comps.items():
                    all_components.setdefault(comp_name, []).append(float(score))
            # Append the averaged value per component into the history for the given detector
            det_hist = self.component_history_by_detector.get(detector_name)
            if not det_hist:
                return
            for comp_name in self.component_names:
                if comp_name in all_components and comp_name in det_hist:
                    avg_val = float(np.mean(all_components[comp_name]))
                    det_hist[comp_name].append(avg_val)
        except Exception as e:
            # Keep UI responsive even if a single frame fails aggregation
            print(f"Error appending component history for {detector_name}: {e}")
    
    def start_benchmark(self):
        """Start benchmarking both detectors"""
        if not self.camera or not self.camera.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        # Disable controls during benchmark
        self.benchmark_button.config(state=tk.DISABLED)
        
        # Start benchmark thread
        self.benchmark_thread = threading.Thread(target=self.benchmark_loop)
        self.benchmark_thread.daemon = True
        self.benchmark_thread.start()
    
    def benchmark_loop(self):
        """Benchmark both detectors"""
        duration = self.benchmark_duration_var.get()
        self.benchmark_data = []
        
        # Clear previous results
        self.benchmark_text.delete(1.0, tk.END)
        self.benchmark_text.insert(tk.END, "Starting benchmark...\n")
        
        detectors = [
            ("OpenCV", self.opencv_detector),
            ("MediaPipe", self.mediapipe_detector)
        ]
        
        for detector_name, detector in detectors:
            self.benchmark_text.insert(tk.END, f"\nTesting {detector_name}...\n")
            self.benchmark_text.see(tk.END)
            
            detector.reset()
            start_time = time.time()
            frame_count = 0
            total_processing_time = 0
            liveness_scores = []
            
            self.benchmark_progress['maximum'] = duration
            
            while time.time() - start_time < duration:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Process frame
                process_start = time.time()
                results = detector.process_frame(frame)
                process_time = time.time() - process_start
                
                frame_count += 1
                total_processing_time += process_time
                
                # Collect liveness scores
                if results['liveness_results']:
                    avg_score = np.mean([r['smoothed_score'] for r in results['liveness_results']])
                    liveness_scores.append(avg_score)
                else:
                    # Ensure coverage parity across detectors by counting no-detection frames
                    liveness_scores.append(0.0)
                
                # Update progress
                elapsed = time.time() - start_time
                self.benchmark_progress['value'] = elapsed
                self.root.update_idletasks()
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            avg_processing_time = total_processing_time / frame_count if frame_count > 0 else 0
            avg_liveness_score = np.mean(liveness_scores) if liveness_scores else 0
            
            # Store results
            benchmark_result = {
                'detector': detector_name,
                'fps': fps,
                'avg_processing_time': avg_processing_time * 1000,  # Convert to ms
                'avg_liveness_score': avg_liveness_score,
                'total_frames': frame_count,
                'duration': elapsed_time
            }
            self.benchmark_data.append(benchmark_result)
            
            # Display results
            result_text = f"""
{detector_name} Results:
- FPS: {fps:.2f}
- Avg Processing Time: {avg_processing_time*1000:.2f}ms
- Avg Liveness Score: {avg_liveness_score:.3f}
- Total Frames: {frame_count}
- Duration: {elapsed_time:.2f}s
"""
            self.benchmark_text.insert(tk.END, result_text)
            self.benchmark_text.see(tk.END)
        
        # Compare results
        if len(self.benchmark_data) == 2:
            opencv_result = self.benchmark_data[0]
            mediapipe_result = self.benchmark_data[1]
            
            comparison_text = f"""
COMPARISON:
- FPS: OpenCV {opencv_result['fps']:.2f} vs MediaPipe {mediapipe_result['fps']:.2f}
- Processing Time: OpenCV {opencv_result['avg_processing_time']:.2f}ms vs MediaPipe {mediapipe_result['avg_processing_time']:.2f}ms
- Accuracy: OpenCV {opencv_result['avg_liveness_score']:.3f} vs MediaPipe {mediapipe_result['avg_liveness_score']:.3f}

Winner:
- Speed: {'OpenCV' if opencv_result['fps'] > mediapipe_result['fps'] else 'MediaPipe'}
- Efficiency: {'OpenCV' if opencv_result['avg_processing_time'] < mediapipe_result['avg_processing_time'] else 'MediaPipe'}
- Accuracy: {'OpenCV' if opencv_result['avg_liveness_score'] > mediapipe_result['avg_liveness_score'] else 'MediaPipe'}
"""
            self.benchmark_text.insert(tk.END, comparison_text)
            self.benchmark_text.see(tk.END)
        
        # Re-enable controls
        self.benchmark_button.config(state=tk.NORMAL)
        self.benchmark_progress['value'] = 0
    
    def update_gui_metrics(self, results):
        """Update GUI with current metrics"""
        try:
            # Performance metrics
            perf = results.get('performance', {})
            self.fps_label.config(text=f"FPS: {perf.get('fps', 0):.1f}")
            self.processing_time_label.config(text=f"Processing Time: {perf.get('avg_processing_time', 0)*1000:.1f}ms")
            
            # Liveness metrics
            if results['liveness_results']:
                avg_score = np.mean([r['smoothed_score'] for r in results['liveness_results']])
                self.needle_score_label.config(text=f"NEEDLE Score: {avg_score:.3f}")
                
                status = "LIVE" if avg_score >= self.threshold_var.get() else "NOT LIVE"
                color = "green" if status == "LIVE" else "red"
                self.liveness_status_label.config(text=f"Status: {status}", foreground=color)
                
                # Component scores
                all_components = {}
                for result in results['liveness_results']:
                    components = result.get('components', {})
                    for comp_name, score in components.items():
                        if comp_name not in all_components:
                            all_components[comp_name] = []
                        all_components[comp_name].append(score)
                
                # Update component labels (histories are handled in detection_loop)
                for comp_name, label in self.component_labels.items():
                    if comp_name in all_components:
                        avg_comp_score = np.mean(all_components[comp_name])
                        label.config(text=f"{comp_name.replace('_', ' ').title()}: {avg_comp_score:.3f}")
                
                # Throttle plot updates to reduce overhead
                now = time.time()
                if now - self.last_plot_update >= self.plot_update_interval:
                    self.update_analysis_plots()
                    self.last_plot_update = now
            else:
                self.needle_score_label.config(text="NEEDLE Score: 0.000")
                self.liveness_status_label.config(text="Status: No Eyes Detected", foreground="orange")
                
                for label in self.component_labels.values():
                    label.config(text=label.cget("text").split(":")[0] + ": 0.000")
        
        except Exception as e:
            print(f"Error updating GUI metrics: {e}")
    
    def display_frame(self, frame):
        """Display frame in GUI"""
        try:
            # Resize frame to fit display - make the camera view wider by targeting a larger width
            aspect_ratio = frame.shape[1] / frame.shape[0]
            target_width = getattr(Config, 'DISPLAY_WIDTH', 800)
            # Compute height from width to preserve aspect
            display_width = int(target_width)
            display_height = int(display_width / aspect_ratio)
            # Optional: clamp to a maximum height to avoid oversized images
            max_height = getattr(Config, 'DISPLAY_MAX_HEIGHT', 600)
            if display_height > max_height:
                display_height = max_height
                display_width = int(display_height * aspect_ratio)
            
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert to RGB and then to PhotoImage
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def update_analysis_plots(self):
        """Update component analysis plots (show both detectors)"""
        try:
            threshold = float(self.threshold_var.get()) if hasattr(self, 'threshold_var') else 0.6
            for ax in self.axes.flat:
                comp_name = self.axis_component_map.get(ax)
                # Preserve existing title based on component
                title = ax.get_title() if ax.get_title() else comp_name.replace('_', ' ').title()
                ax.clear()
                ax.set_title(title, pad=self.analysis_title_pad, fontsize=self.analysis_title_fontsize)
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_xlabel('Frame', fontsize=self.analysis_label_fontsize, labelpad=self.analysis_label_pad)
                ax.set_ylabel('Score', fontsize=self.analysis_label_fontsize, labelpad=self.analysis_label_pad)
                ax.tick_params(axis='both', labelsize=self.analysis_tick_fontsize)
                ax.margins(x=0.02, y=0.1)

                # Retrieve histories for both detectors
                data_opencv = list(self.component_history_by_detector['opencv'].get(comp_name, [])) if comp_name else []
                data_mediapipe = list(self.component_history_by_detector['mediapipe'].get(comp_name, [])) if comp_name else []

                any_data = False
                xlim_max = 50
                if data_mediapipe:
                    ax.plot(data_mediapipe, color='tab:blue', linewidth=1.3, label='MediaPipe')
                    any_data = True
                    xlim_max = max(xlim_max, len(data_mediapipe))
                if data_opencv:
                    # Use a yellow/gold tone for OpenCV as requested
                    ax.plot(data_opencv, color='goldenrod', linewidth=1.3, label='OpenCV')
                    any_data = True
                    xlim_max = max(xlim_max, len(data_opencv))

                if any_data:
                    ax.set_xlim(0, xlim_max)
                else:
                    ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', fontsize=8, alpha=0.6)
                # Threshold line for clearer comparison
                ax.axhline(threshold, color='tab:red', linestyle='--', linewidth=1, label=f'Threshold {threshold:.2f}')
                # Add legend to clarify plotted elements
                ax.legend(loc='upper right', fontsize=self.analysis_legend_fontsize, frameon=False)

            # Improve spacing to avoid overlaps between axes labels and titles
            if not self.fig.get_constrained_layout():
                self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.98, wspace=0.30, hspace=0.40)
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating analysis plots: {e}")
    
    def export_data(self):
        """Export benchmark data to file"""
        if not self.benchmark_data:
            messagebox.showwarning("No Data", "No benchmark data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    df = pd.DataFrame(self.benchmark_data)
                    df.to_csv(filename, index=False)
                else:
                    with open(filename, 'w') as f:
                        json.dump(self.benchmark_data, f, indent=2)
                
                messagebox.showinfo("Export Successful", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = NEEDLELivenessApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
