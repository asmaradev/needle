# NEEDLE Liveness Detection System
## Natural Eye-movement Evaluation for Detecting Live Entities

A novel liveness detection system that uses micro eye-movement pattern analysis to distinguish between real human eyes and static/fake images. This system provides a comprehensive benchmarking platform comparing MediaPipe and OpenCV implementations.

## 🎯 Features

### NEEDLE Algorithm
The core innovation of this system is the **NEEDLE** (Natural Eye-movement Evaluation for Detecting Live Entities) algorithm, which combines multiple micro eye-movement patterns:

- **Blink Pattern Analysis**: Frequency, duration, and regularity of blinks
- **Saccade Pattern Detection**: Large rapid eye movements
- **Microsaccade Pattern Analysis**: Small involuntary eye movements
- **Pupil Variation Tracking**: Natural pupil size changes
- **Temporal Consistency**: Movement pattern consistency over time
- **Movement Naturalness**: Entropy-based naturalness scoring

### Dual Implementation
- **OpenCV-based Detection**: Traditional computer vision approach
- **MediaPipe-based Detection**: Modern ML-powered face mesh analysis
- **Real-time Benchmarking**: Side-by-side performance comparison

### Advanced Features
- Real-time liveness scoring (0.0 - 1.0)
- Component-wise analysis and visualization
- Performance metrics tracking (FPS, processing time, accuracy)
- Comprehensive GUI with live video feed
- Data export capabilities (JSON/CSV)
- Configurable parameters and thresholds

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- Operating System: Windows, macOS, or Linux

### Setup
1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- OpenCV (4.8.1.78)
- MediaPipe (0.10.7)
- NumPy (1.24.3)
- Matplotlib (3.7.2)
- SciPy (1.11.2)
- Pandas (2.0.3)
- Pillow (10.0.0)
- Seaborn (0.12.2)
- Tkinter (included with Python)

## 📖 Usage

### Quick Start
```bash
python main.py
```

### GUI Interface

#### 1. Detector Selection
- **OpenCV**: Traditional cascade-based detection
- **MediaPipe**: ML-powered face mesh detection

#### 2. Live Detection
- Click "Start Detection" to begin real-time analysis
- View live video feed with overlay annotations
- Monitor real-time NEEDLE scores and component analysis

#### 3. Benchmarking
- Click "Start Benchmark" to compare both detectors
- Set benchmark duration (10-300 seconds)
- View detailed performance comparison

#### 4. Configuration
- **NEEDLE Threshold**: Liveness detection threshold (0.0-1.0)
- **Window Size**: Analysis window size (10-60 frames)

### Tabs Overview

#### Live Detection Tab
- Real-time video feed with annotations
- Live metrics panel showing:
  - FPS and processing time
  - NEEDLE liveness score
  - Individual component scores
  - Detection status

#### Benchmark Results Tab
- Automated testing of both detectors
- Performance comparison metrics
- Statistical analysis and winner determination

#### Component Analysis Tab
- Detailed visualization of NEEDLE components
- Real-time plotting of component scores
- Historical trend analysis

## 🧠 NEEDLE Algorithm Details

### Core Components

1. **Blink Pattern Analysis** (Weight: 25%)
   - Frequency analysis (0.1-0.8 Hz optimal)
   - Duration validation (3-15 frames)
   - Regularity scoring using coefficient of variation

2. **Saccade Pattern Detection** (Weight: 20%)
   - Large eye movement detection (>2.0 pixels/frame)
   - Frequency optimization (0.1-0.3 optimal)
   - Natural movement validation

3. **Microsaccade Pattern Analysis** (Weight: 15%)
   - Small involuntary movements (0.5-2.0 pixels/frame)
   - Frequency analysis (0.05-0.15 optimal)
   - Unconscious movement detection

4. **Pupil Variation Tracking** (Weight: 15%)
   - Size variation coefficient analysis
   - Natural variation range (0.05-0.25)
   - Lighting adaptation response

5. **Temporal Consistency** (Weight: 15%)
   - Movement pattern consistency
   - Velocity variation analysis
   - Smoothness evaluation

6. **Movement Naturalness** (Weight: 10%)
   - Entropy-based analysis
   - Spectral analysis of movement patterns
   - Randomness vs. predictability balance

### Scoring System
- Each component contributes to the final NEEDLE score
- Weighted combination based on biological importance
- Final score range: 0.0 (definitely not live) to 1.0 (definitely live)
- Default threshold: 0.6 for liveness determination

## 📊 Performance Metrics

### Tracked Metrics
- **FPS**: Frames processed per second
- **Processing Time**: Average time per frame (milliseconds)
- **Detection Accuracy**: Average liveness score accuracy
- **Component Scores**: Individual algorithm component performance

### Benchmark Comparison
The system automatically compares:
- Speed performance (FPS)
- Processing efficiency (time per frame)
- Detection accuracy (liveness scoring)
- Resource utilization

## 🔧 Configuration

### Config Parameters (`config.py`)
```python
# NEEDLE Algorithm Parameters
NEEDLE_WINDOW_SIZE = 30      # Analysis window size
NEEDLE_THRESHOLD = 0.6       # Liveness threshold

# Detection Parameters
BLINK_THRESHOLD = 0.25       # Eye aspect ratio threshold
SACCADE_THRESHOLD = 2.0      # Saccade velocity threshold
PUPIL_VARIATION_THRESHOLD = 0.15  # Pupil variation threshold

# Component Weights
NEEDLE_WEIGHTS = {
    'blink_pattern': 0.25,
    'saccade_pattern': 0.20,
    'microsaccade_pattern': 0.15,
    'pupil_variation': 0.15,
    'temporal_consistency': 0.15,
    'movement_naturalness': 0.10
}
```

## 📁 Project Structure

```
needle-liveness-detection/
├── main.py                 # Main GUI application
├── config.py              # Configuration parameters
├── utils.py               # Utility functions and classes
├── liveness_analyzer.py   # NEEDLE algorithm implementation
├── opencv_detector.py     # OpenCV-based detection
├── mediapipe_detector.py  # MediaPipe-based detection
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔬 Technical Details

### Eye Landmark Detection
- **OpenCV**: Haar cascade + contour analysis
- **MediaPipe**: 468-point face mesh with iris landmarks

### Movement Analysis
- Optical flow-based tracking
- Velocity calculation and smoothing
- Statistical pattern analysis

### Liveness Determination
- Multi-component scoring system
- Temporal window analysis
- Threshold-based classification

## 📈 Use Cases

### Security Applications
- Access control systems
- Identity verification
- Anti-spoofing measures

### Research Applications
- Biometric research
- Computer vision benchmarking
- Algorithm development

### Commercial Applications
- Mobile device authentication
- Online identity verification
- Surveillance systems

## 🛠️ Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera permissions
   - Verify camera index in config.py
   - Test camera with other applications

2. **Poor Detection Performance**
   - Ensure good lighting conditions
   - Position face clearly in frame
   - Adjust NEEDLE threshold if needed

3. **Low FPS Performance**
   - Close other applications using camera
   - Reduce frame resolution in config
   - Use OpenCV detector for better performance

4. **Installation Issues**
   - Update pip: `pip install --upgrade pip`
   - Install dependencies individually if batch install fails
   - Check Python version compatibility

## 📚 Research Background

### Biological Basis
The NEEDLE algorithm is based on natural human eye movement patterns:
- Involuntary microsaccades occur 1-3 times per second
- Natural blink rates vary between 6-48 blinks per minute
- Pupil size naturally varies with lighting and cognitive load
- Eye movements follow predictable statistical patterns

### Innovation
- First algorithm to combine all major micro-movement patterns
- Novel entropy-based naturalness scoring
- Comprehensive temporal analysis approach
- Real-time implementation with high accuracy

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

### Areas for Contribution
- Additional movement pattern analysis
- Performance optimizations
- New detector implementations
- Enhanced visualization features

## 📄 License

This project is provided for research and educational purposes. Please ensure compliance with local regulations when using biometric detection systems.

## 🙏 Acknowledgments

- MediaPipe team for face mesh technology
- OpenCV community for computer vision tools
- Research community for eye movement analysis foundations

## 📞 Support

For technical support or research collaboration:
- Create an issue in the repository
- Provide detailed error descriptions
- Include system specifications and logs

---

**NEEDLE** - Natural Eye-movement Evaluation for Detecting Live Entities
*Advancing the state of liveness detection through micro eye-movement analysis*
