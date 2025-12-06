# Real-time Surgical Instrument Detection and Tracking System

A real-time surgical instrument detection, segmentation, and angle analysis system based on YOLOv8 and SAM2

## 🌟 Features

- **Real-time Object Detection**: YOLOv8-based surgical instrument detection
- **Precise Instance Segmentation**: High-precision segmentation with SAM2 integration
- **Angle and Ratio Calculation**: Real-time calculation of instrument opening angles and length ratios
- **Anomaly Detection**: FFT and sliding window-based motion anomaly detection
- **Real-time Visualization**: Dot markers, arrow indicators, and text display
- **Memory Optimization**: All data computed in real-time, no disk space usage

## 🎬 Demo

![Surgical Instrument Tracking Demo](./assets/demo.gif)

## 🚀 Quick Start

### Requirements

- Python 3.10+
- CUDA 11.8+ or 12.0+ (GPU recommended)
- 8GB+ GPU Memory

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/your-username/surgical-instrument-tracking.git
cd surgical-instrument-tracking
```

#### 2. Install PyTorch (Choose based on your CUDA version)
```bash
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.0
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu120

```

#### 3. Install Project Dependencies
```bash
cd pro
pip install -e .
```

#### 4. Install Additional Dependencies
```bash
pip install ultralytics  opencv-python
```

#### 5. Download SAM2 Model Weights
```bash
cd checkpoints
./download_ckpts.sh
```

Or download manually:
```bash
# Download large model (recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Download small model (faster)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

In this project, We have downloaded sam2.1 model from tiny to large, if you want more, please visit https://github.com/facebookresearch/sam2 for more models

```

## 📁 Model Setup

### 1. SAM2 Models
Place downloaded SAM2 models in the `checkpoints/` directory:
- `sam2.1_hiera_large.pt` (recommended, higher accuracy)
- `sam2.1_hiera_small.pt` (faster)

### 2. YOLOv8 Model
You need to train or obtain a YOLOv8 model specifically for surgical instrument detection:
- Rename your trained model file 
- Place it in your project root:

**If you don't have a trained YOLO model, you can download a pretrained model for testing:**
```bash
# Download pretrained YOLOv8x model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

```

## ⚙️ Configuration

Modify the following configurations in `main_plus.py`:

```python
# Model path configuration
YOLO_CHECKPOINT  # YOLO model path
SAM2_CHECKPOINT  # SAM2 model path
SAM2_CFG         # SAM2 config file

All of these can use your own trained models.

# Detection parameters
window_size = 30    # Anomaly detection window size
step_size = 15      # Anomaly detection step size
sample_rate = 30    # Sample rate
```

## 🎯 Usage

### Basic Usage

```python
cd pro
python sam2/main_plus.py
```

### Advanced Configuration

```python
from sam2.main_plus import RealtimeTrackingPipeline

# Create pipeline with custom configuration
pipeline = RealtimeTrackingPipeline(
    yolo_checkpoint='path/to/your/yolo/model.pt',
    sam2_checkpoint='path/to/your/sam2/model.pt',
    sam2_cfg='path/to/sam2/config.yaml',
    window_size=30,  # Anomaly detection window size
    step_size=15     # Anomaly detection step size
)

# Use different cameras
pipeline.run(camera_id=0)  # Default camera
pipeline.run(camera_id=1)  # External camera
```

### Interactive Controls

Keyboard controls during runtime:
- **'q'**: Exit the program
- **'r'**: Manually reset the tracker

If you find that segmentation goes wrong, you can press 'r' to renovate the detection.

## 🔧 Technical Architecture

### System Workflow

1. **YOLO Detection**: Use YOLOv8 to detect surgical instrument positions
2. **SAM2 Segmentation**: Perform precise instance segmentation based on detection results
3. **Angle Calculation**: Real-time calculation of instrument opening angles and ratios
4. **Anomaly Detection**: Use FFT and sliding windows to detect motion anomalies
5. **Visualization**: Real-time display of detection results and analysis data

### Core Components

- **ObjectDetector**: YOLOv8 object detector optimized for surgical instrument detection
- **SAM2CameraPredictor**: Real-time video segmentation tracker
- **Angle Calculation Module**: Geometric algorithm-based opening angle analysis
- **Anomaly Detection Module**: FFT signal processing and sliding window analysis

### Algorithm Details

#### Surgical Instrument Detection Strategy
```python
# Scoring system
1. Objects entering from the side (+3 points)
2. Elongated objects with aspect ratio > 2.0 (+2-5 points)
3. Detection confidence weighting (+0-2 points)
4. Exclude oversized objects (area > 50% of frame, -5 points)
```

#### Angle Calculation Method
- Use RANSAC algorithm to fit upper and lower edge line segments
- Calculate line segment intersections and endpoints
- Calculate opening angle based on vector angle formulas

#### Anomaly Detection Algorithm
- **Gradient Detection**: Gradient changes after Gaussian filtering
- **FFT Analysis**: Frequency domain anomaly detection
- **Sliding Window**: Real-time windowed processing

## 🐛 Troubleshooting

### 1. CUDA Out of Memory
```bash
# Try using small model
SAM2_CHECKPOINT = 'sam2.1_hiera_small.pt'
SAM2_CFG = "sam2.1_hiera_s.yaml"
```

### 2. Camera Cannot Open
```python
# Check available cameras
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
```

### 3. SAM2 Model Loading Failed
```bash
# Check if model files exist
ls -la checkpoints/
# Ensure correct model weights are downloaded
```

### 4. Poor Detection Performance
- Adjust `score_thres` and `iou_thres` parameters
- Use a specially trained YOLO model
- Ensure good lighting conditions

## 🤝 Contributing

Issues and Pull Requests are welcome!

```

## 📝 License

This project is open source under the Apache 2.0 License.

## 🙏 Acknowledgments

- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta AI's Segment Anything Model 2
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLOv8 object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Contact

For questions or suggestions, please contact:
- Submit GitHub Issues
- Email: your-email@example.com

---

**Note**: This system is for research and educational purposes only. Conduct sufficient validation and testing before use in medical environments.