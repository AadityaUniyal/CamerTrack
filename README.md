# 🎥 AI Vision Studio

A full-stack **AI/ML Computer Vision** project featuring real-time webcam processing with multiple AI models, controlled via a premium web dashboard.

## ✨ Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| 👤 **Face Detection** | MediaPipe | Real-time face bounding boxes with confidence |
| ✋ **Hand Tracking** | MediaPipe | 21-point skeleton + gesture recognition (5 gestures) |
| 📦 **Object Detection** | MobileNet SSD | 20+ object categories with labels |
| 😊 **Emotion Detection** | DeepFace | 7 emotions with confidence bars |
| 🎛️ **Dashboard** | Flask + JS | Toggle modules, live stats, screenshots |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Open in browser
# → http://localhost:5000
```

## 🏗️ Project Structure

```
├── app.py                 # Flask entry point
├── config.py              # Settings & thresholds
├── camera/
│   └── camera_manager.py  # Webcam capture & streaming
├── ai_modules/
│   ├── face_detector.py   # MediaPipe Face Detection
│   ├── hand_tracker.py    # MediaPipe Hand Tracking
│   ├── object_detector.py # MobileNet SSD Object Detection
│   └── emotion_detector.py # DeepFace Emotion Analysis
├── utils/
│   └── drawing.py         # Styled OpenCV overlays
├── templates/
│   └── index.html         # Dashboard HTML
└── static/
    ├── css/style.css      # Dark glassmorphism theme
    └── js/app.js          # Frontend interactivity
```

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-4` | Toggle individual AI modules |
| `S` | Take screenshot |
| `Space` | Toggle all modules |

## 🛠️ Tech Stack

- **Python 3.10+** — Backend
- **Flask** — Web framework
- **OpenCV** — Camera & image processing
- **MediaPipe** — Face & hand detection
- **DeepFace** — Emotion analysis
- **MobileNet SSD** — Object detection
- **HTML/CSS/JS** — Premium dark dashboard

## 📋 Requirements

- Python 3.10+
- Webcam (optional — shows placeholder if absent)
- ~100MB disk space (for ML models)
