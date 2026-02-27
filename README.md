# AI Vision Studio

A real-time computer vision dashboard built with Python and Flask. Run 13 AI modules simultaneously on a live webcam feed — all togglable from a browser-based dark dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## Features

| Module | Library | What it does |
|---|---|---|
| 👤 Face Detection | OpenCV DNN | Bounding boxes + confidence score |
| ✋ Hand Tracking | MediaPipe | 21-keypoint skeleton, gesture recognition |
| 📦 Object Detection | MobileNet SSD | 20+ categories with labels |
| 😊 Emotion Detection | DeepFace | 7 emotions with confidence |
| 🌡️ Motion Detection | OpenCV | Heatmap motion overlay |
| 🎨 Color Analysis | K-Means | Dominant color extraction |
| 🧊 Face Mesh | MediaPipe | 468-point wireframe overlay |
| 🦴 Pose Estimation | MediaPipe | 17-keypoint body skeleton |
| 📱 QR Scanner | pyzbar | QR codes and barcodes |
| 🎭 Artistic Filters | OpenCV | Cartoon, Sketch, Pencil, Edge, Thermal, X-Ray |
| 🖼️ Background Seg. | MediaPipe | Blur or replace background |
| 👶 Age & Gender | OpenCV DNN | Demographic estimation |
| 🏎️ Speed Tracker | Optical Flow | Per-object speed estimation |

**Dashboard extras:** screenshot, recording, brightness/contrast sliders, multi-camera switch, session analytics, live FPS counter.

---

## Pages

| Route | Description |
|---|---|
| `/dashboard` | Main AI camera view |
| `/about` | Project overview and module list |
| `/problems` | Troubleshooting and FAQ |
| `/contact` | Contact form + LinkedIn |

---

## Quick Start

```bash
# Clone
git clone https://github.com/AadityaUniyal/CamerTrack.git
cd CamerTrack

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open **http://127.0.0.1:5000** in your browser and log in.

Default credentials: `admin` / `admin123`

---

## Project Structure

```
CamerTrack/
├── app.py                        # Flask app, routes, auth, API
├── config.py                     # Settings, password hashing, validation
├── requirements.txt
├── camera/
│   └── camera_manager.py         # Webcam capture, streaming, recording
├── ai_modules/
│   ├── face_detector.py
│   ├── hand_tracker.py
│   ├── object_detector.py
│   ├── emotion_detector.py
│   ├── motion_detector.py
│   ├── color_analyzer.py
│   ├── face_mesh.py
│   ├── pose_estimator.py
│   ├── qr_scanner.py
│   ├── artistic_filters.py
│   ├── background_segmenter.py
│   ├── age_gender_detector.py
│   └── speed_tracker.py
├── utils/
│   └── drawing.py                # OpenCV overlay helpers
├── templates/
│   ├── login.html
│   ├── index.html                # Dashboard
│   ├── about.html
│   ├── problems.html
│   └── contact.html
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── login.css
│   └── js/
│       └── app.js
└── models/                       # Place model weights here (not tracked)
```

---

## Requirements

- Python 3.10+
- Webcam (optional — placeholder shown if absent)
- ~100 MB disk space for ML model weights

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Configuration

All settings live in `config.py`.

**Secret key** — set an environment variable so sessions survive restarts:

```powershell
# Windows
setx AI_VISION_SECRET "your-random-secret-here"
```

```bash
# Linux / macOS
export AI_VISION_SECRET="your-random-secret-here"
```

If the variable is not set, a random key is generated per session (users get logged out on restart).

**Host binding** — by default the server binds to `127.0.0.1` (localhost only). Change `HOST` in `config.py` to `0.0.0.0` only if you need LAN access.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Computer Vision | OpenCV, MediaPipe, DeepFace |
| Object Detection | MobileNet SSD (Caffe) |
| Frontend | HTML, Vanilla CSS, JavaScript |
| Auth | Session-based, PBKDF2-HMAC-SHA256 hashed passwords |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Author

**Aaditya Uniyal**
[linkedin.com/in/aaditya-uniyal-48ab7b342](https://www.linkedin.com/in/aaditya-uniyal-48ab7b342)

---

## License

[MIT](LICENSE)
