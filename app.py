"""
AI Vision Studio — Flask Application
13 AI modules, auth, recording, filters, analytics, artistic modes, multi-cam.
"""

import sys
import os
import time
import re

sys.path.insert(0, os.path.dirname(__file__))

from flask import (Flask, render_template, Response, jsonify, send_file,
                   request, session, redirect, url_for)
from camera.camera_manager import CameraManager
from config import (
    HOST, PORT, DEBUG, SECRET_KEY,
    SESSION_COOKIE_HTTPONLY, SESSION_COOKIE_SAMESITE,
    USERS, verify_password, hash_password,
    VALID_ARTISTIC_MODES, VALID_BG_MODES, VALID_CAMERA_INDICES,
    MIN_PASSWORD_LENGTH, MAX_USERNAME_LENGTH, MIN_USERNAME_LENGTH,
)
import io

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = SESSION_COOKIE_HTTPONLY
app.config["SESSION_COOKIE_SAMESITE"] = SESSION_COOKIE_SAMESITE

camera = CameraManager()

_login_attempts: dict = {}
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60


def _get_ip():
    return request.remote_addr or "unknown"


def _check_rate_limit(ip: str) -> bool:
    entry = _login_attempts.get(ip)
    if not entry:
        return False
    if entry["count"] >= _MAX_ATTEMPTS:
        elapsed = time.time() - entry["first_fail"]
        if elapsed < _LOCKOUT_SECONDS:
            return True
        del _login_attempts[ip]
    return False


def _record_failed_attempt(ip: str):
    entry = _login_attempts.setdefault(ip, {"count": 0, "first_fail": time.time()})
    entry["count"] += 1


def _clear_attempts(ip: str):
    _login_attempts.pop(ip, None)


_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{3,32}$")

def _valid_username(u: str) -> bool:
    return bool(_USERNAME_RE.match(u))


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        ip = _get_ip()
        if _check_rate_limit(ip):
            remaining = int(_LOCKOUT_SECONDS - (time.time() - _login_attempts[ip]["first_fail"]))
            return render_template("login.html",
                                   error=f"Too many failed attempts. Try again in {remaining}s.")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = USERS.get(username)
        if user and verify_password(user["password_hash"], password):
            _clear_attempts(ip)
            session["username"] = username
            session["display_name"] = user["display_name"]
            return redirect(url_for("dashboard"))
        else:
            _record_failed_attempt(ip)
            error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm  = request.form.get("confirm_password", "")
    if not _valid_username(username):
        return render_template("login.html",
                               error=f"Username must be {MIN_USERNAME_LENGTH}–{MAX_USERNAME_LENGTH} "
                                     "characters (letters, numbers, _ or -).")
    if len(password) < MIN_PASSWORD_LENGTH:
        return render_template("login.html",
                               error=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password != confirm:
        return render_template("login.html", error="Passwords do not match.")
    if username in USERS:
        return render_template("login.html", error="Username already exists.")
    USERS[username] = {
        "password_hash": hash_password(password),
        "display_name": username.title(),
    }
    session["username"] = username
    session["display_name"] = USERS[username]["display_name"]
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html",
                           username=session.get("username", ""),
                           display_name=session.get("display_name", "User"))


@app.route("/about")
@login_required
def about():
    return render_template("about.html",
                           username=session.get("username", ""),
                           display_name=session.get("display_name", "User"))


@app.route("/problems")
@login_required
def problems():
    return render_template("problems.html",
                           username=session.get("username", ""),
                           display_name=session.get("display_name", "User"))


@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html",
                           username=session.get("username", ""),
                           display_name=session.get("display_name", "User"))


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(camera.generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/toggle/<module_name>", methods=["POST"])
@login_required
def toggle_module(module_name):
    result = camera.toggle_module(module_name)
    if result is not None:
        return jsonify({"success": True, "module": module_name, "active": result})
    return jsonify({"success": False, "error": "Unknown module"}), 400


@app.route("/api/status")
@login_required
def get_status():
    return jsonify(camera.get_status())


@app.route("/api/modules")
@login_required
def get_modules():
    modules = [
        {"id": "face_detection",         "name": "Face Detection",     "description": "OpenCV DNN face detection",       "icon": "👤", "color": "#00ff88"},
        {"id": "hand_tracking",          "name": "Hand Tracking",      "description": "Gesture recognition + tracking",  "icon": "✋", "color": "#ff8800"},
        {"id": "object_detection",       "name": "Object Detection",   "description": "MobileNet SSD 20+ classes",       "icon": "📦", "color": "#8800ff"},
        {"id": "emotion_detection",      "name": "Emotion Detection",  "description": "DeepFace emotion analysis",       "icon": "😊", "color": "#00c8ff"},
        {"id": "motion_detection",       "name": "Motion Detection",   "description": "Heat-map motion overlay",         "icon": "🌡️", "color": "#ff4444"},
        {"id": "color_analysis",         "name": "Color Analysis",     "description": "K-Means dominant colors",         "icon": "🎨", "color": "#ffcc00"},
        {"id": "face_mesh",              "name": "Face Mesh",          "description": "Wireframe mesh overlay",          "icon": "🧊", "color": "#00ffc8"},
        {"id": "pose_estimation",        "name": "Pose Estimation",    "description": "17-keypoint body skeleton",       "icon": "🦴", "color": "#ff6b9d"},
        {"id": "qr_scanner",             "name": "QR Scanner",         "description": "QR + barcode decoder",            "icon": "📱", "color": "#9d6bff"},
        {"id": "artistic_filters",       "name": "Artistic Filters",   "description": "Cartoon/Sketch/Edge/Thermal",     "icon": "🎭", "color": "#ff9d6b"},
        {"id": "background_segmentation","name": "Background Seg.",    "description": "Blur/replace background",         "icon": "🖼️", "color": "#6bff9d"},
        {"id": "age_gender",             "name": "Age & Gender",       "description": "DNN demographic analysis",        "icon": "👶", "color": "#ffb347"},
        {"id": "speed_tracking",         "name": "Speed Tracker",      "description": "Optical flow speed estimation",   "icon": "🏎️", "color": "#ff4757"},
    ]
    for m in modules:
        m["active"] = camera.module_states.get(m["id"], False)
    return jsonify(modules)


@app.route("/api/screenshot", methods=["POST"])
@login_required
def take_screenshot():
    image_bytes = camera.capture_screenshot()
    if image_bytes:
        return send_file(io.BytesIO(image_bytes), mimetype="image/jpeg",
                         as_attachment=True, download_name="ai_vision_screenshot.jpg")
    return jsonify({"success": False}), 500


@app.route("/api/record/start", methods=["POST"])
@login_required
def start_recording():
    filename = camera.start_recording()
    if filename:
        return jsonify({"success": True, "filename": os.path.basename(filename)})
    return jsonify({"success": False, "error": "Already recording"}), 400


@app.route("/api/record/stop", methods=["POST"])
@login_required
def stop_recording():
    filename = camera.stop_recording()
    if filename:
        return jsonify({"success": True, "filename": os.path.basename(filename)})
    return jsonify({"success": False}), 400


@app.route("/api/filters", methods=["POST"])
@login_required
def set_filters():
    data = request.get_json()
    if not data:
        return jsonify({"success": False}), 400
    if "brightness" in data:
        camera.set_filter("brightness", data["brightness"])
    if "contrast" in data:
        camera.set_filter("contrast", data["contrast"])
    return jsonify({"success": True, "filters": camera.filters})


@app.route("/api/artistic/cycle", methods=["POST"])
@login_required
def cycle_artistic():
    mode = camera.cycle_artistic_filter()
    return jsonify({"success": True, "mode": mode})


@app.route("/api/artistic/set", methods=["POST"])
@login_required
def set_artistic():
    data = request.get_json()
    mode = (data.get("mode", "off") if data else "off").strip().lower()
    if mode not in VALID_ARTISTIC_MODES:
        return jsonify({"success": False, "error": "Invalid mode"}), 400
    camera.set_artistic_filter(mode)
    return jsonify({"success": True, "mode": mode})


@app.route("/api/bg/set", methods=["POST"])
@login_required
def set_bg_mode():
    data = request.get_json()
    mode = (data.get("mode", "blur") if data else "blur").strip().lower()
    if mode not in VALID_BG_MODES:
        return jsonify({"success": False, "error": "Invalid mode"}), 400
    camera.set_bg_mode(mode)
    return jsonify({"success": True, "mode": mode})


@app.route("/api/camera/switch", methods=["POST"])
@login_required
def switch_camera():
    data = request.get_json()
    try:
        idx = int(data.get("index", 0) if data else 0)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid camera index"}), 400
    if idx not in VALID_CAMERA_INDICES:
        return jsonify({"success": False, "error": "Camera index not allowed"}), 400
    success = camera.switch_camera(idx)
    return jsonify({"success": success, "camera_index": idx})


@app.route("/api/analytics")
@login_required
def get_analytics():
    return jsonify(camera.get_analytics())


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI VISION STUDIO")
    print("  Real-Time Camera AI/ML Dashboard — 13 AI Modules")
    print("=" * 60)
    print(f"\n  http://{HOST}:{PORT}")
    print("  Ctrl+C to stop\n")

    camera.start()
    try:
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, use_reloader=False)
    finally:
        camera.stop()
