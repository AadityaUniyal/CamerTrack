"""
AI Vision Studio — Configuration
"""

import os
import hashlib
import secrets


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex() + "$" + key.hex()


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        salt_hex, key_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(key_hex)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return secrets.compare_digest(candidate, expected)
    except Exception:
        return False


# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Flask
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False
SECRET_KEY = os.environ.get("AI_VISION_SECRET") or secrets.token_hex(32)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# Users (passwords hashed at startup)
USERS = {
    "admin": {
        "password_hash": hash_password("admin123"),
        "display_name": "Admin User",
    },
    "demo": {
        "password_hash": hash_password("demo123"),
        "display_name": "Demo User",
    },
}

# Validation whitelists
VALID_ARTISTIC_MODES = {"off", "cartoon", "sketch", "pencil", "edge", "thermal", "xray"}
VALID_BG_MODES = {"blur", "replace", "off"}
VALID_CAMERA_INDICES = {0, 1, 2}

# Password / username policy
MIN_PASSWORD_LENGTH = 6
MAX_USERNAME_LENGTH = 32
MIN_USERNAME_LENGTH = 3

# AI Module Default States
DEFAULT_MODULES = {
    "face_detection": True,
    "hand_tracking": False,
    "object_detection": False,
    "emotion_detection": False,
    "motion_detection": False,
    "color_analysis": False,
    "face_mesh": False,
    "pose_estimation": False,
    "qr_scanner": False,
    "artistic_filters": False,
    "background_segmentation": False,
    "age_gender": False,
    "speed_tracking": False,
}

# Detection thresholds
FACE_CONFIDENCE = 0.5
HAND_CONFIDENCE = 0.5
OBJECT_CONFIDENCE = 0.5

# Filter defaults
DEFAULT_FILTERS = {
    "brightness": 0,
    "contrast": 1.0,
}

# Recording
RECORDING_DIR = "recordings"
RECORDING_FPS = 20
RECORDING_CODEC = "XVID"

# Colors (BGR for OpenCV)
COLORS = {
    "face": (0, 255, 136),
    "hand": (255, 136, 0),
    "object": (136, 0, 255),
    "emotion": (0, 200, 255),
    "motion": (0, 140, 255),
    "color": (255, 200, 0),
    "mesh": (0, 255, 200),
    "pose": (0, 255, 200),
    "qr": (0, 255, 200),
    "artistic": (255, 255, 0),
    "bg": (0, 0, 0),
    "age_gender": (255, 180, 0),
    "speed": (0, 200, 255),
    "text": (255, 255, 255),
    "fps": (0, 255, 255),
}

# MobileNet SSD classes
OBJECT_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]
