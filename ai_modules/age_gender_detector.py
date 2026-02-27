"""
AI Vision Studio — Age & Gender Estimation Module
Uses OpenCV DNN with Caffe models for demographic analysis.
"""

import cv2
import numpy as np
import os
import urllib.request
from utils.drawing import DrawingUtils


class AgeGenderDetector:
    """Age and gender estimation using OpenCV DNN models."""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # Age model
    AGE_PROTO = "age_deploy.prototxt"
    AGE_MODEL = "age_net.caffemodel"
    AGE_PROTO_URL = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt"
    AGE_MODEL_URL = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"

    # Gender model
    GENDER_PROTO = "gender_deploy.prototxt"
    GENDER_MODEL = "gender_net.caffemodel"
    GENDER_PROTO_URL = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt"
    GENDER_MODEL_URL = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"

    AGE_BRACKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60+)"]
    GENDER_LIST = ["Male", "Female"]

    GENDER_COLORS = {
        "Male": (255, 180, 0),    # Blue-ish
        "Female": (180, 0, 255),  # Pink-ish
    }

    MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

    def __init__(self, face_confidence=0.5):
        self.face_confidence = face_confidence
        self.detection_count = 0
        self._face_net = None
        self._age_net = None
        self._gender_net = None
        self._initialized = False
        self._use_fallback = False
        self._face_cascade = None

    def _download_if_missing(self, filename, url):
        """Download a model file if not present."""
        path = os.path.join(self.MODEL_DIR, filename)
        if not os.path.exists(path):
            try:
                print(f"[AgeGender] Downloading {filename}...")
                urllib.request.urlretrieve(url, path)
                return True
            except Exception as e:
                print(f"[AgeGender] Failed to download {filename}: {e}")
                return False
        return True

    def _ensure_models(self):
        """Load all required models."""
        if self._initialized:
            return self._age_net is not None

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        try:
            # Face detector (reuse existing)
            face_proto = os.path.join(self.MODEL_DIR, "face_deploy.prototxt")
            face_model = os.path.join(self.MODEL_DIR, "face_res10_300x300.caffemodel")

            if os.path.exists(face_proto) and os.path.exists(face_model):
                self._face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
            else:
                self._use_fallback = True
                self._face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )

            # Age model
            has_age = (
                self._download_if_missing(self.AGE_PROTO, self.AGE_PROTO_URL) and
                self._download_if_missing(self.AGE_MODEL, self.AGE_MODEL_URL)
            )

            if has_age:
                age_proto_path = os.path.join(self.MODEL_DIR, self.AGE_PROTO)
                age_model_path = os.path.join(self.MODEL_DIR, self.AGE_MODEL)
                self._age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
                print("[AgeGender] Age model loaded!")

            # Gender model
            has_gender = (
                self._download_if_missing(self.GENDER_PROTO, self.GENDER_PROTO_URL) and
                self._download_if_missing(self.GENDER_MODEL, self.GENDER_MODEL_URL)
            )

            if has_gender:
                gender_proto_path = os.path.join(self.MODEL_DIR, self.GENDER_PROTO)
                gender_model_path = os.path.join(self.MODEL_DIR, self.GENDER_MODEL)
                self._gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)
                print("[AgeGender] Gender model loaded!")

        except Exception as e:
            print(f"[AgeGender] Model loading error: {e}")

        self._initialized = True
        return self._age_net is not None

    def _detect_faces(self, frame):
        """Detect face bounding boxes."""
        h, w = frame.shape[:2]
        faces = []

        if self._face_net:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                          (300, 300), (104.0, 177.0, 123.0))
            self._face_net.setInput(blob)
            detections = self._face_net.forward()

            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > self.face_confidence:
                    x1 = max(0, int(detections[0, 0, i, 3] * w))
                    y1 = max(0, int(detections[0, 0, i, 4] * h))
                    x2 = min(w, int(detections[0, 0, i, 5] * w))
                    y2 = min(h, int(detections[0, 0, i, 6] * h))
                    faces.append((x1, y1, x2, y2))
        elif self._face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            for (x, y, fw, fh) in detected:
                faces.append((x, y, x + fw, y + fh))

        return faces

    def process(self, frame):
        """Detect age and gender of faces. Returns: (frame, count)"""
        if not self._ensure_models():
            cv2.putText(frame, "Age/Gender: Models loading...", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            return frame, 0

        faces = self._detect_faces(frame)
        self.detection_count = 0

        for (x1, y1, x2, y2) in faces:
            # Pad face region
            pad = 20
            fy1 = max(0, y1 - pad)
            fy2 = min(frame.shape[0], y2 + pad)
            fx1 = max(0, x1 - pad)
            fx2 = min(frame.shape[1], x2 + pad)

            face_img = frame[fy1:fy2, fx1:fx2]
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue

            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN, swapRB=False)

            # Age prediction
            age_text = "?"
            if self._age_net:
                self._age_net.setInput(blob)
                age_preds = self._age_net.forward()
                age_idx = age_preds[0].argmax()
                age_text = self.AGE_BRACKETS[age_idx]

            # Gender prediction
            gender_text = "?"
            gender_conf = 0
            if self._gender_net:
                self._gender_net.setInput(blob)
                gender_preds = self._gender_net.forward()
                gender_idx = gender_preds[0].argmax()
                gender_text = self.GENDER_LIST[gender_idx]
                gender_conf = gender_preds[0][gender_idx]

            self.detection_count += 1

            # Draw
            color = self.GENDER_COLORS.get(gender_text, (0, 200, 255))
            label = f"{gender_text}, {age_text}"

            DrawingUtils.draw_bbox(frame, x1, y1, x2, y2,
                                   color=color, label=label, confidence=gender_conf)

        return frame, self.detection_count

    def release(self):
        self._face_net = None
        self._age_net = None
        self._gender_net = None
        self._initialized = False
