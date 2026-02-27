"""
AI Vision Studio — Face Detection Module
Uses OpenCV's DNN face detector for real-time face detection.
"""

import cv2
import numpy as np
import os
import urllib.request
from utils.drawing import DrawingUtils


class FaceDetector:
    """Real-time face detection using OpenCV DNN (SSD ResNet)."""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_FILE = "face_deploy.prototxt"
    MODEL_FILE = "face_res10_300x300.caffemodel"

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        self.net = None
        self.detection_count = 0
        self._initialized = False
        # Fallback to Haar cascade
        self._use_haar = False
        self._haar_cascade = None

    def _ensure_model(self):
        """Download DNN model files or fall back to Haar cascade."""
        if self._initialized:
            return True

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        prototxt_path = os.path.join(self.MODEL_DIR, self.PROTOTXT_FILE)
        model_path = os.path.join(self.MODEL_DIR, self.MODEL_FILE)

        try:
            if not os.path.exists(prototxt_path):
                print("[FaceDetector] Downloading prototxt...")
                urllib.request.urlretrieve(self.PROTOTXT_URL, prototxt_path)

            if not os.path.exists(model_path):
                print("[FaceDetector] Downloading model weights (~10MB)...")
                urllib.request.urlretrieve(self.MODEL_URL, model_path)

            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._initialized = True
            print("[FaceDetector] DNN model loaded successfully!")
            return True

        except Exception as e:
            print(f"[FaceDetector] DNN model failed, falling back to Haar cascade: {e}")
            self._use_haar = True
            self._haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self._initialized = True
            return True

    def process(self, frame):
        """
        Detect faces in the frame and draw bounding boxes.
        Returns: (annotated_frame, detection_count)
        """
        if not self._ensure_model():
            return frame, 0

        self.detection_count = 0

        if self._use_haar:
            return self._process_haar(frame)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.min_confidence:
                self.detection_count += 1

                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                DrawingUtils.draw_bbox(
                    frame, x1, y1, x2, y2,
                    color=(0, 255, 136),
                    label="Face",
                    confidence=confidence
                )

        return frame, self.detection_count

    def _process_haar(self, frame):
        """Fallback face detection using Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        self.detection_count = len(faces)

        for (x, y, w, h) in faces:
            DrawingUtils.draw_bbox(
                frame, x, y, x + w, y + h,
                color=(0, 255, 136),
                label="Face",
                confidence=0.0
            )

        return frame, self.detection_count

    def release(self):
        """Release resources."""
        self.net = None
        self._initialized = False
