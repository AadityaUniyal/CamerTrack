"""
AI Vision Studio — Object Detection Module
Uses OpenCV DNN with MobileNet SSD for real-time object detection.
"""

import cv2
import numpy as np
import os
import urllib.request
from utils.drawing import DrawingUtils
from config import OBJECT_CLASSES


class ObjectDetector:
    """Real-time object detection using MobileNet SSD via OpenCV DNN."""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    PROTOTXT_FILE = "MobileNetSSD_deploy.prototxt"
    MODEL_FILE = "MobileNetSSD_deploy.caffemodel"

    # Color palette for different object classes
    CLASS_COLORS = {}

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        self.net = None
        self.detection_count = 0
        self._initialized = False
        self._generate_colors()

    def _generate_colors(self):
        """Generate unique colors for each class."""
        np.random.seed(42)
        for i, cls in enumerate(OBJECT_CLASSES):
            hue = int(255 * i / len(OBJECT_CLASSES))
            color = cv2.cvtColor(
                np.array([[[hue, 200, 255]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR
            )[0][0]
            self.CLASS_COLORS[cls] = tuple(int(c) for c in color)

    def _ensure_model(self):
        """Download model files if not present."""
        if self._initialized:
            return True

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        prototxt_path = os.path.join(self.MODEL_DIR, self.PROTOTXT_FILE)
        model_path = os.path.join(self.MODEL_DIR, self.MODEL_FILE)

        try:
            if not os.path.exists(prototxt_path):
                print("[ObjectDetector] Downloading prototxt...")
                urllib.request.urlretrieve(self.PROTOTXT_URL, prototxt_path)

            if not os.path.exists(model_path):
                print("[ObjectDetector] Downloading model weights (~23MB)...")
                urllib.request.urlretrieve(self.MODEL_URL, model_path)

            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._initialized = True
            print("[ObjectDetector] Model loaded successfully!")
            return True

        except Exception as e:
            print(f"[ObjectDetector] Failed to load model: {e}")
            return False

    def process(self, frame):
        """
        Detect objects in the frame.
        Returns: (annotated_frame, detection_count)
        """
        if not self._ensure_model():
            cv2.putText(frame, "Object Detector: Model loading...",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame, 0

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843,
            (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        self.detection_count = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.min_confidence:
                class_id = int(detections[0, 0, i, 1])

                if class_id < len(OBJECT_CLASSES):
                    class_name = OBJECT_CLASSES[class_id]

                    if class_name == "background":
                        continue

                    self.detection_count += 1

                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    color = self.CLASS_COLORS.get(class_name, (136, 0, 255))

                    DrawingUtils.draw_bbox(
                        frame, x1, y1, x2, y2,
                        color=color,
                        label=class_name.capitalize(),
                        confidence=confidence
                    )

        return frame, self.detection_count

    def release(self):
        """Release resources."""
        self.net = None
        self._initialized = False
