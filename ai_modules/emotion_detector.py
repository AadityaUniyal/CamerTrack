"""
AI Vision Studio — Emotion Detection Module
Uses DeepFace for facial emotion analysis.
"""

import cv2
import numpy as np
import threading
from utils.drawing import DrawingUtils


class EmotionDetector:
    """Real-time emotion detection using DeepFace."""

    EMOTION_COLORS = {
        "happy": (0, 255, 200),
        "sad": (255, 150, 0),
        "angry": (0, 0, 255),
        "surprise": (255, 255, 0),
        "fear": (180, 0, 255),
        "disgust": (0, 180, 0),
        "neutral": (180, 180, 180),
    }

    EMOTION_EMOJIS = {
        "happy": ":)",
        "sad": ":(",
        "angry": ">:(",
        "surprise": ":O",
        "fear": "D:",
        "disgust": ":/",
        "neutral": ":|",
    }

    def __init__(self):
        self.detection_count = 0
        self._last_result = None
        self._analyzing = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._deepface = None

    def _load_deepface(self):
        """Lazy-load DeepFace to avoid startup delay."""
        if self._deepface is None:
            try:
                from deepface import DeepFace
                self._deepface = DeepFace
                print("[EmotionDetector] DeepFace loaded successfully!")
            except ImportError:
                print("[EmotionDetector] DeepFace not installed. Run: pip install deepface")
                return False
        return True

    def _analyze_async(self, frame):
        """Run emotion analysis in a background thread to avoid blocking."""
        try:
            if not self._load_deepface():
                return

            results = self._deepface.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            with self._lock:
                if isinstance(results, list):
                    self._last_result = results
                else:
                    self._last_result = [results]

        except Exception as e:
            pass
        finally:
            self._analyzing = False

    def process(self, frame):
        """
        Detect emotions in faces in the frame.
        Runs analysis every 10 frames to maintain performance.
        Returns: (annotated_frame, detection_count)
        """
        self._frame_count += 1

        # Run analysis every 10 frames in background thread
        if not self._analyzing and self._frame_count % 10 == 0:
            self._analyzing = True
            small_frame = cv2.resize(frame, (320, 240))
            thread = threading.Thread(target=self._analyze_async, args=(small_frame,))
            thread.daemon = True
            thread.start()

        # Draw last known results
        with self._lock:
            results = self._last_result

        self.detection_count = 0

        if results:
            h, w = frame.shape[:2]
            scale_x = w / 320
            scale_y = h / 240

            for result in results:
                try:
                    region = result.get("region", {})
                    emotion_data = result.get("emotion", {})
                    dominant = result.get("dominant_emotion", "unknown")

                    if region and emotion_data:
                        self.detection_count += 1

                        # Scale coordinates back to original frame
                        x = int(region["x"] * scale_x)
                        y = int(region["y"] * scale_y)
                        rw = int(region["w"] * scale_x)
                        rh = int(region["h"] * scale_y)

                        color = self.EMOTION_COLORS.get(dominant, (0, 200, 255))
                        emoji = self.EMOTION_EMOJIS.get(dominant, "?")
                        confidence = emotion_data.get(dominant, 0) / 100.0

                        # Draw bounding box with emotion label
                        DrawingUtils.draw_bbox(
                            frame, x, y, x + rw, y + rh,
                            color=color,
                            label=f"{emoji} {dominant.capitalize()}",
                            confidence=confidence
                        )

                        # Draw top 3 emotion bars below the face
                        sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)[:3]
                        bar_y = y + rh + 10

                        for emo_name, emo_score in sorted_emotions:
                            if bar_y + 20 < h:
                                DrawingUtils.draw_emotion_bar(
                                    frame, emo_name.capitalize(),
                                    emo_score / 100.0,
                                    x, bar_y
                                )
                                bar_y += 22

                except Exception:
                    pass

        return frame, self.detection_count

    def release(self):
        """Release resources."""
        self._last_result = None
        self._analyzing = False
