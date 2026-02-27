"""
AI Vision Studio — Camera Manager
"""

import cv2
import time
import threading
import numpy as np
import os
from datetime import datetime
from utils.drawing import DrawingUtils
from config import (CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, DEFAULT_MODULES,
                    DEFAULT_FILTERS, RECORDING_DIR, RECORDING_FPS, RECORDING_CODEC)


class CameraManager:

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.fps = 0.0
        self._prev_time = time.time()
        self._frame_count = 0

        self.modules = {}
        self.module_states = dict(DEFAULT_MODULES)
        self.detection_counts = {k: 0 for k in DEFAULT_MODULES}

        self.filters = dict(DEFAULT_FILTERS)

        self.is_recording = False
        self._video_writer = None
        self._recording_start = None
        self._recording_filename = None

        self._session_start = None
        self._total_detections = {k: 0 for k in DEFAULT_MODULES}
        self._detection_history = []

        self._camera_index = CAMERA_INDEX

    def _init_modules(self):
        from ai_modules.face_detector import FaceDetector
        from ai_modules.hand_tracker import HandTracker
        from ai_modules.object_detector import ObjectDetector
        from ai_modules.emotion_detector import EmotionDetector
        from ai_modules.motion_detector import MotionDetector
        from ai_modules.color_analyzer import ColorAnalyzer
        from ai_modules.face_mesh import FaceMesh
        from ai_modules.pose_estimator import PoseEstimator
        from ai_modules.qr_scanner import QRScanner
        from ai_modules.artistic_filters import ArtisticFilters
        from ai_modules.background_segmenter import BackgroundSegmenter
        from ai_modules.age_gender_detector import AgeGenderDetector
        from ai_modules.speed_tracker import SpeedTracker

        self.modules = {
            "face_detection": FaceDetector(),
            "hand_tracking": HandTracker(),
            "object_detection": ObjectDetector(),
            "emotion_detection": EmotionDetector(),
            "motion_detection": MotionDetector(),
            "color_analysis": ColorAnalyzer(),
            "face_mesh": FaceMesh(),
            "pose_estimation": PoseEstimator(),
            "qr_scanner": QRScanner(),
            "artistic_filters": ArtisticFilters(),
            "background_segmentation": BackgroundSegmenter(),
            "age_gender": AgeGenderDetector(),
            "speed_tracking": SpeedTracker(),
        }

    def start(self):
        if self.is_running:
            return True

        self.cap = cv2.VideoCapture(self._camera_index)
        if not self.cap.isOpened():
            print("[CameraManager] Warning: Could not open camera!")
            self.cap = None
            self.is_running = True
            self._init_modules()
            self._session_start = time.time()
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self._session_start = time.time()
        self._init_modules()
        print(f"[CameraManager] Camera started at {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        return True

    def stop(self):
        self.is_running = False
        self.stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None
        for module in self.modules.values():
            module.release()
        print("[CameraManager] Camera stopped")

    def switch_camera(self, index):
        self._camera_index = index
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            return True
        return False

    def toggle_module(self, module_name):
        if module_name in self.module_states:
            self.module_states[module_name] = not self.module_states[module_name]
            return self.module_states[module_name]
        return None

    def set_filter(self, filter_name, value):
        if filter_name == "brightness":
            self.filters["brightness"] = max(-100, min(100, int(value)))
        elif filter_name == "contrast":
            self.filters["contrast"] = max(0.5, min(2.0, float(value)))
        return self.filters

    def cycle_artistic_filter(self):
        if "artistic_filters" in self.modules:
            return self.modules["artistic_filters"].cycle_mode()
        return None

    def set_artistic_filter(self, mode):
        if "artistic_filters" in self.modules:
            return self.modules["artistic_filters"].set_mode(mode)
        return False

    def get_artistic_filter_mode(self):
        if "artistic_filters" in self.modules:
            return self.modules["artistic_filters"].current_mode
        return "off"

    def set_bg_mode(self, mode):
        if "background_segmentation" in self.modules:
            return self.modules["background_segmentation"].set_mode(mode)
        return False

    def get_status(self):
        return {
            "fps": round(self.fps, 1),
            "modules": dict(self.module_states),
            "detections": dict(self.detection_counts),
            "camera_active": self.cap is not None and self.cap.isOpened() if self.cap else False,
            "recording": self.is_recording,
            "recording_duration": self._get_recording_duration(),
            "filters": dict(self.filters),
            "camera_index": self._camera_index,
            "artistic_mode": self.get_artistic_filter_mode(),
        }

    def get_analytics(self):
        uptime = time.time() - self._session_start if self._session_start else 0
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        total_all = sum(self._total_detections.values())
        history = self._detection_history[-30:] if self._detection_history else []
        chart_data = [h[1] for h in history]

        return {
            "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "uptime_seconds": int(uptime),
            "total_detections": total_all,
            "detections_by_module": dict(self._total_detections),
            "active_modules": sum(1 for v in self.module_states.values() if v),
            "chart_data": chart_data,
        }

    def start_recording(self):
        if self.is_recording:
            return None
        os.makedirs(RECORDING_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recording_filename = os.path.join(RECORDING_DIR, f"recording_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        self._video_writer = cv2.VideoWriter(
            self._recording_filename, fourcc, RECORDING_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT)
        )
        self.is_recording = True
        self._recording_start = time.time()
        return self._recording_filename

    def stop_recording(self):
        if not self.is_recording:
            return None
        self.is_recording = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        filename = self._recording_filename
        self._recording_filename = None
        self._recording_start = None
        return filename

    def _get_recording_duration(self):
        if self.is_recording and self._recording_start:
            return int(time.time() - self._recording_start)
        return 0

    def _apply_filters(self, frame):
        brightness = self.filters.get("brightness", 0)
        contrast = self.filters.get("contrast", 1.0)
        if brightness != 0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        return frame

    def _process_frame(self, frame):
        total_now = 0
        for module_name, is_active in self.module_states.items():
            if is_active and module_name in self.modules:
                try:
                    frame, count = self.modules[module_name].process(frame)
                    self.detection_counts[module_name] = count
                    self._total_detections[module_name] += count
                    total_now += count
                except Exception as e:
                    print(f"[CameraManager] Error in {module_name}: {e}")
                    self.detection_counts[module_name] = 0
            elif not is_active:
                self.detection_counts[module_name] = 0

        self._detection_history.append((time.time(), total_now))
        if len(self._detection_history) > 300:
            self._detection_history = self._detection_history[-300:]
        return frame

    def _calculate_fps(self):
        self._frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._prev_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._prev_time = current_time

    def get_frame(self):
        if not self.is_running:
            return DrawingUtils.draw_no_camera()
        if self.cap is None or not self.cap.isOpened():
            return DrawingUtils.draw_no_camera()
        ret, frame = self.cap.read()
        if not ret:
            return DrawingUtils.draw_no_camera()

        frame = cv2.flip(frame, 1)
        frame = self._apply_filters(frame)
        frame = self._process_frame(frame)
        self._calculate_fps()
        frame = DrawingUtils.draw_fps(frame, self.fps)
        frame = DrawingUtils.draw_status_bar(frame, self.module_states, self.detection_counts)

        if self.is_recording:
            duration = self._get_recording_duration()
            mins, secs = duration // 60, duration % 60
            cv2.putText(frame, f"REC {mins:02d}:{secs:02d}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (frame.shape[1] - 165, 25), 6, (0, 0, 255), -1)
            if self._video_writer:
                self._video_writer.write(frame)

        return frame

    def generate_stream(self):
        while self.is_running:
            frame = self.get_frame()
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    def capture_screenshot(self):
        frame = self.get_frame()
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ret:
            return buffer.tobytes()
        return None
