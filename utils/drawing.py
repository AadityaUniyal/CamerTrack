"""
AI Vision Studio — Drawing Utilities
Styled overlays, bounding boxes, labels, and HUD for OpenCV frames.
"""

import cv2
import numpy as np
import time


class DrawingUtils:
    """Utility class for drawing styled annotations on video frames."""

    @staticmethod
    def draw_bbox(frame, x1, y1, x2, y2, color, label="", confidence=0.0, thickness=2):
        """Draw a styled bounding box with rounded corners and label."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        # Draw filled corners for style
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

        # Draw thin connecting lines
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Draw label with background
        if label:
            display_text = f"{label} {confidence:.0%}" if confidence > 0 else label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            (tw, th), baseline = cv2.getTextSize(display_text, font, font_scale, text_thickness)

            # Label background
            label_y = max(y1 - 8, th + 8)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, label_y - th - 8), (x1 + tw + 10, label_y + 4), color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Label text
            cv2.putText(frame, display_text, (x1 + 5, label_y - 2),
                        font, font_scale, (0, 0, 0), text_thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, display_text, (x1 + 5, label_y - 2),
                        font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return frame

    @staticmethod
    def draw_fps(frame, fps):
        """Draw FPS counter on the frame."""
        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text
        cv2.putText(frame, text, (15, 15 + th), font, font_scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_status_bar(frame, modules_status, detection_counts):
        """Draw a status bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        bar_height = 40

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Module indicators
        x_pos = 15
        font = cv2.FONT_HERSHEY_SIMPLEX

        module_icons = {
            "face_detection": ("FACE", (0, 255, 136)),
            "hand_tracking": ("HAND", (255, 136, 0)),
            "object_detection": ("OBJ", (136, 0, 255)),
            "emotion_detection": ("EMO", (0, 200, 255)),
        }

        for module_name, (label, color) in module_icons.items():
            is_active = modules_status.get(module_name, False)
            count = detection_counts.get(module_name, 0)

            # Dot indicator
            dot_color = color if is_active else (80, 80, 80)
            cv2.circle(frame, (x_pos + 6, h - bar_height // 2), 5, dot_color, -1)

            # Label
            display = f"{label}: {count}" if is_active else f"{label}: OFF"
            text_color = (255, 255, 255) if is_active else (100, 100, 100)
            cv2.putText(frame, display, (x_pos + 16, h - bar_height // 2 + 5),
                        font, 0.4, text_color, 1, cv2.LINE_AA)

            x_pos += 120

        return frame

    @staticmethod
    def draw_hand_skeleton(frame, landmarks, connections, color=(255, 136, 0)):
        """Draw a styled hand skeleton with landmarks and connections."""
        h, w = frame.shape[:2]

        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Larger dots for fingertips (indices 4, 8, 12, 16, 20)
            radius = 6 if i in [4, 8, 12, 16, 20] else 3
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), radius, color, 1)

        return frame

    @staticmethod
    def draw_emotion_bar(frame, emotion, confidence, x, y):
        """Draw an emotion confidence bar."""
        bar_width = 120
        bar_height = 16

        # Background bar
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        # Filled bar
        fill_width = int(bar_width * confidence)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), (0, 200, 255), -1)
        # Border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 200, 255), 1)

        # Text
        text = f"{emotion}: {confidence:.0%}"
        cv2.putText(frame, text, (x + 4, y + bar_height - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def draw_no_camera(frame_width=1280, frame_height=720):
        """Generate a 'no camera' placeholder frame."""
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)

        # Central icon circle
        cx, cy = frame_width // 2, frame_height // 2
        cv2.circle(frame, (cx, cy - 30), 60, (80, 80, 80), 3)

        # Camera icon lines  
        cv2.rectangle(frame, (cx - 35, cy - 50), (cx + 35, cy - 10), (80, 80, 80), 2)
        cv2.circle(frame, (cx, cy - 30), 12, (80, 80, 80), 2)

        # Slash through
        cv2.line(frame, (cx - 45, cy + 15), (cx + 45, cy - 75), (0, 0, 200), 3)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No Camera Detected"
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.putText(frame, text, (cx - tw // 2, cy + 50),
                    font, 0.8, (120, 120, 120), 2, cv2.LINE_AA)

        sub_text = "Please connect a webcam and restart"
        (tw2, th2), _ = cv2.getTextSize(sub_text, font, 0.5, 1)
        cv2.putText(frame, sub_text, (cx - tw2 // 2, cy + 80),
                    font, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

        return frame
