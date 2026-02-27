"""
AI Vision Studio — Motion Detection Module
Uses frame differencing to detect and highlight motion areas.
"""

import cv2
import numpy as np


class MotionDetector:
    """Real-time motion detection using frame differencing with heat overlay."""

    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        self.detection_count = 0
        self.heat_map = None

    def process(self, frame):
        """
        Detect motion by comparing current frame with previous.
        Returns: (annotated_frame, detection_count)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            self.heat_map = np.zeros((h, w), dtype=np.float32)
            return frame, 0

        # Frame differencing
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Update heat map (accumulate motion intensity)
        motion_float = thresh.astype(np.float32) / 255.0
        self.heat_map = self.heat_map * 0.85 + motion_float * 0.15

        # Normalize heat map for display
        heat_display = np.clip(self.heat_map * 3, 0, 1)
        heat_colored = cv2.applyColorMap(
            (heat_display * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Blend heat map with original frame
        mask = (heat_display > 0.05).astype(np.uint8)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        frame = np.where(mask_3ch > 0,
                         cv2.addWeighted(frame, 0.6, heat_colored, 0.4, 0),
                         frame)

        # Find contours for detection count
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.detection_count = 0

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            self.detection_count += 1
            x, y, cw, ch = cv2.boundingRect(contour)

            # Draw motion indicator
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 140, 255), 1)

        # Motion level indicator
        motion_level = np.mean(heat_display) * 100
        indicator_text = f"Motion: {motion_level:.1f}%"
        cv2.putText(frame, indicator_text, (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)

        # Motion level bar
        bar_x, bar_y = w - 200, 40
        bar_w, bar_h = 170, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * min(motion_level / 30, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 140, 255), -1)

        self.prev_frame = gray
        return frame, self.detection_count

    def release(self):
        """Release resources."""
        self.prev_frame = None
        self.heat_map = None
