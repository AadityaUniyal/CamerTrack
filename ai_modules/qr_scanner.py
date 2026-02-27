"""
AI Vision Studio — QR/Barcode Scanner Module
Detects and decodes QR codes and barcodes in real-time.
"""

import cv2
import numpy as np
from utils.drawing import DrawingUtils


class QRScanner:
    """Real-time QR code and barcode detection and decoding."""

    def __init__(self):
        self.detection_count = 0
        self.qr_detector = cv2.QRCodeDetector()
        self.scan_history = []  # Last N scanned codes
        self._max_history = 10

    def _detect_barcodes(self, frame):
        """Detect barcode-like regions using gradient analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.subtract(cv2.convertScaleAbs(grad_x), cv2.convertScaleAbs(grad_y))

        # Blur and threshold
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)

        # Morphological processing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        barcodes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if 2.0 < aspect < 15.0:  # Barcodes are rectangular
                box = cv2.boxPoints(rect).astype(int)
                barcodes.append(box)

        return barcodes

    def _draw_qr_overlay(self, frame, points, data):
        """Draw a styled QR code detection overlay."""
        if points is None or len(points) < 4:
            return frame

        pts = points.astype(int)

        # Draw polygon outline
        for i in range(len(pts)):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(frame, pt1, pt2, (0, 255, 200), 3, cv2.LINE_AA)

        # Corner markers
        for pt in pts:
            cv2.circle(frame, tuple(pt), 6, (0, 255, 200), -1)
            cv2.circle(frame, tuple(pt), 10, (0, 255, 200), 2)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 200))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Data label
        if data:
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            display_text = data if len(data) <= 40 else data[:37] + "..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(display_text, font, 0.5, 1)

            # Label background
            label_y = min(pts[:, 1]) - 12
            cv2.rectangle(frame, (cx - tw // 2 - 6, label_y - th - 8),
                          (cx + tw // 2 + 6, label_y + 6), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx - tw // 2 - 6, label_y - th - 8),
                          (cx + tw // 2 + 6, label_y + 6), (0, 255, 200), 1)

            cv2.putText(frame, display_text, (cx - tw // 2, label_y),
                        font, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

            # QR icon
            cv2.putText(frame, "QR", (cx - tw // 2 - 30, label_y),
                        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _draw_scan_history(self, frame):
        """Draw recent scan history overlay."""
        if not self.scan_history:
            return frame

        h, w = frame.shape[:2]
        y_start = 60
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Header
        cv2.putText(frame, "SCAN HISTORY", (10, y_start),
                    font, 0.4, (0, 255, 200), 1, cv2.LINE_AA)

        for i, entry in enumerate(self.scan_history[-5:]):
            y = y_start + 18 + i * 16
            text = entry if len(entry) <= 35 else entry[:32] + "..."
            cv2.putText(frame, f"  {i + 1}. {text}", (10, y),
                        font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def process(self, frame):
        """Detect and decode QR codes and barcodes. Returns: (frame, count)"""
        self.detection_count = 0

        # QR Code detection
        try:
            data, points, _ = self.qr_detector.detectAndDecode(frame)

            if points is not None and data:
                self.detection_count += 1
                pts = points[0] if len(points.shape) == 3 else points
                frame = self._draw_qr_overlay(frame, pts, data)

                # Add to history
                if data and data not in self.scan_history[-3:]:
                    self.scan_history.append(data)
                    if len(self.scan_history) > self._max_history:
                        self.scan_history = self.scan_history[-self._max_history:]
        except Exception:
            pass

        # Barcode detection (visual only - OpenCV can't decode barcodes natively)
        try:
            barcodes = self._detect_barcodes(frame)
            for box in barcodes[:3]:
                cv2.drawContours(frame, [box], 0, (255, 200, 0), 2)
                cx = int(np.mean(box[:, 0]))
                cy = int(np.mean(box[:, 1]))
                cv2.putText(frame, "Barcode", (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
                self.detection_count += 1
        except Exception:
            pass

        # Draw scan history
        if self.scan_history:
            frame = self._draw_scan_history(frame)

        return frame, self.detection_count

    def release(self):
        self.scan_history = []
