"""
AI Vision Studio — Color Analysis Module
Extracts dominant colors from the frame using K-Means clustering.
"""

import cv2
import numpy as np


class ColorAnalyzer:
    """Real-time dominant color extraction using K-Means clustering."""

    def __init__(self, num_colors=5):
        self.num_colors = num_colors
        self.detection_count = 0
        self._frame_count = 0
        self._last_palette = None

    def _extract_colors(self, frame, k=5):
        """Extract dominant colors using K-Means."""
        # Resize for speed
        small = cv2.resize(frame, (80, 60))
        pixels = small.reshape(-1, 3).astype(np.float32)

        # K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Count pixels per cluster and sort by frequency
        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(-counts)

        palette = []
        for idx in sorted_indices:
            color = centers[idx].astype(int)
            percentage = counts[idx] / len(labels) * 100
            palette.append({
                "color_bgr": tuple(int(c) for c in color),
                "color_hex": "#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0])),
                "percentage": percentage
            })

        return palette

    def _draw_palette(self, frame, palette):
        """Draw color palette bar on the frame."""
        h, w = frame.shape[:2]

        # Palette background
        palette_h = 50
        palette_y = h - 90  # Above the status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, palette_y - 5), (w - 10, palette_y + palette_h + 5),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw color blocks
        total_width = w - 40
        x_start = 20

        for i, color_info in enumerate(palette[:self.num_colors]):
            block_width = int(total_width * color_info["percentage"] / 100)
            if block_width < 2:
                block_width = 2

            color = color_info["color_bgr"]

            # Color block
            cv2.rectangle(frame,
                          (x_start, palette_y),
                          (x_start + block_width, palette_y + palette_h - 18),
                          color, -1)

            # Rounded border
            cv2.rectangle(frame,
                          (x_start, palette_y),
                          (x_start + block_width, palette_y + palette_h - 18),
                          (255, 255, 255), 1)

            # Hex label
            if block_width > 40:
                hex_text = color_info["color_hex"]
                cv2.putText(frame, hex_text,
                            (x_start + 4, palette_y + palette_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (200, 200, 200), 1, cv2.LINE_AA)

            # Percentage
            if block_width > 25:
                pct_text = f"{color_info['percentage']:.0f}%"
                cv2.putText(frame, pct_text,
                            (x_start + 4, palette_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 0, 0), 1, cv2.LINE_AA)

            x_start += block_width + 2

        # Title
        cv2.putText(frame, "DOMINANT COLORS",
                    (20, palette_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)

        return frame

    def process(self, frame):
        """
        Analyze dominant colors in the frame.
        Runs analysis every 15 frames for performance.
        Returns: (annotated_frame, detection_count)
        """
        self._frame_count += 1

        # Run color extraction every 15 frames
        if self._last_palette is None or self._frame_count % 15 == 0:
            try:
                self._last_palette = self._extract_colors(frame, self.num_colors)
            except Exception:
                self._last_palette = []

        self.detection_count = len(self._last_palette) if self._last_palette else 0

        if self._last_palette:
            frame = self._draw_palette(frame, self._last_palette)

        return frame, self.detection_count

    def release(self):
        """Release resources."""
        self._last_palette = None
