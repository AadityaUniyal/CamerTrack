"""
AI Vision Studio — Artistic Filters Module
6 real-time artistic filter modes: Cartoon, Sketch, Pencil, Edge, Thermal, X-Ray.
"""

import cv2
import numpy as np


class ArtisticFilters:
    """Apply artistic visual filters to camera frames."""

    MODES = ["off", "cartoon", "sketch", "pencil", "edge", "thermal", "xray"]
    MODE_LABELS = {
        "off": "Normal",
        "cartoon": "🎨 Cartoon",
        "sketch": "✏️ Sketch",
        "pencil": "📝 Pencil",
        "edge": "⚡ Neon Edge",
        "thermal": "🌡️ Thermal",
        "xray": "☠️ X-Ray",
    }

    def __init__(self):
        self.current_mode = "off"
        self.detection_count = 0
        self._mode_index = 0

    def set_mode(self, mode):
        """Set the current filter mode."""
        if mode in self.MODES:
            self.current_mode = mode
            self._mode_index = self.MODES.index(mode)
            return True
        return False

    def cycle_mode(self):
        """Cycle to the next filter mode."""
        self._mode_index = (self._mode_index + 1) % len(self.MODES)
        self.current_mode = self.MODES[self._mode_index]
        return self.current_mode

    def _apply_cartoon(self, frame):
        """Cartoon effect: bilateral filter + edge mask."""
        # Downscale for speed, apply bilateral filter, upscale
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        for _ in range(3):
            small = cv2.bilateralFilter(small, 9, 75, 75)
        filtered = cv2.resize(small, (frame.shape[1], frame.shape[0]))

        # Edge mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 2)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Combine
        result = cv2.bitwise_and(filtered, edges_colored)
        return result

    def _apply_sketch(self, frame):
        """Pencil sketch from Gaussian divide."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Divide to get sketch
        sketch = cv2.divide(gray, blur, scale=256)

        # Convert back to BGR
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def _apply_pencil(self, frame):
        """Pencil sketch using OpenCV's built-in function."""
        gray_sketch, color_sketch = cv2.pencilSketch(
            frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05
        )
        return cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)

    def _apply_edge(self, frame):
        """Neon edge detection with color glow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Multi-scale Canny edges
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        edges = cv2.addWeighted(edges1, 0.7, edges2, 0.3, 0)

        # Dilate for glow effect
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Create colored edge image
        result = np.zeros_like(frame)

        # Color the edges based on position (creates rainbow effect)
        h, w = edges.shape
        for channel, color_shift in enumerate([0, 85, 170]):
            shifted = np.roll(edges, color_shift * h // 255, axis=0)
            result[:, :, channel] = shifted

        # Add a subtle glow
        glow = cv2.GaussianBlur(result, (5, 5), 0)
        result = cv2.addWeighted(result, 1.0, glow, 0.5, 0)

        return result

    def _apply_thermal(self, frame):
        """Simulated thermal camera view."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply thermal colormap
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)

        # Add temperature-like scale bar
        h, w = thermal.shape[:2]
        scale_w = 20
        scale = np.zeros((h, scale_w, 3), dtype=np.uint8)
        for i in range(h):
            val = int(255 * (h - i) / h)
            row_color = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET
            )[0][0]
            scale[i, :] = row_color

        thermal[:, w - scale_w:] = scale

        # Temperature labels
        cv2.putText(thermal, "HOT", (w - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(thermal, "COLD", (w - 55, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return thermal

    def _apply_xray(self, frame):
        """X-Ray effect: inverted + enhanced contrast."""
        # Invert
        inverted = cv2.bitwise_not(frame)

        # Convert to grayscale
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)

        # CLAHE for enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply a cool blue tint
        result = np.zeros_like(frame)
        result[:, :, 0] = enhanced  # Blue channel
        result[:, :, 1] = (enhanced * 0.7).astype(np.uint8)  # Green
        result[:, :, 2] = (enhanced * 0.3).astype(np.uint8)  # Red (less)

        return result

    def process(self, frame):
        """Apply current artistic filter. Returns: (frame, mode_active)"""
        if self.current_mode == "off":
            self.detection_count = 0
            return frame, 0

        self.detection_count = 1

        try:
            filters = {
                "cartoon": self._apply_cartoon,
                "sketch": self._apply_sketch,
                "pencil": self._apply_pencil,
                "edge": self._apply_edge,
                "thermal": self._apply_thermal,
                "xray": self._apply_xray,
            }

            if self.current_mode in filters:
                frame = filters[self.current_mode](frame)

            # Draw mode label
            label = self.MODE_LABELS.get(self.current_mode, self.current_mode)
            h = frame.shape[0]
            cv2.putText(frame, label, (10, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            cv2.putText(frame, f"Filter Error: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame, self.detection_count

    def release(self):
        self.current_mode = "off"
