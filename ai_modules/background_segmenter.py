"""
AI Vision Studio — Background Segmentation Module
Foreground/background separation with blur, virtual BG, and removal modes.
"""

import cv2
import numpy as np


class BackgroundSegmenter:
    """Background segmentation using MOG2 background subtractor + morphological refinement."""

    MODES = ["blur", "color", "remove"]
    MODE_LABELS = {"blur": "BG Blur", "color": "Virtual BG", "remove": "BG Remove"}

    # Virtual background colors (BGR)
    BG_COLORS = [
        (60, 40, 20),     # Dark blue
        (20, 60, 20),     # Dark green
        (30, 20, 60),     # Dark purple
        (40, 40, 40),     # Dark gray
    ]

    def __init__(self):
        self.detection_count = 0
        self.current_mode = "blur"
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )
        self._bg_color_idx = 0
        self._learning_frames = 0
        self._mask_smooth = None

    def set_mode(self, mode):
        """Set segmentation mode."""
        if mode in self.MODES:
            self.current_mode = mode
            return True
        return False

    def _get_person_mask(self, frame):
        """Extract person/foreground mask using background subtraction."""
        # Apply background subtractor
        fg_mask = self._bg_subtractor.apply(frame, learningRate=0.01 if self._learning_frames < 60 else 0.001)
        self._learning_frames += 1

        # Threshold
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        fg_mask = cv2.dilate(fg_mask, kernel_close, iterations=1)

        # Smooth the mask edges with GaussianBlur
        fg_mask = cv2.GaussianBlur(fg_mask, (11, 11), 0)
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

        # Temporal smoothing
        if self._mask_smooth is None:
            self._mask_smooth = fg_mask.astype(np.float32) / 255.0
        else:
            self._mask_smooth = self._mask_smooth * 0.7 + (fg_mask.astype(np.float32) / 255.0) * 0.3

        smooth = (self._mask_smooth * 255).astype(np.uint8)
        _, smooth = cv2.threshold(smooth, 128, 255, cv2.THRESH_BINARY)

        return smooth

    def _apply_blur(self, frame, mask):
        """Apply background blur."""
        # Heavy blur for background
        blurred = cv2.GaussianBlur(frame, (51, 51), 30)

        # Soft mask for blending
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        mask_3ch = cv2.GaussianBlur(mask_3ch, (15, 15), 5)

        # Blend: foreground (sharp) + background (blurred)
        result = (frame.astype(np.float32) * mask_3ch +
                  blurred.astype(np.float32) * (1 - mask_3ch))
        return result.astype(np.uint8)

    def _apply_virtual_bg(self, frame, mask):
        """Replace background with solid color."""
        color = self.BG_COLORS[self._bg_color_idx % len(self.BG_COLORS)]
        bg = np.full_like(frame, color)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        mask_3ch = cv2.GaussianBlur(mask_3ch, (15, 15), 5)

        result = (frame.astype(np.float32) * mask_3ch +
                  bg.astype(np.float32) * (1 - mask_3ch))
        return result.astype(np.uint8)

    def _apply_remove(self, frame, mask):
        """Remove background (checkerboard pattern)."""
        h, w = frame.shape[:2]
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        block = 20
        for y in range(0, h, block):
            for x in range(0, w, block):
                if ((y // block) + (x // block)) % 2 == 0:
                    checker[y:y + block, x:x + block] = (40, 40, 40)
                else:
                    checker[y:y + block, x:x + block] = (60, 60, 60)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        mask_3ch = cv2.GaussianBlur(mask_3ch, (11, 11), 3)

        result = (frame.astype(np.float32) * mask_3ch +
                  checker.astype(np.float32) * (1 - mask_3ch))
        return result.astype(np.uint8)

    def process(self, frame):
        """Apply background segmentation. Returns: (frame, detection_count)"""
        mask = self._get_person_mask(frame)
        fg_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        self.detection_count = 1 if fg_pixels > total_pixels * 0.05 else 0

        if self.current_mode == "blur":
            frame = self._apply_blur(frame, mask)
        elif self.current_mode == "color":
            frame = self._apply_virtual_bg(frame, mask)
        elif self.current_mode == "remove":
            frame = self._apply_remove(frame, mask)

        # Mode label
        label = self.MODE_LABELS.get(self.current_mode, self.current_mode)
        cv2.putText(frame, f"BG: {label}", (frame.shape[1] - 180, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame, self.detection_count

    def release(self):
        self._mask_smooth = None
        self._learning_frames = 0
