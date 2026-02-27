"""
AI Vision Studio — Hand Tracking Module
Uses MediaPipe Tasks API for real-time hand landmark detection and gesture recognition.
Falls back to contour-based detection if MediaPipe is unavailable.
"""

import cv2
import numpy as np
import math
from utils.drawing import DrawingUtils


class HandTracker:
    """Real-time hand tracking with gesture recognition."""

    GESTURE_LABELS = {
        "open_palm": "Open Palm",
        "fist": "Fist",
        "peace": "Peace Sign",
        "thumbs_up": "Thumbs Up",
        "pointing": "Pointing",
        "unknown": "Unknown"
    }

    def __init__(self, min_confidence=0.5, max_hands=2):
        self.min_confidence = min_confidence
        self.max_hands = max_hands
        self.detection_count = 0
        self._mp_hands = None
        self._mp_connections = None
        self._use_contour_fallback = False
        self._initialized = False

    def _init(self):
        """Try to initialize MediaPipe, fall back to contour-based detection."""
        if self._initialized:
            return

        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self._mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.min_confidence,
                    min_tracking_confidence=self.min_confidence
                )
                self._mp_connections = mp.solutions.hands.HAND_CONNECTIONS
                print("[HandTracker] MediaPipe Hands loaded successfully!")
            else:
                print("[HandTracker] MediaPipe solutions not available, using contour fallback")
                self._use_contour_fallback = True
        except Exception as e:
            print(f"[HandTracker] MediaPipe not available, using contour fallback: {e}")
            self._use_contour_fallback = True

        self._initialized = True

    def _detect_hands_contour(self, frame):
        """Fallback hand detection using skin color segmentation and contour analysis."""
        h, w = frame.shape[:2]
        detections = []

        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

        lower_skin2 = np.array([170, 30, 60], dtype=np.uint8)
        upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

        mask = mask1 | mask2

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for hand-sized contours
        min_area = (w * h) * 0.01  # At least 1% of frame
        max_area = (w * h) * 0.4   # At most 40% of frame

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_hands]:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)

                # Convexity defects for gesture hint
                hull = cv2.convexHull(contour, returnPoints=False)
                try:
                    defects = cv2.convexityDefects(contour, hull)
                    finger_count = 0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])

                            # Calculate angle between fingers
                            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-6))

                            if angle <= math.pi / 2 and d > 10000:
                                finger_count += 1
                                cv2.circle(frame, far, 4, (0, 255, 255), -1)

                    # Determine gesture
                    if finger_count >= 4:
                        gesture = "open_palm"
                    elif finger_count == 1:
                        gesture = "peace"
                    elif finger_count == 0:
                        gesture = "fist"
                    else:
                        gesture = "unknown"

                    detections.append({
                        "bbox": (x, y, x + cw, y + ch),
                        "gesture": gesture,
                        "contour": contour,
                    })

                except Exception:
                    detections.append({
                        "bbox": (x, y, x + cw, y + ch),
                        "gesture": "unknown",
                        "contour": contour,
                    })

        return detections

    def _classify_gesture_mp(self, landmarks):
        """Classify hand gesture based on MediaPipe finger positions."""
        THUMB_TIP, THUMB_MCP = 4, 2
        INDEX_TIP, INDEX_MCP = 8, 5
        MIDDLE_TIP, MIDDLE_MCP = 12, 9
        RING_TIP, RING_MCP = 16, 13
        PINKY_TIP, PINKY_MCP = 20, 17
        WRIST = 0

        def is_finger_extended(tip_idx, mcp_idx):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            wrist = landmarks[WRIST]
            tip_dist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            mcp_dist = math.sqrt((mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2)
            return tip_dist > mcp_dist * 1.1

        def is_thumb_extended():
            thumb_tip = landmarks[THUMB_TIP]
            thumb_mcp = landmarks[THUMB_MCP]
            wrist = landmarks[WRIST]
            return abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 1.2

        fingers = [
            is_thumb_extended(),
            is_finger_extended(INDEX_TIP, INDEX_MCP),
            is_finger_extended(MIDDLE_TIP, MIDDLE_MCP),
            is_finger_extended(RING_TIP, RING_MCP),
            is_finger_extended(PINKY_TIP, PINKY_MCP),
        ]
        count = sum(fingers)

        if count == 5: return "open_palm"
        elif count == 0: return "fist"
        elif fingers[1] and fingers[2] and count == 2: return "peace"
        elif fingers[0] and count == 1: return "thumbs_up"
        elif fingers[1] and count == 1: return "pointing"
        else: return "unknown"

    def process(self, frame):
        """
        Detect hands and recognize gestures.
        Returns: (annotated_frame, detection_count)
        """
        self._init()

        h, w = frame.shape[:2]
        self.detection_count = 0

        if self._use_contour_fallback:
            # Contour-based detection
            detections = self._detect_hands_contour(frame)
            self.detection_count = len(detections)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                gesture = self.GESTURE_LABELS.get(det["gesture"], "Hand Detected")

                # Draw contour
                if det.get("contour") is not None:
                    cv2.drawContours(frame, [det["contour"]], -1, (255, 136, 0), 2)

                DrawingUtils.draw_bbox(
                    frame, x1, y1, x2, y2,
                    color=(255, 136, 0),
                    label=gesture
                )

        elif self._mp_hands:
            # MediaPipe-based detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                self.detection_count = len(results.multi_hand_landmarks)

                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks = hand_landmarks.landmark

                    DrawingUtils.draw_hand_skeleton(
                        frame, landmarks, self._mp_connections,
                        color=(255, 136, 0)
                    )

                    gesture = self._classify_gesture_mp(landmarks)
                    gesture_label = self.GESTURE_LABELS.get(gesture, "Unknown")

                    x_coords = [int(lm.x * w) for lm in landmarks]
                    y_coords = [int(lm.y * h) for lm in landmarks]
                    x1 = max(0, min(x_coords) - 20)
                    y1 = max(0, min(y_coords) - 20)
                    x2 = min(w, max(x_coords) + 20)
                    y2 = min(h, max(y_coords) + 20)

                    handedness = ""
                    if results.multi_handedness:
                        handedness = results.multi_handedness[hand_idx].classification[0].label + " "

                    DrawingUtils.draw_bbox(
                        frame, x1, y1, x2, y2,
                        color=(255, 136, 0),
                        label=f"{handedness}{gesture_label}"
                    )

        return frame, self.detection_count

    def release(self):
        """Release resources."""
        if self._mp_hands:
            self._mp_hands.close()
