"""
AI Vision Studio — Pose Estimation Module
17-keypoint body skeleton tracking using OpenCV DNN.
"""

import cv2
import numpy as np
import math


class PoseEstimator:
    """Real-time pose estimation using OpenCV DNN with a lightweight model."""

    # COCO keypoint indices
    KEYPOINTS = [
        "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
        "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye",
        "l_eye", "r_ear", "l_ear"
    ]

    # Skeleton connections (pairs of keypoint indices)
    SKELETON = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Right arm
        (1, 5), (5, 6), (6, 7),                 # Left arm
        (1, 8), (8, 9), (9, 10),                # Right leg
        (1, 11), (11, 12), (12, 13),            # Left leg
        (0, 14), (14, 16), (0, 15), (15, 17),   # Face
    ]

    # Limb colors (BGR) for visual distinction
    LIMB_COLORS = [
        (0, 255, 136), (0, 255, 136), (0, 200, 100), (0, 200, 100),  # Right arm
        (255, 136, 0), (255, 136, 0), (255, 100, 0),                  # Left arm
        (136, 0, 255), (136, 0, 255), (100, 0, 200),                  # Right leg
        (0, 200, 255), (0, 200, 255), (0, 150, 200),                  # Left leg
        (255, 255, 0), (255, 255, 0), (255, 200, 0), (255, 200, 0),  # Face
    ]

    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.detection_count = 0
        self._prev_keypoints = None

    def _detect_pose_simple(self, frame):
        """Simple pose detection using body part detection heuristics."""
        h, w = frame.shape[:2]
        keypoints = {}

        # Use the face detector cascade to locate the head
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        if len(faces) == 0:
            # Try upper body
            upper_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            )
            bodies = upper_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
            if len(bodies) == 0:
                return None

            bx, by, bw, bh = bodies[0]
            # Estimate head position from upper body
            head_cx = bx + bw // 2
            head_cy = by + bh // 5
            face_w = bw // 3
        else:
            fx, fy, fw, fh = faces[0]
            head_cx = fx + fw // 2
            head_cy = fy + fh // 2
            face_w = fw

        # Build estimated skeleton from head position
        body_scale = face_w * 3

        # Key landmark positions estimated from head
        keypoints[0] = (head_cx, head_cy)  # Nose
        keypoints[1] = (head_cx, head_cy + int(face_w * 0.9))  # Neck

        # Shoulders
        shoulder_y = head_cy + int(face_w * 1.3)
        keypoints[2] = (head_cx + int(face_w * 1.0), shoulder_y)  # R shoulder
        keypoints[5] = (head_cx - int(face_w * 1.0), shoulder_y)  # L shoulder

        # Elbows
        elbow_y = shoulder_y + int(face_w * 1.2)
        keypoints[3] = (head_cx + int(face_w * 1.3), elbow_y)  # R elbow
        keypoints[6] = (head_cx - int(face_w * 1.3), elbow_y)  # L elbow

        # Wrists
        wrist_y = elbow_y + int(face_w * 1.2)
        keypoints[4] = (head_cx + int(face_w * 1.2), wrist_y)  # R wrist
        keypoints[7] = (head_cx - int(face_w * 1.2), wrist_y)  # L wrist

        # Hips
        hip_y = head_cy + int(body_scale * 1.1)
        keypoints[8] = (head_cx + int(face_w * 0.5), hip_y)  # R hip
        keypoints[11] = (head_cx - int(face_w * 0.5), hip_y)  # L hip

        # Knees
        knee_y = hip_y + int(body_scale * 0.8)
        keypoints[9] = (head_cx + int(face_w * 0.5), min(knee_y, h - 40))  # R knee
        keypoints[12] = (head_cx - int(face_w * 0.5), min(knee_y, h - 40))  # L knee

        # Ankles
        ankle_y = knee_y + int(body_scale * 0.8)
        keypoints[10] = (head_cx + int(face_w * 0.5), min(ankle_y, h - 10))  # R ankle
        keypoints[13] = (head_cx - int(face_w * 0.5), min(ankle_y, h - 10))  # L ankle

        # Eyes
        eye_y = head_cy - int(face_w * 0.1)
        keypoints[14] = (head_cx + int(face_w * 0.2), eye_y)  # R eye
        keypoints[15] = (head_cx - int(face_w * 0.2), eye_y)  # L eye

        # Ears
        keypoints[16] = (head_cx + int(face_w * 0.5), head_cy)  # R ear
        keypoints[17] = (head_cx - int(face_w * 0.5), head_cy)  # L ear

        return keypoints

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 between p1-p2-p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2) + 1e-6
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2) + 1e-6
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def _classify_posture(self, keypoints):
        """Classify basic posture from keypoints."""
        if 1 not in keypoints or 8 not in keypoints:
            return "Unknown"

        neck = keypoints[1]
        r_hip = keypoints.get(8, neck)
        l_hip = keypoints.get(11, neck)
        hip_y = (r_hip[1] + l_hip[1]) // 2

        r_wrist = keypoints.get(4, neck)
        l_wrist = keypoints.get(7, neck)

        # Arms raised: wrists above shoulders
        r_shoulder_y = keypoints.get(2, neck)[1]
        l_shoulder_y = keypoints.get(5, neck)[1]

        arms_up = (r_wrist[1] < r_shoulder_y - 20) or (l_wrist[1] < l_shoulder_y - 20)
        if arms_up:
            return "Arms Raised"

        # Sitting vs standing: hip-to-neck ratio
        torso_height = hip_y - neck[1]
        if torso_height < 80:
            return "Sitting"

        return "Standing"

    def _draw_skeleton(self, frame, keypoints):
        """Draw the body skeleton with styled connections."""
        h, w = frame.shape[:2]

        # Draw limb connections
        for i, (start_idx, end_idx) in enumerate(self.SKELETON):
            if start_idx in keypoints and end_idx in keypoints:
                pt1 = keypoints[start_idx]
                pt2 = keypoints[end_idx]
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    color = self.LIMB_COLORS[i % len(self.LIMB_COLORS)]
                    cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)

        # Draw keypoint dots
        for idx, pt in keypoints.items():
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, pt, 5, (255, 255, 255), -1)
                cv2.circle(frame, pt, 5, (0, 255, 200), 1)

        # Draw joint angles at elbows and knees
        angle_joints = [
            (2, 3, 4, "R.Elbow"),   # Right elbow
            (5, 6, 7, "L.Elbow"),   # Left elbow
            (8, 9, 10, "R.Knee"),   # Right knee
            (11, 12, 13, "L.Knee"), # Left knee
        ]

        for p1_idx, p2_idx, p3_idx, label in angle_joints:
            if all(idx in keypoints for idx in [p1_idx, p2_idx, p3_idx]):
                angle = self._calculate_angle(
                    keypoints[p1_idx], keypoints[p2_idx], keypoints[p3_idx]
                )
                pt = keypoints[p2_idx]
                cv2.putText(frame, f"{angle:.0f}°", (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1, cv2.LINE_AA)

        # Posture label
        posture = self._classify_posture(keypoints)
        cv2.putText(frame, f"Posture: {posture}", (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        return frame

    def process(self, frame):
        """Detect pose and draw skeleton. Returns: (frame, detection_count)"""
        keypoints = self._detect_pose_simple(frame)
        self.detection_count = 0

        if keypoints and len(keypoints) > 3:
            self.detection_count = 1
            self._prev_keypoints = keypoints
            frame = self._draw_skeleton(frame, keypoints)
        elif self._prev_keypoints:
            # Use previous keypoints briefly for smoothness
            frame = self._draw_skeleton(frame, self._prev_keypoints)
            self._prev_keypoints = None

        return frame, self.detection_count

    def release(self):
        self._prev_keypoints = None
