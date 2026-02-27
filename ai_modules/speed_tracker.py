"""
AI Vision Studio — Object Speed Tracker Module
Optical flow + centroid tracking for estimating movement speed in pixels/second.
"""

import cv2
import numpy as np
import time


class SpeedTracker:
    """Track object movement speed using optical flow."""

    def __init__(self, max_tracks=50):
        self.max_tracks = max_tracks
        self.detection_count = 0
        self._prev_gray = None
        self._prev_time = None
        self._tracks = []  # List of tracked point histories
        self._track_ages = []

        # Lucas-Kanade optical flow parameters
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Good features to track parameters
        self._feature_params = dict(
            maxCorners=self.max_tracks,
            qualityLevel=0.3,
            minDistance=10,
            blockSize=7
        )

        self._frame_count = 0

    def _get_speed_color(self, speed):
        """Map speed to color: green (slow) → yellow → red (fast)."""
        max_speed = 500  # px/s
        ratio = min(speed / max_speed, 1.0)

        if ratio < 0.5:
            # Green to yellow
            r = int(255 * (ratio * 2))
            g = 255
        else:
            # Yellow to red
            r = 255
            g = int(255 * (1 - (ratio - 0.5) * 2))

        return (0, g, r)  # BGR

    def process(self, frame):
        """Track points and estimate speed. Returns: (frame, count)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()
        h, w = frame.shape[:2]

        self.detection_count = 0

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_time = current_time
            return frame, 0

        dt = current_time - self._prev_time
        if dt < 0.001:
            dt = 0.033  # Fallback to ~30fps

        # Re-detect features periodically
        self._frame_count += 1
        if self._frame_count % 15 == 0 or len(self._tracks) < 5:
            new_points = cv2.goodFeaturesToTrack(gray, **self._feature_params)
            if new_points is not None:
                for pt in new_points:
                    x, y = pt.ravel()
                    # Don't add if too close to existing track
                    too_close = False
                    for track in self._tracks:
                        if track:
                            lx, ly = track[-1]
                            if abs(x - lx) < 15 and abs(y - ly) < 15:
                                too_close = True
                                break
                    if not too_close and len(self._tracks) < self.max_tracks:
                        self._tracks.append([(x, y)])
                        self._track_ages.append(0)

        # Track existing points using optical flow
        if self._tracks:
            prev_pts = np.array([t[-1] for t in self._tracks], dtype=np.float32).reshape(-1, 1, 2)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, prev_pts, None, **self._lk_params
            )

            new_tracks = []
            new_ages = []
            speeds = []

            for i, (pt_new, st) in enumerate(zip(next_pts, status)):
                if st[0] == 0:
                    continue  # Lost track

                x, y = pt_new.ravel()
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue

                # Calculate speed
                px, py = self._tracks[i][-1]
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                speed = dist / dt  # pixels per second

                # Update track
                track = self._tracks[i]
                track.append((x, y))
                if len(track) > 20:
                    track = track[-20:]

                age = self._track_ages[i] + 1
                if age > 300:  # Expire old tracks
                    continue

                new_tracks.append(track)
                new_ages.append(age)

                # Only draw if significant movement
                if speed > 5:
                    self.detection_count += 1
                    color = self._get_speed_color(speed)

                    # Draw motion trail
                    for j in range(1, len(track)):
                        alpha = j / len(track)
                        thickness = max(1, int(alpha * 3))
                        pt1 = (int(track[j - 1][0]), int(track[j - 1][1]))
                        pt2 = (int(track[j][0]), int(track[j][1]))
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # Current point
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)

                    # Speed label (only for faster objects)
                    if speed > 30:
                        speed_text = f"{speed:.0f} px/s"
                        cv2.putText(frame, speed_text, (int(x) + 8, int(y) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

                    speeds.append(speed)

            self._tracks = new_tracks
            self._track_ages = new_ages

            # Average speed display
            if speeds:
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                color = self._get_speed_color(avg_speed)

                # Speed meter
                meter_x = w - 200
                cv2.putText(frame, f"Avg: {avg_speed:.0f} px/s", (meter_x, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Max: {max_speed:.0f} px/s", (meter_x, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

                # Speed bar
                bar_y = 88
                bar_w = 170
                bar_h = 6
                cv2.rectangle(frame, (meter_x, bar_y), (meter_x + bar_w, bar_y + bar_h),
                              (50, 50, 50), -1)
                fill = int(bar_w * min(avg_speed / 500, 1.0))
                if fill > 0:
                    cv2.rectangle(frame, (meter_x, bar_y), (meter_x + fill, bar_y + bar_h),
                                  color, -1)

        self._prev_gray = gray
        self._prev_time = current_time

        return frame, self.detection_count

    def release(self):
        self._prev_gray = None
        self._tracks = []
        self._track_ages = []
