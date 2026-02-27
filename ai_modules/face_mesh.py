"""
AI Vision Studio — Face Mesh Module
Draws a sci-fi wireframe mesh overlay on detected faces using OpenCV DNN landmarks.
"""

import cv2
import numpy as np
import os
import urllib.request


class FaceMesh:
    """Face mesh wireframe overlay using OpenCV's DNN face detector + Delaunay triangulation."""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    PROTOTXT_FILE = "face_deploy.prototxt"
    MODEL_FILE = "face_res10_300x300.caffemodel"

    # LBF landmark model
    LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
    LBF_MODEL_FILE = "lbfmodel.yaml"

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        self.detection_count = 0
        self._initialized = False
        self._face_net = None
        self._facemark = None

    def _ensure_model(self):
        """Load face detector and landmark model."""
        if self._initialized:
            return True

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        try:
            # Load face detector (reuse the same model as face_detector)
            prototxt_path = os.path.join(self.MODEL_DIR, self.PROTOTXT_FILE)
            model_path = os.path.join(self.MODEL_DIR, self.MODEL_FILE)

            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self._face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            else:
                # Use Haar cascade as fallback
                self._face_net = None
                print("[FaceMesh] DNN model not found, using Haar cascade")

            # Try to load LBF facemark
            lbf_path = os.path.join(self.MODEL_DIR, self.LBF_MODEL_FILE)
            if not os.path.exists(lbf_path):
                try:
                    print("[FaceMesh] Downloading landmark model...")
                    urllib.request.urlretrieve(self.LBF_MODEL_URL, lbf_path)
                except Exception as e:
                    print(f"[FaceMesh] Could not download landmark model: {e}")

            if os.path.exists(lbf_path):
                try:
                    self._facemark = cv2.face.createFacemarkLBF()
                    self._facemark.loadModel(lbf_path)
                    print("[FaceMesh] Landmark model loaded!")
                except Exception as e:
                    print(f"[FaceMesh] Facemark not available: {e}")
                    self._facemark = None

            self._initialized = True
            return True

        except Exception as e:
            print(f"[FaceMesh] Init error: {e}")
            self._initialized = True
            return False

    def _detect_faces(self, frame):
        """Detect face bounding boxes."""
        h, w = frame.shape[:2]
        faces = []

        if self._face_net:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                          (300, 300), (104.0, 177.0, 123.0))
            self._face_net.setInput(blob)
            detections = self._face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.min_confidence:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        else:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, fw, fh) in detected:
                faces.append((x, y, fw, fh))

        return faces

    def _draw_mesh_from_landmarks(self, frame, landmarks):
        """Draw Delaunay triangulation mesh from landmarks."""
        h, w = frame.shape[:2]
        points = []

        for lm in landmarks:
            for pt in lm:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    points.append((x, y))
                    # Draw landmark dots
                    cv2.circle(frame, (x, y), 1, (0, 255, 200), -1)

        if len(points) < 4:
            return frame

        # Delaunay triangulation
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        for pt in points:
            try:
                subdiv.insert(pt)
            except Exception:
                pass

        try:
            triangles = subdiv.getTriangleList()
            for t in triangles:
                pt1 = (int(t[0]), int(t[1]))
                pt2 = (int(t[2]), int(t[3]))
                pt3 = (int(t[4]), int(t[5]))

                # Check all points are within frame
                if all(0 <= p[0] < w and 0 <= p[1] < h for p in [pt1, pt2, pt3]):
                    cv2.line(frame, pt1, pt2, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.line(frame, pt2, pt3, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.line(frame, pt3, pt1, (0, 255, 200), 1, cv2.LINE_AA)
        except Exception:
            pass

        return frame

    def _draw_grid_mesh(self, frame, face_rect):
        """Draw a simple grid-based mesh on the face area (fallback when no landmarks)."""
        x, y, fw, fh = face_rect
        grid_step = 8

        # Semi-transparent overlay
        overlay = frame.copy()

        for gx in range(x, x + fw, grid_step):
            cv2.line(overlay, (gx, y), (gx, y + fh), (0, 255, 200), 1, cv2.LINE_AA)

        for gy in range(y, y + fh, grid_step):
            cv2.line(overlay, (x, gy), (x + fw, gy), (0, 255, 200), 1, cv2.LINE_AA)

        # Blend
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Face outline
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 200), 2)

        # Label
        cv2.putText(frame, "Face Mesh", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

    def process(self, frame):
        """
        Detect faces and draw mesh wireframe overlay.
        Returns: (annotated_frame, detection_count)
        """
        if not self._ensure_model():
            return frame, 0

        faces = self._detect_faces(frame)
        self.detection_count = len(faces)

        if not faces:
            return frame, 0

        if self._facemark:
            # Use LBF landmarks for detailed mesh
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_array = np.array(faces)

            try:
                ok, landmarks = self._facemark.fit(gray, faces_array)
                if ok:
                    frame = self._draw_mesh_from_landmarks(frame, landmarks)
                    return frame, self.detection_count
            except Exception:
                pass

        # Fallback: grid mesh on face regions
        for face_rect in faces:
            frame = self._draw_grid_mesh(frame, face_rect)

        return frame, self.detection_count

    def release(self):
        """Release resources."""
        self._face_net = None
        self._facemark = None
        self._initialized = False
