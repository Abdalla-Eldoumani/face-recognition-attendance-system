import os
import urllib.request
from typing import List, Tuple, Optional

import cv2
import numpy as np

import config


class HaarFaceDetector:
    def __init__(self, cascade_path: str, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60)) -> None:
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces


class DnnFaceDetector:
    def __init__(self, proto_path: str, weights_path: str, confidence_threshold: float = 0.5) -> None:
        if not os.path.exists(proto_path) or not os.path.exists(weights_path):
            self._ensure_dnn_files(proto_path, weights_path)
        # Validate sizes (catch truncated/HTML downloads)
        if os.path.getsize(weights_path) < 1_000_000 or os.path.getsize(proto_path) < 1_000:
            raise RuntimeError("DNN model files appear invalid or truncated. Please re-download.")
        self.net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        self.confidence_threshold = confidence_threshold
        self._fallback_haar: Optional[HaarFaceDetector] = None
        self._warned = False

    @staticmethod
    def _ensure_dnn_files(proto_path: str, weights_path: str) -> None:
        os.makedirs(os.path.dirname(proto_path), exist_ok=True)
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_urls = [
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/4.x/dnn_samples/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        ]
        urllib.request.urlretrieve(proto_url, proto_path)
        # Try mirrors for the caffemodel
        last_error = None
        for url in caffemodel_urls:
            try:
                urllib.request.urlretrieve(url, weights_path)
                if os.path.getsize(weights_path) >= 1_000_000:
                    return
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Failed to download caffemodel from known URLs. Last error: {last_error}")

    def detect(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        (h, w) = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        try:
            self.net.setInput(blob)
            detections = self.net.forward()
        except cv2.error as e:
            # Fallback to Haar on any DNN forward error
            if not self._warned:
                print("Warning: DNN face detector failed; falling back to Haar. Details:", str(e))
                self._warned = True
            if self._fallback_haar is None:
                self._fallback_haar = HaarFaceDetector(
                    cascade_path=config.CASCADE_PATH,
                    scale_factor=config.SCALE_FACTOR,
                    min_neighbors=config.MIN_NEIGHBORS,
                    min_size=config.MIN_SIZE,
                )
            return self._fallback_haar.detect(frame_bgr)

        faces: List[Tuple[int, int, int, int]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence >= self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                x = max(0, start_x)
                y = max(0, start_y)
                ex = min(w - 1, end_x)
                ey = min(h - 1, end_y)
                faces.append((x, y, ex - x, ey - y))
        return faces


def create_face_detector():
    if config.FACE_DETECTOR == "haar":
        return HaarFaceDetector(
            cascade_path=config.CASCADE_PATH,
            scale_factor=config.SCALE_FACTOR,
            min_neighbors=config.MIN_NEIGHBORS,
            min_size=config.MIN_SIZE,
        )
    elif config.FACE_DETECTOR == "dnn":
        return DnnFaceDetector(
            proto_path=config.DNN_PROTO_PATH,
            weights_path=config.DNN_WEIGHTS_PATH,
            confidence_threshold=config.DNN_CONFIDENCE_THRESHOLD,
        )
    else:
        raise ValueError(f"Unknown FACE_DETECTOR '{config.FACE_DETECTOR}'. Use 'haar' or 'dnn'.")
