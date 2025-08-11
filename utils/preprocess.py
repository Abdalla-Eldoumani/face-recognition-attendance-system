import os
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


_facemark = None


def _ensure_lbf_model() -> None:
    os.makedirs(config.FACEMARK_DIR, exist_ok=True)
    if not os.path.exists(config.LBF_MODEL_PATH):
        url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
        urllib.request.urlretrieve(url, config.LBF_MODEL_PATH)


def get_facemark():
    global _facemark
    if _facemark is None:
        _ensure_lbf_model()
        _facemark = cv2.face.createFacemarkLBF()
        _facemark.loadModel(config.LBF_MODEL_PATH)
    return _facemark


def align_face(bgr_image: np.ndarray, face_rect: Tuple[int, int, int, int], output_size: int = 200) -> Optional[np.ndarray]:
    x, y, w, h = face_rect
    roi = bgr_image[y : y + h, x : x + w]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    facemark = get_facemark()

    faces = np.array([[0, 0, w, h]])
    ok, landmarks = facemark.fit(gray, faces)
    if not ok or landmarks is None or len(landmarks) == 0:
        return None

    points = landmarks[0][0]

    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    desired_left = config.ALIGN_LEFT_EYE
    desired_right = config.ALIGN_RIGHT_EYE
    desired_dist = (desired_right[0] - desired_left[0]) * output_size
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    if dist < 1e-6:
        return None
    scale = desired_dist / dist

    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tX = output_size * 0.5
    tY = output_size * desired_left[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    aligned = cv2.warpAffine(gray, M, (output_size, output_size), flags=cv2.INTER_CUBIC)
    return aligned


def is_blurry(bgr_image: np.ndarray, threshold: float) -> bool:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold


def is_well_exposed(bgr_image: np.ndarray, low_thresh: float, high_thresh: float) -> bool:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    return low_thresh < mean_intensity < high_thresh


def is_proper_size(face_rect: Tuple[int, int, int, int], min_size: Tuple[int, int]) -> bool:
    return face_rect[2] >= min_size[0] and face_rect[3] >= min_size[1]


def evaluate_quality_issues(face_bgr: np.ndarray, face_rect: Tuple[int, int, int, int]) -> List[str]:
    issues: List[str] = []
    if not is_proper_size(face_rect, config.QUALITY_MIN_FACE_SIZE):
        issues.append("size")
    if is_blurry(face_bgr, config.QUALITY_BLUR_THRESHOLD):
        issues.append("blur")
    if not is_well_exposed(face_bgr, config.QUALITY_EXPOSURE_LOW, config.QUALITY_EXPOSURE_HIGH):
        issues.append("exposure")
    return issues

