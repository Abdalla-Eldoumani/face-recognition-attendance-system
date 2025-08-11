import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config
from utils.face_detector import create_face_detector
from utils.preprocess import align_face


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in IMAGE_EXTENSIONS


def load_training_data() -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
    detector = create_face_detector()

    images: List[np.ndarray] = []
    labels: List[int] = []
    name_to_id: Dict[str, int] = {}

    people = [d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))]
    people.sort()

    next_id = 0
    for person in people:
        person_dir = os.path.join(config.DATASET_DIR, person)
        file_names = [f for f in os.listdir(person_dir) if is_image_file(os.path.join(person_dir, f))]
        file_names.sort()
        if not file_names:
            continue

        if person not in name_to_id:
            name_to_id[person] = next_id
            next_id += 1

        for file_name in file_names:
            path = os.path.join(person_dir, file_name)
            bgr = cv2.imread(path)
            if bgr is None:
                continue

            faces = detector.detect(bgr)
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
                aligned = align_face(bgr, (x, y, w, h), output_size=config.FACE_SIZE)
                if aligned is None:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    face_img = cv2.resize(gray[y : y + h, x : x + w], (config.FACE_SIZE, config.FACE_SIZE))
                else:
                    face_img = aligned
            else:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                face_img = cv2.resize(gray, (config.FACE_SIZE, config.FACE_SIZE))

            images.append(face_img)
            labels.append(name_to_id[person])

    return images, labels, name_to_id


def train_and_save() -> None:
    config.ensure_directories()

    images, labels, name_to_id = load_training_data()

    if not images or not labels:
        raise RuntimeError("No training data found. Please enroll at least one person.")

    print(f"Training on {len(images)} images across {len(set(labels))} people...")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    recognizer.write(config.MODEL_PATH)

    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(name_to_id, f, ensure_ascii=False, indent=2)

    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Labels saved to {config.LABELS_PATH}")
