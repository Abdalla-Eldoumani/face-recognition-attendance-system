import argparse
import glob
import os
import time
from typing import List

import cv2

import config
from utils.face_detector import create_face_detector
from utils.preprocess import align_face, evaluate_quality_issues
from utils.training import train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a new person via webcam or images, then auto-train")
    parser.add_argument("--name", required=True, help="Full name or identifier of the person to enroll")
    parser.add_argument("--mode", choices=["auto", "camera", "images"], default="auto", help="Enrollment mode")
    parser.add_argument("--images", nargs="*", help="Image files or glob patterns (used in images mode)")
    parser.add_argument("--num-images", type=int, default=30, help="Number of face images to capture (camera mode)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay in seconds between captures")
    parser.add_argument("--auto-snap", action="store_true", help="Auto-snap frames until num-images reached (camera mode)")
    return parser.parse_args()


def expand_image_globs(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for p in patterns:
        matches = glob.glob(p)
        paths.extend(matches)
    # de-duplicate and keep only files
    uniq = []
    seen = set()
    for p in paths:
        if os.path.isfile(p) and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def save_aligned_face(person_dir: str, frame, rect, count_marker: str) -> bool:
    x, y, w, h = rect
    face_bgr = frame[y : y + h, x : x + w]
    issues = evaluate_quality_issues(face_bgr, (x, y, w, h))
    if issues:
        return False
    aligned = align_face(frame, (x, y, w, h), output_size=config.FACE_SIZE)
    if aligned is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aligned = cv2.resize(gray[y : y + h, x : x + w], (config.FACE_SIZE, config.FACE_SIZE))
    ts = int(time.time() * 1000)
    fname = os.path.join(person_dir, f"{count_marker}_{ts}.jpg")
    cv2.imwrite(fname, aligned)
    return True


def enroll_from_camera(name: str, num_images: int, camera_index: int, delay: float, auto_snap: bool) -> int:
    detector = create_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    person_dir = os.path.join(config.DATASET_DIR, name.replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)

    print(f"Enrolling '{name}' from camera — target {num_images} images. Press 'q' to quit.")

    saved = 0
    while saved < num_images:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        faces = detector.detect(frame)
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
            if save_aligned_face(person_dir, frame, (x, y, w, h), f"{name.replace(' ', '_')}_{saved+1:03d}"):
                saved += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Saved {saved}/{num_images}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if delay > 0:
                    time.sleep(delay)

        cv2.imshow("Enroll - Press 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if not auto_snap and key == ord("s") and saved < num_images:
            # Manual snap on 's' if face present
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
                if save_aligned_face(person_dir, frame, (x, y, w, h), f"{name.replace(' ', '_')}_{saved+1:03d}"):
                    saved += 1

    cap.release()
    cv2.destroyAllWindows()
    return saved


def enroll_from_images(name: str, image_paths: List[str]) -> int:
    detector = create_face_detector()
    person_dir = os.path.join(config.DATASET_DIR, name.replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)

    saved = 0
    for idx, path in enumerate(image_paths, start=1):
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Warning: cannot read {path}")
            continue
        faces = detector.detect(bgr)
        if len(faces) == 0:
            print(f"No face detected in {path}")
            continue
        (x, y, w, h) = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
        if save_aligned_face(person_dir, bgr, (x, y, w, h), f"{name.replace(' ', '_')}_{idx:03d}"):
            saved += 1
    return saved


def main() -> None:
    args = parse_args()
    config.ensure_directories()

    mode = args.mode
    if mode == "auto":
        # If images are provided, use images; otherwise use camera
        mode = "images" if args.images else "camera"

    total_saved = 0
    if mode == "camera":
        total_saved = enroll_from_camera(args.name, args.num_images, args.camera, args.delay, args.auto_snap)
    elif mode == "images":
        if not args.images:
            raise RuntimeError("--images mode requires at least one image path or glob pattern")
        paths = expand_image_globs(args.images)
        if not paths:
            raise RuntimeError("No images found for the provided patterns")
        total_saved = enroll_from_images(args.name, paths)

    print(f"Enrollment saved {total_saved} images for {args.name}")

    # Auto-train after any successful enrollment
    if total_saved > 0:
        try:
            train_and_save()
        except RuntimeError as e:
            print("Training skipped:", str(e))


if __name__ == "__main__":
    main()
