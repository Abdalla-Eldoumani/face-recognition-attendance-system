import argparse
import csv
import json
import os
import time
import sqlite3
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Tuple

import cv2

import config
from utils.face_detector import create_face_detector
from utils.preprocess import align_face, evaluate_quality_issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time face recognition attendance")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, help="Camera index (e.g., 0)")
    group.add_argument("--video", type=str, help="Path to video file")

    parser.add_argument(
        "--threshold",
        type=float,
        default=config.RECOGNITION_CONFIDENCE_THRESHOLD,
        help="LBPH confidence threshold (lower = stricter)",
    )
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP, help="Process every Nth frame")
    parser.add_argument("--once-per-day", action="store_true", help="Log at most once per person per day")
    parser.add_argument("--save-unknown", action="store_true", help="Save cropped faces for unknowns to dataset/_unknown for review")
    return parser.parse_args()


def load_model_and_labels() -> Tuple[cv2.face_LBPHFaceRecognizer, Dict[int, str]]:
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.LABELS_PATH):
        raise RuntimeError(
            "Model or labels not found. Train the model first by running 'python train.py' after enrolling people."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(config.MODEL_PATH)

    with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)

    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return recognizer, id_to_name


def append_attendance(name: str, csv_path: str) -> None:
    is_new_file = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def append_attendance_db(name: str) -> None:
    try:
        os.makedirs(config.ATTENDANCE_DIR, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, ts TEXT)"
            )
            conn.execute("INSERT INTO attendance (name, ts) VALUES (?, ?)", (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    config.ensure_directories()

    recognizer, id_to_name = load_model_and_labels()

    detector = create_face_detector()

    # Initialize video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_label = args.video
    else:
        cam_index = args.camera if args.camera is not None else 0
        cap = cv2.VideoCapture(cam_index)
        source_label = f"camera {cam_index}"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_label}")

    attendance_csv = config.get_attendance_csv_path()
    logged_today = set()
    if args.once_per_day and os.path.exists(attendance_csv):
        try:
            with open(attendance_csv, "r", encoding="utf-8") as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if parts:
                        logged_today.add(parts[0])
        except Exception:
            pass

    # Smoothing and cooldown structures
    recent_names: Dict[int, deque] = defaultdict(lambda: deque(maxlen=config.SMOOTHING_WINDOW))
    last_logged_time: Dict[str, float] = {}
    unknown_last_saved_time: Dict[int, float] = {}

    # Overlay counters
    skipped_blur = 0
    skipped_exposure = 0
    skipped_size = 0

    print("Running attendance. Press 'q' to quit.")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        if args.frame_skip > 1 and (frame_index % args.frame_skip) != 0:
            if config.DISPLAY_OVERLAY:
                overlay_text = f"Skipped - Blur:{skipped_blur} Exposure:{skipped_exposure} Size:{skipped_size}"
                cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Attendance", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue

        faces = detector.detect(frame)

        for (x, y, w, h) in faces:
            # Evaluate quality issues for the color crop
            face_bgr = frame[y : y + h, x : x + w]
            issues = evaluate_quality_issues(face_bgr, (x, y, w, h))
            if issues:
                if "blur" in issues:
                    skipped_blur += 1
                if "exposure" in issues:
                    skipped_exposure += 1
                if "size" in issues:
                    skipped_size += 1
                continue

            # Alignment (fallback to grayscale crop)
            aligned = align_face(frame, (x, y, w, h), output_size=config.FACE_SIZE)
            if aligned is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned = cv2.resize(gray[y : y + h, x : x + w], (config.FACE_SIZE, config.FACE_SIZE))

            label_id, confidence = recognizer.predict(aligned)

            if confidence <= args.threshold and label_id in id_to_name:
                name = id_to_name[label_id]
            else:
                name = "Unknown"

            track_key = hash((x // 10, y // 10, w // 10, h // 10))
            recent_names[track_key].append(name)

            smoothed_name = name
            if len(recent_names[track_key]) >= config.SMOOTHING_HITS_REQUIRED:
                counts = {}
                for n in recent_names[track_key]:
                    counts[n] = counts.get(n, 0) + 1
                smoothed_name = max(counts.items(), key=lambda kv: kv[1])[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if smoothed_name != "Unknown" else (0, 0, 255), 2)
            label_text = f"{smoothed_name} ({confidence:.1f})" if smoothed_name != "Unknown" else smoothed_name
            cv2.putText(frame, label_text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if smoothed_name != "Unknown":
                now_ts = time.time()
                if args.once_per_day and smoothed_name in logged_today:
                    pass
                else:
                    if smoothed_name not in last_logged_time or (now_ts - last_logged_time[smoothed_name]) >= config.COOLDOWN_SECONDS:
                        append_attendance(smoothed_name, attendance_csv)
                        append_attendance_db(smoothed_name)
                        last_logged_time[smoothed_name] = now_ts
                        logged_today.add(smoothed_name)
                        # Save a small thumbnail for UI and print a structured log line
                        thumb = frame[max(0, y): y + h, max(0, x): x + w]
                        if thumb.size != 0:
                            try:
                                ts = int(now_ts * 1000)
                                safe_name = smoothed_name.replace(" ", "_")
                                thumb_path = os.path.join(config.THUMBS_DIR, f"{safe_name}_{ts}.jpg")
                                cv2.imwrite(thumb_path, cv2.resize(thumb, (160, 160)))
                                print(f"Recognized {smoothed_name} CONF={confidence:.1f} THUMB={thumb_path}")
                            except Exception:
                                print(f"Recognized {smoothed_name} CONF={confidence:.1f}")
                        else:
                            print(f"Recognized {smoothed_name} CONF={confidence:.1f}")
            else:
                # Optionally save unknown faces for later review/enrollment
                if args.save_unknown:
                    now_ts = time.time()
                    last_save = unknown_last_saved_time.get(track_key, 0.0)
                    if (now_ts - last_save) >= 5.0:  # throttle unknown saves per track
                        unknown_last_saved_time[track_key] = now_ts
                        try:
                            os.makedirs(os.path.join(config.DATASET_DIR, "_unknown"), exist_ok=True)
                            ts = int(now_ts * 1000)
                            crop = frame[max(0, y): y + h, max(0, x): x + w]
                            if crop.size != 0:
                                save_path = os.path.join(config.DATASET_DIR, "_unknown", f"unknown_{ts}.jpg")
                                cv2.imwrite(save_path, cv2.resize(crop, (config.FACE_SIZE, config.FACE_SIZE)))
                        except Exception:
                            pass

        if config.DISPLAY_OVERLAY:
            overlay_text = f"Skipped - Blur:{skipped_blur} Exposure:{skipped_exposure} Size:{skipped_size}"
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Attendance", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()