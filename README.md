# Face Recognition Attendance System

A Python and OpenCV app that recognizes enrolled people from a webcam or video file and logs their attendance to daily CSV files and a local SQLite database. It has a command-line interface and a Tkinter control panel.

## How it works

1. **Detect.** Each frame goes through OpenCV's ResNet-10 SSD face detector (the default) or a Haar cascade. If the SSD model fails while running, the app falls back to Haar.
2. **Align and filter.** Faces are aligned on the eyes with OpenCV's Facemark LBF landmarks. Frames that are blurry, too dark, too bright or too small are skipped, and an overlay counts the skips.
3. **Recognize.** An LBPH recognizer (`opencv-contrib`) trained on the enrolled images predicts a name. A name is logged only after it appears in several recent frames, and the same person is not logged again within a cooldown period.
4. **Log.** Each recognition is appended to `attendance/attendance_YYYYMMDD.csv` and to `attendance/attendance.db`.

Enrolling a person saves face crops under `dataset/<name>/` and retrains the recognizer automatically.

## Setup

Python 3 with Tkinter is required (the standard installers include it).

```bash
python -m venv .venv
source .venv/bin/activate        # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

The LBF landmark model is in the repo. The SSD detector weights are not: `models/face_detector/res10_300x300_ssd_iter_140000.caffemodel` is an empty placeholder, and the app refuses to load it. Before using the default detector, delete that file so the app downloads the weights on first run, or download them yourself:

```bash
curl -L -o models/face_detector/res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/4.x/dnn_samples/face_detector/res10_300x300_ssd_iter_140000.caffemodel
```

Or skip it and use the Haar detector by setting `FACE_DETECTOR=haar`.

## Usage

### Control panel

```bash
python app.py
```

The panel has tabs to enroll people (camera or image files), run attendance, browse history with date and person filters and charts, and review saved unknown faces (assign them to a new or existing person and retrain). It also exports all attendance to one CSV or an XLSX file with a per-person summary. The detector choice and camera index are saved in `ui_settings.json`. The panel defaults to the Haar detector.

### Command line

Enroll a person, then run attendance. Press `q` to quit a camera window.

```bash
# Capture 30 images from camera 0 automatically
python enroll.py --name "Jane Doe" --num-images 30 --camera 0 --auto-snap
# Without --auto-snap, press s to capture each image
# Enroll from image files instead
python enroll.py --name "Jane Doe" --mode images --images "photos/jane/*.jpg"

python run_attendance.py --camera 0
python run_attendance.py --video path/to/video.mp4
```

Useful flags for `run_attendance.py`: `--once-per-day` logs each person at most once a day, `--save-unknown` saves unrecognized faces to `dataset/_unknown/` for review, `--threshold` sets the LBPH confidence cutoff (lower is stricter, default 60), and `--frame-skip N` processes every Nth frame.

Set the detector with an environment variable, for example `export FACE_DETECTOR=haar` (or `set FACE_DETECTOR=haar` in Command Prompt).

## Configuration

`config.py` holds the tuning values: the recognition threshold, frame skip, quality thresholds (blur, exposure, minimum face size), smoothing window and hits required, cooldown in seconds, and whether to draw the overlay.

## Troubleshooting

- `cv2.face` is missing: install `opencv-contrib-python`, not `opencv-python`.
- NumPy or OpenCV import errors on Windows: install `numpy==1.26.4` and `opencv-contrib-python==4.7.0.72`.
- Camera not found: try another index (`--camera 1`) or close other apps using the camera.

## Limitations

LBPH is a classical method. It works for a small group enrolled under similar lighting and camera conditions, and it is less accurate than deep-learning face embeddings. The system does no liveness check, so a photo of an enrolled person can be recognized as that person.

## License

[MIT](LICENSE)
