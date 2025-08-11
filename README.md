## Face Recognition Attendance System (Python + OpenCV)

A real-time attendance app with robust face detection and recognition, alignment, quality gating, temporal smoothing/cooldown, on-screen overlay, and a modern Tk UI. Enrollment auto-trains the recognizer. Reports can be exported as CSV/XLSX.

### Features
- DNN or Haar face detector (toggleable), face alignment (Facemark LBF)
- Quality gating (blur/exposure/size) to skip poor frames
- Temporal smoothing and per-identity cooldown (no duplicate logs)
- On-screen overlay of skip counters
- Enrollment modes: auto/camera/images; auto-train after enroll
- Tkinter + ttkbootstrap Control Panel (`app.py`) with dark/light themes
- Attendance CSV per day; export combined CSV/XLSX; simple summaries

### Setup
- Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
```
- Install dependencies
```bash
pip install -r requirements.txt
```

If you hit NumPy/OpenCV wheel issues on Windows, pin known-good versions:
```bash
pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python
pip install numpy==1.26.4 opencv-contrib-python==4.7.0.72
```

### Models
Models auto-download on first run. Optional manual download (Git Bash):
```bash
# If wget is missing: pacman -S mingw-w64-x86_64-wget or use curl
mkdir -p models/face_detector models/facemark
wget -O models/face_detector/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt
wget -O models/face_detector/res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/4.x/dnn_samples/face_detector/res10_300x300_ssd_iter_140000.caffemodel
wget -O models/facemark/lbfmodel.yaml \
  https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml
```

### Usage (CLI)
- Enroll and auto-train (auto mode chooses images if provided, else camera)
```bash
python enroll.py --name "John Doe" --num-images 30 --camera 0 --auto-snap --delay 0.15
# Images mode
python enroll.py --name "John Doe" --mode images --images "photos/john/*.jpg" "more/*.png"
# Camera mode manual snap (press 's' to capture)
python enroll.py --name "John Doe" --mode camera --num-images 30 --camera 0
```
- Run attendance
```bash
python run_attendance.py --camera 0
# Or
python run_attendance.py --video path/to/video.mp4
```
- Switch detector (default=dnn)
```bash
# Git Bash
export FACE_DETECTOR=haar   # or dnn
# PowerShell
# $env:FACE_DETECTOR="haar"
```

### Usage (UI)
- Launch the control panel
```bash
python app.py
```
- Enroll via camera or images, then start attendance. Detector and camera index are persisted in `ui_settings.json`.

### Configuration
Edit `config.py` to tune:
- `RECOGNITION_CONFIDENCE_THRESHOLD`, `FRAME_SKIP`
- Quality: `QUALITY_BLUR_THRESHOLD`, `QUALITY_EXPOSURE_LOW/HIGH`, `QUALITY_MIN_FACE_SIZE`
- Smoothing/cooldown: `SMOOTHING_WINDOW`, `SMOOTHING_HITS_REQUIRED`, `COOLDOWN_SECONDS`
- Overlay: `DISPLAY_OVERLAY`

### Reports
- Attendance CSVs are saved under `attendance/attendance_YYYYMMDD.csv`.
- In the UI, use Export to save combined CSV/XLSX; or programmatically via `utils/reports.py`.

### Project layout
```
models/           # auto/manual downloaded detector & facemark models
attendance/       # daily CSV logs
dataset/          # enrolled faces
utils/            # detectors, preprocessing, training, reports
  face_detector.py
  preprocess.py
  training.py
  reports.py
config.py
enroll.py
run_attendance.py
app.py
requirements.txt
README.md
```

### Troubleshooting
- DNN errors: models may be truncated; re-download via the URLs above or set `FACE_DETECTOR=haar` (Haar fallback is automatic on runtime errors).
- NumPy import errors: pin to `numpy==1.26.4` and `opencv-contrib-python==4.7.0.72`.
- Camera not found: try a different index (`--camera 1`) or close other apps using the camera.

## License
This project is open source and available under the [MIT License](LICENSE).