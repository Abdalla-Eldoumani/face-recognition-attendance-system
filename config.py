import os
import cv2
from datetime import datetime

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
DB_PATH = os.path.join(ATTENDANCE_DIR, "attendance.db")
THUMBS_DIR = os.path.join(MODELS_DIR, "thumbs")

# Model and label map paths
MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.xml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# Detector selection: 'haar' or 'dnn'
FACE_DETECTOR = os.environ.get("FACE_DETECTOR", "dnn").lower()

# Haar cascade path from OpenCV installation
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# DNN detector (ResNet-10 SSD) files and settings
DNN_DIR = os.path.join(MODELS_DIR, "face_detector")
DNN_PROTO_FILENAME = "deploy.prototxt"
DNN_WEIGHTS_FILENAME = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_PROTO_PATH = os.path.join(DNN_DIR, DNN_PROTO_FILENAME)
DNN_WEIGHTS_PATH = os.path.join(DNN_DIR, DNN_WEIGHTS_FILENAME)
DNN_CONFIDENCE_THRESHOLD = 0.5

# Facemark (LBF) for alignment
FACEMARK_DIR = os.path.join(MODELS_DIR, "facemark")
LBF_MODEL_FILENAME = "lbfmodel.yaml"
LBF_MODEL_PATH = os.path.join(FACEMARK_DIR, LBF_MODEL_FILENAME)
# Desired eye positions within the aligned output image
ALIGN_LEFT_EYE = (0.35, 0.35)
ALIGN_RIGHT_EYE = (0.65, 0.35)

# Detection parameters (used by Haar; some also relevant as general defaults)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (60, 60)

# Recognition parameters
FACE_SIZE = 200  # width and height for LBPH input
RECOGNITION_CONFIDENCE_THRESHOLD = 60.0  # lower = stricter (LBPH confidence)

# Quality gating thresholds
QUALITY_BLUR_THRESHOLD = 100.0  # variance of Laplacian; increase to be stricter
QUALITY_EXPOSURE_LOW = 50      # grayscale mean lower bound
QUALITY_EXPOSURE_HIGH = 200    # grayscale mean upper bound
QUALITY_MIN_FACE_SIZE = (80, 80)  # min (w, h) for gating

# Temporal smoothing and cooldown
SMOOTHING_WINDOW = 5
SMOOTHING_HITS_REQUIRED = 3
COOLDOWN_SECONDS = 60

# Overlay configuration
DISPLAY_OVERLAY = True

# Performance tuning
FRAME_SKIP = 1  # process every Nth frame; set to 2 or 3 if you need more speed


def ensure_directories() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    os.makedirs(DNN_DIR, exist_ok=True)
    os.makedirs(FACEMARK_DIR, exist_ok=True)
    os.makedirs(THUMBS_DIR, exist_ok=True)


def get_attendance_csv_path() -> str:
    today_string = datetime.now().strftime("%Y%m%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today_string}.csv")
