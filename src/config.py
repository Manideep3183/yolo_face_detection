from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATASET_DIR = ROOT_DIR / "img_dataset" / "raw"
MODELS_DIR = ROOT_DIR / "models"
ATTENDANCE_DIR = ROOT_DIR / "attendance"

YOLO_WEIGHTS = MODELS_DIR / "yolov8n-face.pt"
ENCODINGS_PATH = MODELS_DIR / "face_encodings.npz"
LABELS_PATH = MODELS_DIR / "labels.json"

FACE_MATCH_TOLERANCE = 0.62
MIN_CONSISTENT_FRAMES = 5


def ensure_project_dirs() -> None:
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
