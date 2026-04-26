from __future__ import annotations

import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from src.config import ENCODINGS_PATH, LABELS_PATH, RAW_DATASET_DIR, ensure_project_dirs
from src.detector import FaceDetector


def parse_student_folder(folder_name: str) -> Tuple[str, str]:
    if "_" in folder_name:
        roll_no, name = folder_name.split("_", 1)
        return roll_no.strip(), name.strip()
    return folder_name.strip(), folder_name.strip()


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _largest_box(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _face_to_embedding(face_bgr: np.ndarray, model: InceptionResnetV1, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy()[0]
    return _l2_normalize(embedding.astype(np.float32))


def load_main_encoding(
    image_path: str,
    detector: FaceDetector,
    model: InceptionResnetV1,
    device: torch.device,
) -> np.ndarray | None:
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[WARNING] Could not read image {image_path}")
            return None

        detection = detector.detect(frame)
        box = _largest_box(detection.boxes)
        if box is None:
            print(f"[WARNING] No face found in {image_path}")
            return None

        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[WARNING] Invalid crop in {image_path}")
            return None

        return _face_to_embedding(crop, model, device)
    except Exception as exc:
        print(f"[ERROR] Could not process {image_path}: {exc}")
        return None


def train_embeddings() -> None:
    ensure_project_dirs()
    detector = FaceDetector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    known_face_encodings: List[np.ndarray] = []
    known_face_names: List[str] = []
    known_face_rolls: List[str] = []
    label_map: Dict[int, Dict[str, str]] = {}
    per_student_counts: Dict[str, int] = {}

    student_index = 0
    for student_dir in sorted(RAW_DATASET_DIR.glob("*")):
        if not student_dir.is_dir():
            continue

        roll_no, name = parse_student_folder(student_dir.name)
        image_files = list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.jpeg")) + list(student_dir.glob("*.png"))
        if not image_files:
            continue

        local_count = 0
        for image_path in image_files:
            encoding = load_main_encoding(
                str(image_path),
                detector=detector,
                model=embedder,
                device=device,
            )
            if encoding is None:
                continue
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            known_face_rolls.append(roll_no)
            local_count += 1

        if local_count > 0:
            per_student_counts[student_dir.name] = local_count
            label_map[student_index] = {
                "roll_no": roll_no,
                "name": name,
                "folder": student_dir.name,
                "images_used": str(local_count),
            }
            student_index += 1

    if len(known_face_encodings) < 3 or len(per_student_counts) < 2:
        raise RuntimeError(
            "Not enough data to train. Add images in img_dataset/raw/<rollno_name>/ and rerun."
        )

    np.savez_compressed(
        str(ENCODINGS_PATH),
        encodings=np.asarray(known_face_encodings, dtype=np.float64),
        names=np.asarray(known_face_names),
        rolls=np.asarray(known_face_rolls),
    )

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("Training complete")
    print(f"Device used: {device}")
    print(f"Embeddings: {ENCODINGS_PATH}")
    print(f"Labels: {LABELS_PATH}")
    print("Images used per student:")
    for key, value in per_student_counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    train_embeddings()
