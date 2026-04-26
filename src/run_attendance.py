from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from src.attendance import AttendanceWriter
from src.config import ENCODINGS_PATH, FACE_MATCH_TOLERANCE, MIN_CONSISTENT_FRAMES
from src.detector import FaceDetector


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _face_to_embedding(face_bgr: np.ndarray, model: InceptionResnetV1, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy()[0]
    return _l2_normalize(embedding.astype(np.float32))


def main() -> None:
    if not ENCODINGS_PATH.exists():
        raise RuntimeError("Model files are missing. Run: python -m src.train_model")

    loaded = np.load(str(ENCODINGS_PATH), allow_pickle=True)
    known_face_encodings = loaded["encodings"].astype(np.float32)
    known_face_names = loaded["names"]
    known_face_rolls = loaded["rolls"]

    if len(known_face_encodings) == 0:
        raise RuntimeError("No trained encodings found. Run: python -m src.train_model")

    known_face_encodings = np.asarray([_l2_normalize(v) for v in known_face_encodings], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    detector = FaceDetector()

    attendance = AttendanceWriter()

    stable_counts = defaultdict(int)
    marked_rolls = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press q to quit")
    print(f"Device: {device}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        seen_labels = set()

        detection = detector.detect(frame)
        for x1, y1, x2, y2 in detection.boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = _face_to_embedding(crop, embedder, device)
            similarities = known_face_encodings @ embedding

            if len(similarities) == 0:
                continue

            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            info = {
                "roll_no": "Unknown",
                "name": "Unknown",
                "score": best_similarity,
                "recognized": False,
            }

            if best_similarity >= FACE_MATCH_TOLERANCE:
                roll_no = str(known_face_rolls[best_idx])
                name = str(known_face_names[best_idx])
                seen_labels.add(roll_no)
                stable_counts[roll_no] += 1

                if stable_counts[roll_no] >= MIN_CONSISTENT_FRAMES and roll_no not in marked_rolls:
                    is_marked = attendance.mark_present(
                        roll_no=roll_no,
                        name=name,
                        score=best_similarity,
                    )
                    if is_marked:
                        marked_rolls.add(roll_no)
                        print(f"Attendance marked: {roll_no} - {name}")

                info = {
                    "roll_no": roll_no,
                    "name": name,
                    "score": best_similarity,
                    "recognized": True,
                }

            is_recognized = bool(info["recognized"])
            draw_color = (0, 200, 0) if is_recognized else (0, 0, 255)
            score = float(info["score"])

            if is_recognized:
                draw_text = f"{info['roll_no']} {info['name']} {score:.3f}"
            else:
                draw_text = f"Unknown {score:.3f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(
                frame,
                draw_text,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2,
            )

        # Slowly decay counters for identities that disappeared.
        for roll_no in list(stable_counts.keys()):
            if roll_no not in seen_labels:
                stable_counts[roll_no] = max(0, stable_counts[roll_no] - 1)

        cv2.putText(
            frame,
            f"Recognizer: facenet | Detector: {detection.backend}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
