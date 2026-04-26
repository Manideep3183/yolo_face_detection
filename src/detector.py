from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.config import YOLO_WEIGHTS


@dataclass
class DetectionResult:
    boxes: List[Tuple[int, int, int, int]]
    backend: str


class FaceDetector:
    def __init__(self) -> None:
        if not YOLO_WEIGHTS.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {YOLO_WEIGHTS}. Add yolov8n-face.pt to models/."
            )

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError("ultralytics is not available in the current environment") from exc

        self._yolo = YOLO(str(YOLO_WEIGHTS))

    def detect(self, frame: np.ndarray) -> DetectionResult:
        boxes = self._detect_with_yolo(frame)
        return DetectionResult(boxes=boxes, backend="yolo")

    def _detect_with_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        results = self._yolo(frame, verbose=False)
        boxes: List[Tuple[int, int, int, int]] = []

        if not results:
            return boxes

        arr = results[0].boxes.xyxy
        if arr is None:
            return boxes

        for row in arr.cpu().numpy():
            x1, y1, x2, y2 = row[:4].astype(int).tolist()
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

        return boxes
