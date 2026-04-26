from __future__ import annotations

import cv2

from src.config import RAW_DATASET_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()

    student_folder = input("Enter folder name (example: 1601233771057_Manideep): ").strip()
    if not student_folder:
        raise RuntimeError("Folder name is required")

    target_dir = RAW_DATASET_DIR / student_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press c to capture image, q to quit")
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(
            frame,
            f"Saved: {count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            out_path = target_dir / f"img_{count:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            count += 1
            print(f"Saved {out_path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
