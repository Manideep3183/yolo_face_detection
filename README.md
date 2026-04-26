# Student Attendance (YOLO + FaceNet)

Current implementation:
- YOLOv8 face model for face detection
- FaceNet embeddings (facenet-pytorch) for recognition
- CSV attendance logging with roll number, name, timestamp, and score

Assignment report format (ready to submit):
- See [REPORT.md](REPORT.md)

## 1) Install

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Dataset layout

Use one folder per student in rollno_name format:

```text
img_dataset/raw/
  1601233771057_Manideep/
    Manideep1.jpg
    Manideep2.jpg
  160123771055_Rohan/
    Rohan1.jpg
    Rohan2.jpg
  160123771060_Vinod/
    Vinod1.jpg
    Vinod2.jpg
```

## 3) YOLO Weights (required)

This file must exist before training or live detection:

```text
models/yolov8n-face.pt
```

Detection is YOLO-only in the current code.

## 4) Train

```powershell
.\.venv\Scripts\python.exe -m src.train_model
```

This creates:
- `models/face_encodings.npz`
- `models/labels.json`

## 5) Run Live Attendance

```powershell
.\.venv\Scripts\python.exe -m src.run_attendance
```

Controls:
- Press `q` to quit

Output:
- `attendance/attendance_YYYY-MM-DD.csv`

## (Optional) Capture dataset images from webcam

If you want to collect images directly from your webcam into the correct dataset folder:

```powershell
.\.venv\Scripts\python.exe -m src.capture_images
```

## 6) Tuning

Recognition threshold is in `src/config.py`:
- `FACE_MATCH_TOLERANCE` (cosine similarity, higher means stricter match)
- `MIN_CONSISTENT_FRAMES` (stability before writing attendance)
