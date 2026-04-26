# 1. Title

**Project Title:** Automated Attendance System using YOLOv8 Face Detection + FaceNet Face Recognition

**Student Name:** ________________________________  
**Roll No:** _____________________________________  
**Department / Section:** _________________________  
**Date:** ________________________________________

---

# 2. Objective

- To build an automated attendance system that detects multiple faces in a live webcam feed.
- To recognize enrolled students using face embeddings and mark attendance automatically.
- To store attendance in a structured CSV file containing roll number, name, timestamp, and a recognition score.

---

# 3. Problem Statement

In many classrooms and labs, attendance is taken manually. Manual attendance is:

- **Time-consuming**, especially as class size grows.
- **Error-prone**, due to mistakes in roll call or proxy attendance.
- **Difficult to maintain**, because data must be entered/organized after class.

The goal is to automate this process using computer vision so that attendance can be recorded reliably and quickly.

---

# 4. Methodology

This project uses a **two-stage pipeline**:

1) **Face Detection (YOLOv8)** — find face bounding boxes in each frame/image.  
2) **Face Recognition (FaceNet embeddings)** — convert each detected face crop into a 512‑D embedding and match it against enrolled embeddings using cosine similarity.

## 4.1 Enrollment / Training Phase (Embedding Creation)

**Input:** A small dataset of student images (2–3 images per student).  
**Output artifacts:**
- `models/face_encodings.npz` (stored embeddings + labels)
- `models/labels.json` (roll number + name + metadata)

Steps performed by the training script:

1. Load each student folder from `img_dataset/raw/<rollno_name>/`.
2. For each image:
   - Detect faces using YOLOv8 face model (`models/yolov8n-face.pt`).
   - Select the **largest** detected face (to avoid background/false detections).
   - Crop the face and resize to **160×160**.
   - Compute FaceNet embedding using `InceptionResnetV1(pretrained='vggface2')`.
   - L2-normalize the embedding.
3. Save all embeddings and labels to `models/face_encodings.npz`.

## 4.2 Live Attendance Phase (Webcam Recognition)

**Input:** Live webcam frames.  
**Output:** CSV attendance file for the day.

Steps performed during live run:

1. Load enrolled embeddings from `models/face_encodings.npz`.
2. For each webcam frame:
   - Detect faces (YOLO).
   - For each detected face:
     - Crop face region and compute a FaceNet embedding.
     - Compute **cosine similarity** against all known embeddings.
     - Pick the best match.
3. Decision logic:
   - If the best similarity score is **≥ `FACE_MATCH_TOLERANCE`** (currently `0.62`), treat as recognized.
   - Apply a stability gate: the same roll number must be recognized for **`MIN_CONSISTENT_FRAMES`** frames (currently `5`) before marking attendance.
   - Attendance is recorded only once per roll number per day (duplicate protection).

## 4.3 Attendance Logging

- Attendance is written to: `attendance/attendance_YYYY-MM-DD.csv`
- Columns written:
  - `date`, `session_id`, `roll_no`, `name`, `time`, `status`, `score`

---

# 5. Tools Used

**Programming & Runtime**
- Python 3.x

**Libraries**
- OpenCV (`opencv-contrib-python`) — webcam capture, drawing UI overlays, image preprocessing
- Ultralytics YOLOv8 (`ultralytics`) — face detection using a pretrained YOLOv8 face model
- FaceNet (`facenet-pytorch`) — embedding-based face recognition (`InceptionResnetV1`)
- NumPy — vector math (cosine similarity) and model artifact storage (`.npz`)
- Pandas — available for CSV analysis (not required for logging)

**Development Tools**
- VS Code
- Git / GitHub

---

# 6. Dataset

## 6.1 Dataset Description

- **Type:** Private/custom dataset captured from a phone camera or webcam.
- **Classes (students):** 3
- **Images per student:** 2–3
- **Total images used (current training run):** 7

## 6.2 Dataset Folder Structure

The dataset is organized as:

```
img_dataset/raw/
  <rollno_name>/
    image1.jpg
    image2.jpg
    ...
```

Example from this project:

```
img_dataset/raw/
  1601233771057_Manideep/
  160123771055_Rohan/
  160123771060_Vinod/
```

Notes:
- Folder naming is important because the code parses `roll_no` and `name` from `rollno_name`.
- Since the dataset is small, it is recommended to capture images with **different angles** and **lighting** to improve recognition.

---

# 7. Output Screenshots

(Insert screenshots in the final submission. Recommended screenshots:)

1. **Dataset Structure Screenshot**
   - Show `img_dataset/raw/` folders with roll number + name.

2. **Training Output Screenshot**
   - Run `python -m src.train_model` and capture the terminal output showing “Training complete”.

3. **Live Attendance Window Screenshot**
   - Run `python -m src.run_attendance` and capture the webcam window showing:
     - Face bounding boxes
     - Recognized label with roll number + name + score

4. **Attendance CSV Screenshot**
   - Open `attendance/attendance_YYYY-MM-DD.csv` in Excel/Sheets and capture entries.

---

# 8. Conclusion

This project successfully demonstrates an automated attendance system using:

- **YOLOv8** for fast and reliable face detection
- **FaceNet embeddings** for identity recognition
- **CSV logging** for simple attendance storage

The system is designed to work even with a small dataset (2–3 images per student) and includes stability checks to reduce one-frame misclassifications.

---

# 9. Future Scope

- Increase dataset size per student and include more pose/lighting variation for better robustness.
- Add liveness detection (anti-spoofing) to reduce proxy attendance using photos.
- Store attendance in a database (SQLite/MySQL) and generate summary reports.
- Add a simple UI for enrollment (capture images + train + manage students).
- Improve detection/recognition speed using GPU acceleration when available.
