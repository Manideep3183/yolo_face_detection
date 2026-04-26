from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Set

from src.config import ATTENDANCE_DIR, ensure_project_dirs


class AttendanceWriter:
    def __init__(self, session_id: str | None = None) -> None:
        ensure_project_dirs()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M")
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.file_path = ATTENDANCE_DIR / f"attendance_{self.today}.csv"
        self._marked_rolls: Set[str] = set()
        self._init_file()
        self._load_existing_today()

    def _init_file(self) -> None:
        if self.file_path.exists():
            return
        with open(self.file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "date",
                "session_id",
                "roll_no",
                "name",
                "time",
                "status",
                "score",
            ])

    def _load_existing_today(self) -> None:
        if not self.file_path.exists():
            return
        with open(self.file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date") == self.today and row.get("roll_no"):
                    self._marked_rolls.add(row["roll_no"])

    def mark_present(self, roll_no: str, name: str, score: float) -> bool:
        if roll_no in self._marked_rolls:
            return False

        now = datetime.now()
        with open(self.file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                self.session_id,
                roll_no,
                name,
                now.strftime("%H:%M:%S"),
                "Present",
                f"{score:.2f}",
            ])

        self._marked_rolls.add(roll_no)
        return True
