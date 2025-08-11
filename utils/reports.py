import os
import glob
from datetime import datetime
from typing import List

import pandas as pd

import config


def load_attendance_frames() -> List[pd.DataFrame]:
    pattern = os.path.join(config.ATTENDANCE_DIR, "attendance_*.csv")
    frames: List[pd.DataFrame] = []
    for path in glob.glob(pattern):
        try:
            df = pd.read_csv(path)
            df["SourceFile"] = os.path.basename(path)
            frames.append(df)
        except Exception:
            continue
    return frames


def export_reports(dest_path: str) -> None:
    frames = load_attendance_frames()
    if not frames:
        raise RuntimeError("No attendance CSVs found to export.")
    all_df = pd.concat(frames, ignore_index=True)

    # Normalize Timestamp column if present
    if "Timestamp" in all_df.columns:
        try:
            all_df["Timestamp"] = pd.to_datetime(all_df["Timestamp"])
        except Exception:
            pass

    # Summary: counts per Name
    summary = all_df.groupby("Name", dropna=False).size().reset_index(name="Count").sort_values("Count", ascending=False)

    # Export
    ext = os.path.splitext(dest_path)[1].lower()
    if ext == ".csv":
        all_df.to_csv(dest_path, index=False)
    else:
        with pd.ExcelWriter(dest_path, engine="openpyxl") as writer:
            all_df.to_excel(writer, index=False, sheet_name="All Attendance")
            summary.to_excel(writer, index=False, sheet_name="Summary")
