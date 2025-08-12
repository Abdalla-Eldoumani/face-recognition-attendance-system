import os
import sys
import json
import time
import threading
import subprocess
import shutil
from collections import deque
from datetime import datetime

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import config

APP_TITLE = "Face Recognition Attendance — Control Panel"
SETTINGS_JSON = "ui_settings.json"
THUMB_HISTORY = 6


def which_python() -> str:
    return sys.executable or "python"


def load_settings() -> dict:
    if os.path.exists(SETTINGS_JSON):
        try:
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_settings(d: dict) -> None:
    try:
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class ProcessRunner:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.alive = False

    def start(self, args: list[str], env: dict[str, str] | None, on_line):
        if self.proc is not None:
            return
        self.alive = True
        def _run():
            try:
                self.proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    if not self.alive:
                        break
                    on_line(line.rstrip())
            except Exception as e:
                on_line(f"[error] {e}")
            finally:
                self.proc = None
                self.alive = False
        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        self.alive = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


class App(tb.Window):
    def __init__(self):
        super().__init__(title=APP_TITLE, themename="darkly")
        # Slightly wider and taller default; allow vertical scrolling where needed
        self.geometry("1200x820")

        self._settings = load_settings()

        # Top bar
        top = tb.Frame(self)
        top.pack(fill=X, padx=12, pady=8)

        tb.Label(top, text="Face detector:", bootstyle=INFO).pack(side=LEFT)
        self.detector_var = tb.StringVar(value=self._settings.get("detector", os.environ.get("FACE_DETECTOR", "haar")))
        self.detector_combo = tb.Combobox(top, textvariable=self.detector_var, values=["haar", "dnn"], width=10, state="readonly")
        self.detector_combo.pack(side=LEFT, padx=8)

        # Presets (Performance vs Accuracy) and theme buttons
        tb.Label(top, text="Preset:", bootstyle=INFO).pack(side=LEFT, padx=(12,0))
        self.preset_var = tb.StringVar(value=self._settings.get("preset", "Accuracy"))
        self.preset_combo = tb.Combobox(top, textvariable=self.preset_var, values=["Performance", "Accuracy"], width=14, state="readonly")
        self.preset_combo.pack(side=LEFT, padx=8)
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_preset(self.preset_var.get()))

        # Move export/open buttons to the top-right to ensure visibility on smaller screens
        tb.Button(top, text="Export CSV/XLSX", bootstyle=PRIMARY, command=self._export_reports).pack(side=RIGHT)
        tb.Button(top, text="Open Attendance Folder", bootstyle=SECONDARY, command=self._open_attendance_dir).pack(side=RIGHT, padx=(0,8))
        tb.Button(top, text="Light", bootstyle=SECONDARY, command=lambda: self._set_theme("flatly")).pack(side=RIGHT, padx=4)
        tb.Button(top, text="Dark", bootstyle=SECONDARY, command=lambda: self._set_theme("darkly")).pack(side=RIGHT)

        # Notebook
        self.nb = tb.Notebook(self, bootstyle=PRIMARY)
        self.nb.pack(fill=BOTH, expand=True, padx=12, pady=(0,8))

        self.enroll_tab = tb.Frame(self.nb)
        self.run_tab = tb.Frame(self.nb)
        self.history_tab = tb.Frame(self.nb)
        self.review_tab = tb.Frame(self.nb)
        self.nb.add(self.enroll_tab, text="Enroll")
        self.nb.add(self.run_tab, text="Run Attendance")
        self.nb.add(self.history_tab, text="History & Charts")
        self.nb.add(self.review_tab, text="Review Unknowns")

        self._build_enroll_tab()
        self._build_run_tab()
        self._build_history_tab()
        self._build_review_tab()

        # Live/Recent panel
        side = tb.Labelframe(self, text="Live Preview & Recent Recognitions", bootstyle=INFO)
        side.pack(fill=BOTH, expand=False, padx=12, pady=(0,8))

        live_row = tb.Frame(side)
        live_row.pack(fill=X, padx=8, pady=8)
        self.thumb_images: deque[ImageTk.PhotoImage] = deque(maxlen=THUMB_HISTORY)
        self.thumb_labels: list[tb.Label] = []
        for _ in range(THUMB_HISTORY):
            lbl = tb.Label(live_row, image=None, width=120, bootstyle=SECONDARY)
            lbl.pack(side=LEFT, padx=4)
            self.thumb_labels.append(lbl)

        recent_frame = tb.Frame(side)
        recent_frame.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.recent_tree = tb.Treeview(recent_frame, columns=("name", "time", "confidence"), show="headings", height=6)
        self.recent_tree.heading("name", text="Name")
        self.recent_tree.heading("time", text="Last Seen")
        self.recent_tree.heading("confidence", text="Conf.")
        self.recent_tree.column("name", width=200)
        self.recent_tree.column("time", width=160)
        self.recent_tree.column("confidence", width=80, anchor=E)
        self.recent_tree.pack(fill=BOTH, expand=True)

        # Log area
        log_frame = tb.Labelframe(self, text="Logs", bootstyle=INFO)
        log_frame.pack(fill=BOTH, expand=True, padx=12, pady=(0,8))
        self.log = tb.ScrolledText(log_frame, height=10)
        self.log.pack(fill=BOTH, expand=True, padx=8, pady=8)

        # Bottom status bar only
        bot = tb.Frame(self)
        bot.pack(fill=X, padx=12, pady=(0,10))
        self.status = tb.Label(bot, text="Ready", anchor=W, bootstyle=SECONDARY)
        self.status.pack(fill=X, side=LEFT)

        self.enroll_runner = ProcessRunner()
        self.run_runner = ProcessRunner()

        # Persist detector changes
        self.detector_combo.bind("<<ComboboxSelected>>", lambda e: self._save_setting("detector", self.detector_var.get()))

    def _set_theme(self, theme):
        try:
            self.style.theme_use(theme)
            self._save_setting("theme", theme)
        except Exception:
            messagebox.showwarning("Theme", f"Theme '{theme}' not available.")

    def _apply_preset(self, preset: str):
        # Store preset
        self._save_setting("preset", preset)
        # Apply simple defaults: Performance (skip frames) vs Accuracy (no skip)
        try:
            if preset == "Performance":
                # Prefer Haar for speed; skip frames
                self.detector_var.set("haar")
                self._save_setting("detector", "haar")
                # Log hint; frame skip is passed at runtime; we keep UI simple
                self._append_log("[preset] Performance: detector=haar; consider --frame-skip 2 for more speed.")
            else:
                # Prefer DNN for accuracy
                self.detector_var.set("dnn")
                self._save_setting("detector", "dnn")
                self._append_log("[preset] Accuracy: detector=dnn; process every frame.")
        except Exception as e:
            self._append_log(f"[preset] Failed to apply preset: {e}")

    def _append_log(self, line: str):
        self.log.insert(END, line + "\n")
        self.log.see(END)
        self.status.configure(text=line[:120])
        # Parse structured recognition logs from run_attendance:
        # Formats:
        #   "Marked present: NAME"
        #   "Recognized NAME CONF=XX.X THUMB=path"
        if "Marked present:" in line:
            name = line.split(":", 1)[1].strip()
            self._update_recent(name, confidence=None, thumb_path=None)
        elif line.startswith("Recognized "):
            try:
                parts = line[len("Recognized "):].strip().split()
                name = parts[0]
                conf_val = None
                thumb_path = None
                for p in parts[1:]:
                    if p.startswith("CONF="):
                        conf_val = float(p.split("=", 1)[1])
                    if p.startswith("THUMB="):
                        thumb_path = p.split("=", 1)[1]
                self._update_recent(name, confidence=conf_val, thumb_path=thumb_path)
            except Exception:
                pass

    def _save_setting(self, key: str, value):
        self._settings[key] = value
        save_settings(self._settings)

    # Enroll tab UI
    def _build_enroll_tab(self):
        f = self.enroll_tab
        pad = dict(padx=10, pady=8)

        row1 = tb.Frame(f); row1.pack(fill=X, **pad)
        tb.Label(row1, text="Name:").pack(side=LEFT)
        self.en_name = tb.Entry(row1, width=30); self.en_name.pack(side=LEFT, padx=8)

        row2 = tb.Frame(f); row2.pack(fill=X, **pad)
        tb.Label(row2, text="Mode:").pack(side=LEFT)
        self.mode_var = tb.StringVar(value="auto")
        for m in ("auto", "camera", "images"):
            tb.Radiobutton(row2, text=m.capitalize(), value=m, variable=self.mode_var, bootstyle=SUCCESS).pack(side=LEFT, padx=6)

        row3 = tb.Frame(f); row3.pack(fill=X, **pad)
        tb.Label(row3, text="Camera index:").pack(side=LEFT)
        self.cam_index = tb.Spinbox(row3, from_=0, to=8, width=6)
        self.cam_index.delete(0, END); self.cam_index.insert(0, str(self._settings.get("camera_index", 0)))
        self.cam_index.pack(side=LEFT, padx=8)
        self.cam_index.bind("<FocusOut>", lambda e: self._save_setting("camera_index", int(self.cam_index.get())))

        tb.Label(row3, text="Num images:").pack(side=LEFT)
        self.num_images = tb.Spinbox(row3, from_=5, to=200, width=6)
        self.num_images.delete(0, END); self.num_images.insert(0, "30")
        self.num_images.pack(side=LEFT, padx=8)

        tb.Label(row3, text="Delay(s):").pack(side=LEFT)
        self.delay = tb.Spinbox(row3, from_=0.0, to=3.0, increment=0.05, width=6)
        self.delay.delete(0, END); self.delay.insert(0, "0.15")
        self.delay.pack(side=LEFT, padx=8)

        self.auto_snap = tb.BooleanVar(value=True)
        tb.Checkbutton(row3, text="Auto-snap", variable=self.auto_snap, bootstyle=INFO).pack(side=LEFT, padx=8)

        row4 = tb.Frame(f); row4.pack(fill=X, **pad)
        self.images_paths: list[str] = []
        tb.Button(row4, text="Select images...", bootstyle=PRIMARY, command=self._select_images).pack(side=LEFT)
        self.images_label = tb.Label(row4, text="No images selected", bootstyle=SECONDARY)
        self.images_label.pack(side=LEFT, padx=10)

        row5 = tb.Frame(f); row5.pack(fill=X, **pad)
        tb.Button(row5, text="Start Enroll", bootstyle=SUCCESS, command=self._start_enroll).pack(side=LEFT)
        tb.Button(row5, text="Stop", bootstyle=DANGER, command=lambda: self.enroll_runner.stop()).pack(side=LEFT, padx=8)

    def _select_images(self):
        paths = filedialog.askopenfilenames(title="Select images", filetypes=[("Images", ".jpg .jpeg .png .bmp")])
        if paths:
            self.images_paths = list(paths)
            self.images_label.config(text=f"{len(self.images_paths)} images selected")

    def _start_enroll(self):
        name = self.en_name.get().strip()
        if not name:
            messagebox.showerror("Missing", "Please enter a name")
            return
        mode = self.mode_var.get()
        args = [which_python(), "enroll.py", "--name", name, "--mode", mode]
        if mode in ("auto", "camera"):
            args += ["--num-images", str(self.num_images.get()), "--camera", str(self.cam_index.get())]
            if self.auto_snap.get():
                args.append("--auto-snap")
            args += ["--delay", str(self.delay.get())]
        if mode in ("auto", "images") and self.images_paths:
            args += ["--images", *self.images_paths]

        env = os.environ.copy()
        env["FACE_DETECTOR"] = self.detector_var.get()
        self._append_log("$ " + " ".join(args))
        self.enroll_runner.start(args, env, self._append_log)

    # Run tab UI
    def _build_run_tab(self):
        f = self.run_tab
        pad = dict(padx=10, pady=8)

        row1 = tb.Frame(f); row1.pack(fill=X, **pad)
        self.run_source = tb.StringVar(value="camera")
        for m in ("camera", "video"):
            tb.Radiobutton(row1, text=m.capitalize(), value=m, variable=self.run_source, bootstyle=INFO).pack(side=LEFT, padx=6)

        row2 = tb.Frame(f); row2.pack(fill=X, **pad)
        tb.Label(row2, text="Camera index:").pack(side=LEFT)
        self.run_cam_index = tb.Spinbox(row2, from_=0, to=8, width=6)
        self.run_cam_index.delete(0, END); self.run_cam_index.insert(0, str(self._settings.get("camera_index", 0)))
        self.run_cam_index.pack(side=LEFT, padx=8)
        self.run_cam_index.bind("<FocusOut>", lambda e: self._save_setting("camera_index", int(self.run_cam_index.get())))

        tb.Label(row2, text="Video file:").pack(side=LEFT)
        self.video_path_var = tb.StringVar()
        self.video_entry = tb.Entry(row2, textvariable=self.video_path_var, width=40)
        self.video_entry.pack(side=LEFT, padx=8)
        tb.Button(row2, text="Browse...", bootstyle=PRIMARY, command=self._select_video).pack(side=LEFT)

        row2b = tb.Frame(f); row2b.pack(fill=X, **pad)
        tb.Label(row2b, text="Threshold (LBPH, lower=stricter):").pack(side=LEFT)
        self.threshold_entry = tb.Entry(row2b, width=8)
        try:
            self.threshold_entry.insert(0, str(config.RECOGNITION_CONFIDENCE_THRESHOLD))
        except Exception:
            self.threshold_entry.insert(0, "60.0")
        self.threshold_entry.pack(side=LEFT, padx=8)

        row3 = tb.Frame(f); row3.pack(fill=X, **pad)
        tb.Button(row3, text="Start", bootstyle=SUCCESS, command=self._start_run).pack(side=LEFT)
        tb.Button(row3, text="Stop", bootstyle=DANGER, command=lambda: self.run_runner.stop()).pack(side=LEFT, padx=8)
        # Default flags for attendance
        row4 = tb.Frame(f); row4.pack(fill=X, **pad)
        self.once_per_day_var = tb.BooleanVar(value=self._settings.get("once_per_day", True))
        self.save_unknown_var = tb.BooleanVar(value=self._settings.get("save_unknown", True))
        tb.Checkbutton(row4, text="Once per day", variable=self.once_per_day_var, bootstyle=INFO,
                       command=lambda: self._save_setting("once_per_day", bool(self.once_per_day_var.get()))).pack(side=LEFT, padx=6)
        tb.Checkbutton(row4, text="Save unknown", variable=self.save_unknown_var, bootstyle=INFO,
                       command=lambda: self._save_setting("save_unknown", bool(self.save_unknown_var.get()))).pack(side=LEFT, padx=6)

    def _select_video(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Videos", ".mp4 .avi .mov .mkv"), ("All", ".*")])
        if path:
            self.video_path_var.set(path)

    def _start_run(self):
        args = [which_python(), "run_attendance.py"]
        if self.run_source.get() == "camera":
            args += ["--camera", str(self.run_cam_index.get())]
        else:
            video = self.video_path_var.get()
            if not video:
                messagebox.showerror("Missing", "Please select a video file")
                return
            args += ["--video", video]
        # threshold
        thr = self.threshold_entry.get().strip()
        if thr:
            args += ["--threshold", thr]
        if self.once_per_day_var.get():
            args.append("--once-per-day")
        if self.save_unknown_var.get():
            args.append("--save-unknown")
        env = os.environ.copy()
        env["FACE_DETECTOR"] = self.detector_var.get()
        self._append_log("$ " + " ".join(args))
        self.run_runner.start(args, env, self._append_log)

    # History & Charts tab
    def _build_history_tab(self):
        import sqlite3
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt

        f = self.history_tab
        pad = dict(padx=10, pady=8)

        filters = tb.Frame(f); filters.pack(fill=X, **pad)
        tb.Label(filters, text="From:").pack(side=LEFT)
        self.hist_from = tb.Entry(filters, width=12)
        self.hist_from.insert(0, "YYYY-MM-DD")
        self.hist_from.pack(side=LEFT, padx=6)
        tb.Label(filters, text="To:").pack(side=LEFT)
        self.hist_to = tb.Entry(filters, width=12)
        self.hist_to.insert(0, "YYYY-MM-DD")
        self.hist_to.pack(side=LEFT, padx=6)
        tb.Label(filters, text="Person:").pack(side=LEFT, padx=(12,0))
        self.hist_person = tb.Entry(filters, width=20)
        self.hist_person.pack(side=LEFT, padx=6)
        tb.Button(filters, text="Refresh", bootstyle=PRIMARY, command=self._refresh_history).pack(side=LEFT, padx=8)

        # Chart area
        self.hist_fig = plt.Figure(figsize=(6,3), dpi=100)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=f)
        self.hist_canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Table area
        table_frame = tb.Frame(f)
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0,10))
        self.hist_tree = tb.Treeview(table_frame, columns=("name","ts"), show="headings")
        self.hist_tree.heading("name", text="Name")
        self.hist_tree.heading("ts", text="Timestamp")
        self.hist_tree.column("name", width=220)
        self.hist_tree.column("ts", width=200)
        self.hist_tree.pack(fill=BOTH, expand=True)

        # Initial load
        self._refresh_history()

    def _refresh_history(self):
        import sqlite3
        from datetime import datetime as dt

        # read filters
        date_from = self.hist_from.get().strip()
        date_to = self.hist_to.get().strip()
        person = self.hist_person.get().strip()

        where = []
        params = []
        if date_from and date_from != "YYYY-MM-DD":
            where.append("ts >= ?")
            params.append(date_from + " 00:00:00")
        if date_to and date_to != "YYYY-MM-DD":
            where.append("ts <= ?")
            params.append(date_to + " 23:59:59")
        if person:
            where.append("name = ?")
            params.append(person)
        sql = "SELECT name, ts FROM attendance"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC"

        rows = []
        try:
            conn = sqlite3.connect(config.DB_PATH)
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, ts TEXT)")
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            self._append_log(f"[history] DB error: {e}")
            rows = []

        # update table
        self.hist_tree.delete(*self.hist_tree.get_children())
        for name, ts in rows:
            self.hist_tree.insert("", "end", values=(name, ts))

        # aggregate counts per person
        counts = {}
        for name, _ in rows:
            counts[name] = counts.get(name, 0) + 1

        # redraw chart
        self._draw_counts_chart(counts)

    def _draw_counts_chart(self, counts: dict):
        import matplotlib.pyplot as plt
        self.hist_ax.clear()
        if counts:
            names = list(counts.keys())
            values = [counts[n] for n in names]
            self.hist_ax.bar(names, values, color="#4e79a7")
            self.hist_ax.set_ylabel("Count")
            self.hist_ax.set_title("Attendance Counts per Person")
            self.hist_ax.tick_params(axis='x', rotation=45)
            self.hist_fig.tight_layout()
        else:
            self.hist_ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=self.hist_ax.transAxes)
        self.hist_canvas.draw_idle()

    # Review Unknowns tab UI
    def _build_review_tab(self):
        f = self.review_tab
        pad = dict(padx=10, pady=8)

        top = tb.Frame(f); top.pack(fill=X, **pad)
        tb.Label(top, text="Unknown faces folder:").pack(side=LEFT)
        self.unknown_dir_label = tb.Label(top, text=os.path.abspath(os.path.join(config.DATASET_DIR, "_unknown")), bootstyle=SECONDARY)
        self.unknown_dir_label.pack(side=LEFT, padx=8)
        tb.Button(top, text="Refresh", bootstyle=PRIMARY, command=self._refresh_unknowns).pack(side=RIGHT)

        body = tb.Frame(f); body.pack(fill=BOTH, expand=True, **pad)
        left = tb.Frame(body); left.pack(side=LEFT, fill=Y)
        right = tb.Frame(body); right.pack(side=LEFT, fill=BOTH, expand=True, padx=(16,0))

        # Unknowns list
        self.unknown_tree = tb.Treeview(left, columns=("file"), show="headings", height=16)
        self.unknown_tree.heading("file", text="Filename")
        self.unknown_tree.column("file", width=280)
        self.unknown_tree.pack(fill=Y, expand=False)
        # Enable multi-select and keyboard shortcuts
        self.unknown_tree.configure(selectmode="extended")
        self.unknown_tree.bind("<Control-a>", lambda e: self._select_all_unknowns())
        self.unknown_tree.bind("<Delete>", lambda e: self._delete_selected_unknowns())
        self.unknown_tree.bind("<<TreeviewSelect>>", lambda e: self._show_unknown_preview())

        btns = tb.Frame(left); btns.pack(fill=X, pady=8)
        tb.Button(btns, text="Delete Selected", bootstyle=DANGER, command=self._delete_selected_unknowns).pack(side=LEFT)

        # Preview + Assign
        preview_frame = tb.Labelframe(right, text="Preview", bootstyle=INFO); preview_frame.pack(fill=BOTH, expand=True)
        self.preview_label = tb.Label(preview_frame, image=None, bootstyle=SECONDARY)
        self.preview_label.pack(padx=8, pady=8)

        assign = tb.Labelframe(right, text="Assign", bootstyle=INFO); assign.pack(fill=X, pady=8)
        rowp = tb.Frame(assign); rowp.pack(fill=X, padx=8, pady=6)
        tb.Label(rowp, text="Existing person:").pack(side=LEFT)
        self.people_combo_var = tb.StringVar()
        self.people_combo = tb.Combobox(rowp, textvariable=self.people_combo_var, state="readonly", width=30)
        self.people_combo.pack(side=LEFT, padx=8)
        tb.Button(rowp, text="Reload", bootstyle=SECONDARY, command=self._reload_people).pack(side=LEFT)

        rown = tb.Frame(assign); rown.pack(fill=X, padx=8, pady=6)
        tb.Label(rown, text="New person name:").pack(side=LEFT)
        self.new_person_entry = tb.Entry(rown, width=30)
        self.new_person_entry.pack(side=LEFT, padx=8)

        rowo = tb.Frame(assign); rowo.pack(fill=X, padx=8, pady=6)
        self.auto_retrain_var = tb.BooleanVar(value=True)
        tb.Checkbutton(rowo, text="Auto-retrain after assign", variable=self.auto_retrain_var, bootstyle=INFO).pack(side=LEFT)

        rowb = tb.Frame(assign); rowb.pack(fill=X, padx=8, pady=6)
        tb.Button(rowb, text="Assign to Existing", bootstyle=SUCCESS, command=self._assign_to_existing).pack(side=LEFT)
        tb.Button(rowb, text="Assign to New", bootstyle=PRIMARY, command=self._assign_to_new).pack(side=LEFT, padx=8)

        self._reload_people()
        self._refresh_unknowns()

    def _refresh_unknowns(self):
        self.unknown_tree.delete(*self.unknown_tree.get_children())
        unk_dir = os.path.join(config.DATASET_DIR, "_unknown")
        os.makedirs(unk_dir, exist_ok=True)
        try:
            items = [f for f in os.listdir(unk_dir) if os.path.isfile(os.path.join(unk_dir, f))]
            items.sort(reverse=True)
            for f in items:
                self.unknown_tree.insert("", "end", values=(f,))
        except Exception as e:
            self._append_log(f"[error] Failed listing unknowns: {e}")

    def _reload_people(self):
        people = []
        try:
            for d in os.listdir(config.DATASET_DIR):
                p = os.path.join(config.DATASET_DIR, d)
                if os.path.isdir(p) and d != "_unknown":
                    people.append(d)
        except Exception:
            pass
        people.sort()
        self.people_combo.configure(values=people)
        if people:
            self.people_combo_var.set(people[0])

    def _get_selected_unknown_paths(self) -> list[str]:
        sel = self.unknown_tree.selection()
        paths: list[str] = []
        unk_dir = os.path.join(config.DATASET_DIR, "_unknown")
        for item in sel:
            filename = self.unknown_tree.item(item, "values")[0]
            paths.append(os.path.join(unk_dir, filename))
        return paths

    def _show_unknown_preview(self):
        paths = self._get_selected_unknown_paths()
        if not paths:
            return
        path = paths[0]
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 270))
            tkimg = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=tkimg)
            self.preview_label.image = tkimg
        except Exception:
            self.preview_label.configure(image=None)
            self.preview_label.image = None

    def _delete_selected_unknowns(self):
        paths = self._get_selected_unknown_paths()
        if not paths:
            return
        removed = 0
        for p in paths:
            try:
                os.remove(p)
                removed += 1
            except Exception as e:
                self._append_log(f"[error] Delete failed for {p}: {e}")
        self._append_log(f"Deleted {removed} unknown file(s)")
        self._refresh_unknowns()

    def _select_all_unknowns(self):
        for iid in self.unknown_tree.get_children():
            self.unknown_tree.selection_add(iid)

    def _assign_list_to_person(self, person_name: str, paths: list[str]):
        if not person_name:
            messagebox.showerror("Assign", "Person name is required")
            return
        dest_dir = os.path.join(config.DATASET_DIR, person_name.replace(" ", "_"))
        os.makedirs(dest_dir, exist_ok=True)
        moved = 0
        for p in paths:
            try:
                base = os.path.basename(p)
                ts = int(time.time() * 1000)
                new_name = f"assigned_{ts}_{base}"
                shutil.move(p, os.path.join(dest_dir, new_name))
                moved += 1
            except Exception as e:
                self._append_log(f"[error] Assign failed for {p}: {e}")
        self._append_log(f"Assigned {moved} file(s) to {person_name}")
        self._refresh_unknowns()
        if self.auto_retrain_var.get() and moved > 0:
            self._start_retrain()

    def _assign_to_existing(self):
        paths = self._get_selected_unknown_paths()
        if not paths:
            messagebox.showinfo("Assign", "No unknown files selected")
            return
        person = self.people_combo_var.get().strip()
        self._assign_list_to_person(person, paths)

    def _assign_to_new(self):
        paths = self._get_selected_unknown_paths()
        if not paths:
            messagebox.showinfo("Assign", "No unknown files selected")
            return
        person = self.new_person_entry.get().strip()
        if not person:
            messagebox.showerror("Assign", "Please enter a new person name")
            return
        self._assign_list_to_person(person, paths)

    def _start_retrain(self):
        # Modal spinner window
        spinner = tb.Toplevel(self)
        spinner.title("Retraining...")
        spinner.geometry("300x120")
        spinner.transient(self)
        spinner.grab_set()
        tb.Label(spinner, text="Retraining model, please wait...", bootstyle=INFO).pack(pady=12)
        pb = tb.Progressbar(spinner, mode="indeterminate")
        pb.pack(fill=X, padx=16, pady=8)
        pb.start(10)

        def _work():
            try:
                self._append_log("[train] Starting retrain...")
                from utils.training import train_and_save
                train_and_save()
                self._append_log("[train] Retrain complete.")
            except Exception as e:
                self._append_log(f"[train] Retrain failed: {e}")
            finally:
                try:
                    pb.stop()
                    spinner.grab_release()
                    spinner.destroy()
                except Exception:
                    pass
        t = threading.Thread(target=_work, daemon=True)
        t.start()

    # Update recent recognitions table and thumbnails
    def _update_recent(self, name: str, confidence: float | None, thumb_path: str | None = None):
        now = datetime.now().strftime("%H:%M:%S")
        self.recent_tree.insert("", 0, values=(name, now, f"{confidence:.1f}" if confidence is not None else "-"))
        # Update thumbnails
        if thumb_path and os.path.exists(thumb_path):
            try:
                img = Image.open(thumb_path).convert("RGB")
                img.thumbnail((120, 90))
                tkimg = ImageTk.PhotoImage(img)
            except Exception:
                tkimg = ImageTk.PhotoImage(Image.new("RGB", (120, 90), (30, 30, 30)))
        else:
            tkimg = ImageTk.PhotoImage(Image.new("RGB", (120, 90), (30, 30, 30)))
        self.thumb_images.appendleft(tkimg)
        for i, lbl in enumerate(self.thumb_labels):
            img = self.thumb_images[i] if i < len(self.thumb_images) else None
            lbl.configure(image=img)
            lbl.image = img

    def _export_reports(self):
        try:
            from utils.reports import export_reports
        except Exception as e:
            messagebox.showerror("Export", f"Reports module not available: {e}")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", ".xlsx"), ("CSV", ".csv")])
        if not dest:
            return
        try:
            export_reports(dest)
            messagebox.showinfo("Export", f"Exported to {dest}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    def _open_attendance_dir(self):
        try:
            path = os.path.abspath(config.ATTENDANCE_DIR)
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Open Folder", f"Failed to open: {e}")


if __name__ == "__main__":
    App().mainloop()
