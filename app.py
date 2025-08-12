import os
import sys
import json
import time
import threading
import subprocess
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
        self.geometry("1100x760")

        self._settings = load_settings()

        # Top bar
        top = tb.Frame(self)
        top.pack(fill=X, padx=12, pady=8)

        tb.Label(top, text="Face detector:", bootstyle=INFO).pack(side=LEFT)
        self.detector_var = tb.StringVar(value=self._settings.get("detector", os.environ.get("FACE_DETECTOR", "haar")))
        self.detector_combo = tb.Combobox(top, textvariable=self.detector_var, values=["haar", "dnn"], width=10, state="readonly")
        self.detector_combo.pack(side=LEFT, padx=8)

        tb.Button(top, text="Light", bootstyle=SECONDARY, command=lambda: self._set_theme("flatly")).pack(side=RIGHT, padx=4)
        tb.Button(top, text="Dark", bootstyle=SECONDARY, command=lambda: self._set_theme("darkly")).pack(side=RIGHT)

        # Notebook
        self.nb = tb.Notebook(self, bootstyle=PRIMARY)
        self.nb.pack(fill=BOTH, expand=True, padx=12, pady=(0,8))

        self.enroll_tab = tb.Frame(self.nb)
        self.run_tab = tb.Frame(self.nb)
        self.nb.add(self.enroll_tab, text="Enroll")
        self.nb.add(self.run_tab, text="Run Attendance")

        self._build_enroll_tab()
        self._build_run_tab()

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

        # Bottom bar: attendance export
        bot = tb.Frame(self)
        bot.pack(fill=X, padx=12, pady=(0,10))
        tb.Button(bot, text="Export CSV/XLSX", bootstyle=PRIMARY, command=self._export_reports).pack(side=RIGHT)
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

    def _append_log(self, line: str):
        self.log.insert(END, line + "\n")
        self.log.see(END)
        self.status.configure(text=line[:120])
        # Parse recognition events for recent panel and thumbnails
        # Expected log pattern we print in run_attendance: "Marked present: NAME"
        if "Marked present:" in line:
            name = line.split(":", 1)[1].strip()
            self._update_recent(name, confidence=None)
        # If we extend run_attendance to print "Recognized NAME CONF=<value>", parse here and update thumbnail.

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

        row3 = tb.Frame(f); row3.pack(fill=X, **pad)
        tb.Button(row3, text="Start", bootstyle=SUCCESS, command=self._start_run).pack(side=LEFT)
        tb.Button(row3, text="Stop", bootstyle=DANGER, command=lambda: self.run_runner.stop()).pack(side=LEFT, padx=8)

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
        env = os.environ.copy()
        env["FACE_DETECTOR"] = self.detector_var.get()
        self._append_log("$ " + " ".join(args))
        self.run_runner.start(args, env, self._append_log)

    # Update recent recognitions table and thumbnails
    def _update_recent(self, name: str, confidence: float | None):
        now = datetime.now().strftime("%H:%M:%S")
        self.recent_tree.insert("", 0, values=(name, now, f"{confidence:.1f}" if confidence is not None else "-"))
        # Optionally, update thumbnails with a placeholder until we wire frame taps from run_attendance
        placeholder = Image.new("RGB", (120, 90), (30, 30, 30))
        tkimg = ImageTk.PhotoImage(placeholder)
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


if __name__ == "__main__":
    App().mainloop()
