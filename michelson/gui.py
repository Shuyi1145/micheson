from __future__ import annotations

import math
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .signal import AnalysisResult, WavelengthResult, analyze_signal, calculate_wavelength
from .storage import save_measurement


class MichelsonApp:
    def __init__(self, root: tk.Tk, camera, config: dict, *, demo_mode: bool = False):
        self.root = root
        self.camera = camera
        self.config = config
        self.demo_mode = demo_mode
        self.measurement_config = config["measurement"]
        self.recording = False
        self.timestamps: list[float] = []
        self.intensities: list[float] = []
        self.preview_values: deque[float] = deque(maxlen=360)
        self.roi_center: tuple[int, int] | None = None
        self.last_frame_shape: tuple[int, int] | None = None
        self.last_analysis: AnalysisResult | None = None
        self.last_wavelength: WavelengthResult | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.video_item: int | None = None

        self.root.title("迈克耳孙干涉波长测量")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._update_frame()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Value.TLabel", font=("Consolas", 12, "bold"))

        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="迈克耳孙干涉波长测量", style="Title.TLabel").pack(side="left")
        self.state_var = tk.StringVar(value="演示模式" if self.demo_mode else "预览中")
        ttk.Label(header, textvariable=self.state_var).pack(side="right")

        content = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        content.pack(fill="both", expand=True)
        self.video_canvas = tk.Canvas(content, width=640, height=480, bg="#111111", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.bind("<Button-1>", self._select_roi)

        right = ttk.Frame(content, padding=(12, 0, 0, 0))
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="中心区域平均灰度（最近 360 帧）").pack(anchor="w")
        self.wave_canvas = tk.Canvas(right, width=520, height=230, bg="#101820", highlightthickness=0)
        self.wave_canvas.pack(fill="x", pady=(4, 12))

        readings = ttk.LabelFrame(right, text="千分尺读数", padding=10)
        readings.pack(fill="x")
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.resolution_var = tk.StringVar(
            value="" if self.measurement_config.get("micrometer_resolution_um") is None
            else str(self.measurement_config["micrometer_resolution_um"])
        )
        self._entry_row(readings, 0, "起始读数 / mm", self.start_var)
        self._entry_row(readings, 1, "终止读数 / mm", self.end_var)
        self._entry_row(readings, 2, "分辨率 / μm", self.resolution_var)

        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=10)
        self.start_button = ttk.Button(buttons, text="开始记录", command=self.start_recording)
        self.stop_button = ttk.Button(buttons, text="停止并分析", command=self.stop_and_analyze, state="disabled")
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.stop_button.pack(side="left", expand=True, fill="x", padx=(4, 0))
        ttk.Button(right, text="按当前读数计算并保存", command=self.calculate_and_save).pack(fill="x")

        results = ttk.LabelFrame(right, text="测量结果", padding=10)
        results.pack(fill="both", expand=True, pady=(10, 0))
        self.result_vars = {
            "frames": tk.StringVar(value="—"),
            "cycles": tk.StringVar(value="—"),
            "rate": tk.StringVar(value="—"),
            "contrast": tk.StringVar(value="—"),
            "wavelength": tk.StringVar(value="—"),
            "deviation": tk.StringVar(value="—"),
            "quality": tk.StringVar(value="—"),
        }
        labels = [
            ("frames", "采样"),
            ("cycles", "周期数（相位 / 过零）"),
            ("rate", "帧率 / 条纹频率"),
            ("contrast", "条纹对比度"),
            ("wavelength", "测得波长"),
            ("deviation", "相对 632.8 nm 偏差"),
            ("quality", "数据质量"),
        ]
        for row, (key, label) in enumerate(labels):
            ttk.Label(results, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            ttk.Label(results, textvariable=self.result_vars[key], style="Value.TLabel").grid(
                row=row, column=1, sticky="w", pady=2
            )
        self.warning_var = tk.StringVar(value="单击画面可移动采样区域；测量时必须单向、缓慢移动 M2。")
        ttk.Label(results, textvariable=self.warning_var, foreground="#b45309", wraplength=480).grid(
            row=len(labels), column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

    @staticmethod
    def _entry_row(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)

    def _select_roi(self, event: tk.Event) -> None:
        if self.last_frame_shape is None:
            return
        frame_height, frame_width = self.last_frame_shape
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)
        self.roi_center = (
            int(np.clip(event.x * frame_width / canvas_width, 0, frame_width - 1)),
            int(np.clip(event.y * frame_height / canvas_height, 0, frame_height - 1)),
        )

    def _roi_mean(self, gray: np.ndarray) -> float:
        height, width = gray.shape
        if self.roi_center is None:
            self.roi_center = (width // 2, height // 2)
        x, y = self.roi_center
        radius = int(self.measurement_config.get("roi_radius_px", 8))
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        local = gray[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        selected = local[mask]
        return float(np.mean(selected)) if selected.size else float(gray[y, x])

    def _update_frame(self) -> None:
        ok, frame, timestamp = self.camera.read()
        if not ok or frame is None:
            self.state_var.set("摄像头读取失败")
            self.root.after(100, self._update_frame)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_frame_shape = gray.shape
        intensity = self._roi_mean(gray)
        self.preview_values.append(intensity)
        if self.recording:
            self.timestamps.append(timestamp)
            self.intensities.append(intensity)
            self.result_vars["frames"].set(f"{len(self.timestamps)} 帧")

        display = frame.copy()
        x, y = self.roi_center or (gray.shape[1] // 2, gray.shape[0] // 2)
        radius = int(self.measurement_config.get("roi_radius_px", 8))
        cv2.circle(display, (x, y), radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            display,
            f"ROI mean: {intensity:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        canvas_width = max(self.video_canvas.winfo_width(), 640)
        canvas_height = max(self.video_canvas.winfo_height(), 480)
        image = Image.fromarray(rgb).resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image=image)
        if self.video_item is None:
            self.video_item = self.video_canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.video_canvas.itemconfigure(self.video_item, image=self.photo)
        self._draw_waveform()
        self.root.after(1, self._update_frame)

    def _draw_waveform(self) -> None:
        canvas = self.wave_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)
        values = np.asarray(self.preview_values, dtype=float)
        for fraction in (0.25, 0.5, 0.75):
            y = int(height * fraction)
            canvas.create_line(0, y, width, y, fill="#263746")
        if len(values) < 2:
            return
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1
        x_coordinates = np.linspace(0, width - 1, len(values))
        y_coordinates = height - 8 - (values - low) / (high - low) * (height - 16)
        points = [coordinate for pair in zip(x_coordinates, y_coordinates) for coordinate in pair]
        canvas.create_line(*points, fill="#38bdf8", width=2, smooth=False)
        canvas.create_text(8, 8, text=f"{low:.1f}–{high:.1f}", fill="#d7e5ee", anchor="nw")

    def start_recording(self) -> None:
        if not self.start_var.get().strip():
            messagebox.showinfo("需要起始读数", "请先填写千分尺起始读数，再开始记录。")
            return
        try:
            float(self.start_var.get())
        except ValueError:
            messagebox.showerror("读数格式错误", "起始读数必须是数字，单位为 mm。")
            return
        self.timestamps.clear()
        self.intensities.clear()
        self.last_analysis = None
        self.last_wavelength = None
        self.recording = True
        self.state_var.set("正在记录：请单向缓慢移动 M2")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.warning_var.set("保持 M2 单向移动；若波形变化太密，请立即降低速度。")

    def stop_and_analyze(self) -> None:
        self.recording = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.state_var.set("正在分析")
        try:
            self.last_analysis = analyze_signal(
                self.timestamps,
                self.intensities,
                safe_fringe_fraction_of_fps=float(
                    self.measurement_config.get("safe_fringe_fraction_of_fps", 0.25)
                ),
                minimum_samples=int(self.measurement_config.get("minimum_samples", 60)),
            )
        except ValueError as error:
            self.state_var.set("分析失败")
            self.warning_var.set(str(error))
            messagebox.showerror("无法分析", str(error))
            return

        result = self.last_analysis
        self.result_vars["frames"].set(f"{len(result.timestamps)} 帧 / {result.duration_s:.2f} s")
        self.result_vars["cycles"].set(f"{result.cycles_phase:.3f} / {result.cycles_crossing}")
        self.result_vars["rate"].set(
            f"{result.sample_rate_hz:.1f} fps / {result.median_fringe_hz:.2f} Hz"
        )
        self.result_vars["contrast"].set(f"{result.contrast * 100:.1f}%")
        self.result_vars["quality"].set(result.quality)
        self.warning_var.set("；".join(result.warnings) if result.warnings else "数据检查通过，可以输入终止读数计算。")
        self.state_var.set("分析完成")

    def calculate_and_save(self) -> None:
        if self.last_analysis is None:
            messagebox.showinfo("尚无分析结果", "请先完成一次记录并点击“停止并分析”。")
            return
        try:
            start = float(self.start_var.get())
            end = float(self.end_var.get())
            resolution_text = self.resolution_var.get().strip()
            resolution = float(resolution_text) if resolution_text else None
            cycle_uncertainty = abs(
                self.last_analysis.cycles_phase - self.last_analysis.cycles_crossing
            )
            self.last_wavelength = calculate_wavelength(
                start,
                end,
                self.last_analysis.cycles_phase,
                micrometer_resolution_um=resolution,
                cycle_uncertainty=cycle_uncertainty,
            )
        except ValueError as error:
            messagebox.showerror("无法计算", str(error))
            return

        result = self.last_wavelength
        uncertainty = "" if result.uncertainty_nm is None else f" ± {result.uncertainty_nm:.3f}"
        self.result_vars["wavelength"].set(f"{result.wavelength_nm:.3f}{uncertainty} nm")
        reference = float(self.measurement_config.get("reference_wavelength_nm", 632.8))
        deviation = (result.wavelength_nm - reference) / reference * 100
        self.result_vars["deviation"].set(f"{deviation:+.3f}%")

        paths = save_measurement(
            self.config.get("output_directory", "data"),
            self.last_analysis,
            self.last_wavelength,
            {
                "start_reading_mm": start,
                "end_reading_mm": end,
                "roi_center_px": self.roi_center,
                "roi_radius_px": int(self.measurement_config.get("roi_radius_px", 8)),
                "demo_mode": self.demo_mode,
            },
        )
        self.state_var.set("结果已保存")
        messagebox.showinfo("保存完成", f"数据已保存：\n{paths[0]}\n{paths[1]}")

    def close(self) -> None:
        self.camera.close()
        self.root.destroy()


def run_gui(camera, config: dict, *, demo_mode: bool = False) -> None:
    root = tk.Tk()
    root.minsize(1120, 620)
    MichelsonApp(root, camera, config, demo_mode=demo_mode)
    root.mainloop()
