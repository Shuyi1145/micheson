# -*- coding: gb2312 -*-
from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .roi import ROI_NAMES, analyze_rois, roi_positions, sample_rois
from .preview import morphology_preview
from .signal import calculate_wavelength
from .storage import save_measurement

_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_LABELS = ("上", "右", "下", "左")
_COLORS = ("#4ade80", "#60a5fa", "#f87171", "#22d3ee")
_BGR = ((80, 220, 80), (240, 160, 80), (80, 80, 240), (220, 220, 60))


class MichelsonApp:
    def __init__(self, root, camera, config, *, demo_mode=False):
        self.root, self.camera, self.config = root, camera, config
        self.demo_mode = demo_mode
        self.settings = config["measurement"]
        self.radius = max(1, int(self.settings.get("roi_radius_px", 8)))
        self.offset = max(2 * self.radius + 1, int(self.settings.get("roi_offset_px", 40)))
        self.center = None
        self.shape = None
        self.points = None
        self.recording = False
        self.closed = False
        self.fullscreen = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.timestamps, self.samples = [], []
        self.preview = deque(maxlen=360)
        self.latest = None
        self.camera_error = ""
        self.recording_error = ""
        self.sequence = 0
        self.drawn_sequence = -1
        self.last_wave_time = 0.0
        self.last_multi = None
        self.last_wavelength = None
        self.measurement_start = None
        self.geometry_snapshot = None
        self.display_mapping = None
        self.photo = None
        self.video_item = None
        self.comparison_photos = [None, None]
        self.comparison_items = [None, None]
        self.last_comparison_time = 0.0
        self.after_id = None
        root.title("迈克耳孙干涉测量 - 四点 ROI")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<F11>", self.toggle_fullscreen)
        root.bind("<Escape>", self.exit_fullscreen)
        self._build_ui()
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker.start()
        self._update_frame()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=6)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="四点 ROI 干涉测量", font=("sans-serif", 12, "bold")).pack(side="left")
        ttk.Button(header, text="全屏 F11", command=self.toggle_fullscreen).pack(side="right")
        self.state_var = tk.StringVar(value="演示模式" if self.demo_mode else "预览中")
        ttk.Label(header, textvariable=self.state_var).pack(side="right", padx=8)

        content = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2, minsize=260)
        content.rowconfigure(0, weight=1)
        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=3)
        left.rowconfigure(2, weight=2)
        left.rowconfigure(4, weight=1)
        ttk.Label(left, text="测量路：原始画面（单击选择中心）").grid(row=0, column=0, sticky="w")
        self.video_canvas = tk.Canvas(left, width=1, height=1, bg="#111111", highlightthickness=0)
        self.video_canvas.grid(row=1, column=0, sticky="nsew")
        self.video_canvas.bind("<Button-1>", self._select_roi)
        comparison = ttk.Frame(left)
        comparison.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        comparison.columnconfigure((0, 1), weight=1, uniform="comparison")
        comparison.rowconfigure(1, weight=1)
        self.comparison_canvases = []
        for index, label in enumerate(("显示路：开运算（去小亮点）", "显示路：闭运算（补小断裂）")):
            ttk.Label(comparison, text=label, font=("sans-serif", 9)).grid(row=0, column=index, sticky="w")
            canvas = tk.Canvas(comparison, width=1, height=1, bg="#111111", highlightthickness=0)
            canvas.grid(row=1, column=index, sticky="nsew", padx=(0, 2) if index == 0 else (2, 0))
            self.comparison_canvases.append(canvas)
        ttk.Label(left, text="单击图像设置中心 | 四点独立波形", font=("sans-serif", 9)).grid(
            row=3, column=0, sticky="w")
        self.wave_canvas = tk.Canvas(left, width=1, height=1, bg="#101820", highlightthickness=0)
        self.wave_canvas.grid(row=4, column=0, sticky="nsew")

        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.panel_canvas = tk.Canvas(right, width=280, height=1, highlightthickness=0)
        self.panel_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.panel_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.panel_canvas.configure(yscrollcommand=scrollbar.set)
        self.panel = ttk.Frame(self.panel_canvas, padding=6)
        self.panel_item = self.panel_canvas.create_window(0, 0, window=self.panel, anchor="nw")
        self.panel.bind("<Configure>", lambda event: self.panel_canvas.configure(
            scrollregion=self.panel_canvas.bbox("all")))
        self.panel_canvas.bind("<Configure>", self._resize_panel)
        self.root.bind("<MouseWheel>", self._scroll_panel, add="+")
        self.root.bind("<Button-4>", self._scroll_panel, add="+")
        self.root.bind("<Button-5>", self._scroll_panel, add="+")

        results = ttk.LabelFrame(self.panel, text="结果（四点周期数中位数）", padding=6)
        results.pack(fill="x")
        self.result_vars = {name: tk.StringVar(value="-") for name in
                            ("wavelength", "cycles", "frames", "rate", "agreement", "quality")}
        fields = (("wavelength", "波长"), ("cycles", "合并周期数"), ("frames", "采样"),
                  ("rate", "帧率 / 条纹频率"), ("agreement", "有效点 / 极差"), ("quality", "状态"))
        for key, label in fields:
            row = ttk.Frame(results)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Label(row, textvariable=self.result_vars[key]).pack(side="right")
        self.channel_vars = [tk.StringVar(value=f"{label}: 等待采集") for label in _LABELS]
        for variable, color in zip(self.channel_vars, _COLORS):
            ttk.Label(results, textvariable=variable, foreground=color).pack(anchor="w")
        self.warning_var = tk.StringVar(value="先定位中心；测量时单向缓慢移动 M2。")
        self.warning_label = ttk.Label(results, textvariable=self.warning_var,
                                       foreground="#b45309", wraplength=260)
        self.warning_label.pack(fill="x", pady=(5, 0))

        geometry = ttk.LabelFrame(self.panel, text="四点位置", padding=6)
        geometry.pack(fill="x", pady=6)
        self.offset_var = tk.IntVar(value=self.offset)
        ttk.Label(geometry, text="距中心 / px").pack(side="left")
        self.offset_spin = ttk.Spinbox(geometry, from_=2 * self.radius + 1, to=500,
                                      width=6, textvariable=self.offset_var)
        self.offset_spin.pack(side="left", padx=4)
        self.apply_button = ttk.Button(geometry, text="应用", command=self.apply_geometry)
        self.apply_button.pack(side="right")

        readings = ttk.LabelFrame(self.panel, text="千分尺读数", padding=6)
        readings.pack(fill="x")
        self.start_var, self.end_var = tk.StringVar(), tk.StringVar()
        resolution = self.settings.get("micrometer_resolution_um")
        self.resolution_var = tk.StringVar(value="" if resolution is None else str(resolution))
        self.start_entry = self._entry_row(readings, 0, "起始 / mm", self.start_var)
        self._entry_row(readings, 1, "终止 / mm", self.end_var)
        self._entry_row(readings, 2, "分辨率 / um", self.resolution_var)

        processing = ttk.LabelFrame(self.panel, text="显示处理（不参与计数）", padding=6)
        processing.pack(fill="x", pady=6)
        self.kernel_var = tk.StringVar(value="3")
        self.denoise_var = tk.BooleanVar(value=True)
        self.auto_threshold_var = tk.BooleanVar(value=True)
        self.threshold_var = tk.StringVar(value="128")
        ttk.Label(processing, text="核大小 / px").grid(row=0, column=0, sticky="w")
        ttk.Combobox(processing, textvariable=self.kernel_var, values=(1, 3, 5, 7),
                     state="readonly", width=6).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(processing, text="3x3 中值降噪", variable=self.denoise_var).grid(
            row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(processing, text="自动二值化阈值", variable=self.auto_threshold_var).grid(
            row=2, column=0, columnspan=2, sticky="w")
        self._entry_row(processing, 3, "手动阈值 / 0-255", self.threshold_var)
        self.processing_var = tk.StringVar(value="两图使用同一二值图；仅作显示")
        ttk.Label(processing, textvariable=self.processing_var, wraplength=240).grid(
            row=4, column=0, columnspan=2, sticky="w")

        actions = ttk.Frame(right, padding=(0, 5, 0, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        actions.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(actions, text="开始记录", command=self.start_recording)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.stop_button = ttk.Button(actions, text="停止并分析", command=self.stop_and_analyze, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.save_button = ttk.Button(actions, text="计算并保存", command=self.calculate_and_save, state="disabled")
        self.save_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    @staticmethod
    def _entry_row(parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)
        return entry

    def _resize_panel(self, event):
        self.panel_canvas.itemconfigure(self.panel_item, width=event.width)
        self.warning_label.configure(wraplength=max(160, event.width - 30))

    def _scroll_panel(self, event):
        # Only scroll when the pointer is inside the right panel.
        widget = event.widget
        while widget is not None:
            if widget is self.panel_canvas:
                direction = -1 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1
                self.panel_canvas.yview_scroll(direction * 3, "units")
                return "break"
            widget = getattr(widget, "master", None)

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        return "break"

    def exit_fullscreen(self, event=None):
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)
        return "break"

    def apply_geometry(self):
        try:
            offset = int(self.offset_var.get())
            if offset < 2 * self.radius + 1:
                raise ValueError("间距太小，四点采样区域会重叠。")
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("ROI 参数", str(error))
            return
        with self.lock:
            if self.recording:
                return
            if self.shape is not None:
                center, points, offset = roi_positions(self.center, self.shape, offset, self.radius)
                self.center, self.points = center, points
            self.offset = offset
            self.preview.clear()
        self.offset_var.set(offset)
        self.warning_var.set("间距已设置；靠近边界时位置会自动限制在图像内。")

    def _select_roi(self, event):
        if self.display_mapping is None:
            return
        scale, x0, y0, width, height = self.display_mapping
        x, y = (event.x - x0) / scale, (event.y - y0) / scale
        if not (0 <= x < width and 0 <= y < height):
            return
        with self.lock:
            if self.recording:
                self.warning_var.set("记录中已锁定 ROI；停止后才能移动。")
                return
            self.center = (int(x), int(y))
            self.preview.clear()

    def _capture_loop(self):
        try:
            while not self.stop_event.is_set():
                ok, frame, timestamp = self.camera.read()
                if not ok or frame is None:
                    with self.lock:
                        self.camera_error = "摄像头读取失败"
                        if self.recording:
                            self.recording_error = "记录期间读取图像失败，存在遗漏；请重新采集。"
                    self.stop_event.wait(0.1)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self.lock:
                    if self.recording and self.shape != gray.shape:
                        raise ValueError("记录中图像尺寸发生变化，请重新采集。")
                    center = self.center or (gray.shape[1] // 2, gray.shape[0] // 2)
                    center, points, actual_offset = roi_positions(center, gray.shape, self.offset, self.radius)
                    values = sample_rois(gray, points, self.radius)
                    self.center, self.points, self.shape = center, points, gray.shape
                    self.camera_error = ""
                    self.sequence += 1
                    self.latest = (frame, points, values, self.sequence, actual_offset)
                    self.preview.append(values)
                    if self.recording:
                        self.timestamps.append(timestamp)
                        self.samples.append(values)
        except Exception as error:
            with self.lock:
                self.camera_error = str(error)
        finally:
            self.camera.close()

    def _update_frame(self):
        if self.closed:
            return
        with self.lock:
            latest = self.latest
            preview = list(self.preview)
            error = self.camera_error
            count = len(self.timestamps)
        if error:
            self.state_var.set("采集异常")
            self.warning_var.set(error)
        if latest is not None and latest[3] != self.drawn_sequence:
            frame, points, values, sequence, offset = latest
            self._display_video(frame, points)
            self.drawn_sequence = sequence
            if self.last_multi is None:
                for variable, label, value in zip(self.channel_vars, _LABELS, values):
                    variable.set(f"{label}: 灰度 {value:.1f}")
            if self.recording:
                self.result_vars["frames"].set(f"{count} 帧")
        now = time.monotonic()
        if now - self.last_wave_time >= 0.125:
            self._draw_waveform(preview)
            self.last_wave_time = now
        self.after_id = self.root.after(66, self._update_frame)

    def _display_video(self, frame, points):
        display = frame.copy()
        for index, ((x, y), color) in enumerate(zip(points, _BGR)):
            cv2.circle(display, (x, y), self.radius, color, 2, cv2.LINE_AA)
            cv2.putText(display, str(index + 1), (x + self.radius + 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        height, width = frame.shape[:2]
        cw, ch = max(1, self.video_canvas.winfo_width()), max(1, self.video_canvas.winfo_height())
        target_width = max(1, min(cw, int(ch * width / height)))
        target_height = max(1, round(target_width * height / width))
        x0, y0 = (cw - target_width) // 2, (ch - target_height) // 2
        self.display_mapping = (target_width / width, x0, y0, width, height)
        image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        image = image.resize((target_width, target_height), _BILINEAR)
        self.photo = ImageTk.PhotoImage(image=image)
        if self.video_item is None:
            self.video_item = self.video_canvas.create_image(x0, y0, image=self.photo, anchor="nw")
        else:
            self.video_canvas.coords(self.video_item, x0, y0)
            self.video_canvas.itemconfigure(self.video_item, image=self.photo)
        now = time.monotonic()
        # Keep expensive display processing out of the acquisition thread.
        if now - self.last_comparison_time >= 0.125:
            self._display_comparison(frame)
            self.last_comparison_time = now

    def _display_comparison(self, frame):
        try:
            threshold = None if self.auto_threshold_var.get() else float(self.threshold_var.get())
            result = morphology_preview(frame, kernel_size=int(self.kernel_var.get()),
                                        denoise=self.denoise_var.get(), threshold=threshold)
        except (ValueError, tk.TclError, cv2.error) as error:
            self.processing_var.set(f"显示参数无效：{error}")
            for canvas in self.comparison_canvases:
                canvas.delete("all")
            self.comparison_items = [None, None]
            self.comparison_photos = [None, None]
            return
        self.processing_var.set(f"二值化阈值：{result.threshold:.1f}；仅显示，不参与计数")
        for index, pixels in enumerate((result.opening, result.closing)):
            canvas = self.comparison_canvases[index]
            cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            height, width = pixels.shape
            scale = min(cw / width, ch / height)
            tw, th = max(1, int(width * scale)), max(1, int(height * scale))
            # Nearest-neighbor preserves binary black/white instead of inventing gray edges.
            nearest = getattr(Image, "Resampling", Image).NEAREST
            photo = ImageTk.PhotoImage(Image.fromarray(pixels).resize((tw, th), nearest))
            self.comparison_photos[index] = photo
            x0, y0 = (cw - tw) // 2, (ch - th) // 2
            if self.comparison_items[index] is None:
                self.comparison_items[index] = canvas.create_image(x0, y0, image=photo, anchor="nw")
            else:
                canvas.coords(self.comparison_items[index], x0, y0)
                canvas.itemconfigure(self.comparison_items[index], image=photo)

    def _draw_waveform(self, samples):
        canvas = self.wave_canvas
        canvas.delete("all")
        width, height = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
        if len(samples) < 2:
            return
        values = np.asarray(samples)
        xx = np.linspace(0, width - 1, len(values))
        # Separate lanes preserve each channel's visibility despite DC differences.
        lane = height / 4
        for index, color in enumerate(_COLORS):
            signal = values[:, index]
            low, high = float(signal.min()), float(signal.max())
            yy = (index + 1) * lane - 3 - (signal - low) / max(high - low, 1.0) * max(1, lane - 14)
            coordinates = [float(v) for pair in zip(xx, yy) for v in pair]
            canvas.create_line(*coordinates, fill=color, width=1)
            canvas.create_text(3, index * lane + 2, text=f"{index + 1}: {signal[-1]:.1f}",
                               fill=color, anchor="nw")

    def start_recording(self):
        try:
            start = float(self.start_var.get())
            if not np.isfinite(start):
                raise ValueError("起始读数必须是有限数字。")
        except ValueError:
            messagebox.showerror("起始读数", "请填写有效起始读数，单位 mm。")
            return
        with self.lock:
            if self.points is None or self.camera_error or not self.worker.is_alive():
                messagebox.showerror("采集未就绪", "等待摄像头图像正常后再开始记录。")
                return
            self.timestamps.clear()
            self.samples.clear()
            center, points, offset = roi_positions(self.center, self.shape, self.offset, self.radius)
            self.center, self.points, self.offset = center, points, offset
            self.recording_error = ""
            self.geometry_snapshot = {"center_px": center, "points_px": points,
                                      "offset_px": offset, "radius_px": self.radius}
            self.recording = True
        self.measurement_start = start
        self.last_multi, self.last_wavelength = None, None
        for variable in self.result_vars.values():
            variable.set("-")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.save_button.configure(state="disabled")
        self.start_entry.configure(state="disabled")
        self.offset_spin.configure(state="disabled")
        self.apply_button.configure(state="disabled")
        self.state_var.set("记录中")
        self.warning_var.set("ROI 已锁定；保持单向缓慢移动，四点不能修复混叠漏计。")

    def stop_and_analyze(self):
        with self.lock:
            self.recording = False
            timestamps, samples = list(self.timestamps), list(self.samples)
            error = self.recording_error or self.camera_error
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.start_entry.configure(state="normal")
        self.offset_spin.configure(state="normal")
        self.apply_button.configure(state="normal")
        try:
            if error:
                raise ValueError(error)
            result = analyze_rois(timestamps, samples,
                safe_fringe_fraction_of_fps=float(self.settings.get("safe_fringe_fraction_of_fps", 0.25)),
                minimum_samples=int(self.settings.get("minimum_samples", 60)))
        except ValueError as error:
            self.state_var.set("分析失败")
            self.warning_var.set(str(error))
            messagebox.showerror("无法合并四点", str(error))
            return
        self.last_multi = result
        analysis = result.analysis
        self.result_vars["cycles"].set(f"{result.cycles:.3f}")
        self.result_vars["frames"].set(f"{len(timestamps)} / {analysis.duration_s:.1f} s")
        self.result_vars["rate"].set(f"{analysis.sample_rate_hz:.1f} / {analysis.median_fringe_hz:.2f} Hz")
        self.result_vars["agreement"].set(f"{len(result.accepted)}/4 / {result.spread_cycles:.3f} 周期")
        self.result_vars["quality"].set(analysis.quality)
        for name, label, variable in zip(ROI_NAMES, _LABELS, self.channel_vars):
            channel = result.channels.get(name)
            if name in result.rejected:
                variable.set(f"{label}: 排除 ({result.rejected[name]})")
            elif channel is not None:
                variable.set(f"{label}: {channel.cycles_phase:.3f} 周期 / 对比度 {channel.contrast:.0%}")
        self.warning_var.set("；".join(analysis.warnings) or "四点一致性通过；输入终止读数计算。")
        self.state_var.set("分析完成")
        self.save_button.configure(state="normal")
        self.panel_canvas.yview_moveto(0)

    def calculate_and_save(self):
        if self.last_multi is None:
            return
        try:
            end = float(self.end_var.get())
            if not np.isfinite(end):
                raise ValueError("终止读数必须是有限数字。")
            resolution_text = self.resolution_var.get().strip()
            resolution = float(resolution_text) if resolution_text else None
            if resolution is not None and (not np.isfinite(resolution) or resolution <= 0):
                raise ValueError("分辨率必须为正数或留空。")
            result = calculate_wavelength(self.measurement_start, end, self.last_multi.cycles)
            self.last_wavelength = result
            # No calibrated cycle uncertainty exists yet; do not label agreement as it.
            self.result_vars["wavelength"].set(f"{result.wavelength_nm:.3f} nm")
            metadata = {"start_reading_mm": self.measurement_start, "end_reading_mm": end,
                        "micrometer_resolution_um": resolution, "demo_mode": self.demo_mode,
                        "roi_geometry": self.geometry_snapshot,
                        "merged_cycles": self.last_multi.cycles,
                        "spread_cycles": self.last_multi.spread_cycles,
                        "accepted_rois": self.last_multi.accepted,
                        "rejected_rois": self.last_multi.rejected,
                        "representative_roi": self.last_multi.representative,
                        "uncertainty_note": "尚无完整标定，不报告合成不确定度"}
            paths = save_measurement(self.config.get("output_directory", "data"),
                self.last_multi.analysis, result, metadata,
                channels=self.last_multi.channels,
                raw_timestamps=self.timestamps, raw_samples=self.samples)
        except (ValueError, OSError) as error:
            messagebox.showerror("计算或保存失败", str(error))
            return
        self.state_var.set("结果已保存")
        self.panel_canvas.yview_moveto(0)
        messagebox.showinfo("保存完成", f"数据已保存：\n{paths[0]}\n{paths[1]}")

    def close(self):
        self.closed = True
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.stop_event.set()
        self.worker.join(timeout=0.5)
        self.root.destroy()


def run_gui(camera, config, *, demo_mode=False):
    root = tk.Tk()
    root.minsize(640, 360)
    width = max(640, min(1200, root.winfo_screenwidth() - 40))
    height = max(360, min(760, root.winfo_screenheight() - 80))
    root.geometry(f"{width}x{height}+10+10")
    MichelsonApp(root, camera, config, demo_mode=demo_mode)
    root.mainloop()
