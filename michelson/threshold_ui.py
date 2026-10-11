# -*- coding: gb2312 -*-
"""Visual threshold selection; all images remain display-only."""
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .preview import morphology_preview


class ThresholdWindow:
    def __init__(self, parent, get_frame, apply_threshold, display_settings):
        self.get_frame = get_frame
        self.apply_threshold = apply_threshold
        self.settings = display_settings
        self.window = tk.Toplevel(parent)
        self.window.title("二值化阈值可视化调节")
        self.window.geometry("980x650")
        self.window.minsize(640, 420)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        self.after_id = None
        self.frame = None
        self.photos = [None] * 4
        self.items = [None] * 4
        options, _ = display_settings.snapshot()
        self.auto = tk.BooleanVar(value=options["threshold"] is None)
        self.freeze = tk.BooleanVar(value=False)
        self.slider = tk.DoubleVar(value=128 if options["threshold"] is None else options["threshold"])
        self.status = tk.StringVar(value="等待视频画面")
        controls = ttk.Frame(self.window, padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="阈值 0-255：").pack(side="left")
        self.scale = ttk.Scale(controls, from_=0, to=255, variable=self.slider,
                               command=self.on_slide)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Checkbutton(controls, text="自动阈值", variable=self.auto,
                        command=self.draw).pack(side="left")
        ttk.Checkbutton(controls, text="冻结对比帧", variable=self.freeze,
                        command=self.draw).pack(side="left", padx=8)
        ttk.Button(controls, text="应用到主界面/网页", command=self.apply).pack(side="left")
        ttk.Label(self.window, textvariable=self.status, padding=(8, 2)).pack(fill="x")
        grid = ttk.Frame(self.window, padding=8)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure((0, 1), weight=1, uniform="threshold")
        grid.rowconfigure((0, 1), weight=1, uniform="threshold")
        self.canvases = []
        for index, label in enumerate(("原始灰度", "二值化（未做形态学）", "开运算", "闭运算")):
            cell = ttk.Frame(grid)
            cell.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
            ttk.Label(cell, text=label).pack(anchor="w")
            canvas = tk.Canvas(cell, bg="#111111", width=1, height=1, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            self.canvases.append(canvas)
        ttk.Label(self.window, text="降低阈值：更多白色；提高阈值：更少白色。以条纹可分辨、不粘连为准；不参与测量计数。",
                  padding=8, wraplength=900).pack(fill="x")
        self.tick()

    def on_slide(self, value):
        self.auto.set(False)
        self.draw()

    def draw(self):
        if self.frame is None or self.closed:
            return
        options, _ = self.settings.snapshot()
        threshold = None if self.auto.get() else int(round(self.slider.get()))
        result = morphology_preview(self.frame, **dict(options, threshold=threshold))
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        pixels = (gray, result.binary, result.opening, result.closing)
        white = np.count_nonzero(result.binary) / result.binary.size
        self.status.set(f"实际阈值：{result.threshold:.0f} | 白色占比：{white:.1%} | 核：{options['kernel_size']} | 仅显示")
        resampling = getattr(Image, "Resampling", Image)
        for index, (canvas, array) in enumerate(zip(self.canvases, pixels)):
            height, width = array.shape
            cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            ratio = min(cw / width, ch / height)
            size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            photo = ImageTk.PhotoImage(Image.fromarray(array).resize(
                size, resampling.BILINEAR if index == 0 else resampling.NEAREST))
            self.photos[index] = photo
            x, y = (cw - size[0]) // 2, (ch - size[1]) // 2
            if self.items[index] is None:
                self.items[index] = canvas.create_image(x, y, image=photo, anchor="nw")
            else:
                canvas.coords(self.items[index], x, y)
                canvas.itemconfigure(self.items[index], image=photo)

    def tick(self):
        if self.closed:
            return
        if not self.freeze.get() or self.frame is None:
            self.frame = self.get_frame()
        self.draw()
        self.after_id = self.window.after(125, self.tick)

    def apply(self):
        self.apply_threshold(None if self.auto.get() else int(round(self.slider.get())))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.after_id is not None:
            self.window.after_cancel(self.after_id)
        self.window.destroy()
