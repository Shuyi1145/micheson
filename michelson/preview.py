# -*- coding: gb2312 -*-
"""Display-only processing. Never feed these images into measurement."""
from dataclasses import dataclass
import threading

import cv2
import numpy as np


@dataclass(frozen=True)
class MorphologyPreview:
    opening: np.ndarray
    closing: np.ndarray
    threshold: float
    binary: np.ndarray


class DisplaySettings:
    """Thread-safe display options shared by Tk and web encoding."""

    def __init__(self, options=None):
        self.lock = threading.Lock()
        self.options = {"kernel_size": 3, "denoise": True, "threshold": None}
        self.revision = 0
        self.update(**(options or {}))

    def update(self, **options):
        with self.lock:
            candidate = dict(self.options, **options)
            morphology_preview(np.zeros((1, 1, 3), np.uint8), **candidate)
            if candidate != self.options:
                self.options = candidate
                self.revision += 1

    def snapshot(self):
        with self.lock:
            return dict(self.options), self.revision


def morphology_preview(frame, *, kernel_size=3, denoise=True, threshold=None):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Expected an unsigned 8-bit image.")
    if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a nonempty BGR image.")
    if kernel_size not in (1, 3, 5, 7):
        raise ValueError("Kernel size must be 1, 3, 5 or 7.")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 255):
        raise ValueError("Threshold must be between 0 and 255.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    mode = cv2.THRESH_BINARY
    if threshold is None:
        mode |= cv2.THRESH_OTSU
    used_threshold, binary = cv2.threshold(gray, threshold or 0, 255, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Independent branches from the SAME binary image, not a chained open-close.
    return MorphologyPreview(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
        cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
        float(used_threshold),
        binary,
    )
