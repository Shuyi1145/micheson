from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraStatus:
    width: int
    height: int
    requested_fps: float
    reported_fps: float


class UsbCamera:
    """OpenCV USB 摄像头封装。

    不同 UVC 驱动对曝光参数的取值定义不同，因此只有配置中明确给出
    exposure/gain 时才尝试写入，不能把 set() 成功当成硬件一定已接受。
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._capture = cv2.VideoCapture(int(settings.get("index", 0)))
        if not self._capture.isOpened():
            raise RuntimeError("无法打开 USB 摄像头，请检查设备编号和访问权限。")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings.get("width", 640)))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings.get("height", 480)))
        self._capture.set(cv2.CAP_PROP_FPS, float(settings.get("fps", 30)))

        # 自动曝光/自动白平衡会抹平真实的明暗变化。部分 Linux UVC 驱动
        # 不接受 OpenCV 的统一值，所以界面仍会提醒用户用设备工具核验。
        self._capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        exposure = settings.get("exposure")
        gain = settings.get("gain")
        if exposure is not None:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        if gain is not None:
            self._capture.set(cv2.CAP_PROP_GAIN, float(gain))

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self._capture.read()
        return ok, frame if ok else None, time.monotonic()

    def status(self) -> CameraStatus:
        return CameraStatus(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            requested_fps=float(self._settings.get("fps", 30)),
            reported_fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
        )

    def close(self) -> None:
        self._capture.release()


class DemoCamera:
    """无相机时用于界面演示和算法联调的合成干涉图样。"""

    def __init__(self, settings: dict, fringe_hz: float = 3.0):
        self.width = int(settings.get("width", 640))
        self.height = int(settings.get("height", 480))
        self.fps = float(settings.get("fps", 30))
        self.fringe_hz = fringe_hz
        self._start = time.monotonic()
        self._last_frame = 0.0
        y, x = np.mgrid[: self.height, : self.width]
        self._radius_squared = (x - self.width / 2) ** 2 + (y - self.height / 2) ** 2
        self._rng = np.random.default_rng(2026)

    def read(self) -> tuple[bool, np.ndarray, float]:
        target_interval = 1.0 / max(self.fps, 1.0)
        now = time.monotonic()
        if self._last_frame:
            remaining = target_interval - (now - self._last_frame)
            if remaining > 0:
                time.sleep(remaining)
        now = time.monotonic()
        self._last_frame = now
        elapsed = now - self._start
        phase = 2 * math.pi * self.fringe_hz * elapsed
        rings = np.cos(0.00105 * self._radius_squared + phase)
        vignette = np.clip(1.0 - self._radius_squared / (self.width * self.height), 0.45, 1.0)
        gray = 126 + 92 * rings * vignette
        gray += self._rng.normal(0, 2.5, gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return True, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), now

    def status(self) -> CameraStatus:
        return CameraStatus(self.width, self.height, self.fps, self.fps)

    def close(self) -> None:
        return None

