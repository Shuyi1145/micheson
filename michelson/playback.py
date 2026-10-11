# -*- coding: gb2312 -*-
"""Paced preview plus one-pass recording using source video timestamps."""
import math
import threading
import time
from pathlib import Path

import cv2

from .camera import CameraStatus


class VideoPlayback:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not self.capture.isOpened() or not math.isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise ValueError("Cannot decode video or invalid frame rate")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.closed = threading.Event()
        self.restart_requested = threading.Event()
        self.control = threading.Condition()
        self.decoder_lock = threading.Lock()
        self.paused = False
        self.position_s = 0.0
        self.pass_number = 1
        self.next_frame = 0.0
        self.frame_index = 0
        self.measurement_token = 0
        self.requested_measurement = 0
        self.active_measurement = 0
        self.completed_measurement = 0
        self.stop_at_end = False
        self.previous_media_s = -1.0

    def toggle_pause(self):
        with self.control:
            self.paused = not self.paused
            self.control.notify_all()
            return self.paused

    def restart(self):
        # Only the read owner touches the decoder, avoiding seek/read races.
        with self.control:
            self.restart_requested.set()
            self.requested_measurement = 0
            self.stop_at_end = False
            self.paused = False
            self.control.notify_all()

    def begin_measurement(self):
        with self.control:
            self.measurement_token += 1
            self.requested_measurement = self.measurement_token
            self.completed_measurement = 0
            self.restart_requested.set()
            self.paused = False
            self.control.notify_all()
            return self.measurement_token

    def cancel_measurement(self):
        with self.control:
            self.stop_at_end = False
            self.requested_measurement = 0
            self.active_measurement = 0

    def playback_info(self):
        with self.control:
            return {"name": self.path.name, "position_s": self.position_s,
                    "duration_s": self.frames / self.fps,
                    "pass": self.pass_number, "paused": self.paused,
                    "active_measurement": self.active_measurement,
                    "completed_measurement": self.completed_measurement}

    def _rewind(self):
        with self.decoder_lock:
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise ValueError("Cannot rewind video")
        with self.control:
            self.frame_index = 0
            self.position_s = 0.0
            self.pass_number += 1
            self.previous_media_s = -1.0
        self.next_frame = 0.0

    def read(self):
        with self.control:
            while self.paused and not self.closed.is_set():
                self.control.wait(timeout=0.1)
                self.next_frame = 0.0
        if self.closed.is_set():
            return False, None, time.perf_counter()
        if self.restart_requested.is_set():
            with self.control:
                self.restart_requested.clear()
                self.active_measurement = self.requested_measurement
                self.stop_at_end = bool(self.active_measurement)
                self.requested_measurement = 0
            self._rewind()
        while self.next_frame > time.perf_counter():
            if self.closed.wait(self.next_frame - time.perf_counter()):
                return False, None, time.perf_counter()
        with self.decoder_lock:
            ok, frame = self.capture.read()
        if not ok:
            if self.frames <= 0 or self.frame_index < self.frames:
                raise ValueError("Video decode failed before end of file")
            with self.control:
                if self.stop_at_end:
                    self.completed_measurement = self.active_measurement
                    self.stop_at_end = False
                    self.paused = True
                    return False, None, time.perf_counter()
            self._rewind()
            with self.decoder_lock:
                ok, frame = self.capture.read()
            if not ok:
                raise ValueError("Video contains no readable frames")
        timestamp = time.perf_counter()
        with self.control:
            media_s = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(media_s) or media_s < 0 or media_s <= self.previous_media_s:
                media_s = self.frame_index / self.fps
            self.position_s = media_s
            self.previous_media_s = media_s
            self.frame_index += 1
        # Keep every frame; slow decoding slows playback instead of skipping frames.
        self.next_frame = timestamp + 1.0 / self.fps
        return True, frame, timestamp

    def status(self):
        return CameraStatus(self.width, self.height, self.fps, self.fps)

    def close(self):
        self.closed.set()
        with self.control:
            self.control.notify_all()

    def release(self):
        # The owner releases the decoder after the reading worker has stopped.
        self.close()
        with self.decoder_lock:
            self.capture.release()
