# -*- coding: gb2312 -*-
"""Read-only LAN preview, independent from measurement and GUI refresh."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from .preview import morphology_preview


class FrameRelay:
    """Bounded latest-frame slot; only the camera owner calls offer()."""

    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.received_at = 0.0
        self.error = "Waiting for camera"
        self.jpeg = None
        self.jpeg_sequence = 0
        self.closed = False

    def offer(self, frame):
        with self.condition:
            # Camera output is not modified by either preview branch.
            self.frame = frame
            self.sequence += 1
            self.received_at = time.monotonic()
            self.error = ""
            self.condition.notify_all()

    def fail(self, message):
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def status(self):
        with self.condition:
            age = time.monotonic() - self.received_at if self.received_at else None
            return {"healthy": not self.closed and not self.error and age is not None and age < 3,
                    "error": self.error, "age_s": age, "frame": self.jpeg_sequence}


class StreamCamera:
    """Publish existing reads without opening a second camera device."""

    def __init__(self, camera, relay):
        self.camera, self.relay = camera, relay

    def read(self):
        result = self.camera.read()
        ok, frame, _ = result
        if ok and frame is not None:
            self.relay.offer(frame)
        else:
            self.relay.fail("Camera read failed")
        return result

    def status(self):
        return self.camera.status()

    def close(self):
        self.relay.fail("Camera stopped")
        self.camera.close()


def encode_comparison(frame):
    """Fixed display size bounds network cost; never replace the capture frame."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / width, 480 / height)
    raw = cv2.resize(frame, (max(2, int(width * scale)), max(2, int(height * scale))),
                     interpolation=cv2.INTER_AREA)
    result = morphology_preview(raw)
    h, w = raw.shape[:2]
    half = w // 2
    comparison_height = max(1, round(h * half / w))
    # One composite JPEG ensures all three views represent the same frame.
    composite = np.zeros((h + comparison_height + 48, w, 3), dtype=np.uint8)
    composite[24:24 + h] = raw
    cv2.putText(composite, "RAW - measurement source", (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1)
    for index, image in enumerate((result.opening, result.closing)):
        x = index * half
        target_width = half if index == 0 else w - half
        small = cv2.resize(image, (target_width, comparison_height), interpolation=cv2.INTER_NEAREST)
        composite[h + 48:, x:x + target_width] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        cv2.putText(composite, "OPEN" if index == 0 else "CLOSE", (x + 5, h + 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1)
    ok, encoded = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


class PreviewServer:
    def __init__(self, relay, host="127.0.0.1", port=8080):
        self.relay = relay
        self.stop = threading.Event()
        self.stream_slots = threading.BoundedSemaphore(6)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def do_GET(self):
                path = urlsplit(self.path).path
                try:
                    if path == "/":
                        self.send_bytes(Path(__file__).with_name("webstream.html").read_bytes(), "text/html; charset=utf-8")
                    elif path == "/status":
                        self.send_bytes(json.dumps(relay.status()).encode("utf-8"), "application/json")
                    elif path == "/stream.mjpg":
                        self.stream()
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    pass

            def send_bytes(self, data, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)

            def stream(self):
                if not owner.stream_slots.acquire(blocking=False):
                    self.send_error(503, "Too many viewers")
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = -1
                    while not owner.stop.is_set():
                        with relay.condition:
                            relay.condition.wait_for(lambda: relay.closed or relay.jpeg_sequence != last, timeout=1)
                            if relay.closed:
                                break
                            jpeg, sequence = relay.jpeg, relay.jpeg_sequence
                            healthy = (not relay.error and time.monotonic() - relay.received_at < 3)
                        if not healthy or jpeg is None or sequence == last:
                            owner.stop.wait(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                        last = sequence
                finally:
                    owner.stream_slots.release()

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.http.daemon_threads = True
        self.encoder = threading.Thread(target=self._encode_loop, daemon=True)
        self.listener = threading.Thread(target=self.http.serve_forever, daemon=True)

    @property
    def address(self):
        return self.http.server_address

    def start(self):
        self.encoder.start()
        self.listener.start()

    def _encode_loop(self):
        last = -1
        while not self.stop.is_set():
            with self.relay.condition:
                frame, sequence = self.relay.frame, self.relay.sequence
            if frame is not None and sequence != last:
                try:
                    jpeg = encode_comparison(frame)
                    with self.relay.condition:
                        self.relay.jpeg = jpeg
                        self.relay.jpeg_sequence = sequence
                        self.relay.condition.notify_all()
                except (ValueError, cv2.error) as error:
                    self.relay.fail(str(error))
                last = sequence
            self.stop.wait(0.125)

    def close(self):
        self.stop.set()
        with self.relay.condition:
            self.relay.closed = True
            self.relay.condition.notify_all()
        self.http.shutdown()
        self.http.server_close()
        self.encoder.join(timeout=2)
        self.listener.join(timeout=2)
