# -*- coding: gb2312 -*-
import json
import time
import unittest
from urllib.request import ProxyHandler, build_opener

# Loopback tests must not be routed through the developer's HTTP proxy.
urlopen = build_opener(ProxyHandler({})).open

import cv2
import numpy as np

from michelson.webstream import FrameRelay, PreviewServer, StreamCamera, encode_comparison


class FakeCamera:
    def __init__(self):
        self.frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        self.reads = 0
        self.closed = False

    def read(self):
        self.reads += 1
        return True, self.frame, 123.5

    def close(self):
        self.closed = True


class WebStreamTests(unittest.TestCase):
    def test_camera_read_preserves_values_and_timestamp(self):
        relay, source = FrameRelay(), FakeCamera()
        camera = StreamCamera(source, relay)
        ok, frame, timestamp = camera.read()
        self.assertTrue(ok)
        self.assertIs(frame, source.frame)
        self.assertEqual(timestamp, 123.5)
        self.assertEqual(source.reads, 1)
        camera.close()
        self.assertTrue(source.closed)
        self.assertFalse(relay.status()["healthy"])

    def test_encoding_does_not_mutate_source(self):
        frame = np.random.default_rng(42).integers(0, 256, (48, 65, 3), dtype=np.uint8)
        original = frame.copy()
        jpeg = encode_comparison(frame)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[1], 65)
        self.assertGreater(image.shape[0], 48)
        np.testing.assert_array_equal(frame, original)

    def test_http_page_status_and_stream(self):
        relay = FrameRelay()
        server = PreviewServer(relay, port=0)
        server.start()
        base = f"http://127.0.0.1:{server.address[1]}"
        try:
            with urlopen(base, timeout=3) as response:
                page = response.read().decode("utf-8")
                self.assertIn("/stream.mjpg", page)
                self.assertIn("/status", page)
            relay.offer(np.full((48, 64, 3), 100, np.uint8))
            deadline = time.monotonic() + 3
            while relay.jpeg is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(relay.jpeg)
            with urlopen(base + "/status", timeout=3) as response:
                self.assertTrue(json.load(response)["healthy"])
            with urlopen(base + "/stream.mjpg", timeout=3) as response:
                self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
                self.assertEqual(response.readline(), b"--frame\r\n")
                self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                length = int(response.readline().split(b":")[1])
                self.assertEqual(response.readline(), b"\r\n")
                self.assertTrue(response.read(length).startswith(b"\xff\xd8"))
            relay.fail("Disconnected")
            with urlopen(base + "/status", timeout=3) as response:
                self.assertFalse(json.load(response)["healthy"])
        finally:
            server.close()
        self.assertFalse(server.encoder.is_alive())
        self.assertFalse(server.listener.is_alive())

    def test_stale_frame_is_unhealthy(self):
        relay = FrameRelay()
        relay.offer(np.zeros((4, 4, 3), np.uint8))
        relay.received_at -= 4
        self.assertFalse(relay.status()["healthy"])


if __name__ == "__main__":
    unittest.main()
