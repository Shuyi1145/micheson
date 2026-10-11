# -*- coding: gb2312 -*-
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from michelson.playback import VideoPlayback
from michelson.webstream import FrameRelay, StreamCamera


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.avi"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 120, (64, 48))
        if not writer.isOpened():
            self.skipTest("MJPG encoder unavailable")
        for value in (30, 80, 130):
            writer.write(np.full((48, 64, 3), value, np.uint8))
        writer.release()

    def source(self):
        source = VideoPlayback(self.path)
        self.addCleanup(source.release)
        return source

    def test_pacing_loop_and_frame_order(self):
        source = self.source()
        frames = [source.read() for _ in range(5)]
        self.assertTrue(all(frame[0] for frame in frames))
        self.assertEqual([round(float(frame[1].mean())) for frame in frames],
                         [30, 80, 130, 30, 80])
        self.assertGreaterEqual(frames[1][2] - frames[0][2], 1 / 120 - 0.002)
        self.assertTrue(all(b[2] > a[2] for a, b in zip(frames, frames[1:])))
        self.assertEqual(source.playback_info()["pass"], 2)
        self.assertEqual(source.status().width, 64)

    def test_restart_and_shared_web_source(self):
        source = self.source()
        relay = FrameRelay()
        relay.source_info = source.playback_info
        stream = StreamCamera(source, relay)
        stream.read()
        stream.read()
        source.restart()
        ok, frame, _ = stream.read()
        self.assertTrue(ok)
        self.assertAlmostEqual(float(frame.mean()), 30, delta=2)
        self.assertIs(relay.frame, frame)
        self.assertEqual(relay.status()["playback"]["position_s"], 0)

    def test_close_unblocks_paused_reader(self):
        source = self.source()
        self.assertTrue(source.toggle_pause())
        result = []
        reader = threading.Thread(target=lambda: result.append(source.read()))
        reader.start()
        time.sleep(0.03)
        self.assertTrue(reader.is_alive())
        source.close()
        reader.join(timeout=1)
        self.assertFalse(reader.is_alive())
        self.assertFalse(result[0][0])

    def test_missing_or_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            VideoPlayback(self.path.with_name("missing.mp4"))
        with self.assertRaises(ValueError):
            VideoPlayback(Path(__file__))

    def test_entrypoint_never_opens_usb_camera(self):
        import main
        with patch("sys.argv", ["main.py", "--play-video", str(self.path)]), \
                patch("main.UsbCamera", side_effect=AssertionError("USB opened")), \
                patch("michelson.gui.run_gui") as gui:
            main.main()
            source = gui.call_args.args[0]
            self.assertIs(gui.call_args.kwargs["playback"], source)
            self.assertTrue(source.closed.is_set())

    def test_desktop_video_pause_restart_and_close(self):
        import json
        import tkinter as tk
        from michelson.gui import MichelsonApp
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        source = self.source()
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        config["measurement"]["roi_radius_px"] = 3
        config["measurement"]["roi_offset_px"] = 12
        errors = []
        root.report_callback_exception = lambda kind, error, trace: errors.append(error)
        root.geometry("800x480")
        app = MichelsonApp(root, source, config, playback=source, center=(32, 24))

        def check():
            try:
                self.assertIsNotNone(app.latest)
                self.assertIsNotNone(app.photo)
                self.assertTrue(all(p is not None for p in app.comparison_photos))
                self.assertNotIn("disabled", app.start_button.state())
                app.toggle_playback()
                self.assertTrue(source.playback_info()["paused"])
                app.restart_playback()
                self.assertFalse(source.playback_info()["paused"])
            except Exception as error:
                errors.append(error)
            finally:
                app.close()

        root.after(400, check)
        root.mainloop()
        self.assertFalse(app.worker.is_alive())
        if errors:
            raise errors[0]

    def test_one_pass_recording_includes_first_and_last_frame(self):
        source = self.source()
        source.read()
        token = source.begin_measurement()
        frames = [source.read() for _ in range(3)]
        self.assertEqual([round(float(frame[1].mean())) for frame in frames], [30, 80, 130])
        self.assertEqual(source.playback_info()["active_measurement"], token)
        self.assertAlmostEqual(source.playback_info()["position_s"], 2 / 120)
        self.assertFalse(source.read()[0])
        self.assertEqual(source.playback_info()["completed_measurement"], token)
        self.assertTrue(source.playback_info()["paused"])

    def test_default_entrypoint_uses_configured_video(self):
        import main
        config = {"video": {"path": str(self.path), "center_px": [32, 24]}, "camera": {}}
        with patch("sys.argv", ["main.py"]), patch("main.load_config", return_value=config), \
                patch("main.UsbCamera", side_effect=AssertionError("USB opened")), \
                patch("michelson.gui.run_gui") as gui:
            main.main()
            self.assertEqual(gui.call_args.kwargs["center"], (32, 24))
            self.assertIsNotNone(gui.call_args.kwargs["playback"])


if __name__ == "__main__":
    unittest.main()
