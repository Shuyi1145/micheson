# -*- coding: gb2312 -*-
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from michelson.gui import MichelsonApp
from michelson.playback import VideoPlayback
from michelson.preview import DisplaySettings
from michelson.webstream import FrameRelay, PreviewServer


class VideoGuiTests(unittest.TestCase):
    def test_full_video_measurement_and_threshold_window(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        errors = []
        root.report_callback_exception = lambda kind, value, trace: errors.append(value)
        root.geometry("800x480")
        with tempfile.TemporaryDirectory() as directory, \
                patch("michelson.gui.messagebox.showerror") as error_dialog, \
                patch("michelson.gui.messagebox.showinfo"):
            path = Path(directory) / "source.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 120, (96, 96))
            if not writer.isOpened():
                root.destroy()
                self.skipTest("MJPG unavailable")
            for index in range(300):
                value = round(120 + 80 * np.cos(2 * np.pi * 4 * index / 120))
                writer.write(np.full((96, 96, 3), value, np.uint8))
            writer.release()
            source = VideoPlayback(path)
            settings = DisplaySettings()
            relay = FrameRelay(settings)
            config = {"measurement": {"roi_radius_px": 3, "roi_offset_px": 12},
                      "video": {"start_reading_cm": 1, "end_reading_cm": 1.001},
                      "output_directory": directory}
            app = MichelsonApp(root, source, config, playback=source,
                               center=(48, 48), display_settings=settings)

            def start():
                try:
                    self.assertEqual(float(app.start_var.get()), 10)
                    self.assertAlmostEqual(float(app.end_var.get()), 10.01)
                    app.start_recording()
                    self.assertTrue(app.recording)
                    app.open_threshold_window()
                    window = app.threshold_window
                    window.freeze.set(True)
                    window.slider.set(70)
                    window.on_slide("70")
                    window.apply()
                    self.assertEqual(settings.snapshot()[0]["threshold"], 70)
                    self.assertEqual(relay.status()["display_options"]["threshold"], 70)
                    self.assertTrue(all(p is not None for p in window.photos))
                    app.playback.toggle_pause()
                    root.after(100, app.playback.toggle_pause)
                except Exception as error:
                    errors.append(error)

            def finish():
                try:
                    self.assertFalse(app.recording)
                    self.assertTrue(app.video_complete)
                    self.assertEqual(len(app.samples), 300)
                    self.assertEqual(app.timestamps[0], 0)
                    self.assertAlmostEqual(app.timestamps[-1], 299 / 120)
                    self.assertIsNotNone(app.last_multi, error_dialog.call_args)
                    self.assertAlmostEqual(app.last_multi.cycles, 10, delta=0.5)
                    # Threshold changes do not alter the original sampled gray values.
                    np.testing.assert_allclose(np.asarray(app.samples)[:, 0],
                        [round(120 + 80 * np.cos(2 * np.pi * 4 * i / 120)) for i in range(300)], atol=2)
                    app.calculate_and_save()
                    self.assertIsNotNone(app.last_wavelength)
                    self.assertFalse(error_dialog.called)
                except Exception as error:
                    errors.append(error)
                finally:
                    app.close()
                    source.release()

            root.after(200, start)
            root.after(7000, finish)
            root.mainloop()
        if errors:
            raise errors[0]

    def test_web_reencodes_frozen_frame_after_threshold_change(self):
        import time
        settings = DisplaySettings({"threshold": 20, "denoise": False})
        relay = FrameRelay(settings)
        server = PreviewServer(relay, port=0)
        server.start()
        try:
            relay.offer(np.full((48, 64, 3), 100, np.uint8))
            deadline = time.monotonic() + 3
            while relay.jpeg is None and time.monotonic() < deadline:
                time.sleep(0.02)
            first, sequence = relay.jpeg, relay.jpeg_sequence
            self.assertIsNotNone(first)
            settings.update(threshold=200)
            deadline = time.monotonic() + 3
            while relay.jpeg_sequence == sequence and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertGreater(relay.jpeg_sequence, sequence)
            self.assertNotEqual(relay.jpeg, first)
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
