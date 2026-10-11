# -*- coding: gb2312 -*-
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from michelson.camera import DemoCamera
from michelson.gui import MichelsonApp


class SmallScreenTests(unittest.TestCase):
    def test_small_window_acquisition_and_save(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"No graphical display: {error}")
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        errors = []
        root.report_callback_exception = lambda kind, value, trace: errors.append(value)
        root.geometry("800x480+10+10")
        with tempfile.TemporaryDirectory() as directory, \
                patch("michelson.gui.messagebox.showinfo"), \
                patch("michelson.gui.messagebox.showerror") as dialog:
            config["output_directory"] = directory
            app = MichelsonApp(root, DemoCamera(config["camera"]), config, demo_mode=True)

            def start():
                try:
                    app.start_var.set("1.0")
                    app.start_recording()
                    self.assertTrue(app.recording)
                    for button in (app.start_button, app.stop_button, app.save_button):
                        bottom = button.winfo_rooty() + button.winfo_height()
                        self.assertLessEqual(bottom, root.winfo_rooty() + root.winfo_height())
                    self.assertGreater(app.video_canvas.winfo_height(), 50)
                    self.assertEqual(len(app.points), 4)
                    self.assertEqual(len(app.comparison_canvases), 2)
                    self.assertTrue(all(photo is not None for photo in app.comparison_photos))
                    self.assertLess(app.video_canvas.winfo_rooty(),
                                    app.comparison_canvases[0].winfo_rooty())
                    # Invalid display settings must not stop measurement.
                    app.auto_threshold_var.set(False)
                    app.threshold_var.set("invalid")
                    app._display_comparison(app.latest[0])
                    app.threshold_var.set("128")
                    self.assertLess(app.panel_canvas.yview()[1], 1.0)
                    app.toggle_fullscreen()
                    self.assertTrue(app.fullscreen)
                    app.exit_fullscreen()
                    self.assertFalse(app.fullscreen)
                except Exception as error:
                    errors.append(error)

            def finish():
                try:
                    app.stop_and_analyze()
                    self.assertIsNotNone(app.last_multi, dialog.call_args)
                    self.assertGreater(len(app.timestamps), 60)
                    # Acquisition exceeds the 15 Hz preview rate.
                    duration = app.timestamps[-1] - app.timestamps[0]
                    self.assertGreater((len(app.timestamps) - 1) / duration, 20)
                    app.end_var.set(str(1 + app.last_multi.cycles * 632.8 / 2e6))
                    app.calculate_and_save()
                    self.assertAlmostEqual(app.last_wavelength.wavelength_nm, 632.8, places=5)
                    self.assertEqual(len(app.video_canvas.find_all()), 1)
                    for canvas in app.comparison_canvases:
                        self.assertEqual(len(canvas.find_all()), 1)
                    self.assertFalse(dialog.called)
                except Exception as error:
                    errors.append(error)
                finally:
                    app.close()

            root.after(350, start)
            root.after(3350, finish)
            root.mainloop()
        if errors:
            raise errors[0]


if __name__ == "__main__":
    unittest.main()
