# -*- coding: gb2312 -*-
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from michelson.video import analyze_video, inferred_displacement


class VideoTests(unittest.TestCase):
    def test_reference_displacement_identity(self):
        distance = inferred_displacement(100, 632.8)
        self.assertAlmostEqual(distance, 0.03164)
        for count in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                inferred_displacement(count, 632.8)

    def test_every_frame_count_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (96, 96))
            if not writer.isOpened():
                self.skipTest("MJPG writer unavailable")
            for index in range(300):
                value = round(120 + 80 * np.cos(2 * np.pi * 2 * index / 30))
                writer.write(np.full((96, 96, 3), value, np.uint8))
            writer.release()
            config = {"measurement": {"roi_radius_px": 3, "roi_offset_px": 12},
                      "output_directory": str(Path(temporary) / "results")}
            directory, result = analyze_video(source, config, export_preview=True)
            self.assertEqual(result["frames_analyzed"], 300)
            self.assertEqual(len(result["accepted_rois"]), 4)
            self.assertAlmostEqual(result["merged_cycles"], 20, delta=0.5)
            self.assertAlmostEqual(result["formula_check"]["wavelength_nm"], 632.8)
            self.assertIn("not measured", result["displacement_source"])
            with (directory / "raw.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 301)
            self.assertEqual(int(rows[-1][0]), 299)
            self.assertTrue((directory / "comparison.mp4").is_file())
            self.assertEqual(json.loads((directory / "result.json").read_text(encoding="utf-8"))["frames_analyzed"], 300)
            _, partial = analyze_video(source, config, start_s=2, end_s=3, export_preview=False)
            self.assertEqual(partial["frames_analyzed"], 30)
            self.assertIsNone(partial["merged_cycles"])
            self.assertIsNotNone(partial["merge_error"])


if __name__ == "__main__":
    unittest.main()
