# -*- coding: gb2312 -*-
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from michelson.roi import analyze_rois, roi_positions, sample_rois
from michelson.signal import calculate_wavelength
from michelson.storage import save_measurement


class FourROITests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0, 10, 1 / 60)
        self.phases = np.arange(4) * np.pi / 2
        self.samples = 120 + 65 * np.cos(
            2 * np.pi * 3 * self.times[:, None] + self.phases)
        self.samples += np.random.default_rng(6).normal(0, 0.6, self.samples.shape)

    def test_different_phases_do_not_cancel(self):
        result = analyze_rois(self.times, self.samples)
        self.assertEqual(len(result.accepted), 4)
        self.assertAlmostEqual(result.cycles, 3 * self.times[-1], delta=0.4)
        self.assertLess(np.std(self.samples.mean(axis=1)), 1)

    def test_one_blocked_channel_is_rejected(self):
        self.samples[:, 0] = 100
        result = analyze_rois(self.times, self.samples)
        self.assertEqual(len(result.accepted), 3)
        self.assertIn("top", result.rejected)
        self.assertAlmostEqual(result.cycles, 3 * self.times[-1], delta=0.4)

    def test_one_wrong_frequency_is_rejected(self):
        self.samples[:, 0] = 120 + 65 * np.cos(2 * np.pi * 4 * self.times)
        result = analyze_rois(self.times, self.samples)
        self.assertNotIn("top", result.accepted)
        self.assertEqual(len(result.accepted), 3)

    def test_two_bad_channels_fail(self):
        self.samples[:, :2] = 100
        with self.assertRaises(ValueError):
            analyze_rois(self.times, self.samples)

    def test_no_three_channel_agreement_fails(self):
        for index, frequency in enumerate((2, 3, 4, 5)):
            self.samples[:, index] = 120 + 65 * np.cos(2 * np.pi * frequency * self.times)
        with self.assertRaises(ValueError):
            analyze_rois(self.times, self.samples)

    def test_edge_rois_are_full_and_non_overlapping(self):
        center, points, offset = roi_positions((0, 0), (480, 640), 40, 8)
        self.assertEqual(center, (48, 48))
        self.assertEqual(sample_rois(np.full((480, 640), 99), points, 8), [99] * 4)
        self.assertGreater(offset, 16)
        with self.assertRaises(ValueError):
            roi_positions((0, 0), (20, 20), 40, 8)

    def test_export_keeps_four_raw_channels(self):
        result = analyze_rois(self.times, self.samples)
        wavelength = calculate_wavelength(0, result.cycles * 632.8 / 2e6, result.cycles)
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = save_measurement(directory, result.analysis, wavelength,
                {"merged_cycles": result.cycles}, channels=result.channels,
                raw_timestamps=self.times, raw_samples=self.samples)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            raw_path = Path(directory) / payload["raw_csv"]
            with raw_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), len(self.times) + 1)
            self.assertEqual(len(rows[0]), 5)
            self.assertEqual(len(payload["channels"]), 4)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(next(csv.reader(handle))), 16)


if __name__ == "__main__":
    unittest.main()
