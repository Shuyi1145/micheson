# -*- coding: gb2312 -*-
import unittest

import cv2
import numpy as np

from michelson.preview import morphology_preview


class PreviewTests(unittest.TestCase):
    def test_independent_branches_and_input_unchanged(self):
        gray = np.zeros((40, 60), dtype=np.uint8)
        gray[8:32, 12:48] = 255
        gray[20, 30] = 0
        gray[3, 3] = 255
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        original = frame.copy()
        result = morphology_preview(frame, denoise=False, threshold=128)
        np.testing.assert_array_equal(frame, original)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        np.testing.assert_array_equal(result.opening, cv2.dilate(cv2.erode(gray, kernel), kernel))
        np.testing.assert_array_equal(result.closing, cv2.erode(cv2.dilate(gray, kernel), kernel))
        self.assertEqual(result.opening[3, 3], 0)
        self.assertEqual(result.closing[20, 30], 255)
        self.assertEqual(result.closing[3, 3], 255)

    def test_binary_output_and_otsu(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        result = morphology_preview(frame)
        for pixels in (result.opening, result.closing):
            self.assertEqual(pixels.shape, frame.shape[:2])
            self.assertTrue(set(np.unique(pixels)).issubset({0, 255}))
        self.assertTrue(0 <= result.threshold <= 255)

    def test_constant_small_frames(self):
        for value in (0, 128, 255):
            frame = np.full((1, 1, 3), value, dtype=np.uint8)
            result = morphology_preview(frame)
            self.assertEqual(result.opening.shape, (1, 1))

    def test_invalid_parameters(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        for kwargs in ({"kernel_size": 4}, {"threshold": -1}, {"threshold": 256},
                       {"threshold": float("nan")}):
            with self.assertRaises(ValueError):
                morphology_preview(frame, **kwargs)
        with self.assertRaises(ValueError):
            morphology_preview(frame.astype(float))


if __name__ == "__main__":
    unittest.main()
