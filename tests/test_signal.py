import unittest

import numpy as np

from michelson.signal import analyze_signal, calculate_wavelength


class SignalAnalysisTests(unittest.TestCase):
    def test_constant_frequency_cycle_count(self) -> None:
        fps = 60.0
        duration = 8.0
        frequency = 3.0
        times = np.arange(0, duration, 1 / fps)
        rng = np.random.default_rng(7)
        intensities = 120 + 60 * np.cos(2 * np.pi * frequency * times + 0.35)
        intensities += rng.normal(0, 1.2, len(times))
        result = analyze_signal(times, intensities)
        expected = frequency * (times[-1] - times[0])
        self.assertAlmostEqual(result.cycles_phase, expected, delta=0.4)
        self.assertAlmostEqual(result.median_fringe_hz, frequency, delta=0.15)
        self.assertEqual(result.quality, "良好")

    def test_wavelength_formula(self) -> None:
        # 316.4 μm 的镜面位移对应 632.8 nm 激光的 1000 个周期。
        result = calculate_wavelength(1.0000, 1.3164, 1000.0)
        self.assertAlmostEqual(result.wavelength_nm, 632.8, places=8)

    def test_uncertainty_requires_resolution(self) -> None:
        result = calculate_wavelength(0.0, 0.3164, 1000.0)
        self.assertIsNone(result.uncertainty_nm)
        result_with_resolution = calculate_wavelength(
            0.0,
            0.3164,
            1000.0,
            micrometer_resolution_um=1.0,
            cycle_uncertainty=0.2,
        )
        self.assertIsNotNone(result_with_resolution.uncertainty_nm)


if __name__ == "__main__":
    unittest.main()

