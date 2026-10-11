# -*- coding: gb2312 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnalysisResult:
    timestamps: np.ndarray
    raw_signal: np.ndarray
    filtered_signal: np.ndarray
    phase: np.ndarray
    cycles_phase: float
    cycles_crossing: int
    duration_s: float
    sample_rate_hz: float
    median_fringe_hz: float
    maximum_fringe_hz: float
    contrast: float
    quality: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class WavelengthResult:
    wavelength_nm: float
    displacement_mm: float
    cycles: float
    uncertainty_nm: float | None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _analytic_signal(values: np.ndarray) -> np.ndarray:
    """用 FFT 构造 Hilbert 解析信号，避免在香橙派上依赖 SciPy。"""
    count = len(values)
    spectrum = np.fft.fft(values)
    multiplier = np.zeros(count)
    if count % 2 == 0:
        multiplier[0] = 1
        multiplier[count // 2] = 1
        multiplier[1 : count // 2] = 2
    else:
        multiplier[0] = 1
        multiplier[1 : (count + 1) // 2] = 2
    return np.fft.ifft(spectrum * multiplier)


def _positive_zero_crossings(values: np.ndarray, hysteresis: float = 0.15) -> np.ndarray:
    crossings: list[float] = []
    armed = values[0] < -hysteresis
    for index in range(1, len(values)):
        if values[index] < -hysteresis:
            armed = True
        if armed and values[index - 1] <= 0 < values[index]:
            denominator = values[index] - values[index - 1]
            fraction = 0.0 if denominator == 0 else -values[index - 1] / denominator
            crossings.append(index - 1 + float(fraction))
            armed = False
    return np.asarray(crossings, dtype=float)


def analyze_signal(
    timestamps: list[float] | np.ndarray,
    intensities: list[float] | np.ndarray,
    *,
    safe_fringe_fraction_of_fps: float = 0.25,
    minimum_samples: int = 60,
) -> AnalysisResult:
    """分析单调移动反射镜时的中心区域灰度波形。

    phase 周期数可利用首尾的非整数周期；crossing 周期数用于独立交叉检查。
    单通道强度无法判断反射镜运动方向，因此实验过程中必须保持单向移动。
    """
    times = np.asarray(timestamps, dtype=float)
    raw = np.asarray(intensities, dtype=float)
    valid = np.isfinite(times) & np.isfinite(raw)
    times, raw = times[valid], raw[valid]
    if len(times) < minimum_samples:
        raise ValueError(f"有效样本不足：至少需要 {minimum_samples} 帧，当前 {len(times)} 帧。")
    if np.any(np.diff(times) <= 0):
        order = np.argsort(times)
        times, raw = times[order], raw[order]
        unique = np.concatenate(([True], np.diff(times) > 0))
        times, raw = times[unique], raw[unique]
    if len(times) < minimum_samples:
        raise ValueError("去除重复时间戳后，有效样本不足。")

    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise ValueError("采集时长必须大于零。")
    uniform_times = np.linspace(times[0], times[-1], len(times))
    uniform_raw = np.interp(uniform_times, times, raw)
    sample_rate = float((len(times) - 1) / duration)

    median = float(np.median(uniform_raw))
    mad = float(np.median(np.abs(uniform_raw - median)))
    if mad > 0:
        uniform_raw = np.clip(uniform_raw, median - 8 * mad, median + 8 * mad)
    smoothed = _moving_average(uniform_raw, 3)

    normalized_time = np.linspace(-1.0, 1.0, len(smoothed))
    trend_degree = 2 if len(smoothed) >= 100 else 1
    trend = np.polyval(np.polyfit(normalized_time, smoothed, trend_degree), normalized_time)
    detrended = smoothed - trend
    low, high = np.percentile(detrended, [5, 95])
    amplitude = float((high - low) / 2)
    if amplitude < 1e-9:
        raise ValueError("灰度变化过小，无法识别干涉周期。")
    filtered = detrended / amplitude

    analytic = _analytic_signal(filtered)
    phase = np.unwrap(np.angle(analytic))
    phase = _moving_average(phase, 5)
    cycles_phase = abs(float(phase[-1] - phase[0])) / (2 * np.pi)

    crossing_positions = _positive_zero_crossings(filtered)
    cycles_crossing = max(0, len(crossing_positions) - 1)
    median_fringe_hz = 0.0
    maximum_fringe_hz = 0.0
    if len(crossing_positions) >= 2:
        crossing_times = np.interp(
            crossing_positions,
            np.arange(len(uniform_times), dtype=float),
            uniform_times,
        )
        periods = np.diff(crossing_times)
        periods = periods[periods > 0]
        if len(periods):
            median_fringe_hz = float(1.0 / np.median(periods))
            maximum_fringe_hz = float(1.0 / np.min(periods))

    original_low, original_high = np.percentile(uniform_raw, [5, 95])
    denominator = original_high + original_low
    contrast = 0.0 if denominator <= 0 else float((original_high - original_low) / denominator)

    warnings: list[str] = []
    if contrast < 0.08:
        warnings.append("条纹对比度偏低；请改善准直、减小环境光或调整曝光。")
    safe_frequency = sample_rate * safe_fringe_fraction_of_fps
    if maximum_fringe_hz > safe_frequency:
        warnings.append(
            f"局部条纹频率达到 {maximum_fringe_hz:.2f} Hz，超过当前帧率的安全范围；可能漏计。"
        )
    disagreement = abs(cycles_phase - cycles_crossing)
    # Crossing count spans only the first to last detected crossing; two partial
    # endpoint cycles can be missing without indicating an acquisition failure.
    tolerance = max(2.0, 0.05 * max(cycles_phase, 1.0))
    if disagreement > tolerance:
        warnings.append("相位法与过零计数差异较大；可能存在运动反向、曝光变化或漏帧。")
    if cycles_phase < 5:
        warnings.append("有效周期数过少；建议扩大单向位移以降低计数相对误差。")

    if any("可能漏计" in item or "差异较大" in item for item in warnings):
        quality = "需复核"
    elif warnings:
        quality = "可用"
    else:
        quality = "良好"

    return AnalysisResult(
        timestamps=uniform_times - uniform_times[0],
        raw_signal=uniform_raw,
        filtered_signal=filtered,
        phase=phase - phase[0],
        cycles_phase=cycles_phase,
        cycles_crossing=cycles_crossing,
        duration_s=duration,
        sample_rate_hz=sample_rate,
        median_fringe_hz=median_fringe_hz,
        maximum_fringe_hz=maximum_fringe_hz,
        contrast=contrast,
        quality=quality,
        warnings=tuple(warnings),
    )


def calculate_wavelength(
    start_reading_mm: float,
    end_reading_mm: float,
    cycles: float,
    *,
    micrometer_resolution_um: float | None = None,
    cycle_uncertainty: float | None = None,
) -> WavelengthResult:
    if cycles <= 0:
        raise ValueError("周期数必须大于零。")
    displacement_mm = abs(float(end_reading_mm) - float(start_reading_mm))
    if displacement_mm <= 0:
        raise ValueError("千分尺起始读数与终止读数不能相同。")
    wavelength_nm = 2.0 * displacement_mm * 1_000_000.0 / cycles

    uncertainty_nm: float | None = None
    if micrometer_resolution_um is not None and micrometer_resolution_um > 0:
        # 单次读数按均匀分布估计，位移由两次独立读数之差得到。
        sigma_d_mm = np.sqrt(2.0) * (micrometer_resolution_um / 1000.0) / np.sqrt(12.0)
        sigma_n = 0.25 if cycle_uncertainty is None else max(float(cycle_uncertainty), 0.0)
        relative = np.sqrt((sigma_d_mm / displacement_mm) ** 2 + (sigma_n / cycles) ** 2)
        uncertainty_nm = float(wavelength_nm * relative)

    return WavelengthResult(
        wavelength_nm=float(wavelength_nm),
        displacement_mm=displacement_mm,
        cycles=float(cycles),
        uncertainty_nm=uncertainty_nm,
    )

