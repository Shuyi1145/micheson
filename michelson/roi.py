# -*- coding: gb2312 -*-
"""Independent four-region acquisition and conservative count agreement."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .signal import AnalysisResult, analyze_signal

ROI_NAMES = ("top", "right", "bottom", "left")


def roi_positions(center, shape, offset, radius):
    """Keep all circles inside the frame, disjoint and on one radius."""
    height, width = shape[:2]
    radius = max(1, int(radius))
    minimum_offset = 2 * radius + 1
    maximum_offset = min(width, height) // 2 - radius - 1
    if maximum_offset < minimum_offset:
        raise ValueError("Frame too small for four non-overlapping ROIs.")
    offset = int(np.clip(offset, minimum_offset, maximum_offset))
    margin = offset + radius
    x = int(np.clip(center[0], margin, width - margin - 1))
    y = int(np.clip(center[1], margin, height - margin - 1))
    points = ((x, y - offset), (x + offset, y), (x, y + offset), (x - offset, y))
    return (x, y), points, offset


def sample_rois(gray, points, radius):
    """Spatial averaging within each circle; never average across channels."""
    radius = int(radius)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = xx * xx + yy * yy <= radius * radius
    values = []
    for x, y in points:
        patch = gray[y - radius:y + radius + 1, x - radius:x + radius + 1]
        if patch.shape != mask.shape:
            raise ValueError("ROI extends outside the frame.")
        values.append(float(patch[mask].mean()))
    return values


@dataclass(frozen=True)
class MultiROIResult:
    channels: dict[str, AnalysisResult]
    rejected: dict[str, str]
    accepted: tuple[str, ...]
    representative: str
    cycles: float
    spread_cycles: float
    warnings: tuple[str, ...]

    @property
    def analysis(self):
        """The stored single-channel waveform remains explicitly representative."""
        result = self.channels[self.representative]
        return replace(result, quality="需复核" if self.warnings else result.quality,
                       warnings=tuple(dict.fromkeys(result.warnings + self.warnings)))


def analyze_rois(timestamps, samples, **kwargs):
    minimum = int(kwargs.get("minimum_samples", 60))
    if len(timestamps) < minimum:
        raise ValueError(f"有效样本不足：至少需要 {minimum} 帧，当前 {len(timestamps)} 帧。")
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape != (len(timestamps), 4):
        raise ValueError("Expected one row of four ROI values for every timestamp.")
    if not np.isfinite(samples).all() or not np.isfinite(timestamps).all():
        raise ValueError("ROI samples and timestamps must be finite.")
    channels = {}
    rejected = {}
    candidates = []
    for index, name in enumerate(ROI_NAMES):
        try:
            result = analyze_signal(timestamps, samples[:, index], **kwargs)
            channels[name] = result
            if result.contrast < 0.08:
                rejected[name] = "对比度不足"
            elif result.quality == "需复核":
                rejected[name] = "计数或采样质量异常"
            else:
                candidates.append(name)
        except ValueError as error:
            rejected[name] = str(error)
    if len(candidates) < 3:
        raise ValueError("至少需要三个有效 ROI；请调整中心、间距、曝光并重新采集。")

    counts = np.array([channels[name].cycles_phase for name in candidates])
    # A fixed tolerance is an engineering check, not a statistical uncertainty.
    tolerance = max(0.75, 0.02 * float(np.median(counts)))
    clusters = [tuple(name for name in candidates
                      if 0 <= channels[name].cycles_phase - channels[pivot].cycles_phase
                      <= tolerance) for pivot in candidates]
    accepted = max(clusters, key=lambda group: (len(group),
                   -float(np.ptp([channels[n].cycles_phase for n in group]))))
    if len(accepted) < 3:
        raise ValueError("四点周期数不一致，不能可靠合并；请检查漂移或局部干扰。")
    for name in candidates:
        if name not in accepted:
            rejected[name] = "周期数偏离其他 ROI"
    accepted_counts = [channels[name].cycles_phase for name in accepted]
    cycles = float(np.median(accepted_counts))
    representative = min(accepted, key=lambda name: abs(channels[name].cycles_phase - cycles))
    warnings = tuple(f"{name}: {reason}" for name, reason in rejected.items())
    return MultiROIResult(channels, rejected, accepted, representative, cycles,
                          float(np.ptp(accepted_counts)), warnings)
