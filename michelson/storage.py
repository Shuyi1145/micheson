# -*- coding: gb2312 -*-
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .roi import ROI_NAMES
from .signal import AnalysisResult, WavelengthResult


def save_measurement(
    output_directory: str | Path,
    analysis: AnalysisResult,
    wavelength: WavelengthResult | None,
    metadata: dict,
    *,
    channels: dict[str, AnalysisResult] | None = None,
    raw_timestamps=None,
    raw_samples=None,
) -> tuple[Path, Path]:
    channel_names = tuple(channels or {})
    for name in channel_names:
        if not np.array_equal(channels[name].timestamps, analysis.timestamps):
            raise ValueError("Channel timestamps do not match the representative timeline.")
    if (raw_timestamps is None) != (raw_samples is None):
        raise ValueError("Raw timestamps and samples must be supplied together.")
    if raw_timestamps is not None:
        raw_times = np.asarray(raw_timestamps, dtype=float)
        raw_values = np.asarray(raw_samples, dtype=float)
        if raw_times.ndim != 1 or len(raw_times) == 0 or raw_values.shape != (len(raw_times), 4):
            raise ValueError("Invalid four-ROI raw data shape.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = directory / f"measurement_{stamp}.csv"
    json_path = directory / f"measurement_{stamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        columns = [analysis.timestamps, analysis.raw_signal, analysis.filtered_signal, analysis.phase]
        headings = ["time_s", "raw_gray", "filtered", "phase_rad"]
        for name in channel_names:
            channel = channels[name]
            columns.extend([channel.raw_signal, channel.filtered_signal, channel.phase])
            headings.extend([f"{name}_gray", f"{name}_filtered", f"{name}_phase_rad"])
        writer.writerow(headings)
        for row in zip(*columns, strict=True):
            writer.writerow([f"{float(value):.10g}" for value in row])

    raw_path = None
    if raw_timestamps is not None:
        raw_path = directory / f"measurement_{stamp}_raw.csv"
        with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s"] + [f"{name}_gray" for name in ROI_NAMES])
            for timestamp, values in zip(raw_times - raw_times[0], raw_values, strict=True):
                writer.writerow([f"{timestamp:.10g}"] + [f"{value:.10g}" for value in values])

    summary = {
        "metadata": metadata,
        "analysis": {
            "cycles_phase": analysis.cycles_phase,
            "cycles_crossing": analysis.cycles_crossing,
            "duration_s": analysis.duration_s,
            "sample_rate_hz": analysis.sample_rate_hz,
            "median_fringe_hz": analysis.median_fringe_hz,
            "maximum_fringe_hz": analysis.maximum_fringe_hz,
            "contrast": analysis.contrast,
            "quality": analysis.quality,
            "warnings": list(analysis.warnings),
        },
        "wavelength": None if wavelength is None else asdict(wavelength),
        "channels": {
            name: {"cycles_phase": channels[name].cycles_phase,
                   "cycles_crossing": channels[name].cycles_crossing,
                   "contrast": channels[name].contrast,
                   "quality": channels[name].quality,
                   "warnings": list(channels[name].warnings)}
            for name in channel_names
        },
        "raw_csv": None if raw_path is None else raw_path.name,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path

