from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .signal import AnalysisResult, WavelengthResult


def save_measurement(
    output_directory: str | Path,
    analysis: AnalysisResult,
    wavelength: WavelengthResult | None,
    metadata: dict,
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = directory / f"measurement_{stamp}.csv"
    json_path = directory / f"measurement_{stamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "raw_gray", "filtered", "phase_rad"])
        for row in zip(
            analysis.timestamps,
            analysis.raw_signal,
            analysis.filtered_signal,
            analysis.phase,
            strict=True,
        ):
            writer.writerow([f"{float(value):.10g}" for value in row])

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
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path

