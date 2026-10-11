# -*- coding: gb2312 -*-
"""Offline, every-frame video analysis and display-only comparison export."""
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .roi import ROI_NAMES, analyze_rois, roi_positions, sample_rois
from .signal import analyze_signal, calculate_wavelength
from .webstream import encode_comparison


def inferred_displacement(cycles, reference_nm):
    if not np.isfinite(cycles) or cycles <= 0:
        raise ValueError("A positive cycle count is required.")
    if not np.isfinite(reference_nm) or reference_nm <= 0:
        raise ValueError("A positive reference wavelength is required.")
    return cycles * reference_nm / 2e6


def analyze_video(path, config, *, center=None, start_s=0.0, end_s=None,
                  export_preview=True, progress=None):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not np.isfinite(start_s) or start_s < 0:
        raise ValueError("Start time must be nonnegative.")
    if end_s is not None and (not np.isfinite(end_s) or end_s <= start_s):
        raise ValueError("End time must exceed start time.")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot open video.")
    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate.")
        directory = Path(config.get("output_directory", "data")) / (
            "video_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        directory.mkdir(parents=True, exist_ok=False)
        settings = config["measurement"]
        radius = int(settings.get("roi_radius_px", 8))
        offset = int(settings.get("roi_offset_px", 40))
        times, samples = [], []
        index, previous_time = 0, -1.0
        points, shape, actual_center = None, None, None
        timestamp_fallbacks = 0
        preview_stride = max(1, round(fps / 8))
        raw_path = directory / "raw.csv"
        with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["frame_index", "video_time_s"] + list(ROI_NAMES))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                # Prefer source presentation time; frame/FPS is an explicit fallback.
                stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not np.isfinite(stamp) or stamp < 0 or stamp <= previous_time:
                    stamp = index / fps
                    timestamp_fallbacks += 1
                previous_time = stamp
                current_index = index
                index += 1
                if stamp < start_s:
                    continue
                if end_s is not None and stamp >= end_s:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if shape is None:
                    shape = gray.shape
                    requested = center or (shape[1] // 2, shape[0] // 2)
                    actual_center, points, offset = roi_positions(requested, shape, offset, radius)
                elif gray.shape != shape:
                    raise ValueError("Video frame dimensions changed.")
                values = sample_rois(gray, points, radius)
                times.append(stamp)
                samples.append(values)
                csv_writer.writerow([current_index, stamp] + values)
                if export_preview and (len(times) - 1) % preview_stride == 0:
                    # Markers must not be fed into thresholding or morphology either.
                    image = cv2.imdecode(np.frombuffer(encode_comparison(frame), np.uint8), cv2.IMREAD_COLOR)
                    scale = min(1.0, 640 / frame.shape[1], 480 / frame.shape[0])
                    for number, point in enumerate(points, 1):
                        marker = (int(point[0] * scale), 24 + int(point[1] * scale))
                        cv2.circle(image, marker, max(1, int(radius * scale)), (0, 255, 255), 1)
                        cv2.putText(image, str(number), marker, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 255, 255), 1)
                    if writer is None:
                        writer = cv2.VideoWriter(str(directory / "comparison.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps / preview_stride,
                            (image.shape[1], image.shape[0]))
                        if not writer.isOpened():
                            raise ValueError("Cannot write comparison MP4 with this OpenCV build.")
                    writer.write(image)
                if progress and len(times) % 300 == 0:
                    progress(len(times), stamp)
        minimum = int(settings.get("minimum_samples", 60))
        parameters = {"minimum_samples": minimum,
            "safe_fringe_fraction_of_fps": float(settings.get("safe_fringe_fraction_of_fps", 0.25))}
        channels = {}
        for number, name in enumerate(ROI_NAMES):
            try:
                a = analyze_signal(times, np.asarray(samples)[:, number], **parameters)
                channels[name] = {"cycles": a.cycles_phase, "crossings": a.cycles_crossing,
                    "contrast": a.contrast, "quality": a.quality, "warnings": list(a.warnings)}
            except (ValueError, IndexError) as error:
                channels[name] = {"error": str(error)}
        merged, merge_error = None, None
        try:
            merged = analyze_rois(times, samples, **parameters)
        except ValueError as error:
            merge_error = str(error)
        reference = float(settings.get("reference_wavelength_nm", 632.8))
        displacement = None if merged is None else inferred_displacement(merged.cycles, reference)
        # This is a derived quantity, not an invented micrometer reading.
        summary = {"source_video": source.name, "frames_analyzed": len(times),
            "nominal_fps": fps, "video_start_s": times[0] if times else None,
            "video_end_s": times[-1] if times else None,
            "timestamp_fallbacks": timestamp_fallbacks,
            "roi_geometry": {"center_px": actual_center, "points_px": points,
                             "radius_px": radius, "offset_px": offset},
            "channels": channels, "merged_cycles": None if merged is None else merged.cycles,
            "accepted_rois": [] if merged is None else list(merged.accepted),
            "merge_error": merge_error, "reference_wavelength_nm": reference,
            "inferred_displacement_mm": displacement,
            "displacement_source": "derived from counted cycles and reference wavelength; not measured",
            "formula_check": None if merged is None else asdict(
                calculate_wavelength(0, displacement, merged.cycles)),
            "raw_csv": "raw.csv", "preview_video": "comparison.mp4" if export_preview else None}
        (directory / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory, summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
