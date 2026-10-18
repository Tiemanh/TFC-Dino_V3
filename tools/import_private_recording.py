"""Import one real phone recording, preserving its explicit gesture label.

Example: python tools/import_private_recording.py --input take.csv
  --label waving_left --participant p01 --take 001 --units ms2
Column arguments support phone export formats such as phyphox.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def import_recording(input_path, output_root, label, participant, take,
                     columns=("timestamp", "x", "y", "z"), units="ms2", target_hz=200):
    for value in (label, participant, take):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Use letters, numbers, underscores or hyphens for IDs and labels")
    if units not in ("ms2", "g") or target_hz <= 0:
        raise ValueError("Units must be ms2 or g, and target rate must be positive")
    frame = pd.read_csv(input_path, sep=None, engine="python")
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Required columns: {columns}; found: {list(frame.columns)}")
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    values.columns = ["timestamp", "x", "y", "z"]
    values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp"])
    values = values.sort_values("timestamp").drop_duplicates("timestamp")
    values[["x", "y", "z"]] = values[["x", "y", "z"]].interpolate(limit_direction="both")
    values = values.dropna()
    if len(values) < 2:
        raise ValueError("Recording needs at least two finite timestamps and all three axes")
    times = values["timestamp"].to_numpy()
    times = times - times[0]  # Input timestamps must be seconds.
    duration = float(times[-1])
    if duration <= 0 or duration > 300:
        raise ValueError("Recordings must last 0–300 seconds; check timestamp units are seconds")
    grid = np.arange(0, duration, 1.0 / target_hz)
    if len(grid) < 206:
        raise ValueError("Record a longer gesture: need at least 206 resampled points")
    # Standardize physical units only; statistical normalization happens on train later.
    output = pd.DataFrame({"timestamp": grid})
    for axis in ("x", "y", "z"):
        output[axis] = np.interp(grid, times, values[axis]) / (9.80665 if units == "ms2" else 1)
    recording_id = f"{participant}_{label}_{take}"
    output["label"] = label
    output["recording_id"] = recording_id
    output["participant_id"] = participant
    destination = Path(output_root) / label / f"{recording_id}.csv"
    destination.parent.mkdir(exist_ok=True, parents=True)
    if destination.exists():
        raise FileExistsError(f"Recording already imported: {destination}")
    output.to_csv(destination, index=False)
    provenance = {"source_type": "user_recorded", "label": label, "recording_id": recording_id,
                  "participant_id": participant, "original_filename": Path(input_path).name,
                  "original_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                  "sampling_rate_hz": target_hz, "output_acceleration_units": "g",
                  "input_units": units, "samples": len(output), "duration_seconds": duration}
    destination.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--take", required=True)
    parser.add_argument("--units", choices=("ms2", "g"), required=True)
    parser.add_argument("--target-hz", type=int, default=200)
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--z-column", default="z")
    args = parser.parse_args()
    print(import_recording(args.input, args.output_dir, args.label, args.participant, args.take,
                           (args.time_column, args.x_column, args.y_column, args.z_column),
                           args.units, args.target_hz))


if __name__ == "__main__":
    main()
