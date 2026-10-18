"""Clean raw accelerometer CSV files and build TF-C train/test tensors.

Expected columns are: timestamp, ax, ay, az, label, recording_id.  The script
also accepts x/y/z and accel_x/accel_y/accel_z aliases. One recording must
contain one gesture label. Splitting happens by recording before segmentation,
which prevents neighbouring windows from leaking into both splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


AXIS_ALIASES = {
    "ax": ("ax", "x", "accel_x", "accelerometer_x"),
    "ay": ("ay", "y", "accel_y", "accelerometer_y"),
    "az": ("az", "z", "accel_z", "accelerometer_z"),
}


def _find_column(columns, candidates):
    lookup = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def read_recordings(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(raw_dir.rglob("*.csv")):
        frame = pd.read_csv(csv_path)
        rename = {}
        for target, aliases in AXIS_ALIASES.items():
            source = _find_column(frame.columns, aliases)
            if source is None:
                raise ValueError(f"{csv_path}: missing axis column {target}")
            rename[source] = target
        frame = frame.rename(columns=rename)
        if "label" not in frame.columns:
            frame["label"] = csv_path.parent.name
        if "recording_id" not in frame.columns:
            frame["recording_id"] = str(csv_path.relative_to(raw_dir).with_suffix(""))
        frame["recording_id"] = frame["recording_id"].astype(str)
        frame["label"] = frame["label"].astype(str)
        frame["source_file"] = str(csv_path.relative_to(raw_dir))
        frames.append(frame[["ax", "ay", "az", "label", "recording_id", "source_file"]])
    if not frames:
        raise ValueError(f"No CSV files found under {raw_dir}")
    data = pd.concat(frames, ignore_index=True)
    for axis in ("ax", "ay", "az"):
        data[axis] = pd.to_numeric(data[axis], errors="coerce")
    data[["ax", "ay", "az"]] = data[["ax", "ay", "az"]].replace([np.inf, -np.inf], np.nan)
    data[["ax", "ay", "az"]] = data.groupby("recording_id")[["ax", "ay", "az"]].transform(
        lambda group: group.interpolate(limit_direction="both")
    )
    data = data.dropna(subset=["ax", "ay", "az", "label", "recording_id"])
    return data


def split_recordings(data: pd.DataFrame, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    recording_labels = data.groupby("recording_id")["label"].agg(lambda values: values.mode().iat[0])
    for label in sorted(recording_labels.unique()):
        ids = recording_labels[recording_labels == label].index.to_numpy(copy=True)
        if len(ids) < 2:
            raise ValueError(f"Label '{label}' needs at least 2 separate recordings")
        rng.shuffle(ids)
        test_count = max(1, int(round(len(ids) * test_ratio)))
        test_count = min(test_count, len(ids) - 1)
        test_ids.extend(ids[:test_count].tolist())
        train_ids.extend(ids[test_count:].tolist())
    return set(train_ids), set(test_ids)


def fit_normalization(data: pd.DataFrame, train_ids: set[str]):
    train = data[data["recording_id"].isin(train_ids)][["ax", "ay", "az"]]
    low = train.quantile(0.005).to_numpy(dtype=np.float32)
    high = train.quantile(0.995).to_numpy(dtype=np.float32)
    clipped = np.clip(train.to_numpy(dtype=np.float32), low, high)
    mean = clipped.mean(axis=0)
    std = np.maximum(clipped.std(axis=0), 1e-6)
    return low, high, mean, std


def make_windows(data, recording_ids, label_to_id, window_size, stride, stats):
    low, high, mean, std = stats
    samples, labels = [], []
    for recording_id in sorted(recording_ids):
        recording = data[data["recording_id"] == recording_id]
        unique_labels = recording["label"].astype(str).unique()
        if len(unique_labels) != 1:
            raise ValueError(f"Recording '{recording_id}' contains multiple labels")
        values = recording[["ax", "ay", "az"]].to_numpy(dtype=np.float32)
        values = (np.clip(values, low, high) - mean) / std
        for start in range(0, len(values) - window_size + 1, stride):
            samples.append(values[start:start + window_size].T)
            labels.append(label_to_id[unique_labels[0]])
    if not samples:
        raise ValueError("No windows were created; recordings may be shorter than window_size")
    return {
        "samples": torch.tensor(np.stack(samples), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/Gesture"))
    parser.add_argument("--window-size", type=int, default=206)
    parser.add_argument("--stride", type=int, default=103)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = read_recordings(args.raw_dir)
    train_ids, test_ids = split_recordings(data, args.test_ratio, args.seed)
    class_names = sorted(data["label"].astype(str).unique().tolist())
    if len(class_names) != 8:
        raise ValueError(f"Expected exactly 8 gesture labels, found {len(class_names)}: {class_names}")
    label_to_id = {name: index for index, name in enumerate(class_names)}
    stats = fit_normalization(data, train_ids)
    train = make_windows(data, train_ids, label_to_id, args.window_size, args.stride, stats)
    test = make_windows(data, test_ids, label_to_id, args.window_size, args.stride, stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, args.output_dir / "train.pt")
    torch.save(test, args.output_dir / "test.pt")
    low, high, mean, std = stats
    metadata = {
        "class_names": class_names,
        "label_to_id": label_to_id,
        "window_size": args.window_size,
        "stride": args.stride,
        "channels": ["ax", "ay", "az"],
        "normalization": {"clip_low": low.tolist(), "clip_high": high.tolist(), "mean": mean.tolist(), "std": std.tolist()},
        "train_recordings": sorted(train_ids),
        "test_recordings": sorted(test_ids),
        "train_windows": len(train["labels"]),
        "test_windows": len(test["labels"]),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
