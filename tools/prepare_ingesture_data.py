"""Convert official InGesture segmented CSV files to TF-C tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


GESTURES = {
    "other": 0,
    "fluid_intake": 1,
    "answering_cell_phone": 2,
    "scratching_head": 3,
    "passing_hand_over_face": 4,
    "adjusting_glasses_touching_temples": 5,
    "holding_chin": 6,
    "stretching_with_hands_behind_neck": 7,
}


def segmented_files(root: Path):
    files = []
    for path in sorted(root.glob("subject_*/*__*.csv")):
        gesture = path.stem.split("__", 1)[0]
        if gesture in GESTURES:
            files.append((path, path.parent.name, gesture))
    if not files:
        raise ValueError(f"No segmented InGesture CSV files found under {root}")
    return files


def split_subjects(files, test_ratio: float, seed: int):
    subjects = sorted({subject for _, subject, _ in files}, key=lambda value: int(value.split("_")[-1]))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    test_count = max(1, int(round(len(subjects) * test_ratio)))
    return set(subjects[test_count:]), set(subjects[:test_count])


def read_xyz(path: Path):
    frame = pd.read_csv(path, usecols=["x", "y", "z"])
    values = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(limit_direction="both").dropna()
    return values.to_numpy(dtype=np.float32)


def fit_stats(files, train_subjects, max_points=1_000_000, seed=42):
    chunks = [read_xyz(path) for path, subject, _ in files if subject in train_subjects]
    values = np.concatenate(chunks, axis=0)
    if len(values) > max_points:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), size=max_points, replace=False)]
    low = np.quantile(values, 0.005, axis=0).astype(np.float32)
    high = np.quantile(values, 0.995, axis=0).astype(np.float32)
    clipped = np.clip(values, low, high)
    return low, high, clipped.mean(axis=0), np.maximum(clipped.std(axis=0), 1e-6)


def build_split(files, subjects, stats, window_size, stride):
    low, high, mean, std = stats
    samples, labels = [], []
    per_class = {name: 0 for name in GESTURES}
    for path, subject, gesture in files:
        if subject not in subjects:
            continue
        values = (np.clip(read_xyz(path), low, high) - mean) / std
        for start in range(0, len(values) - window_size + 1, stride):
            samples.append(values[start:start + window_size].T)
            labels.append(GESTURES[gesture])
            per_class[gesture] += 1
    if not samples:
        raise ValueError("No windows created from the selected subjects")
    return {
        "samples": torch.tensor(np.stack(samples), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/InGesture"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/InGesture"))
    parser.add_argument("--window-size", type=int, default=206)
    parser.add_argument("--stride", type=int, default=103)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="Fraction of all subjects for validation; held out before normalization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_ratio < 1 or not 0 <= args.val_ratio < 1 - args.test_ratio:
        parser.error("Ratios must leave at least one training subject")

    files = segmented_files(args.input_dir)
    train_subjects, test_subjects = split_subjects(files, args.test_ratio, args.seed)
    val_subjects = set()
    if args.val_ratio:
        candidates = sorted(train_subjects)
        np.random.default_rng(args.seed + 1).shuffle(candidates)
        n_val = max(1, round(len({s for _, s, _ in files}) * args.val_ratio))
        val_subjects = set(candidates[:n_val])
        train_subjects -= val_subjects
        if not train_subjects:
            raise ValueError("Validation split leaves no training subjects")
    stats = fit_stats(files, train_subjects, seed=args.seed)
    train, train_counts = build_split(files, train_subjects, stats, args.window_size, args.stride)
    test, test_counts = build_split(files, test_subjects, stats, args.window_size, args.stride)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, args.output_dir / "train.pt")
    torch.save(test, args.output_dir / "test.pt")
    if val_subjects:
        val, val_counts = build_split(files, val_subjects, stats, args.window_size, args.stride)
        torch.save(val, args.output_dir / "val.pt")
    low, high, mean, std = stats
    class_names = [name for name, _ in sorted(GESTURES.items(), key=lambda item: item[1])]
    metadata = {
        "dataset": "InGesture v5",
        "doi": "10.17632/fdxst56tcj.5",
        "license": "CC BY 4.0",
        "sensor": "WT901BLECL5 wrist-worn IMU",
        "sampling_rate_hz": 200,
        "class_names": class_names,
        "label_to_id": GESTURES,
        "window_size": args.window_size,
        "stride": args.stride,
        "channels": ["x", "y", "z"],
        "normalization": {"clip_low": low.tolist(), "clip_high": high.tolist(), "mean": mean.tolist(), "std": std.tolist()},
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "train_windows": len(train["labels"]),
        "test_windows": len(test["labels"]),
        "train_windows_by_class": train_counts,
        "test_windows_by_class": test_counts,
        "validation_subjects": sorted(val_subjects),
        "validation_windows": len(val["labels"]) if val_subjects else 0,
        "validation_windows_by_class": val_counts if val_subjects else {},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
