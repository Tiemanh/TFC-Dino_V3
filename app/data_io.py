"""Shared CSV/TXT parsing for the FastAPI and Streamlit frontends."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


AXIS_ALIASES = {
    "x": ("x", "ax", "accx", "accel_x", "accelerometer_x"),
    "y": ("y", "ay", "accy", "accel_y", "accelerometer_y"),
    "z": ("z", "az", "accz", "accel_z", "accelerometer_z"),
}


def parse_xyz_bytes(contents: bytes) -> np.ndarray:
    """Read a delimited text file and return finite/interpolated X/Y/Z values."""
    try:
        frame = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
    except Exception as error:
        raise ValueError(f"Không đọc được CSV/TXT: {error}") from error

    lookup = {str(column).strip().lower(): column for column in frame.columns}
    selected = {}
    for target, aliases in AXIS_ALIASES.items():
        source = next((lookup[name] for name in aliases if name in lookup), None)
        if source is None:
            raise ValueError(f"Thiếu cột trục {target.upper()}; cần X/Y/Z hoặc AX/AY/AZ")
        selected[target] = pd.to_numeric(frame[source], errors="coerce")

    xyz = pd.DataFrame(selected).replace([np.inf, -np.inf], np.nan)
    xyz = xyz.interpolate(limit_direction="both").dropna()
    if xyz.empty:
        raise ValueError("File không chứa mẫu cảm biến hợp lệ")
    return xyz.to_numpy(dtype=np.float32)
