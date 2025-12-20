"""
data_loader.py
Author: Moye Nyuysoni
Load HighD recordings Data
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class HighDAll:
    tracks: pd.DataFrame
    recording_meta: pd.DataFrame
    fps: float


def load_all_highd(data_root: str | Path, n_recordings: int = 60) -> HighDAll:
    root = Path(data_root)

    usecols = [
        "frame", "id", "x", "laneId",
        "xVelocity", "xAcceleration",
        "dhw", "precedingXVelocity"
    ]

    tracks_list = []
    rec_meta_list = []

    for rec in range(1, n_recordings + 1):
        r = f"{rec:02d}"
        tracks_path = root / f"{r}_tracks.csv"
        rec_meta_path = root / f"{r}_recordingMeta.csv"

        if not tracks_path.exists() or not rec_meta_path.exists():
            continue

        rm = pd.read_csv(rec_meta_path)
        rm["rec"] = rec
        rec_meta_list.append(rm)

        df = pd.read_csv(tracks_path, usecols=lambda c: c in usecols)
        df["rec"] = rec
        df["id"] = (df["rec"].astype(np.int64) * 1_000_000 + df["id"].astype(np.int64)).astype(np.int64)
        tracks_list.append(df)

    tracks = pd.concat(tracks_list, ignore_index=True)
    recording_meta = pd.concat(rec_meta_list, ignore_index=True)

    fps = float(np.median(recording_meta["frameRate"].values)) if "frameRate" in recording_meta.columns else 25.0
    return HighDAll(tracks=tracks, recording_meta=recording_meta, fps=fps)
