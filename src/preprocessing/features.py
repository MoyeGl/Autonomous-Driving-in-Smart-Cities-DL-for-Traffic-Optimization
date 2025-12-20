"""
features.py
Author: Moye Nyuysoni
Aggregate HighD tracks into lane-segment traffic states.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AggConfig:
    segment_len_m: float = 100.0
    dt_s: float = 1.0
    clip_speed: float = 70.0


def aggregate_lane_segment(tracks, fps: float, cfg: AggConfig = AggConfig()):
    import pandas as pd

    df = tracks.copy()
    df = df.sort_values(["id", "frame"])

    df["t"] = (df["frame"] / float(fps)).astype(np.float32)
    step = max(1, int(round(cfg.dt_s * float(fps))))
    df["tbin"] = (df["frame"] // step).astype(np.int64)

    if "laneId" not in df.columns:
        df["laneId"] = 0

    x = pd.to_numeric(df.get("x", 0.0), errors="coerce").fillna(0.0)
    df["segment"] = np.floor(x / float(cfg.segment_len_m)).astype(np.int64)

    v = pd.to_numeric(df.get("xVelocity", 0.0), errors="coerce").fillna(0.0).abs()
    df["speed"] = v.clip(0.0, float(cfg.clip_speed)).astype(np.float32)

    df["lane_change"] = (df.groupby("id")["laneId"].diff().fillna(0).abs() > 0).astype(np.int32)

    g = df.groupby(["tbin", "laneId", "segment"], sort=False)

    out = g.agg(
        speed_mean=("speed", "mean"),
        veh_count=("id", "nunique"),
        lc_rate=("lane_change", "mean"),
    ).reset_index()

    out["density"] = (out["veh_count"].astype(np.float32) / float(cfg.segment_len_m)) * 1000.0  # veh/km
    out["flow"] = out["veh_count"].astype(np.float32) / float(cfg.dt_s)  # veh/s

    out = out.sort_values(["laneId", "segment", "tbin"]).reset_index(drop=True)
    return out
