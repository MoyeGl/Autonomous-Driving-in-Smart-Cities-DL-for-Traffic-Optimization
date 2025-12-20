"""
sequences.py
Author: Moye Nyuysoni
Lazy datasets for (speed forecasting + lane-change activity prediction).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class StandardScaler:
    mu: np.ndarray
    sig: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mu = x.mean(axis=0).astype(np.float32)
        sig = x.std(axis=0).astype(np.float32)
        sig = np.where(sig < 1e-6, 1.0, sig).astype(np.float32)
        return cls(mu=mu, sig=sig)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu) / self.sig).astype(np.float32)


def split_by_group(df, train_frac: float = 0.8, seed: int = 42):
    rng = np.random.default_rng(seed)
    keys = df[["laneId", "segment"]].drop_duplicates().to_numpy()
    rng.shuffle(keys)
    cut = int(round(train_frac * len(keys)))
    tr = set(map(tuple, keys[:cut]))
    va = set(map(tuple, keys[cut:]))

    m_tr = df.apply(lambda r: (int(r["laneId"]), int(r["segment"])) in tr, axis=1)
    return df[m_tr].copy(), df[~m_tr].copy()


class LazyTrafficDataset(Dataset):
    def __init__(
        self,
        df,
        seq_len: int,
        horizon: int,
        feature_cols: Sequence[str],
        speed_col: str = "speed_mean",
        lc_col: str = "lc_rate",
        x_scaler: StandardScaler | None = None,
        stride: int = 1,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        import pandas as pd

        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.feature_cols = list(feature_cols)
        self.speed_col = speed_col
        self.lc_col = lc_col
        self.stride = max(1, int(stride))

        data = df.sort_values(["laneId", "segment", "tbin"]).copy()

        for c in ["tbin", "laneId", "segment"] + self.feature_cols + [self.speed_col, self.lc_col]:
            if c not in data.columns:
                data[c] = 0.0

        data[self.feature_cols] = data[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        data[self.speed_col] = data[self.speed_col].astype(np.float32)
        data[self.lc_col] = data[self.lc_col].astype(np.float32)

        self.x_scaler = x_scaler or StandardScaler.fit(data[self.feature_cols].values.astype(np.float32))

        self.groups: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.index: List[Tuple[int, int]] = []

        for _, g in data.groupby(["laneId", "segment"], sort=False):
            X = self.x_scaler.transform(g[self.feature_cols].values.astype(np.float32))
            ys = g[self.speed_col].values.astype(np.float32)
            yl = g[self.lc_col].values.astype(np.float32)

            n = len(g)
            end = n - self.seq_len - self.horizon + 1
            if end <= 0:
                continue

            gi = len(self.groups)
            self.groups.append((X, ys, yl))
            for start in range(0, end, self.stride):
                self.index.append((gi, start))

        if max_samples is not None and max_samples < len(self.index):
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(self.index), size=max_samples, replace=False)
            self.index = [self.index[i] for i in pick.tolist()]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gi, start = self.index[idx]
        X, ys, yl = self.groups[gi]

        x_seq = X[start : start + self.seq_len]
        y_speed = ys[start + self.seq_len + self.horizon - 1]

        future_lc = yl[start + self.seq_len : start + self.seq_len + self.horizon]
        y_lc = 1 if future_lc.max() > 0.0 else 0

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_speed, dtype=torch.float32),
            torch.tensor(y_lc, dtype=torch.int64),
        )
