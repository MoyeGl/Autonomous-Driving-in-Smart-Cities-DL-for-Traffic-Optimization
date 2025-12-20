"""
optimizer.py
Author: Moye Nyuysoni
Step 3:  speed advisory choice using predicted speed + lane-change risk.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VSLConfig:
    candidates_kmh: tuple[int, ...] = (60, 80, 100, 120)
    max_change_kmh: int = 20
    risk_thresh: float = 0.5


def choose_speed_limit(current_kmh: int, pred_speed_ms: float, lc_risk: float, cfg: VSLConfig = VSLConfig()) -> int:
    pred_kmh = pred_speed_ms * 3.6
    target = int(round(pred_kmh))

    if lc_risk >= cfg.risk_thresh:
        target = min(target, current_kmh)

    target = max(cfg.candidates_kmh[0], min(cfg.candidates_kmh[-1], target))
    target = max(current_kmh - cfg.max_change_kmh, min(current_kmh + cfg.max_change_kmh, target))

    return min(cfg.candidates_kmh, key=lambda s: abs(s - target))
