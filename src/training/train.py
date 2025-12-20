"""
train.py
Author: Moye Nyuysoni
Multi-task training: speed (regression) + lane-change activity (classification).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 4
    min_delta: float = 1e-4
    num_workers: int = 0
    lc_weight: float = 1.0
    pos_weight_cap: float = 30.0
    pos_weight_probe: int = 100_000


def _estimate_pos_weight(ds, max_items: int, seed: int = 42) -> float:
    n = len(ds)
    k = min(max_items, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)

    pos = 0
    for i in idx:
        _, _, ylc = ds[int(i)]
        pos += int(ylc.item())
    neg = k - pos
    return float(neg / max(1, pos))


def _eval(model, loader, device: torch.device):
    model.eval()
    ys_true, ys_pred = [], []
    lc_true, lc_prob = [], []

    with torch.no_grad():
        for X, y_speed, y_lc in loader:
            X = X.to(device)

            pred_speed, lc_logits = model(X)
            pred_speed = pred_speed.cpu().numpy()
            prob = torch.sigmoid(lc_logits).cpu().numpy()

            ys_true.append(y_speed.numpy())
            ys_pred.append(pred_speed)
            lc_true.append(y_lc.numpy())
            lc_prob.append(prob)

    ys_true = np.concatenate(ys_true)
    ys_pred = np.concatenate(ys_pred)
    lc_true = np.concatenate(lc_true)
    lc_prob = np.concatenate(lc_prob)

    mae = float(np.mean(np.abs(ys_true - ys_pred)))
    rmse = float(np.sqrt(np.mean((ys_true - ys_pred) ** 2)))

    lc_pred = (lc_prob >= 0.5).astype(int)
    prec = precision_score(lc_true, lc_pred, zero_division=0)
    rec = recall_score(lc_true, lc_pred, zero_division=0)
    f1 = f1_score(lc_true, lc_pred, zero_division=0)
    ap = average_precision_score(lc_true, lc_prob) if len(np.unique(lc_true)) > 1 else float("nan")

    return mae, rmse, prec, rec, f1, ap


def train_multitask(
    model,
    train_ds,
    val_ds,
    cfg: TrainConfig = TrainConfig(),
    seed: int = 42,
    device: str | None = None
):
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    pw = _estimate_pos_weight(train_ds, max_items=cfg.pos_weight_probe, seed=seed)
    pw = float(min(cfg.pos_weight_cap, max(1.0, pw)))
    pos_weight = torch.tensor([pw], dtype=torch.float32, device=dev)

    loss_speed = torch.nn.HuberLoss(delta=1.0)
    loss_lc = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist = {k: [] for k in [
        "train_total", "train_speed", "train_lc",
        "val_speed_mae", "val_speed_rmse",
        "val_lc_prec", "val_lc_rec", "val_lc_f1", "val_lc_ap",
        "pos_weight"
    ]}

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_total, run_speed, run_lc, seen = 0.0, 0.0, 0.0, 0

        for X, y_speed, y_lc in train_loader:
            X = X.to(dev)
            y_speed = y_speed.to(dev)
            y_lc = y_lc.float().to(dev)

            opt.zero_grad(set_to_none=True)
            pred_speed, lc_logits = model(X)

            ls = loss_speed(pred_speed, y_speed)
            ll = loss_lc(lc_logits, y_lc)
            loss = ls + cfg.lc_weight * ll

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = X.size(0)
            run_total += float(loss.item()) * bs
            run_speed += float(ls.item()) * bs
            run_lc += float(ll.item()) * bs
            seen += bs

        train_total = run_total / max(1, seen)
        train_speed = run_speed / max(1, seen)
        train_lc = run_lc / max(1, seen)

        mae, rmse, prec, rec, f1, ap = _eval(model, val_loader, dev)

        hist["train_total"].append(train_total)
        hist["train_speed"].append(train_speed)
        hist["train_lc"].append(train_lc)

        hist["val_speed_mae"].append(mae)
        hist["val_speed_rmse"].append(rmse)

        hist["val_lc_prec"].append(float(prec))
        hist["val_lc_rec"].append(float(rec))
        hist["val_lc_f1"].append(float(f1))
        hist["val_lc_ap"].append(float(ap))

        hist["pos_weight"].append(float(pw))

        print(
            f"Epoch {ep:02d}/{cfg.epochs} | "
            f"train_total {train_total:.4f} | train_speed {train_speed:.4f} | train_lc {train_lc:.4f} | "
            f"val MAE {mae:.3f} RMSE {rmse:.3f} | "
            f"LC prec {prec:.3f} rec {rec:.3f} f1 {f1:.3f} AP {ap:.3f} | pw {pw:.1f}"
        )

        val_key = float(mae)  # early stopping on speed MAE
        if val_key < best_val - cfg.min_delta:
            best_val = val_key
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping (best MAE={best_val:.3f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist
