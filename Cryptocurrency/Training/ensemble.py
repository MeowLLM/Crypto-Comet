import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np
from datetime import datetime

from config.process import PatternDataset
from config.DITRM import DeepImprovementTRM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

LOG_DIR = "train_logs"
CKPT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "sliding_train_log.csv")

# ============================================================
def load_df(symbol):
    df = pd.read_parquet(fr"M:\Dataset\{symbol}\Label\{symbol}_patterns.parquet")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

# ============================================================
def window_ckpt(symbol, start, end):
    return os.path.join(CKPT_DIR, f"{symbol}_{start}_{end}.pt")

# ============================================================
def window_already_trained(symbol, start, end):
    if not os.path.exists(LOG_FILE):
        return False
    log = pd.read_csv(LOG_FILE)
    key = (
        (log.symbol == symbol)
        & (log.start == str(start))
        & (log.end == str(end))
    )
    return key.any()

def append_log(row: dict):
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

def evaluate_single(model, loader):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(DEVICE, dtype=torch.float32)
            logits = model(X)[:, 0, :]
            ps.append(logits.argmax(dim=1).cpu())
            ys.append(y)

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p, average="macro", zero_division=0
    )

    return prec, rec, f1, classification_report(y, p, zero_division=0)

def load_ensemble_models(symbol, feat_dim, dim, max_models=10):
    if not os.path.exists(LOG_FILE):
        return []

    log = pd.read_csv(LOG_FILE)
    log = log[log.symbol == symbol]
    log = log.sort_values("end", ascending=False).head(max_models)

    models = []
    now = pd.Timestamp.utcnow()

    for _, r in log.iterrows():
        ckpt = window_ckpt(symbol, r.start, r.end)
        if not os.path.exists(ckpt):
            continue

        m = DeepImprovementTRM(
            vocab_size=3,
            dim=dim,
            input_dim=feat_dim,
            num_kv=8,
            num_outer_steps=18,
            num_latent_steps=18,
            moe_experts_dim=[[4,8,18],[2,4,8,16],[8,16,32],[16,32,64],[32,64]],
            moe_latent_dim=0.5,
            moe_top_k=2,
            moe_routing="topk",
            moe_shared=True,
            moe_use_core=True,
            moe_use_erc_loss=False,
            moe_cost_lambda=5e-4,
        ).to(DEVICE)

        state = torch.load(ckpt, map_location=DEVICE)
        m.load_state_dict(state["model"])
        m.eval()

        age = (now - pd.to_datetime(r.end)).days

        models.append({
            "model": m,
            "f1": float(r.f1),
            "age": max(age, 0)
        })

    return models

@torch.no_grad()
def ensemble_logits(models, X, decay=0.05):
    logits_sum = 0.0
    weight_sum = 0.0

    for m in models:
        w = m["f1"] * np.exp(-decay * m["age"])
        logits = m["model"](X)[:, 0, :]
        logits_sum += w * logits
        weight_sum += w

    return logits_sum / (weight_sum + 1e-8)

def evaluate_ensemble(models, loader):
    ys, ps = [], []

    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(DEVICE, dtype=torch.float32)
            logits = ensemble_logits(models, X)
            ps.append(logits.argmax(dim=1).cpu())
            ys.append(y)

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p, average="macro", zero_division=0
    )

    return prec, rec, f1, classification_report(y, p, zero_division=0)

def train_DITRM_sliding_FIXED(
    symbol="WLDUSDC",
    window_days=7,
    step_days=1,
    epochs_per_window=30,
    horizon=10,
    dim=32,
    batch_size=1024,
    lr=6e-4
):
    df = load_df(symbol)
    full_ds = PatternDataset(df, horizon)
    feat_dim = full_ds[0][0].shape[0]

    model = DeepImprovementTRM(
        vocab_size=3,
        dim=dim,
        input_dim=feat_dim,
        num_kv=8,
        num_outer_steps=18,
        num_latent_steps=18,
        moe_experts_dim=[[4,8,18],[2,4,8,16],[8,16,32],[16,32,64],[32,64]],
        moe_latent_dim=0.5,
        moe_top_k=2,
        moe_routing="topk",
        moe_shared=True,
        moe_use_core=True,
        moe_use_erc_loss=False,
        moe_cost_lambda=5e-4,
    ).to(DEVICE)

    start, end = df.date.min(), df.date.max()
    print(f"[Sliding] {symbol} {start} → {end}")

    while start < end:
        w_end = start + pd.Timedelta(days=window_days)

        if window_already_trained(symbol, start.date(), w_end.date()):
            start += pd.Timedelta(days=step_days)
            continue

        chunk = df[(df.date >= start) & (df.date < w_end)]
        if len(chunk) < 1200:
            start += pd.Timedelta(days=step_days)
            continue

        dataset = PatternDataset(chunk, horizon)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
        )

        c = np.bincount(dataset.y, minlength=3).astype(np.float32)
        w = torch.tensor(1.0 / (c + 1e-6), device=DEVICE)
        w = w / w.sum() * 3.0

        ce = nn.CrossEntropyLoss(weight=w, reduction="none")
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)

        print(f"\n📆 {start.date()} → {w_end.date()}  samples={len(dataset)}")

        for ep in range(epochs_per_window):
            model.train()
            for X, y, r in loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                r = r.to(DEVICE)

                logits = model(X)[:, 0, :]
                loss_raw = ce(logits, y)
                scale = torch.clamp(1 + 2 * r.abs(), max=3)
                loss = (scale * loss_raw).mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

        ckpt = window_ckpt(symbol, start.date(), w_end.date())
        torch.save({"model": model.state_dict()}, ckpt)

        ensemble = load_ensemble_models(symbol, feat_dim, dim)

        if len(ensemble) >= 3:
            prec, rec, f1, report = evaluate_ensemble(ensemble, loader)
            print("[ENSEMBLE]")
        else:
            prec, rec, f1, report = evaluate_single(model, loader)
            print("[SINGLE]")

        print(report)

        append_log({
            "symbol": symbol,
            "start": str(start.date()),
            "end": str(w_end.date()),
            "samples": len(dataset),
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        start += pd.Timedelta(days=step_days)


if __name__ == "__main__":
    train_DITRM_sliding_FIXED()
