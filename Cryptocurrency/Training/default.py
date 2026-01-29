import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np

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

def load_df(symbol):
    df = pd.read_parquet(fr"M:\Dataset\{symbol}\Label\{symbol}_patterns.parquet")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def window_already_trained(symbol, start, end):
    if not os.path.exists(LOG_FILE):
        return False
    log = pd.read_csv(LOG_FILE)
    key = (log.symbol == symbol) & (log.start == str(start)) & (log.end == str(end))
    return key.any()

def append_log(row: dict):
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

def evaluate(model, loader):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.long)

            logits = model(X)[:, 0, :]
            ps.append(logits.argmax(dim=1).cpu())
            ys.append(y.cpu())

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

    ckpt_path = os.path.join(CKPT_DIR, f"{symbol}_MUON_SLIDE.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print("[✓] Loaded model weights")

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

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
        )

        eval_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        )

        # ---------- class weights (FLOAT32 SAFE) ----------
        c = np.bincount(dataset.y, minlength=3).astype(np.float32)
        w = torch.tensor(1.0 / (c + 1e-6), device=DEVICE, dtype=torch.float32)
        w = w / w.sum() * 3.0

        ce = nn.CrossEntropyLoss(weight=w, reduction="none")

        # ---------- reset optimizer per window ----------
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)

        print(f"\n📆 {start.date()} → {w_end.date()}  samples={len(dataset)}")

        for ep in range(epochs_per_window):
            model.train()
            total_loss = 0.0

            for X, y, r in train_loader:
                X = X.to(DEVICE, dtype=torch.float32)
                y = y.to(DEVICE, dtype=torch.long)
                r = r.to(DEVICE, dtype=torch.float32)

                logits = model(X)[:, 0, :].float()

                loss_raw = ce(logits, y)

                scale = torch.clamp(1.0 + 2.0 * r.abs(), max=3.0)
                loss = (scale * loss_raw).mean()

                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                loss = loss - 0.01 * entropy

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                total_loss += loss.item()

            print(f"  Epoch {ep+1}/{epochs_per_window}  Loss={total_loss/len(train_loader):.6f}")

        prec, rec, f1, report = evaluate(model, eval_loader)
        print(report)

        torch.save({"model": model.state_dict()}, ckpt_path)

        append_log({
            "symbol": symbol,
            "start": str(start.date()),
            "end": str(w_end.date()),
            "samples": len(dataset),
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        with open(os.path.join(LOG_DIR, f"{symbol}_{start.date()}_{w_end.date()}.json"), "w") as f:
            json.dump({"classification_report": report}, f, indent=2)

        start += pd.Timedelta(days=step_days)

if __name__ == "__main__":
    train_DITRM_sliding_FIXED(
        symbol="WLDUSDC",
        window_days=7,
        step_days=1,
        epochs_per_window=30,
        dim=32,
        batch_size=1024,
        lr=6e-4
    )
