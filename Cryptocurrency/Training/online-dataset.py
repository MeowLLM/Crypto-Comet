import os
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, precision_recall_fscore_support

from config.DITRM import (DeepImprovementTRM)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 32
BATCH_SIZE = 128
EPOCHS = 30
LR = 3e-4
EPS = 0.002

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

ds = load_dataset("WinkingFace/CryptoLM-Bitcoin-BTC-USDT")
df = ds["train"].to_pandas()


df = df.sort_values("timestamp")
df["volume"] = df["volume"].astype(float)

feature_cols = [
    "open", "high", "low", "close",
    "volume",
    "MA_20", "MA_50", "MA_200",
    "RSI", "%K", "%D",
    "ADX", "ATR",
    "Trendline", "MACD",
    "BL_Upper", "BL_Lower",
    "Signal", "Histogram",
    "MN_Upper", "MN_Lower",
    "month",
]

df = df.dropna().reset_index(drop=True)

scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols].values)
close = df["close"].values


X_seq, y_seq = [], []

for i in range(len(X) - SEQ_LEN - 1):
    seq_x = X[i : i + SEQ_LEN]

    r = (close[i + SEQ_LEN + 1] - close[i + SEQ_LEN]) / close[i + SEQ_LEN]

    if r > EPS:
        y = 2  # BUY
    elif r < -EPS:
        y = 0  # SELL
    else:
        y = 1  # HOLD

    X_seq.append(seq_x)
    y_seq.append(y)

X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.long)

split = int(0.8 * len(X_seq))

X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

feat_dim = X_train.shape[-1]
dim = 128

model = DeepImprovementTRM(
    vocab_size=3,
    dim=dim,
    input_dim=feat_dim,
    num_kv=8,
    num_outer_steps=18,
    num_latent_steps=18,
    moe_experts_dim=[
        [4, 8, 18],
        [2, 4, 8, 16],
        [8, 16, 32],
        [16, 32, 64],
        [32, 64],
    ],
    moe_latent_dim=0.5,
    moe_top_k=2,
    moe_routing="topk",
    moe_shared=True,
    moe_use_core=True,
    moe_use_erc_loss=False,
    moe_cost_lambda=5e-4,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

best_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
  
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)

            all_logits.append(logits.cpu())
            all_targets.append(yb)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    y_true = all_targets.numpy()
    y_pred = all_logits.argmax(dim=1).numpy()
    prec, rec, f1s, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)
    acc = (y_pred == y_true).mean()

    if f1s > best_f1:
        best_f1 = f1s
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                 "f1": f1s,
                "accuracy": acc,
                "SEQ_LEN": SEQ_LEN,
                "EPS": EPS,
                "features": feature_cols,
            },
            os.path.join(SAVE_DIR, "best_model.pt"),
        )

    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, f"last_epoch_{epoch}.pt"),
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Acc {acc:.4f} | "
        f"F1 {f1s:.4f} | "
        f"Prec {prec:.4f} | "
        f"Recall {rec:.4f}"
    )

# ckpt = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
# model.load_state_dict(ckpt["model_state_dict"])
# model.eval()
