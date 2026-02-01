# ============================================================
# FULLY PYTHON — LOAD & EVALUATE SAVED MODELS
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# ============================================================
# 0. Environment
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Paths
# ============================================================

DATA_FILE = r"C:\Users\maste\OneDrive\Desktop\Crypto-Omni\archive\btc_15m_data_2018_to_2025.csv"
CKPT_DIR  = r"C:\Users\maste\OneDrive\Desktop\Crypto-Omni\archive\checkpoints"

LSTM_CKPT = os.path.join(CKPT_DIR, "lstm_best.pt")
TRM_CKPT  = os.path.join(CKPT_DIR, "trm_best.pt")

# ============================================================
# 2. Load & preprocess data (same as training)
# ============================================================

df = pd.read_csv(DATA_FILE)
df = df.drop(columns=["Open time", "Close time", "Ignore"], errors="ignore")
df = df.astype(float)

scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df)

FEATURES = df.drop(columns=["Close"])
TARGET   = df["Close"]

# ============================================================
# 3. Sequence builder
# ============================================================

def make_sequences(X, y, lookback=64, horizon=1):
    xs, ys = [], []
    for i in range(lookback, len(X) - horizon):
        xs.append(X.iloc[i - lookback:i].values)
        ys.append(y.iloc[i + horizon])
    return np.array(xs), np.array(ys)

LOOKBACK = 64
HORIZON  = 1

X_seq, y_seq = make_sequences(FEATURES, TARGET, LOOKBACK, HORIZON)

# ============================================================
# 4. Temporal split
# ============================================================

split = int(len(X_seq) * 0.8)
X_test = X_seq[split:]
y_test = y_seq[split:]

# ============================================================
# 5. Dataset
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_loader = DataLoader(
    TimeSeriesDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

# ============================================================
# 6. Models
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

from config.DITRM import DeepImprovementTRM

lstm = LSTMModel(X_test.shape[2]).to(DEVICE)
trm  = DeepImprovementTRM(
    vocab_size=1,
    dim=128,
    input_dim=X_test.shape[2],
    num_outer_steps=2,
    num_latent_steps=3,
    moe_experts_dim=[[64, 32], [128], [256, 128]],
    moe_top_k=2,
    moe_cost_lambda=5e-4
).to(DEVICE)

# ============================================================
# 7. Load checkpoints
# ============================================================

lstm.load_state_dict(torch.load(LSTM_CKPT, map_location=DEVICE)["model_state"])
trm.load_state_dict(torch.load(TRM_CKPT, map_location=DEVICE)["model_state"])

lstm.eval()
trm.eval()

# ============================================================
# 8. Evaluation functions
# ============================================================

def evaluate_rmse(model):
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            p = model(x)

            if p.dim() == 3:
                p = p[:, -1, 0]
            elif p.dim() == 2:
                p = p[:, 0]

            preds.extend(p.cpu().numpy())
            trues.extend(y.numpy())

    return mean_squared_error(trues, preds)

def to_direction(y, thr=0.002):
    y = np.asarray(y).reshape(-1)
    dy = np.diff(y, prepend=y[0])
    return np.where(dy > thr, 2,
           np.where(dy < -thr, 0, 1))

def evaluate_direction(model):
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            p = model(x)

            if p.dim() == 3:
                p = p[:, -1, 0]
            elif p.dim() == 2:
                p = p[:, 0]

            preds.extend(p.cpu().numpy())
            trues.extend(y.numpy())

    return accuracy_score(
        to_direction(trues),
        to_direction(preds)
    )

# ============================================================
# 9. Run evaluation
# ============================================================

print("\n=== LOADED MODEL EVALUATION ===")
print("LSTM RMSE:", evaluate_rmse(lstm))
print("TRM  RMSE:", evaluate_rmse(trm))

print("\n=== DIRECTION ACCURACY ===")
print("LSTM Direction Acc:", evaluate_direction(lstm))
print("TRM  Direction Acc:", evaluate_direction(trm))

# ============================================================
# 10. Output shape sanity check
# ============================================================

with torch.no_grad():
    x, _ = next(iter(test_loader))
    print("\nTRM output shape:", tuple(trm(x.to(DEVICE)).shape))
