# ============================================================
# 0. Environment
# ============================================================
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Load & clean data (NO leakage)
# ============================================================
FILE = r"C:\Users\maste\OneDrive\Desktop\Crypto-Omni\archive\btc_15m_data_2018_to_2025.csv"

SAVE_DIR = r"C:\Users\maste\OneDrive\Desktop\Crypto-Omni\archive\checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

LSTM_CKPT = os.path.join(SAVE_DIR, "lstm_best.pt")
TRM_CKPT  = os.path.join(SAVE_DIR, "trm_best.pt")

df = pd.read_csv(FILE)

# drop non-numeric / future-identifying columns
df = df.drop(columns=["Open time", "Close time", "Ignore"], errors="ignore")
df = df.astype(float)

# scale AFTER drop, BEFORE sequence construction
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df)

FEATURES = df.drop(columns=["Close"])
TARGET = df["Close"]

# ============================================================
# 2. Sequence builder (forecasting)
# X[t] = past window
# y[t] = Close(t + horizon)
# ============================================================
def make_sequences(X, y, lookback=64, horizon=1):
    xs, ys = [], []
    for i in range(lookback, len(X) - horizon):
        xs.append(X.iloc[i - lookback:i].values)
        ys.append(y.iloc[i + horizon])
    return np.array(xs), np.array(ys)

LOOKBACK = 64
HORIZON = 1

X_seq, y_seq = make_sequences(FEATURES, TARGET, LOOKBACK, HORIZON)

# ============================================================
# 3. Temporal split (NO shuffling)
# ============================================================
split = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# ============================================================
# 4. PyTorch Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
test_ds  = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# ============================================================
# 5. LSTM baseline (sanity anchor)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

# ============================================================
# 6. Your model: DeepImprovementTRM
# ============================================================
from config.DITRM import DeepImprovementTRM

trm = DeepImprovementTRM(
    vocab_size=1,                 # regression head
    dim=16,
    input_dim=X_train.shape[2],
    num_outer_steps=5,
    num_latent_steps=10,
    moe_experts_dim=[[4,8,16,], [8,16,32], [64], [32,64]],
    moe_top_k=2,
    moe_cost_lambda=5e-4
).to(DEVICE)

# ============================================================
# 8. Evaluation (RMSE)
# ============================================================
def evaluate_rmse(model):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            p = model(x)
            if p.dim() == 3:            # [B, T, 1]
                p = p[:, -1, 0]
            p = p.numpy()
            preds.extend(p)
            trues.extend(y.numpy())

    return mean_squared_error(trues, preds)

# ============================================================
# 7. Shared training loop
# ============================================================
def train(model, epochs=5,f=""):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_rmse = float("inf")

    for e in range(epochs):
        model.train()
        total = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            if pred.dim() == 3:            # [B, T, 1]
                pred = pred[:, -1, 0]      # → [B]
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()

        rmse = evaluate_rmse(model)
        
        print(f"Epoch {e+1}: loss={total/len(train_loader):.6f} | {rmse}")
        
        # 🔒 Save best model
        if f and rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "model_state": model.state_dict(),
                "rmse": rmse,
                "epoch": e + 1
            }, f)
            print(f"  ✓ Saved best model → {f}")

# ============================================================
# 9. Benchmark
# ============================================================
lstm = LSTMModel(X_train.shape[2]).to(DEVICE)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

lstm_total, lstm_trainable, lstm_frozen = count_parameters(lstm)
trm_total, trm_trainable, trm_frozen   = count_parameters(trm)

print("=== PARAMETER COUNT ===")
print(f"LSTM Total Params     : {lstm_total:,}")
print(f"LSTM Trainable Params : {lstm_trainable:,}")
print(f"LSTM Frozen Params    : {lstm_frozen:,}")

print(f"TRM Total Params      : {trm_total:,}")
print(f"TRM Trainable Params  : {trm_trainable:,}")
print(f"TRM Frozen Params     : {trm_frozen:,}")

# ============================================================
# Optional: parameter ratio
# ============================================================

print()
print(f"TRM / LSTM param ratio: {trm_total / lstm_total:.2f}x")

print("\nTraining DeepImprovementTRM")
train(trm, epochs=50,f=TRM_CKPT)
print("TRM RMSE:", evaluate_rmse(trm))

print("\nTraining LSTM")
train(lstm, epochs=50,f=LSTM_CKPT)
print("LSTM RMSE:", evaluate_rmse(lstm))

# ============================================================
# 10. Optional: Direction classification metric
# ============================================================
def to_direction(y, thr=0.002):
    dy = np.diff(y, prepend=y[0])
    return np.where(dy > thr, 2, np.where(dy < -thr, 0, 1))

def evaluate_direction(model):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            p = model(x).cpu().numpy()
            preds.extend(p)
            trues.extend(y.numpy())

    return accuracy_score(
        to_direction(np.array(trues)),
        to_direction(np.array(preds))
    )

print("LSTM Direction Acc:", evaluate_direction(lstm))
print("TRM Direction Acc:", evaluate_direction(trm))
