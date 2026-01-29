#!/usr/bin/env python3
"""
FULL PATTERN PIPELINE + PARALLEL PROCESSING + TORCH DATASET
Author: MeowLLM & o5

This file:
 - Loads raw OHLCV
 - Builds all feature sets
 - Builds candle pattern engines
 - Builds pattern momentum / velocity / stacks
 - Builds volatility-regime interactions
 - Builds OFI-orderflow interactions
 - Builds meta-pattern clusters (KMeans)
 - Builds strategy signals
 - Saves per-symbol datasets
 - Executes parallel dataset generation
 - Provides torch-ready PatternDataset
"""

import os
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

BASE = r""


def symbol_path(symbol: str) -> str:
    return os.path.join(BASE, symbol)


def label_path(symbol: str) -> str:
    p = os.path.join(BASE, symbol, "Label")
    os.makedirs(p, exist_ok=True)
    return p

def load_all_raw(symbol: str) -> pd.DataFrame:
    folder = symbol_path(symbol)
    files = glob.glob(os.path.join(folder, "*.csv")) + \
            glob.glob(os.path.join(folder, "*.json"))

    dfs = []
    for f in files:
        if f.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_json(f)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No raw files for {symbol}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret"] = df["close"].pct_change()
    df["logret"] = np.log(df["close"]).diff()
    df["vol_20"] = df["logret"].rolling(20).std()
    df["vol_50"] = df["logret"].rolling(50).std()
    df["vol_200"] = df["logret"].rolling(200).std()

    df["range"] = df["high"] - df["low"]
    df["true_range"] = np.maximum(df["high"] - df["low"],
                           np.maximum(abs(df["high"] - df["close"].shift()),
                                      abs(df["low"] - df["close"].shift())))
    df["atr_20"] = df["true_range"].rolling(20).mean()

    df["liq"] = df["volume"] * df["range"]
    df["oflow"] = (df["close"] - df["open"]) * df["volume"]

    return df

def detect_basic_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body = abs(df["close"] - df["open"])
    candle = df["high"] - df["low"]

    df["is_doji"] = (body < 0.1 * candle).astype(int)
    df["is_bigbody"] = (body > 0.6 * candle).astype(int)
    df["is_bull"] = (df["close"] > df["open"]).astype(int)
    df["is_bear"] = (df["close"] < df["open"]).astype(int)
    df["is_engulf"] = (
        (df["is_bull"] & (df["close"] > df["open"].shift()) &
         (df["open"] < df["close"].shift()))
        |
        (df["is_bear"] & (df["open"] > df["close"].shift()) &
         (df["close"] < df["open"].shift()))
    ).astype(int)

    return df

def add_future_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df[f"future_ret_{horizon}"] = df["close"].shift(-horizon) / df["close"] - 1
    return df


def add_pattern_strategy(df: pd.DataFrame, horizon: int, score_thr: float) -> pd.DataFrame:
    future = f"future_ret_{horizon}"
    df["signal"] = 0

    df.loc[(df["is_bull"] == 1) & (df[future] > score_thr), "signal"] = 1
    df.loc[(df["is_bear"] == 1) & (df[future] < -score_thr), "signal"] = -1

    return df

def add_pattern_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["pattern_mom"] = df["is_bigbody"].rolling(5).sum()
    return df


def add_pattern_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df["pattern_vel"] = df["is_engulf"].rolling(10).mean()
    return df


def add_pattern_stacks(df: pd.DataFrame) -> pd.DataFrame:
    df["pattern_stack"] = (
        df["is_bull"].rolling(3).sum() -
        df["is_bear"].rolling(3).sum()
    )
    return df


def add_pattern_ofi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["pattern_ofi"] = df["oflow"] * (df["is_bigbody"] + 1)
    return df


def add_pattern_vol_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["pattern_volmix"] = df["vol_20"] * (1 + df["pattern_vel"])
    return df


def add_pattern_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    df["pos_size"] = 1 / (1 + df["vol_20"])
    return df


def add_super_patterns(df: pd.DataFrame, embed_dim: int, n_clusters: int) -> pd.DataFrame:
    feats = ["logret", "range", "vol_20", "vol_50", "is_bigbody", "is_engulf"]
    sub = df[feats].fillna(0).values

    km = KMeans(n_clusters=n_clusters, n_init="auto")
    meta = km.fit_predict(sub)

    df["meta_pattern"] = meta
    return df

def detect_all_candle_patterns(
    df: pd.DataFrame,
    backtest_horizons=(10,),
    backtest_window=2000,
    embed_dim=32,
):
    df = detect_basic_patterns(df)
    return df

def build_full_pattern_dataset(
    df: pd.DataFrame,
    backtest_horizons=(10,),
    backtest_window=2000,
    embed_dim=32,
    score_thr=0.15,
) -> pd.DataFrame:

    df = build_features(df)
    df = detect_all_candle_patterns(df)

    # Use first horizon
    H = backtest_horizons[0]

    df = add_future_return(df, H)
    df = add_pattern_strategy(df, H, score_thr)

    df = add_pattern_stacks(df)
    df = add_pattern_velocity(df)
    df = add_pattern_momentum(df)
    df = add_pattern_ofi_interaction(df)
    df = add_pattern_vol_interaction(df)
    df = add_pattern_position_sizing(df)

    df = add_super_patterns(df, embed_dim=embed_dim, n_clusters=12)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

def process_symbol_to_disk(
    symbol: str,
    backtest_horizons=(10,),
    backtest_window=2000,
    embed_dim=32,
    score_thr=0.15,
    fmt="parquet",
):
    try:
        print(f"[{symbol}] Loading raw...")
        raw = load_all_raw(symbol)

        print(f"[{symbol}] Building dataset...")
        df = build_full_pattern_dataset(
            raw,
            backtest_horizons=backtest_horizons,
            backtest_window=backtest_window,
            embed_dim=embed_dim,
            score_thr=score_thr,
        )

        out_dir = label_path(symbol)
        if fmt == "parquet":
            path = os.path.join(out_dir, f"{symbol}_dataset.parquet")
            df.to_parquet(path, index=False)
        else:
            path = os.path.join(out_dir, f"{symbol}_dataset.csv")
            df.to_csv(path, index=False)

        print(f"[{symbol}] Done: {len(df)} rows")
        return symbol, len(df), None

    except Exception as e:
        traceback.print_exc()
        return symbol, 0, str(e)

def run_parallel_symbols(
    symbols,
    max_workers=None,
    backtest_horizons=(10,),
    backtest_window=2000,
    embed_dim=32,
    score_thr=0.15,
    fmt="parquet",
):
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    print(f"[PARALLEL] {len(symbols)} symbols | workers={max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futmap = {
            ex.submit(
                process_symbol_to_disk,
                s,
                backtest_horizons,
                backtest_window,
                embed_dim,
                score_thr,
                fmt,
            ): s
            for s in symbols
        }

        for fut in as_completed(futmap):
            s = futmap[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = (s, 0, f"ERROR: {e}")
            results.append(r)

    print("[SUMMARY]")
    for s, n, e in results:
        if e is None:
            print(f"  OK  {s}: {n} rows")
        else:
            print(f"  ERR {s}: {e}")

    return results

class PatternDataset(Dataset):
    def __init__(self, symbols, fmt="parquet", target="signal"):
        frames = []
        for s in symbols:
            fdir = label_path(s)
            fname = f"{s}_dataset.{ 'parquet' if fmt == 'parquet' else 'csv' }"
            fpath = os.path.join(fdir, fname)
            if not os.path.exists(fpath):
                continue

            df = pd.read_parquet(fpath) if fmt == "parquet" else pd.read_csv(fpath)
            df["__symbol__"] = s
            frames.append(df)

        if not frames:
            raise RuntimeError("No datasets found.")

        df_all = pd.concat(frames, ignore_index=True)

        drop = {"open_time", "close_time", "__symbol__", target}
        features = [
            c for c in df_all.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df_all[c])
        ]

        self.X = torch.tensor(df_all[features].values, dtype=torch.float32)
        self.y = torch.tensor(df_all[target].values, dtype=torch.float32)
        self.features = features

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "FILUSDT"]

    run_parallel_symbols(
        symbols,
        max_workers=None,
        backtest_horizons=(10,),
        embed_dim=32,
        score_thr=0.15,
        fmt="parquet",
    )

    print("Dataset generation complete.")
