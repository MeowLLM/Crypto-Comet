#!/usr/bin/env python3
import os
import re
import time
import threading
import requests
import pandas as pd
import datetime as dt
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = r"M:\Dataset"
NOW_YEAR = 2025

INTERVAL = "1m"
MAX_LIMIT = 1000

GLOBAL_THREADS = 8          # how many symbols in parallel
SYMBOL_THREADS = 16         # how many shards per symbol in parallel

REQS_PER_SEC = 18           # global rate-limit
RATE_INTERVAL = 1.0 / REQS_PER_SEC

PAIRS = [
    {"pair": "FILUSDT", "year": 2021},
  {"pair": "FILUSDC", "year": 2025},
  {"pair": "FILETH", "year": 2023},
  {"pair": "FILBTC", "year": 2021},
  {"pair": "WBETHUSDT", "year": 2024},
  {"pair": "WBETHETH", "year": 2024},
  {"pair": "LDOUSDT", "year": 2023},
  {"pair": "LDOUSDC", "year": 2025},
  {"pair": "LDOBTC", "year": 2023},
  {"pair": "USUALUSDC", "year": 2025},
  {"pair": "USUALUSDT", "year": 2025},
  {"pair": "POLYXBTC", "year": 2023},
  {"pair": "POLYXUSDT", "year": 2023},
  {"pair": "USDCUSDT", "year": 2019},
  {"pair": "TUSDUSDT", "year": 2019},
  {"pair": "BNBBTC", "year": 2018},
  {"pair": "BNBETH", "year": 2018},
  {"pair": "BNBEUR", "year": 2020},
  {"pair": "BNBUSDT", "year": 2018},
  {"pair": "BNBUSDC", "year": 2019},
  {"pair": "BNBTUSD", "year": 2020},
  {"pair": "XRPBNB", "year": 2019},
  {"pair": "XRPEUR", "year": 2018},
  {"pair": "XRPUSDC", "year": 2019},
  {"pair": "XRPTUSD", "year": 2019},
  {"pair": "XRPUSDT", "year": 2019},
  {"pair": "XRPBTC", "year": 2018},
  {"pair": "XRPETH", "year": 2018},
  {"pair": "SOLUSDT", "year": 2021},
  {"pair": "SOLEUR", "year": 2022},
  {"pair": "SOLTUSD", "year": 2024},
  {"pair": "SOLUSDC", "year": 2022},
  {"pair": "SOLBNB", "year": 2021},
  {"pair": "SOLBTC", "year": 2021},
  {"pair": "SOLETH", "year": 2022},
  {"pair": "ETHUSDT", "year": 2019},
  {"pair": "ETHUSDC", "year": 2019},
  {"pair": "ETHEUR", "year": 2020},
  {"pair": "ETCTUSD", "year": 2020},
  {"pair": "ETHBTC", "year": 2018},
  {"pair": "LTCEUR", "year": 2021},
  {"pair": "LTCUSDC", "year": 2019},
  {"pair": "LTCUSDT", "year": 2018},
  {"pair": "LTCBTC", "year": 2018},
  {"pair": "LTCETH", "year": 2018},
  {"pair": "LTCBNB", "year": 2018},
  {"pair": "BTCEUR", "year": 2020},
  {"pair": "BTCTUSD", "year": 2019},
  {"pair": "BTCUSDT", "year": 2018},
  {"pair": "BTCUSDC", "year": 2019},
  {"pair": "ZECBTC", "year": 2018},
  {"pair": "ZECETH", "year": 2018},
  {"pair": "ZECUSDC", "year": 2020},
  {"pair": "ZECUSDT", "year": 2020},
  {"pair": "AAVEUSDT", "year": 2021},
  {"pair": "AAVEUSDC", "year": 2025},
  {"pair": "AAVEBTC", "year": 2021},
  {"pair": "AAVEETH", "year": 2021},
  {"pair": "UNIUSDT", "year": 2021},
  {"pair": "UNIUSDC", "year": 2025},
  {"pair": "UNIETH", "year": 2022},
  {"pair": "UNIBTC", "year": 2021},
  {"pair": "SANDUSDT", "year": 2021},
  {"pair": "SANDUSDC", "year": 2025},
  {"pair": "SANDBTC", "year": 2021},
  {"pair": "IMXBTC", "year": 2022},
  {"pair": "IMXUSDT", "year": 2022},
  {"pair": "FLOKIUSDC", "year": 2024},
  {"pair": "FLOKIUSDT", "year": 2024},
  {"pair": "PEPEUSDT", "year": 2024},
  {"pair": "PEPEUSDC", "year": 2025},
  {"pair": "TRUMPUSDT", "year": 2025},
  {"pair": "TRUMPUSDC", "year": 2025},
  {"pair": "SHIBDOGE", "year": 2022},
  {"pair": "SHIBUSDT", "year": 2022},
  {"pair": "SHIBUSDC", "year": 2025},
  {"pair": "DOGEUSDT", "year": 2020},
  {"pair": "DOGEUSDC", "year": 2025},
  {"pair": "DOGEEUR", "year": 2021},
  {"pair": "DOGEBTC", "year": 2020},
  {"pair": "TAOUSDT", "year": 2025},
  {"pair": "TAOUSDC", "year": 2025},
  {"pair": "FETBTC", "year": 2020},
  {"pair": "FETBNB", "year": 2020},
  {"pair": "FETUSDC", "year": 2025},
  {"pair": "FETUSDT", "year": 2020},
  {"pair": "WLDUSDT", "year": 2024},
  {"pair": "WLDUSDC", "year": 2025},
  {"pair": "WLDBTC", "year": 2024},
  {"pair": "COTIUSDT", "year": 2021},
  {"pair": "COTIBTC", "year": 2021},
  {"pair": "BCHUSDT", "year": 2020},
  {"pair": "BCHBNB", "year": 2020},
  {"pair": "BCHUSDC", "year": 2020},
  {"pair": "BCHBTC", "year": 2020}
]

progress_lock = threading.Lock()

PROGRESS = {
    "total_tasks": 0,
    "done_tasks": 0,
    "years": {}   # (symbol,year) -> {"shards":N, "done":K}
}

def show_progress(symbol: str, year: int, final=False):
    """Thread-safe progress printer."""
    with progress_lock:
        yinfo = PROGRESS["years"][(symbol, year)]
        done = yinfo["done"]
        total = yinfo["shards"]

        pct = (done / total * 100) if total else 0.0
        gp = (PROGRESS["done_tasks"] / PROGRESS["total_tasks"] * 100) if PROGRESS["total_tasks"] else 0

    status = "DONE" if final else f"{pct:5.1f}%"

    print(
        f"[{symbol} {year}] shards {done}/{total} ({status}) | GLOBAL {gp:5.1f}%",
        flush=True
    )

YEAR_RE = re.compile(r"(20[1-3][0-9])")

def extract_years(path: str):
    years = set()
    for root, _, files in os.walk(path):
        for f in files:
            m = YEAR_RE.search(f)
            if m:
                years.add(int(m.group(1)))
    return years

def scan_dataset():
    report = {}
    for item in PAIRS:
        pair = item["pair"]
        start = item["year"]
        folder = os.path.join(BASE, pair)
        expected = set(range(start, NOW_YEAR + 1))

        if not os.path.exists(folder):
            report[pair] = {"exists": False, "found": [], "missing": sorted(expected)}
            continue

        found = extract_years(folder)
        missing = expected - found

        report[pair] = {
            "exists": True,
            "found": sorted(found),
            "missing": sorted(missing)
        }

    print("\n========== DATASET SCAN REPORT ==========\n")
    for pair, info in report.items():
        print("====", pair, "====")
        if not info["exists"]:
            print("Folder missing entirely")
        else:
            print("FOUND YEARS :", info["found"])
            print("MISSING     :", info["missing"])
        print()

    return report

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Binance-Kline-Collector/1.0)"})

_last_req_time = 0.0
_rate_lock = threading.Lock()

import random
def ratelimit():
    global _last_req_time
    with _rate_lock:
        now = time.time()
        delta = now - _last_req_time

        # add jitter to break mass sync sleep
        if delta < RATE_INTERVAL:
            sleep_time = (RATE_INTERVAL - delta) * (1 + 0.35 * random.random())
            time.sleep(sleep_time)

        _last_req_time = time.time()

def make_shards(start_ts: int, end_ts: int, shard_minutes: int = 1000):
    step = shard_minutes * 60 * 1000
    shards = []
    t = start_ts
    while t < end_ts:
        e = min(t + step, end_ts)
        shards.append((t, e))
        t = e + 1
    return shards

def fetch_shard(symbol: str, interval: str, start_ts: int, end_ts: int):
    url = "https://api.binance.com/api/v3/klines"
    out = []
    t = start_ts

    empty_count = 0
    network_fail = 0

    MAX_EMPTY = 5         # beyond this, shard ends
    MAX_FAIL  = 5         # avoids freeze on network errors
    MAX_SPIN  = 200       # overall safety

    spin = 0

    while t <= end_ts and spin < MAX_SPIN:
        spin += 1

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": t,
            "endTime": end_ts,
            "limit": MAX_LIMIT
        }

        ratelimit()

        try:
            r = SESSION.get(url, params=params, timeout=5)

            # Hard 429 fix: apply variable backoff
            if r.status_code == 429:
                time.sleep(0.25 + 0.25 * random.random())
                continue

            if r.status_code != 200:
                network_fail += 1
                if network_fail > MAX_FAIL:
                    break
                time.sleep(0.15)
                continue

            data = r.json()

        except Exception:
            network_fail += 1
            if network_fail > MAX_FAIL:
                break
            time.sleep(0.15 + 0.15 * random.random())
            continue

        # --- empty response case (danger zone) ---
        if not data:
            empty_count += 1
            if empty_count > MAX_EMPTY:
                break
            # skip ahead 1 candle
            t += 60_000
            continue

        # --- valid data ---
        empty_count = 0
        network_fail = 0
        out.extend(data)

        # next time window
        t = data[-1][6] + 1

    return out

def fetch_year(symbol: str, year: int):
    print(f"[{symbol}] {year} starting")

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)

    utc_today = datetime.utcnow().date()
    if year == utc_today.year:
        end_date = utc_today

    start_ts = int(time.mktime(start_date.timetuple())) * 1000
    end_ts   = int(time.mktime(end_date.timetuple())) * 1000

    shards = make_shards(start_ts, end_ts, shard_minutes=1000)

    # Register shard count
    with progress_lock:
        PROGRESS["years"][(symbol, year)] = {"shards": len(shards), "done": 0}

    print(f"[{symbol}] {year} shards:", len(shards))

    all_rows = []

    def done_callback(_):
        with progress_lock:
            PROGRESS["years"][(symbol, year)]["done"] += 1
        show_progress(symbol, year)

    # Parallel shard executor
    with ThreadPoolExecutor(max_workers=SYMBOL_THREADS) as pool:
        futures = [pool.submit(fetch_shard, symbol, INTERVAL, s, e) for (s, e) in shards]

        for f in futures:
            f.add_done_callback(done_callback)

        for f in as_completed(futures):
            rows = f.result()
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        print(f"[{symbol}] no data for {year}")
        return

    # Sort globally
    all_rows.sort(key=lambda x: x[0])

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_base_vol","taker_quote_vol","ignore"
    ])

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    os.makedirs(os.path.join(BASE, symbol), exist_ok=True)
    path = os.path.join(BASE, symbol, f"{symbol}_{INTERVAL}_{year}.csv")
    df.to_csv(path, index=False)

    print(f"[{symbol}] saved {year} → {path}")

    with progress_lock:
        PROGRESS["done_tasks"] += 1

    show_progress(symbol, year, final=True)

def download_missing(report):
    # Build task list
    tasks = []
    for item in PAIRS:
        symbol = item["pair"]
        missing = report.get(symbol, {}).get("missing", [])
        for y in missing:
            tasks.append((symbol, y))

    # Initialize progress
    with progress_lock:
        PROGRESS["total_tasks"] = len(tasks)
        PROGRESS["done_tasks"] = 0

    if not tasks:
        print("Nothing missing. All up to date.")
        return

    # Run tasks
    with ThreadPoolExecutor(max_workers=GLOBAL_THREADS) as pool:
        futures = [pool.submit(fetch_year, s, y) for (s, y) in tasks]

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("THREAD ERROR:", e)

if __name__ == "__main__":
    report = scan_dataset()
    download_missing(report)
