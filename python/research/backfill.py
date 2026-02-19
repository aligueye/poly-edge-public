"""Historical backfill from Binance data.binance.vision for initial model training.

Downloads 1-min kline CSVs (OHLCV + taker buy volume), reconstructs 5-min windows,
computes available features (~19 of 26), labels outcomes.

Missing features (need live L2 book / DVOL / liquidation stream):
  order_flow_imbalance_L5, order_flow_imbalance_L10, normalized_spread,
  vwap_to_mid_bid, vwap_to_mid_ask, liquidation_imbalance_60s, dvol_level

Usage:
  .venv/bin/python python/research/backfill.py --days 30 --output-dir data/training/
"""

import argparse
import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m"

# All 26 feature names — must match model.py FEATURE_NAMES exactly.
FEATURE_NAMES = [
    "order_flow_imbalance_L5",
    "order_flow_imbalance_L10",
    "normalized_spread",
    "vwap_to_mid_bid",
    "vwap_to_mid_ask",
    "trade_aggressor_ratio_30s",
    "trade_aggressor_ratio_60s",
    "trade_aggressor_ratio_120s",
    "cvd_60s",
    "liquidation_imbalance_60s",
    "rsi_7",
    "rsi_14",
    "rsi_30",
    "macd_signal",
    "stoch_k",
    "momentum_30s",
    "momentum_60s",
    "momentum_120s",
    "ema_9_vs_21",
    "atr_14",
    "bollinger_pct_b",
    "hourly_trend",
    "price_vs_open",
    "time_decay",
    "dvol_level",
    "mean_reversion_signal",
]

SNAPSHOT_OFFSETS = [60, 120, 180, 240]  # seconds into 5-min window
TA_WARMUP_MINUTES = 120  # 2h of candles before first valid window

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_day(date: datetime) -> pd.DataFrame:
    """Download 1-min klines for a single day from data.binance.vision."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/BTCUSDT-1m-{date_str}.zip"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        logger.warning("no data for %s (404)", date_str)
        return pd.DataFrame()
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = zf.namelist()[0]
    with zf.open(csv_name) as f:
        df = pd.read_csv(f, names=KLINE_COLUMNS, header=0)

    for col in ["open", "high", "low", "close", "volume",
                "taker_buy_base_vol", "taker_buy_quote_vol"]:
        df[col] = df[col].astype(float)
    df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def download_range(start: datetime, end: datetime) -> pd.DataFrame:
    """Download klines for a date range, concatenate into single DataFrame."""
    frames = []
    date = start.date()
    end_date = end.date()
    total_days = (end_date - date).days + 1
    downloaded = 0

    while date <= end_date:
        dt = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
        df = download_day(dt)
        if not df.empty:
            frames.append(df)
            downloaded += 1
        else:
            logger.warning("skipping %s (no data)", date)
        date += timedelta(days=1)
        # Be nice to the CDN
        if downloaded % 10 == 0 and downloaded > 0:
            logger.info("downloaded %d/%d days...", downloaded, total_days)
            time.sleep(0.5)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    logger.info("total klines: %d (%d days)", len(combined), downloaded)
    return combined


# ---------------------------------------------------------------------------
# TA indicator computation (matches Go ta/engine.go logic)
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA with multiplier 2/(period+1)."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def compute_rsi(closes: np.ndarray, period: int) -> float:
    """RSI using EMA smoothing (Wilder's approach via EMA(2N-1))."""
    if len(closes) < period + 1:
        return np.nan
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder's smoothing = EMA with period 2*N-1
    avg_gain = _ema(gains, period * 2 - 1)
    avg_loss = _ema(losses, period * 2 - 1)
    if avg_loss[-1] == 0:
        return 100.0
    rs = avg_gain[-1] / avg_loss[-1]
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd_signal(closes: np.ndarray) -> float:
    """MACD(12,26,9) signal line."""
    if len(closes) < 26:
        return np.nan
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    return float(signal[-1])


def compute_stoch_k(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    period: int = 14) -> float:
    """%K = 100 * (C - LL) / (HH - LL) over period."""
    if len(closes) < period:
        return np.nan
    hh = highs[-period:].max()
    ll = lows[-period:].min()
    if hh == ll:
        return 50.0
    return float(100.0 * (closes[-1] - ll) / (hh - ll))


def compute_bollinger_pctb(closes: np.ndarray, period: int = 20,
                           num_std: float = 2.0) -> float:
    """%B = (close - lower) / (upper - lower)."""
    if len(closes) < period:
        return np.nan
    window = closes[-period:]
    sma = window.mean()
    std = window.std(ddof=0)
    if std == 0:
        return 0.5
    upper = sma + num_std * std
    lower = sma - num_std * std
    return float((closes[-1] - lower) / (upper - lower))


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> float:
    """ATR using EMA of true range."""
    if len(closes) < 2:
        return np.nan
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    if len(tr) < period:
        return np.nan
    return float(_ema(tr, period)[-1])


def compute_ema_cross(closes: np.ndarray) -> float:
    """(EMA9 - EMA21) / last_close."""
    if len(closes) < 21:
        return np.nan
    ema9 = _ema(closes, 9)[-1]
    ema21 = _ema(closes, 21)[-1]
    if closes[-1] == 0:
        return 0.0
    return float((ema9 - ema21) / closes[-1])


# ---------------------------------------------------------------------------
# Vectorized TA arrays (compute once over full kline series)
# ---------------------------------------------------------------------------

def _rsi_array(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI at every candle index. NaN where insufficient data."""
    N = len(closes)
    result = np.full(N, np.nan)
    if N < period + 1:
        return result
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = _ema(gains, period * 2 - 1)
    avg_loss = _ema(losses, period * 2 - 1)
    with np.errstate(divide="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # rsi[i] corresponds to closes[i+1] (since deltas starts at index 1)
    result[period:] = rsi[period - 1:]
    return result


def _macd_signal_array(closes: np.ndarray) -> np.ndarray:
    """MACD(12,26,9) signal line at every candle index."""
    N = len(closes)
    result = np.full(N, np.nan)
    if N < 26:
        return result
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    result[:] = signal
    result[:25] = np.nan
    return result


def _stoch_k_array(highs: np.ndarray, lows: np.ndarray,
                   closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Stochastic %K at every candle index using rolling window."""
    hh = pd.Series(highs).rolling(period, min_periods=period).max().values
    ll = pd.Series(lows).rolling(period, min_periods=period).min().values
    denom = hh - ll
    result = np.where(denom > 0, 100.0 * (closes - ll) / denom, 50.0)
    result[:period - 1] = np.nan
    return result


def _bollinger_pctb_array(closes: np.ndarray, period: int = 20,
                          num_std: float = 2.0) -> np.ndarray:
    """%B at every candle index using rolling window."""
    s = pd.Series(closes)
    sma = s.rolling(period, min_periods=period).mean().values
    std = s.rolling(period, min_periods=period).std(ddof=0).values
    upper = sma + num_std * std
    lower = sma - num_std * std
    denom = upper - lower
    result = np.where(denom > 0, (closes - lower) / denom, 0.5)
    result[:period - 1] = np.nan
    return result


def _atr_array(highs: np.ndarray, lows: np.ndarray,
               closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR at every candle index."""
    N = len(closes)
    result = np.full(N, np.nan)
    if N < 2:
        return result
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr = _ema(tr, period)
    result[period:] = atr[period - 1:]
    return result


def _ema_cross_array(closes: np.ndarray) -> np.ndarray:
    """(EMA9 - EMA21) / close at every candle index."""
    N = len(closes)
    result = np.full(N, np.nan)
    if N < 21:
        return result
    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    result = np.where(closes > 0, (ema9 - ema21) / closes, 0.0)
    result[:20] = np.nan
    return result


def precompute_all_indicators(klines: pd.DataFrame) -> dict:
    """Pre-compute all TA indicators and derived features as full arrays.

    Returns dict of feature_name -> np.ndarray (one value per candle).
    """
    closes = klines["close"].values.astype(np.float64)
    highs = klines["high"].values.astype(np.float64)
    lows = klines["low"].values.astype(np.float64)
    volumes = klines["volume"].values.astype(np.float64)
    taker_buy = klines["taker_buy_base_vol"].values.astype(np.float64)
    N = len(closes)

    ind = {}

    # TA indicators
    ind["rsi_7"] = _rsi_array(closes, 7)
    ind["rsi_14"] = _rsi_array(closes, 14)
    ind["rsi_30"] = _rsi_array(closes, 30)
    ind["macd_signal"] = _macd_signal_array(closes)
    ind["stoch_k"] = _stoch_k_array(highs, lows, closes, 14)
    ind["bollinger_pct_b"] = _bollinger_pctb_array(closes, 20)
    ind["atr_14"] = _atr_array(highs, lows, closes, 14)
    ind["ema_9_vs_21"] = _ema_cross_array(closes)

    # Momentum
    ind["momentum_30s"] = np.full(N, np.nan)
    ind["momentum_60s"] = np.full(N, np.nan)
    ind["momentum_120s"] = np.full(N, np.nan)
    if N > 1:
        m1 = np.where(closes[:-1] > 0, (closes[1:] - closes[:-1]) / closes[:-1], np.nan)
        ind["momentum_30s"][1:] = m1
        ind["momentum_60s"][1:] = m1
    if N > 2:
        ind["momentum_120s"][2:] = np.where(
            closes[:-2] > 0, (closes[2:] - closes[:-2]) / closes[:-2], np.nan
        )

    # Hourly trend (60-bar lookback)
    hourly = np.zeros(N)
    if N > 60:
        hourly[60:] = np.sign(closes[60:] - closes[:-60])
    ind["hourly_trend"] = hourly

    # Aggressor ratios
    aggr_1 = np.where(volumes > 0, taker_buy / volumes, 0.5)
    ind["trade_aggressor_ratio_30s"] = aggr_1
    ind["trade_aggressor_ratio_60s"] = aggr_1
    vol_2 = pd.Series(volumes).rolling(2, min_periods=2).sum().values
    buy_2 = pd.Series(taker_buy).rolling(2, min_periods=2).sum().values
    ind["trade_aggressor_ratio_120s"] = np.where(vol_2 > 0, buy_2 / vol_2, 0.5)

    # CVD 60s
    sell_vol = volumes - taker_buy
    ind["cvd_60s"] = (taker_buy - sell_vol) * closes

    return ind


# ---------------------------------------------------------------------------
# Feature computation for one snapshot (used by live collector)
# ---------------------------------------------------------------------------

def compute_snapshot_features(
    candles: pd.DataFrame,
    snapshot_idx: int,
    window_open_price: float,
    seconds_remaining: float,
) -> dict:
    """Compute features from candles up to snapshot_idx (inclusive).

    Expects candles with float columns: open, high, low, close, volume,
    taker_buy_base_vol.
    """
    c = candles.iloc[: snapshot_idx + 1]
    closes = c["close"].values.astype(np.float64)
    highs = c["high"].values.astype(np.float64)
    lows = c["low"].values.astype(np.float64)

    features = {name: np.nan for name in FEATURE_NAMES}

    # --- TA indicators ---
    features["rsi_7"] = compute_rsi(closes, 7)
    features["rsi_14"] = compute_rsi(closes, 14)
    features["rsi_30"] = compute_rsi(closes, 30)
    features["macd_signal"] = compute_macd_signal(closes)
    features["stoch_k"] = compute_stoch_k(highs, lows, closes, 14)
    features["bollinger_pct_b"] = compute_bollinger_pctb(closes, 20)
    features["atr_14"] = compute_atr(highs, lows, closes, 14)
    features["ema_9_vs_21"] = compute_ema_cross(closes)

    # --- Momentum (1-min approximation) ---
    if len(closes) >= 2 and closes[-2] > 0:
        features["momentum_30s"] = (closes[-1] - closes[-2]) / closes[-2]
        features["momentum_60s"] = (closes[-1] - closes[-2]) / closes[-2]
    if len(closes) >= 3 and closes[-3] > 0:
        features["momentum_120s"] = (closes[-1] - closes[-3]) / closes[-3]

    # --- Hourly trend ---
    if len(closes) >= 61 and closes[-61] > 0:
        diff = closes[-1] - closes[-61]
        features["hourly_trend"] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
    else:
        features["hourly_trend"] = 0.0

    # --- Aggressor ratio (from kline taker buy volume) ---
    for n_bars, feat_name in [(1, "trade_aggressor_ratio_30s"),
                               (1, "trade_aggressor_ratio_60s"),
                               (2, "trade_aggressor_ratio_120s")]:
        window = c.iloc[-min(n_bars, len(c)):]
        total_vol = window["volume"].sum()
        buy_vol = window["taker_buy_base_vol"].sum()
        features[feat_name] = float(buy_vol / total_vol) if total_vol > 0 else 0.5

    # --- CVD 60s ---
    last_bar = c.iloc[-1:]
    buy_vol = last_bar["taker_buy_base_vol"].sum()
    sell_vol = last_bar["volume"].sum() - buy_vol
    features["cvd_60s"] = float((buy_vol - sell_vol) * closes[-1])

    # --- Derived ---
    if window_open_price > 0:
        features["price_vs_open"] = float((closes[-1] - window_open_price) / window_open_price)
    else:
        features["price_vs_open"] = 0.0

    features["time_decay"] = seconds_remaining / 300.0

    if window_open_price > 0:
        pct_move = abs(closes[-1] - window_open_price) / window_open_price
        features["mean_reversion_signal"] = 1.0 if pct_move > 0.003 else 0.0
    else:
        features["mean_reversion_signal"] = 0.0

    # OFI, spread, VWAP, liq imbalance, dvol: remain NaN (need live data)
    return features


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def backfill(days: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    # data.binance.vision has ~1-2 day lag
    end = (now - timedelta(days=2)).replace(hour=23, minute=59, second=0, microsecond=0)
    start = end - timedelta(days=days)

    # Extra days for TA warmup
    warmup_start = start - timedelta(minutes=TA_WARMUP_MINUTES)

    logger.info("backfill range: %s to %s (warmup from %s)",
                start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                warmup_start.strftime("%Y-%m-%d"))

    klines = download_range(warmup_start, end)
    if klines.empty:
        logger.error("no klines downloaded")
        return

    # Build 5-min windows
    # Align start to 5-min boundary
    window_start = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    window_start_ms = int(window_start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    # Reset index once; extract numpy arrays for fast lookups
    klines = klines.reset_index(drop=True)
    open_times = klines["open_time"].values  # int64 ms timestamps
    closes_arr = klines["close"].values.astype(np.float64)

    # Pre-compute all TA indicators once over the full series
    logger.info("pre-computing TA indicators...")
    t0 = time.time()
    indicators = precompute_all_indicators(klines)
    logger.info("pre-computation done in %.1f seconds", time.time() - t0)

    rows = []
    windows_ok = 0
    windows_skip = 0

    while window_start_ms < end_ms:
        window_end_ms = window_start_ms + 300_000  # 5 min in ms

        # Find candles in this window using numpy (much faster than pandas mask)
        window_indices = np.where(
            (open_times >= window_start_ms) & (open_times < window_end_ms)
        )[0]

        if len(window_indices) < 5:
            windows_skip += 1
            window_start_ms = window_end_ms
            continue

        window_open = float(klines.at[window_indices[0], "open"])
        window_close = float(klines.at[window_indices[-1], "close"])
        outcome = 1 if window_close >= window_open else 0

        wstart_dt = datetime.fromtimestamp(window_start_ms / 1000, tz=timezone.utc)
        wend_dt = datetime.fromtimestamp(window_end_ms / 1000, tz=timezone.utc)

        for offset_s in SNAPSHOT_OFFSETS:
            offset_min = offset_s // 60
            snapshot_ms = window_start_ms + offset_s * 1000

            # Binary search for snapshot candle index (O(log n) vs O(n))
            snapshot_iloc = int(np.searchsorted(open_times, snapshot_ms, side="right")) - 1
            if snapshot_iloc < 31:
                continue

            # Index into pre-computed arrays (O(1) per feature)
            features = {name: np.nan for name in FEATURE_NAMES}
            for feat_name, arr in indicators.items():
                features[feat_name] = float(arr[snapshot_iloc])

            # Window-specific features (can't be pre-computed)
            snap_price = closes_arr[snapshot_iloc]
            if window_open > 0:
                features["price_vs_open"] = float((snap_price - window_open) / window_open)
                pct_move = abs(snap_price - window_open) / window_open
                features["mean_reversion_signal"] = 1.0 if pct_move > 0.003 else 0.0
            else:
                features["price_vs_open"] = 0.0
                features["mean_reversion_signal"] = 0.0
            features["time_decay"] = (300.0 - offset_s) / 300.0

            snap_candle_idx = min(window_indices[0] + offset_min - 1, window_indices[-1])
            row = {
                "window_start": wstart_dt,
                "window_end": wend_dt,
                "snapshot_offset_s": offset_s,
                "btc_open": window_open,
                "btc_close": window_close,
                "btc_price": float(klines.at[snap_candle_idx, "close"]),
                "outcome": outcome,
                "source": "backfill",
            }
            row.update(features)
            rows.append(row)

        windows_ok += 1
        if windows_ok % 2000 == 0:
            logger.info("processed %d windows...", windows_ok)

        window_start_ms = window_end_ms

    if not rows:
        logger.error("no rows generated")
        return

    df = pd.DataFrame(rows)

    # Verify all 26 feature columns present
    missing_cols = [n for n in FEATURE_NAMES if n not in df.columns]
    if missing_cols:
        logger.error("missing feature columns: %s", missing_cols)
        return

    # Save per-day Parquet
    df["date"] = df["window_start"].dt.date
    total_saved = 0
    for date, group in df.groupby("date"):
        path = output_dir / f"backfill_{date}.parquet"
        group.drop(columns=["date"]).to_parquet(path, index=False, engine="pyarrow")
        total_saved += len(group)
        logger.info("saved %s (%d rows)", path, len(group))

    logger.info(
        "backfill complete: %d windows, %d rows, %d days, %d skipped",
        windows_ok, total_saved, df["date"].nunique(), windows_skip,
    )
    logger.info("outcome distribution:\n%s", df["outcome"].value_counts().to_string())

    _validate_features(df)
    return df


def _validate_features(df: pd.DataFrame):
    """Validate feature ranges and log results."""
    checks = {
        "rsi_7": (0, 100), "rsi_14": (0, 100), "rsi_30": (0, 100),
        "stoch_k": (0, 100),
        "bollinger_pct_b": (-2, 3),
        "trade_aggressor_ratio_30s": (0, 1),
        "trade_aggressor_ratio_60s": (0, 1),
        "trade_aggressor_ratio_120s": (0, 1),
        "time_decay": (0, 1),
        "hourly_trend": (-1, 1),
        "mean_reversion_signal": (0, 1),
    }

    problems = []
    for feat, (lo, hi) in checks.items():
        vals = df[feat].dropna()
        if vals.empty:
            problems.append(f"{feat}: all NaN")
            continue
        fmin, fmax = vals.min(), vals.max()
        if fmin < lo - 0.01 or fmax > hi + 0.01:
            problems.append(f"{feat}: [{fmin:.4f}, {fmax:.4f}] outside [{lo}, {hi}]")
        else:
            logger.info("  %s: [%.4f, %.4f] OK", feat, fmin, fmax)

    nan_expected = [
        "order_flow_imbalance_L5", "order_flow_imbalance_L10",
        "normalized_spread", "vwap_to_mid_bid", "vwap_to_mid_ask",
        "liquidation_imbalance_60s", "dvol_level",
    ]
    for feat in nan_expected:
        non_nan = df[feat].notna().sum()
        if non_nan > 0:
            problems.append(f"{feat}: should be all NaN but has {non_nan} values")

    if problems:
        logger.warning("validation issues:\n  %s", "\n  ".join(problems))
    else:
        logger.info("all feature validation checks passed")


def main():
    parser = argparse.ArgumentParser(description="Backfill training data from Binance historical klines")
    parser.add_argument("--days", type=int, default=30, help="Days to backfill (default: 30)")
    parser.add_argument("--output-dir", type=str, default="data/training",
                        help="Output directory (default: data/training)")
    args = parser.parse_args()
    backfill(args.days, Path(args.output_dir))


if __name__ == "__main__":
    main()
