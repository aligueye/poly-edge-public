"""Live training data collector.

Connects to Binance Futures WS (aggTrade, depth20, forceOrder), Deribit WS (DVOL),
and Polymarket RTDS (BTC/USD Chainlink price). Records feature snapshots within
5-min windows and labels outcomes.

Outputs Parquet files with the same schema as backfill.py (all 26 features).

Usage:
  .venv/bin/python python/research/collect_training_data.py --output-dir data/training/

Note: Binance WS may be geo-restricted from some locations. Run from a VPS
in an accessible region if needed.
"""

import asyncio
import json
import logging
import os
import signal
import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets")

try:
    from python_socks.async_.asyncio import Proxy
    HAS_SOCKS = True
except ImportError:
    HAS_SOCKS = False

# Reuse TA computation from backfill
from backfill import (
    FEATURE_NAMES,
    compute_rsi,
    compute_macd_signal,
    compute_stoch_k,
    compute_bollinger_pctb,
    compute_atr,
    compute_ema_cross,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# WS endpoints
BINANCE_WS = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@aggTrade/btcusdt@forceOrder"
DERIBIT_WS = "wss://www.deribit.com/ws/api/v2"
RTDS_WS = "wss://ws-live-data.polymarket.com"


def _get_proxy_url() -> str | None:
    """Build SOCKS5 proxy URL from .env variables, or None."""
    host = os.environ.get("SOCKS5_PROXY_HOST")
    port = os.environ.get("SOCKS5_PROXY_PORT")
    if not host or not port:
        return None
    user = os.environ.get("SOCKS5_PROXY_USER", "")
    pw = os.environ.get("SOCKS5_PROXY_PASS", "")
    if user:
        return f"socks5://{user}:{pw}@{host}:{port}"
    return f"socks5://{host}:{port}"


async def _connect_ws(url: str, **kwargs):
    """Connect to a WebSocket, optionally via SOCKS5 proxy for Binance."""
    proxy_url = _get_proxy_url()
    needs_proxy = "binance.com" in url

    if needs_proxy and proxy_url and HAS_SOCKS:
        # Route Binance through SOCKS5 proxy
        from urllib.parse import urlparse
        parsed = urlparse(url)
        dest_host = parsed.hostname
        dest_port = parsed.port or 443

        proxy = Proxy.from_url(proxy_url)
        sock = await proxy.connect(dest_host=dest_host, dest_port=dest_port)
        logger.info("connecting to %s via SOCKS5 proxy", dest_host)
        return await websockets.connect(url, sock=sock, server_hostname=dest_host, **kwargs)
    elif needs_proxy and proxy_url and not HAS_SOCKS:
        logger.warning("SOCKS5 proxy configured but python-socks not installed, connecting direct")

    return await websockets.connect(url, **kwargs)

SNAPSHOT_OFFSETS = [30, 60, 90, 120, 180, 240]  # seconds into 5-min window
MAX_CANDLES = 120
MAX_TRADES = 200
MAX_LIQS = 50


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    price: float
    qty: float
    is_buyer_maker: bool
    timestamp_ms: int


@dataclass
class BookLevel:
    price: float
    qty: float


@dataclass
class Liquidation:
    side: str  # "BUY" or "SELL"
    price: float
    qty: float
    timestamp_ms: int


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_vol: float
    start_ms: int


@dataclass
class MarketState:
    """Shared mutable state updated by WS handlers."""
    # Binance
    btc_price: float = 0.0
    bids: list = field(default_factory=list)  # list[BookLevel]
    asks: list = field(default_factory=list)
    trades: deque = field(default_factory=lambda: deque(maxlen=MAX_TRADES))
    liqs: deque = field(default_factory=lambda: deque(maxlen=MAX_LIQS))
    funding_rate: float = 0.0

    # Deribit
    dvol: float = 0.0

    # Polymarket
    poly_price: float = 0.0  # Chainlink BTC/USD

    # 1-min candle builder
    candles: list = field(default_factory=list)
    current_candle: Candle = None

    # Window state
    window_open_price: float = 0.0
    window_start_ms: int = 0

    def update_candle(self, price: float, qty: float, ts_ms: int, is_buyer_maker: bool):
        """Build 1-min candles from trades (matches Go TA engine logic)."""
        candle_start = (ts_ms // 60_000) * 60_000  # truncate to minute

        if self.current_candle is None:
            self.current_candle = Candle(
                open=price, high=price, low=price, close=price,
                volume=qty, taker_buy_vol=0.0 if is_buyer_maker else qty,
                start_ms=candle_start,
            )
            return

        if candle_start > self.current_candle.start_ms:
            # Close current candle
            self.candles.append(self.current_candle)
            if len(self.candles) > MAX_CANDLES:
                self.candles = self.candles[-MAX_CANDLES:]
            # Start new candle
            self.current_candle = Candle(
                open=price, high=price, low=price, close=price,
                volume=qty, taker_buy_vol=0.0 if is_buyer_maker else qty,
                start_ms=candle_start,
            )
        else:
            self.current_candle.close = price
            self.current_candle.high = max(self.current_candle.high, price)
            self.current_candle.low = min(self.current_candle.low, price)
            self.current_candle.volume += qty
            if not is_buyer_maker:
                self.current_candle.taker_buy_vol += qty


# ---------------------------------------------------------------------------
# Feature computation (live — all 26 features available)
# ---------------------------------------------------------------------------

def compute_live_features(state: MarketState, seconds_remaining: float) -> dict:
    """Compute all 26 features from live state."""
    features = {name: np.nan for name in FEATURE_NAMES}
    now_ms = int(_time.time() * 1000)

    # --- Order flow from L2 book ---
    if state.bids and state.asks:
        features["order_flow_imbalance_L5"] = _ofi(state.bids, state.asks, 5)
        features["order_flow_imbalance_L10"] = _ofi(state.bids, state.asks, 10)
        features["normalized_spread"] = _spread(state.bids, state.asks)
        vb, va = _vwap_dist(state.bids, state.asks)
        features["vwap_to_mid_bid"] = vb
        features["vwap_to_mid_ask"] = va

    # --- Aggressor ratio from trades ---
    trades = list(state.trades)
    for window_ms, feat in [(30_000, "trade_aggressor_ratio_30s"),
                             (60_000, "trade_aggressor_ratio_60s"),
                             (120_000, "trade_aggressor_ratio_120s")]:
        features[feat] = _aggressor_ratio(trades, now_ms, window_ms)

    features["cvd_60s"] = _cvd(trades, now_ms, 60_000)

    # --- Liquidation imbalance ---
    liqs = list(state.liqs)
    features["liquidation_imbalance_60s"] = _liq_imbalance(liqs, now_ms, 60_000)

    # --- TA from 1-min candles ---
    if len(state.candles) >= 31:
        closes = np.array([c.close for c in state.candles], dtype=np.float64)
        highs = np.array([c.high for c in state.candles], dtype=np.float64)
        lows = np.array([c.low for c in state.candles], dtype=np.float64)

        features["rsi_7"] = compute_rsi(closes, 7)
        features["rsi_14"] = compute_rsi(closes, 14)
        features["rsi_30"] = compute_rsi(closes, 30)
        features["macd_signal"] = compute_macd_signal(closes)
        features["stoch_k"] = compute_stoch_k(highs, lows, closes, 14)
        features["bollinger_pct_b"] = compute_bollinger_pctb(closes, 20)
        features["atr_14"] = compute_atr(highs, lows, closes, 14)
        features["ema_9_vs_21"] = compute_ema_cross(closes)

        if len(closes) >= 61 and closes[-61] > 0:
            diff = closes[-1] - closes[-61]
            features["hourly_trend"] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        else:
            features["hourly_trend"] = 0.0

    # --- Momentum from trades ---
    if trades:
        price_now = trades[-1].price
        for window_ms, feat in [(30_000, "momentum_30s"),
                                 (60_000, "momentum_60s"),
                                 (120_000, "momentum_120s")]:
            cutoff = now_ms - window_ms
            for t in reversed(trades):
                if t.timestamp_ms <= cutoff:
                    if t.price > 0:
                        features[feat] = (price_now - t.price) / t.price
                    break

    # --- Derived ---
    if state.window_open_price > 0 and state.btc_price > 0:
        features["price_vs_open"] = (state.btc_price - state.window_open_price) / state.window_open_price
    else:
        features["price_vs_open"] = 0.0

    features["time_decay"] = seconds_remaining / 300.0
    features["dvol_level"] = state.dvol

    if state.window_open_price > 0 and state.btc_price > 0:
        pct_move = abs(state.btc_price - state.window_open_price) / state.window_open_price
        features["mean_reversion_signal"] = 1.0 if pct_move > 0.003 else 0.0
    else:
        features["mean_reversion_signal"] = 0.0

    return features


def _ofi(bids, asks, levels):
    bid_qty = sum(b.qty for b in bids[:levels])
    ask_qty = sum(a.qty for a in asks[:levels])
    total = bid_qty + ask_qty
    return bid_qty / total if total > 0 else 0.5


def _spread(bids, asks):
    if not bids or not asks:
        return 0.0
    mid = (bids[0].price + asks[0].price) / 2
    if mid == 0:
        return 0.0
    return (asks[0].price - bids[0].price) / mid


def _vwap_dist(bids, asks):
    if not bids or not asks:
        return 0.0, 0.0
    mid = (bids[0].price + asks[0].price) / 2
    if mid == 0:
        return 0.0, 0.0

    def vwap(levels):
        pv, v = 0.0, 0.0
        for l in levels[:10]:
            pv += l.price * l.qty
            v += l.qty
        return pv / v if v > 0 else 0.0

    return (vwap(bids) - mid) / mid, (vwap(asks) - mid) / mid


def _aggressor_ratio(trades, now_ms, window_ms):
    if not trades:
        return 0.5
    cutoff = now_ms - window_ms
    buy_vol, total_vol = 0.0, 0.0
    for t in trades:
        if t.timestamp_ms < cutoff:
            continue
        vol = t.price * t.qty
        total_vol += vol
        if not t.is_buyer_maker:
            buy_vol += vol
    return buy_vol / total_vol if total_vol > 0 else 0.5


def _cvd(trades, now_ms, window_ms):
    if not trades:
        return 0.0
    cutoff = now_ms - window_ms
    cvd = 0.0
    for t in trades:
        if t.timestamp_ms < cutoff:
            continue
        vol = t.price * t.qty
        cvd += vol if not t.is_buyer_maker else -vol
    return cvd


def _liq_imbalance(liqs, now_ms, window_ms):
    if not liqs:
        return 0.0
    cutoff = now_ms - window_ms
    imb = 0.0
    for l in liqs:
        if l.timestamp_ms < cutoff:
            continue
        vol = l.price * l.qty
        imb += vol if l.side == "SELL" else -vol
    return imb


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------

async def binance_loop(state: MarketState, stop_event: asyncio.Event):
    """Connect to Binance Futures combined stream."""
    while not stop_event.is_set():
        try:
            ws = await _connect_ws(BINANCE_WS, ping_interval=20)
            async with ws:
                logger.info("binance WS connected")
                async for raw in ws:
                    if stop_event.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                        stream = msg.get("stream", "")
                        data = msg.get("data", {})

                        if "depth20" in stream:
                            state.bids = [
                                BookLevel(float(b[0]), float(b[1]))
                                for b in data.get("b", [])
                            ]
                            state.asks = [
                                BookLevel(float(a[0]), float(a[1]))
                                for a in data.get("a", [])
                            ]

                        elif "aggTrade" in stream:
                            price = float(data["p"])
                            qty = float(data["q"])
                            ts = data["T"]
                            is_bm = data["m"]
                            state.trades.append(Trade(price, qty, is_bm, ts))
                            state.btc_price = price
                            state.update_candle(price, qty, ts, is_bm)

                        elif "forceOrder" in stream:
                            o = data.get("o", {})
                            state.liqs.append(Liquidation(
                                side=o.get("S", ""),
                                price=float(o.get("p", 0)),
                                qty=float(o.get("q", 0)),
                                timestamp_ms=o.get("T", 0),
                            ))

                    except (KeyError, ValueError) as e:
                        logger.debug("binance parse error: %s", e)

        except Exception as e:
            if stop_event.is_set():
                return
            logger.warning("binance WS error: %s, reconnecting in 5s", e)
            await asyncio.sleep(5)


async def deribit_loop(state: MarketState, stop_event: asyncio.Event):
    """Connect to Deribit for DVOL."""
    while not stop_event.is_set():
        try:
            ws = await _connect_ws(DERIBIT_WS, ping_interval=30)
            async with ws:
                logger.info("deribit WS connected")
                sub = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "public/subscribe",
                    "params": {"channels": ["deribit_volatility_index.btc_usd"]},
                }
                await ws.send(json.dumps(sub))

                async for raw in ws:
                    if stop_event.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                        if msg.get("method") == "subscription":
                            params = msg.get("params", {})
                            data = params.get("data", {})
                            if "volatility" in data:
                                state.dvol = float(data["volatility"])
                    except (KeyError, ValueError) as e:
                        logger.debug("deribit parse error: %s", e)

        except Exception as e:
            if stop_event.is_set():
                return
            logger.warning("deribit WS error: %s, reconnecting in 5s", e)
            await asyncio.sleep(5)


async def rtds_loop(state: MarketState, stop_event: asyncio.Event):
    """Connect to Polymarket RTDS for Chainlink BTC/USD."""
    while not stop_event.is_set():
        try:
            ws = await _connect_ws(RTDS_WS, ping_interval=30)
            async with ws:
                logger.info("polymarket RTDS connected")
                sub = {
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                        "filters": '{"symbol":"btc/usd"}',
                    }],
                }
                await ws.send(json.dumps(sub))

                async for raw in ws:
                    if stop_event.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                        if (msg.get("topic") == "crypto_prices_chainlink"
                                and msg.get("type") == "update"):
                            payload = msg.get("payload", {})
                            if "value" in payload:
                                state.poly_price = float(payload["value"])
                    except (KeyError, ValueError) as e:
                        logger.debug("rtds parse error: %s", e)

        except Exception as e:
            if stop_event.is_set():
                return
            logger.warning("rtds WS error: %s, reconnecting in 5s", e)
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Window scheduler + snapshot recorder
# ---------------------------------------------------------------------------

async def window_loop(
    state: MarketState,
    output_dir: Path,
    stop_event: asyncio.Event,
):
    """Manage 5-min windows: record snapshots, label outcomes, save Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wait for initial data
    logger.info("waiting for market data...")
    while state.btc_price == 0 and not stop_event.is_set():
        await asyncio.sleep(1)
    if stop_event.is_set():
        return

    logger.info("got initial BTC price: %.2f, starting window loop", state.btc_price)

    day_rows = []
    current_date = None
    windows_total = 0

    while not stop_event.is_set():
        # Align to next 5-min boundary
        now = datetime.now(timezone.utc)
        minute = now.minute
        mins_to_next = (5 - minute % 5) % 5
        next_boundary = now.replace(second=0, microsecond=0) + timedelta(
            minutes=mins_to_next
        )
        wait_secs = (next_boundary - now).total_seconds()

        if wait_secs < 1:
            # Already on a 5-min boundary — start immediately
            logger.info("on 5-min boundary, starting window now")
        else:
            logger.info("next window at %s (%.0fs)", next_boundary.strftime("%H:%M"), wait_secs)
            # Wait for window start
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_secs)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # timer expired, start window

        window_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=5)
        state.window_start_ms = int(window_start.timestamp() * 1000)
        state.window_open_price = state.btc_price

        logger.info(
            "window started: %s, open=%.2f",
            window_start.strftime("%H:%M"), state.window_open_price,
        )

        snapshots = []

        # Take snapshots at each offset
        for offset_s in SNAPSHOT_OFFSETS:
            target = window_start + timedelta(seconds=offset_s)
            now = datetime.now(timezone.utc)
            wait = (target - now).total_seconds()
            if wait > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=wait)
                    break
                except asyncio.TimeoutError:
                    pass

            if stop_event.is_set():
                break

            secs_remaining = (window_end - datetime.now(timezone.utc)).total_seconds()
            features = compute_live_features(state, max(secs_remaining, 0))

            snapshots.append({
                "snapshot_offset_s": offset_s,
                "btc_price": state.btc_price,
                "features": features,
            })

            n_candles = len(state.candles)
            logger.debug(
                "snapshot T+%ds: price=%.2f, candles=%d, dvol=%.1f",
                offset_s, state.btc_price, n_candles, state.dvol,
            )

        if stop_event.is_set():
            break

        # Wait for window close
        now = datetime.now(timezone.utc)
        wait = (window_end - now).total_seconds()
        if wait > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait)
                if stop_event.is_set():
                    break
            except asyncio.TimeoutError:
                pass

        # Label outcome
        window_close_price = state.btc_price
        outcome = 1 if window_close_price >= state.window_open_price else 0

        # Build rows
        for snap in snapshots:
            row = {
                "window_start": window_start,
                "window_end": window_end,
                "snapshot_offset_s": snap["snapshot_offset_s"],
                "btc_open": state.window_open_price,
                "btc_close": window_close_price,
                "btc_price": snap["btc_price"],
                "outcome": outcome,
                "source": "live",
            }
            row.update(snap["features"])
            day_rows.append(row)

        windows_total += 1
        logger.info(
            "window %d done: open=%.2f close=%.2f outcome=%d (%d snapshots)",
            windows_total, state.window_open_price, window_close_price,
            outcome, len(snapshots),
        )

        # Save daily Parquet (rotate at midnight UTC)
        today = window_start.date()
        if current_date is not None and today != current_date:
            _save_day(day_rows, current_date, output_dir)
            day_rows = []
        current_date = today

        # Also save periodically (every 12 windows = 1 hour)
        if windows_total % 12 == 0 and day_rows:
            _save_day(day_rows, current_date, output_dir)

    # Save remaining
    if day_rows and current_date:
        _save_day(day_rows, current_date, output_dir)


def _save_day(rows: list, date, output_dir: Path):
    """Save rows to a Parquet file for the given date."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = output_dir / f"live_{date}.parquet"

    # Append if file exists
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["window_start", "snapshot_offset_s"], keep="last"
        )

    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("saved %s (%d rows total)", path, len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(output_dir: Path):
    state = MarketState()
    stop_event = asyncio.Event()

    # Handle SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    tasks = [
        asyncio.create_task(binance_loop(state, stop_event)),
        asyncio.create_task(deribit_loop(state, stop_event)),
        asyncio.create_task(rtds_loop(state, stop_event)),
        asyncio.create_task(window_loop(state, output_dir, stop_event)),
    ]

    # Wait for stop or any task to fail
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    stop_event.set()

    for t in done:
        if t.exception():
            logger.error("task failed: %s", t.exception())

    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    logger.info("collector stopped")


def main():
    import argparse

    # Load .env for proxy credentials
    try:
        from dotenv import load_dotenv
        load_dotenv()  # loads from .env in cwd or parent
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Live training data collector")
    parser.add_argument("--output-dir", type=str, default="data/training",
                        help="Output directory (default: data/training)")
    args = parser.parse_args()

    proxy = _get_proxy_url()
    if proxy:
        logger.info("SOCKS5 proxy configured: %s", proxy.split("@")[-1])  # log host only
    else:
        logger.info("no proxy configured, connecting direct")

    asyncio.run(run(Path(args.output_dir)))


if __name__ == "__main__":
    main()
