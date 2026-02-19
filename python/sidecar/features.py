"""Feature engineering for the ML model.

Computes 25 features from MarketState proto message:
- 10 order flow features (from raw book/trade data)
- 12 pre-computed TA features (passed through from Go)
- 3 derived/contextual features
"""

import numpy as np


def compute_features(state) -> dict:
    """Compute all 25 features from a MarketState proto message.

    Args:
        state: MarketState protobuf message from Go.

    Returns:
        dict with 25 feature name -> float value mappings.
    """
    features = {}

    # --- Order flow features (computed from raw book/trade data) ---
    bids = list(state.binance_bids)  # flattened [price, qty, price, qty, ...]
    asks = list(state.binance_asks)

    features["order_flow_imbalance_L5"] = _ofi(bids, asks, levels=5)
    features["order_flow_imbalance_L10"] = _ofi(bids, asks, levels=10)
    features["normalized_spread"] = _normalized_spread(bids, asks)

    vwap_bid, vwap_ask = _vwap_distances(bids, asks)
    features["vwap_to_mid_bid"] = vwap_bid
    features["vwap_to_mid_ask"] = vwap_ask

    trades = list(state.recent_trades)
    features["trade_aggressor_ratio_30s"] = _aggressor_ratio(trades, window_ms=30_000)
    features["trade_aggressor_ratio_60s"] = _aggressor_ratio(trades, window_ms=60_000)
    features["trade_aggressor_ratio_120s"] = _aggressor_ratio(trades, window_ms=120_000)
    features["cvd_60s"] = _cvd(trades, window_ms=60_000)

    liqs = list(state.recent_liquidations)
    features["liquidation_imbalance_60s"] = _liquidation_imbalance(liqs, window_ms=60_000)

    # --- Pre-computed TA from Go (pass-through) ---
    features["rsi_7"] = state.rsi_7
    features["rsi_14"] = state.rsi_14
    features["rsi_30"] = state.rsi_30
    features["macd_signal"] = state.macd_signal
    features["stoch_k"] = state.stoch_k
    features["momentum_30s"] = state.momentum_30s
    features["momentum_60s"] = state.momentum_60s
    features["momentum_120s"] = state.momentum_120s
    features["ema_9_vs_21"] = state.ema_9_vs_21
    features["atr_14"] = state.atr_14
    features["bollinger_pct_b"] = state.bollinger_pct_b
    features["hourly_trend"] = state.hourly_trend

    # --- Derived/contextual features ---
    if state.window_open_price > 0:
        features["price_vs_open"] = (
            (state.btc_price - state.window_open_price) / state.window_open_price
        )
    else:
        features["price_vs_open"] = 0.0

    features["time_decay"] = state.seconds_remaining / 300.0
    features["dvol_level"] = state.dvol

    # Mean reversion signal: 1.0 if price moved > 0.3% from open
    if state.window_open_price > 0:
        pct_move = abs(state.btc_price - state.window_open_price) / state.window_open_price
        features["mean_reversion_signal"] = 1.0 if pct_move > 0.003 else 0.0
    else:
        features["mean_reversion_signal"] = 0.0

    return features


def _ofi(bids: list, asks: list, levels: int) -> float:
    """Order flow imbalance: sum(bid_qty) / sum(all_qty) for top N levels."""
    bid_qty = 0.0
    ask_qty = 0.0
    for i in range(min(levels, len(bids) // 2)):
        bid_qty += bids[i * 2 + 1]
    for i in range(min(levels, len(asks) // 2)):
        ask_qty += asks[i * 2 + 1]
    total = bid_qty + ask_qty
    if total == 0:
        return 0.5
    return bid_qty / total


def _normalized_spread(bids: list, asks: list) -> float:
    """(best_ask - best_bid) / midprice."""
    if len(bids) < 2 or len(asks) < 2:
        return 0.0
    best_bid = bids[0]
    best_ask = asks[0]
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        return 0.0
    return (best_ask - best_bid) / mid


def _vwap_distances(bids: list, asks: list) -> tuple:
    """VWAP distance from mid for bid and ask sides (top 10 levels)."""
    if len(bids) < 2 or len(asks) < 2:
        return 0.0, 0.0
    mid = (bids[0] + asks[0]) / 2
    if mid == 0:
        return 0.0, 0.0

    bid_vwap = _side_vwap(bids, 10)
    ask_vwap = _side_vwap(asks, 10)

    return (bid_vwap - mid) / mid, (ask_vwap - mid) / mid


def _side_vwap(levels: list, n: int) -> float:
    """Volume-weighted average price for one side of the book."""
    total_pv = 0.0
    total_v = 0.0
    for i in range(min(n, len(levels) // 2)):
        price = levels[i * 2]
        qty = levels[i * 2 + 1]
        total_pv += price * qty
        total_v += qty
    if total_v == 0:
        return 0.0
    return total_pv / total_v


def _aggressor_ratio(trades, window_ms: int) -> float:
    """Buy volume / total volume over the last window_ms."""
    if not trades:
        return 0.5
    cutoff = trades[-1].timestamp_ms - window_ms
    buy_vol = 0.0
    total_vol = 0.0
    for t in trades:
        if t.timestamp_ms < cutoff:
            continue
        vol = t.price * t.quantity
        total_vol += vol
        if not t.is_buyer_maker:  # taker is buyer
            buy_vol += vol
    if total_vol == 0:
        return 0.5
    return buy_vol / total_vol


def _cvd(trades, window_ms: int) -> float:
    """Cumulative volume delta (buy_vol - sell_vol) over the last window_ms."""
    if not trades:
        return 0.0
    cutoff = trades[-1].timestamp_ms - window_ms
    cvd = 0.0
    for t in trades:
        if t.timestamp_ms < cutoff:
            continue
        vol = t.price * t.quantity
        if not t.is_buyer_maker:
            cvd += vol
        else:
            cvd -= vol
    return cvd


def _liquidation_imbalance(liqs, window_ms: int) -> float:
    """Long liquidation volume - short liquidation volume over window_ms."""
    if not liqs:
        return 0.0
    cutoff = liqs[-1].timestamp_ms - window_ms
    imbalance = 0.0
    for liq in liqs:
        if liq.timestamp_ms < cutoff:
            continue
        vol = liq.price * liq.quantity
        if liq.side == "SELL":  # long liquidation
            imbalance += vol
        else:
            imbalance -= vol
    return imbalance
