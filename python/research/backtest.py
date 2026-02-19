"""Backtest directional strategy against historical data.

Simulates Polymarket trading on test set:
- Model predicts at T+60s snapshot per window
- Trade only when confidence > threshold
- Sweep thresholds: 0.52, 0.55, 0.58, 0.60, 0.62, 0.65
- Fee model: PostOnly 0%, FOK taker when confidence > 0.65 & time < 120s
- Position sizing: quarter-Kelly
- Output: PnL curve, Sharpe, max drawdown, win rate, trades/day

Usage:
  .venv/bin/python python/research/backtest.py --data-dir data/training/
  .venv/bin/python python/research/backtest.py --data-dir data/training/ --model-dir data/models/
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

THRESHOLDS = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
INITIAL_CAPITAL = 1000.0  # USDC for simulation
FOK_FEE_RATE = 0.0025     # taker fee rate


def load_model(model_dir: Path):
    """Load best model and optional calibrator."""
    meta_path = model_dir / "training_meta.json"
    if not meta_path.exists():
        logger.error("no training_meta.json in %s", model_dir)
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    best_name = meta["best_model"]
    features = meta["features"]

    if best_name == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_dir / "model.cbm"))
    elif best_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(model_dir / "model_xgb.json"))
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_dir / "model_lgbm.txt"))

    # Load calibrator if exists
    calibrator = None
    cal_path = model_dir / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        logger.info("loaded calibrator (%s)", meta.get("calibration_method", "unknown"))

    logger.info("loaded %s model with %d features", best_name, len(features))
    return model, calibrator, features


def predict_proba(model, calibrator, X, model_name: str) -> np.ndarray:
    """Get P(up) predictions, using calibrator if available."""
    if calibrator is not None:
        proba = calibrator.predict_proba(X)[:, 1]
    elif model_name == "lightgbm" and hasattr(model, "predict"):
        # LightGBM Booster returns raw scores
        raw = model.predict(X)
        proba = raw  # Already probabilities for binary classification
    else:
        proba = model.predict_proba(X)[:, 1]
    return proba


def simulate_threshold(df: pd.DataFrame, proba: np.ndarray,
                       threshold: float, capital: float) -> dict:
    """Simulate trading at a given confidence threshold.

    For each window, uses the T+60s snapshot prediction.
    Binary market: buy YES at 0.50, payout $1 if correct, $0 if wrong.
    Position sizing: quarter-Kelly.
    """
    # Use only T+60s snapshots (one prediction per window)
    mask_60s = df["snapshot_offset_s"] == 60
    if mask_60s.sum() == 0:
        # Fallback to first available offset
        first_offset = df["snapshot_offset_s"].min()
        mask_60s = df["snapshot_offset_s"] == first_offset

    window_df = df[mask_60s].copy()
    window_proba = proba[mask_60s.values]

    balance = capital
    trades = []
    pnl_curve = [capital]
    dates = []

    for i in range(len(window_df)):
        row = window_df.iloc[i]
        p_up = window_proba[i]
        outcome = int(row["outcome"])

        # Determine action and confidence
        if p_up > 0.5:
            action = "buy_yes"
            confidence = p_up
            market_price = 0.50  # assume fair market
        elif p_up < 0.5:
            action = "buy_no"
            confidence = 1.0 - p_up
            market_price = 0.50
        else:
            continue

        if confidence < threshold:
            continue

        # Quarter-Kelly sizing (on initial capital, not compounding balance)
        edge = confidence - market_price
        if edge <= 0:
            continue
        kelly = edge / (1.0 - market_price)
        size_frac = 0.25 * kelly
        position_size = size_frac * capital  # flat sizing on initial capital

        # Min/max position — cap at 10% of initial capital per trade
        position_size = max(1.0, min(position_size, capital * 0.10))
        if position_size > balance:
            continue

        # Fee model
        offset_s = row["snapshot_offset_s"]
        if confidence > 0.65 and offset_s < 120:
            # FOK taker order
            fee = position_size * FOK_FEE_RATE * market_price * (1 - market_price)
        else:
            # PostOnly — 0% fee
            fee = 0.0

        # Determine win/loss
        if action == "buy_yes":
            won = outcome == 1
        else:
            won = outcome == 0

        if won:
            # Buy at market_price, payout $1 per share
            # Shares = position_size / market_price
            # Profit = shares * (1 - market_price) - fee
            profit = position_size * (1.0 - market_price) / market_price - fee
        else:
            # Lose entire position
            profit = -position_size - fee

        balance += profit
        trades.append({
            "date": row["window_start"],
            "action": action,
            "confidence": confidence,
            "size": position_size,
            "fee": fee,
            "won": won,
            "profit": profit,
            "balance": balance,
        })
        pnl_curve.append(balance)
        dates.append(row["window_start"])

    if not trades:
        return {
            "threshold": threshold, "trades": 0, "win_rate": 0,
            "total_pnl": 0, "sharpe": 0, "max_drawdown": 0,
            "final_balance": capital, "pnl_curve": [capital],
            "dates": [],
        }

    trades_df = pd.DataFrame(trades)
    total_pnl = balance - capital
    win_rate = trades_df["won"].mean()
    n_trades = len(trades_df)

    # Daily PnL for Sharpe
    trades_df["date_only"] = pd.to_datetime(trades_df["date"]).dt.date
    daily_pnl = trades_df.groupby("date_only")["profit"].sum()
    n_days = max(1, len(daily_pnl))

    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    # Max drawdown
    curve = np.array(pnl_curve)
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    max_dd = float(drawdown.max())

    total_fees = trades_df["fee"].sum()

    return {
        "threshold": threshold,
        "trades": n_trades,
        "trades_per_day": n_trades / n_days,
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "total_fees": float(total_fees),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "final_balance": float(balance),
        "pnl_curve": pnl_curve,
        "dates": dates,
    }


def simulate_random_baseline(df: pd.DataFrame, capital: float) -> dict:
    """Random 50% predictor baseline with same sizing logic."""
    rng = np.random.RandomState(42)
    n = len(df[df["snapshot_offset_s"] == 60])
    random_proba = rng.uniform(0.4, 0.6, size=len(df))
    return simulate_threshold(df, random_proba, 0.52, capital)


def main():
    parser = argparse.ArgumentParser(description="Backtest directional strategy")
    parser.add_argument("--data-dir", type=str, default="data/training")
    parser.add_argument("--model-dir", type=str, default="data/models")
    parser.add_argument("--output-dir", type=str, default="data/backtest")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, calibrator, features = load_model(model_dir)
    if model is None:
        return

    # Load test data (last 20% by time)
    from train_model import load_data, time_split
    df = load_data(data_dir)
    if df.empty:
        return

    df = df.dropna(subset=["outcome"])
    df["outcome"] = df["outcome"].astype(int)
    _, test_df = time_split(df, 0.8)

    logger.info("backtest on %d test rows (%s to %s)",
                len(test_df),
                test_df["window_start"].iloc[0],
                test_df["window_start"].iloc[-1])

    # Get predictions
    meta_path = model_dir / "training_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    X_test = test_df[features]
    proba = predict_proba(model, calibrator, X_test, meta["best_model"])

    # Sweep thresholds
    results = []
    logger.info("=== Threshold Sweep ===")
    for thr in THRESHOLDS:
        r = simulate_threshold(test_df, proba, thr, INITIAL_CAPITAL)
        results.append(r)
        logger.info("  thr=%.2f  trades=%d  win=%.1f%%  pnl=$%.2f  sharpe=%.2f  maxdd=%.1f%%",
                    thr, r["trades"], r["win_rate"] * 100, r["total_pnl"],
                    r["sharpe"], r["max_drawdown"] * 100)

    # Random baseline
    baseline = simulate_random_baseline(test_df, INITIAL_CAPITAL)
    logger.info("  baseline  trades=%d  win=%.1f%%  pnl=$%.2f",
                baseline["trades"], baseline["win_rate"] * 100, baseline["total_pnl"])

    # Find best threshold by Sharpe
    best = max(results, key=lambda r: r["sharpe"] if r["trades"] > 0 else -999)
    logger.info("best threshold: %.2f (Sharpe=%.2f, PnL=$%.2f)",
                best["threshold"], best["sharpe"], best["total_pnl"])

    # --- Plot PnL curves ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: PnL curves for all thresholds
    ax = axes[0]
    for r in results:
        if r["trades"] > 0:
            ax.plot(range(len(r["pnl_curve"])), r["pnl_curve"],
                    label=f"thr={r['threshold']:.2f} (${r['total_pnl']:.0f})")
    if baseline["trades"] > 0:
        ax.plot(range(len(baseline["pnl_curve"])), baseline["pnl_curve"],
                "--", color="gray", label=f"random (${baseline['total_pnl']:.0f})")
    ax.axhline(y=INITIAL_CAPITAL, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Balance ($)")
    ax.set_title("Cumulative PnL by Threshold")
    ax.legend(fontsize=8)

    # Bottom: bar chart of metrics
    ax = axes[1]
    thrs = [r["threshold"] for r in results if r["trades"] > 0]
    pnls = [r["total_pnl"] for r in results if r["trades"] > 0]
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax.bar(range(len(thrs)), pnls, color=colors, alpha=0.7)
    ax.set_xticks(range(len(thrs)))
    ax.set_xticklabels([f"{t:.2f}" for t in thrs])
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("PnL by Threshold")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "backtest_pnl.png", dpi=150)
    plt.close(fig)
    logger.info("saved PnL plot to %s", output_dir / "backtest_pnl.png")

    # Save results JSON
    # Strip non-serializable fields
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in ("pnl_curve", "dates")}
        save_results.append(sr)

    summary = {
        "thresholds": save_results,
        "baseline": {k: v for k, v in baseline.items() if k not in ("pnl_curve", "dates")},
        "best_threshold": best["threshold"],
        "best_sharpe": best["sharpe"],
        "best_pnl": best["total_pnl"],
        "initial_capital": INITIAL_CAPITAL,
        "test_rows": len(test_df),
    }
    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("saved results to %s", output_dir / "backtest_results.json")


if __name__ == "__main__":
    main()
