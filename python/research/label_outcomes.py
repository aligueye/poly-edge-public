"""Validate and fix labels in training data.

Reads Parquet files from data/training/, validates:
  - No NaN outcomes
  - No future data leakage (features don't use info past snapshot time)
  - Feature ranges are reasonable
  - Labels match open/close prices

Can fill missing outcomes using Binance klines from data.binance.vision.

Usage:
  .venv/bin/python python/research/label_outcomes.py --data-dir data/training/
  .venv/bin/python python/research/label_outcomes.py --data-dir data/training/ --fix
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "order_flow_imbalance_L5", "order_flow_imbalance_L10",
    "normalized_spread", "vwap_to_mid_bid", "vwap_to_mid_ask",
    "trade_aggressor_ratio_30s", "trade_aggressor_ratio_60s",
    "trade_aggressor_ratio_120s", "cvd_60s", "liquidation_imbalance_60s",
    "rsi_7", "rsi_14", "rsi_30", "macd_signal", "stoch_k",
    "momentum_30s", "momentum_60s", "momentum_120s",
    "ema_9_vs_21", "atr_14", "bollinger_pct_b", "hourly_trend",
    "price_vs_open", "time_decay", "dvol_level", "mean_reversion_signal",
]

RANGE_CHECKS = {
    "rsi_7": (0, 100), "rsi_14": (0, 100), "rsi_30": (0, 100),
    "stoch_k": (0, 100),
    "bollinger_pct_b": (-3, 4),
    "trade_aggressor_ratio_30s": (0, 1),
    "trade_aggressor_ratio_60s": (0, 1),
    "trade_aggressor_ratio_120s": (0, 1),
    "order_flow_imbalance_L5": (0, 1),
    "order_flow_imbalance_L10": (0, 1),
    "time_decay": (0, 1),
    "hourly_trend": (-1, 1),
    "mean_reversion_signal": (0, 1),
}


def load_all(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all Parquet files in data_dir."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        logger.error("no Parquet files in %s", data_dir)
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["_file"] = f.name
        frames.append(df)
        logger.info("  loaded %s: %d rows", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("total: %d rows from %d files", len(combined), len(files))
    return combined


def validate(df: pd.DataFrame) -> list:
    """Run all validation checks. Returns list of (severity, message) tuples."""
    issues = []

    # 1. Schema check
    for col in FEATURE_NAMES + ["window_start", "window_end", "outcome", "btc_open", "btc_close"]:
        if col not in df.columns:
            issues.append(("ERROR", f"missing column: {col}"))

    if not issues:
        # 2. NaN outcomes
        nan_outcomes = df["outcome"].isna().sum()
        if nan_outcomes > 0:
            issues.append(("ERROR", f"{nan_outcomes} rows with NaN outcome"))

        # 3. Outcome values
        bad_outcomes = ~df["outcome"].isin([0, 1])
        if bad_outcomes.any():
            issues.append(("ERROR", f"{bad_outcomes.sum()} rows with outcome not in {{0, 1}}"))

        # 4. Label consistency: outcome should match btc_close >= btc_open
        if "btc_open" in df.columns and "btc_close" in df.columns:
            expected = (df["btc_close"] >= df["btc_open"]).astype(int)
            mismatched = (df["outcome"] != expected).sum()
            if mismatched > 0:
                issues.append(("WARN", f"{mismatched} rows where outcome != (close >= open)"))

        # 5. Future data leakage: time_decay should be > 0 (snapshot before window end)
        if "time_decay" in df.columns:
            zero_decay = (df["time_decay"] <= 0).sum()
            if zero_decay > 0:
                issues.append(("WARN", f"{zero_decay} rows with time_decay <= 0 (snapshot at/after window end)"))

        # 6. Feature ranges
        for feat, (lo, hi) in RANGE_CHECKS.items():
            if feat not in df.columns:
                continue
            vals = df[feat].dropna()
            if vals.empty:
                continue
            out_of_range = ((vals < lo - 0.1) | (vals > hi + 0.1)).sum()
            if out_of_range > 0:
                issues.append(("WARN", f"{feat}: {out_of_range} values outside [{lo}, {hi}]"))

        # 7. Feature availability by source
        for source in df["source"].unique():
            subset = df[df["source"] == source]
            non_nan_counts = {f: subset[f].notna().sum() for f in FEATURE_NAMES}
            available = sum(1 for v in non_nan_counts.values() if v > 0)
            total = len(FEATURE_NAMES)
            logger.info("  source=%s: %d/%d features available, %d rows", source, available, total, len(subset))

        # 8. Duplicate check
        dupes = df.duplicated(subset=["window_start", "snapshot_offset_s", "source"]).sum()
        if dupes > 0:
            issues.append(("WARN", f"{dupes} duplicate (window_start, offset, source) rows"))

        # 9. Outcome distribution
        dist = df["outcome"].value_counts()
        total = len(df)
        for val, count in dist.items():
            pct = count / total * 100
            if pct < 30 or pct > 70:
                issues.append(("WARN", f"outcome={val} is {pct:.1f}% (expected ~50%)"))
            logger.info("  outcome=%d: %d (%.1f%%)", val, count, pct)

    return issues


def fix_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix NaN outcomes using btc_open/btc_close when available."""
    nan_mask = df["outcome"].isna()
    n_nan = nan_mask.sum()
    if n_nan == 0:
        logger.info("no NaN outcomes to fix")
        return df

    has_prices = nan_mask & df["btc_open"].notna() & df["btc_close"].notna()
    fixable = has_prices.sum()

    if fixable > 0:
        df.loc[has_prices, "outcome"] = (
            df.loc[has_prices, "btc_close"] >= df.loc[has_prices, "btc_open"]
        ).astype(int)
        logger.info("fixed %d/%d NaN outcomes from btc_open/close", fixable, n_nan)

    still_nan = df["outcome"].isna().sum()
    if still_nan > 0:
        logger.warning("%d outcomes still NaN (no price data available)", still_nan)

    return df


def main():
    parser = argparse.ArgumentParser(description="Validate training data labels")
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Directory with Parquet files")
    parser.add_argument("--fix", action="store_true",
                        help="Fix NaN outcomes and save back")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for cleaned dataset (single Parquet)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_all(data_dir)
    if df.empty:
        return

    issues = validate(df)

    if not issues:
        logger.info("all validation checks passed")
    else:
        for severity, msg in issues:
            if severity == "ERROR":
                logger.error(msg)
            else:
                logger.warning(msg)

    if args.fix:
        df = fix_outcomes(df)

        # Re-validate after fix
        post_issues = validate(df)
        errors = [i for i in post_issues if i[0] == "ERROR"]
        if errors:
            logger.error("errors remain after fix:")
            for _, msg in errors:
                logger.error("  %s", msg)
        else:
            logger.info("post-fix validation passed")

        # Save
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = data_dir / "training_clean.parquet"

        # Drop the _file column
        if "_file" in df.columns:
            df = df.drop(columns=["_file"])

        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("saved cleaned dataset: %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
