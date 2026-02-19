"""Train CatBoost / XGBoost / LightGBM models on backfill + live data.

- Time-based 80/20 split (never random)
- Optuna hyperparameter tuning with TimeSeriesSplit CV
- Probability calibration (Platt scaling / isotonic regression)
- Evaluation: accuracy, AUC-ROC, Brier score, log loss, SHAP
- Saves best model as .cbm / .json and calibrator as .pkl

Usage:
  .venv/bin/python python/research/train_model.py --data-dir data/training/
  .venv/bin/python python/research/train_model.py --data-dir data/training/ --trials 50
"""

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Must match model.py FEATURE_NAMES
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

# Features that are NaN in backfill-only data
BACKFILL_MISSING = {
    "order_flow_imbalance_L5", "order_flow_imbalance_L10",
    "normalized_spread", "vwap_to_mid_bid", "vwap_to_mid_ask",
    "liquidation_imbalance_60s", "dvol_level",
}


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all Parquet files, sorted by window_start."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        logger.error("no Parquet files in %s", data_dir)
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("window_start").reset_index(drop=True)
    logger.info("loaded %d rows from %d files", len(df), len(files))
    return df


def get_available_features(df: pd.DataFrame) -> list:
    """Return feature names that have non-NaN data."""
    available = []
    for f in FEATURE_NAMES:
        if f in df.columns and df[f].notna().sum() > len(df) * 0.5:
            available.append(f)
    return available


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Time-based split. First train_frac rows for train, rest for test."""
    n = len(df)
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _catboost_objective(trial, X_train, y_train, n_splits=3):
    params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "depth": trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [1, 3, 5, 7]),
        "iterations": trial.suggest_categorical("iterations", [200, 500, 1000]),
        "verbose": 0,
        "random_seed": 42,
        "nan_mode": "Min",
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=50, verbose=0)
        proba = model.predict_proba(Xv)[:, 1]
        scores.append(roc_auc_score(yv, proba))
    return np.mean(scores)


def _xgb_objective(trial, X_train, y_train, n_splits=3):
    params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1, 3, 5, 7]),
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 500, 1000]),
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": 42,
        "use_label_encoder": False,
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = XGBClassifier(**params)
        model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=0)
        proba = model.predict_proba(Xv)[:, 1]
        scores.append(roc_auc_score(yv, proba))
    return np.mean(scores)


def _lgbm_objective(trial, X_train, y_train, n_splits=3):
    params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1, 3, 5, 7]),
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 500, 1000]),
        "verbose": -1,
        "random_state": 42,
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = LGBMClassifier(**params)
        model.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[])
        proba = model.predict_proba(Xv)[:, 1]
        scores.append(roc_auc_score(yv, proba))
    return np.mean(scores)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_catboost(X_train, y_train, X_test, y_test, best_params: dict) -> CatBoostClassifier:
    params = {**best_params, "verbose": 0, "random_seed": 42, "nan_mode": "Min"}
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test),
              early_stopping_rounds=50, verbose=0)
    return model


def train_xgboost(X_train, y_train, X_test, y_test, best_params: dict) -> XGBClassifier:
    params = {**best_params, "eval_metric": "logloss", "verbosity": 0,
              "random_state": 42, "use_label_encoder": False}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, best_params: dict) -> LGBMClassifier:
    params = {**best_params, "verbose": -1, "random_state": 42}
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[])
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, name: str, output_dir: Path) -> dict:
    """Evaluate a trained model. Returns metrics dict."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    ll = log_loss(y_test, proba)

    logger.info("  %s — acc=%.4f  auc=%.4f  brier=%.4f  logloss=%.4f", name, acc, auc, brier, ll)

    # Accuracy by confidence bucket
    buckets = {"top_10pct": 0.9, "top_20pct": 0.8, "top_50pct": 0.5}
    bucket_acc = {}
    for label, quantile in buckets.items():
        conf = np.abs(proba - 0.5)
        threshold = np.quantile(conf, quantile)
        mask = conf >= threshold
        if mask.sum() > 0:
            bucket_acc[label] = float(accuracy_score(y_test[mask], preds[mask]))

    if acc > 0.75:
        logger.warning("  %s accuracy %.1f%% is suspiciously high — check for data leakage!", name, acc * 100)

    return {
        "name": name,
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "accuracy_by_confidence": bucket_acc,
        "n_test": len(y_test),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(model, X_val, y_val, X_test, y_test,
                    output_dir: Path, name: str):
    """Apply Platt scaling and isotonic regression, pick best by Brier score.

    Uses manual calibration (LogisticRegression / IsotonicRegression on
    model's predicted probabilities) since CalibratedClassifierCV(cv='prefit')
    was removed in sklearn 1.8.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    raw_proba_test = model.predict_proba(X_test)[:, 1]
    raw_brier = brier_score_loss(y_test, raw_proba_test)
    results = {"raw": {"brier": float(raw_brier), "proba": raw_proba_test}}

    # Get model's predicted probabilities on validation set
    raw_proba_val = model.predict_proba(X_val)[:, 1]

    calibrators = {}

    # Platt scaling (sigmoid): logistic regression on raw probabilities
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(raw_proba_val.reshape(-1, 1), y_val)
    platt_proba = platt.predict_proba(raw_proba_test.reshape(-1, 1))[:, 1]
    platt_brier = brier_score_loss(y_test, platt_proba)
    results["sigmoid"] = {"brier": float(platt_brier), "proba": platt_proba}
    calibrators["sigmoid"] = platt
    logger.info("  %s calibration (sigmoid): brier %.4f -> %.4f",
                name, raw_brier, platt_brier)

    # Isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_proba_val, y_val)
    iso_proba = iso.predict(raw_proba_test)
    iso_brier = brier_score_loss(y_test, iso_proba)
    results["isotonic"] = {"brier": float(iso_brier), "proba": iso_proba}
    calibrators["isotonic"] = iso
    logger.info("  %s calibration (isotonic): brier %.4f -> %.4f",
                name, raw_brier, iso_brier)

    # Pick best
    best_method = min(["sigmoid", "isotonic"], key=lambda m: results[m]["brier"])
    best_cal = calibrators[best_method]

    if results[best_method]["brier"] >= raw_brier:
        logger.info("  calibration didn't improve Brier score, skipping")
        best_cal = None
        best_method = "none"
    else:
        logger.info("  best calibration: %s (brier %.4f)", best_method, results[best_method]["brier"])

    # Plot calibration curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for label, data in results.items():
        prob_true, prob_pred = calibration_curve(y_test, data["proba"], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=f"{label} (brier={data['brier']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"calibration_{name}.png", dpi=150)
    plt.close(fig)

    return best_cal, best_method


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def plot_shap(model, X_test, name: str, output_dir: Path):
    """Generate SHAP feature importance plot."""
    try:
        import shap
        # Use a sample for speed
        sample = X_test.sample(min(500, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, sample, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_{name}.png", dpi=150)
        plt.close("all")
        logger.info("  saved SHAP plot for %s", name)
    except Exception as e:
        logger.warning("  SHAP plot failed for %s: %s", name, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ML models for BTC direction prediction")
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Directory with Parquet training files")
    parser.add_argument("--output-dir", type=str, default="data/models",
                        help="Directory to save models and plots")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per model (default: 30)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_dir)
    if df.empty:
        return

    # Determine available features
    features = get_available_features(df)
    logger.info("using %d features: %s", len(features), features)

    # Drop rows with NaN in outcome
    df = df.dropna(subset=["outcome"])
    df["outcome"] = df["outcome"].astype(int)

    # Use only T+60s snapshots (one per window) — later snapshots leak
    # because price_vs_open at T+240 is nearly the outcome itself
    if "snapshot_offset_s" in df.columns:
        before = len(df)
        df = df[df["snapshot_offset_s"] == 60].reset_index(drop=True)
        logger.info("filtered to T+60s snapshots: %d -> %d rows", before, len(df))

    # Time-based split: 80% train, 20% test
    train_df, test_df = time_split(df, 0.8)

    # Further split train into train/val for calibration (last 20% of train)
    val_split = int(len(train_df) * 0.8)
    cal_train_df = train_df.iloc[:val_split]
    cal_val_df = train_df.iloc[val_split:]

    X_train = train_df[features]
    y_train = train_df["outcome"]
    X_test = test_df[features]
    y_test = test_df["outcome"]
    X_cal_train = cal_train_df[features]
    y_cal_train = cal_train_df["outcome"]
    X_cal_val = cal_val_df[features]
    y_cal_val = cal_val_df["outcome"]

    logger.info("train: %d rows (%s to %s)", len(train_df),
                train_df["window_start"].iloc[0], train_df["window_start"].iloc[-1])
    logger.info("test:  %d rows (%s to %s)", len(test_df),
                test_df["window_start"].iloc[0], test_df["window_start"].iloc[-1])
    logger.info("outcome distribution — train: %.1f%% up, test: %.1f%% up",
                y_train.mean() * 100, y_test.mean() * 100)

    all_results = []
    best_model = None
    best_score = -1
    best_name = ""
    best_calibrator = None

    # --- CatBoost ---
    logger.info("=== CatBoost: Optuna tuning (%d trials) ===", args.trials)
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(lambda t: _catboost_objective(t, X_train, y_train), n_trials=args.trials)
    logger.info("  best CV AUC: %.4f  params: %s", study_cb.best_value, study_cb.best_params)

    cb_model = train_catboost(X_train, y_train, X_test, y_test, study_cb.best_params)
    cb_metrics = evaluate(cb_model, X_test, y_test, "catboost", output_dir)
    cb_metrics["best_params"] = study_cb.best_params
    all_results.append(cb_metrics)
    plot_shap(cb_model, X_test, "catboost", output_dir)

    if cb_metrics["auc_roc"] > best_score:
        best_score = cb_metrics["auc_roc"]
        best_model = cb_model
        best_name = "catboost"

    # --- XGBoost ---
    logger.info("=== XGBoost: Optuna tuning (%d trials) ===", args.trials)
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda t: _xgb_objective(t, X_train, y_train), n_trials=args.trials)
    logger.info("  best CV AUC: %.4f  params: %s", study_xgb.best_value, study_xgb.best_params)

    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, study_xgb.best_params)
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "xgboost", output_dir)
    xgb_metrics["best_params"] = study_xgb.best_params
    all_results.append(xgb_metrics)
    plot_shap(xgb_model, X_test, "xgboost", output_dir)

    if xgb_metrics["auc_roc"] > best_score:
        best_score = xgb_metrics["auc_roc"]
        best_model = xgb_model
        best_name = "xgboost"

    # --- LightGBM ---
    logger.info("=== LightGBM: Optuna tuning (%d trials) ===", args.trials)
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(lambda t: _lgbm_objective(t, X_train, y_train), n_trials=args.trials)
    logger.info("  best CV AUC: %.4f  params: %s", study_lgbm.best_value, study_lgbm.best_params)

    lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test, study_lgbm.best_params)
    lgbm_metrics = evaluate(lgbm_model, X_test, y_test, "lightgbm", output_dir)
    lgbm_metrics["best_params"] = study_lgbm.best_params
    all_results.append(lgbm_metrics)
    plot_shap(lgbm_model, X_test, "lightgbm", output_dir)

    if lgbm_metrics["auc_roc"] > best_score:
        best_score = lgbm_metrics["auc_roc"]
        best_model = lgbm_model
        best_name = "lightgbm"

    # --- Summary ---
    logger.info("=== Model Comparison ===")
    for r in all_results:
        logger.info("  %-10s  acc=%.4f  auc=%.4f  brier=%.4f  logloss=%.4f",
                    r["name"], r["accuracy"], r["auc_roc"], r["brier_score"], r["log_loss"])
    logger.info("  best model: %s (AUC=%.4f)", best_name, best_score)

    # --- Calibration on best model ---
    logger.info("=== Calibrating %s ===", best_name)
    best_calibrator, cal_method = calibrate_model(
        best_model, X_cal_val, y_cal_val, X_test, y_test, output_dir, best_name
    )

    # --- Save best model ---
    if best_name == "catboost":
        model_path = output_dir / "model.cbm"
        best_model.save_model(str(model_path))
    elif best_name == "xgboost":
        model_path = output_dir / "model_xgb.json"
        best_model.save_model(str(model_path))
    else:
        model_path = output_dir / "model_lgbm.txt"
        best_model.booster_.save_model(str(model_path))
    logger.info("saved best model to %s", model_path)

    if best_calibrator is not None:
        cal_path = output_dir / "calibrator.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(best_calibrator, f)
        logger.info("saved calibrator to %s (%s)", cal_path, cal_method)

    # Save feature list used for training
    meta = {
        "features": features,
        "best_model": best_name,
        "best_params": all_results[[r["name"] for r in all_results].index(best_name)]["best_params"],
        "calibration_method": cal_method,
        "metrics": all_results,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("saved training metadata to %s", output_dir / "training_meta.json")


if __name__ == "__main__":
    main()
