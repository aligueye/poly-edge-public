"""Model loading and inference.

Loads a trained model from disk using training_meta.json to determine
the model type (catboost, xgboost, or lightgbm) and feature order.
If no model or meta file exists, returns "hold" with confidence 0.0.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

# Feature order must match training (Sprint 5). Defined here as single source of truth.
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


class Model:
    """Model wrapper for directional prediction. Supports CatBoost, XGBoost, and LightGBM."""

    def __init__(self, model_dir: str = "."):
        self._model = None
        self._model_name = None
        self._features = FEATURE_NAMES
        self._load(model_dir)

    def _load(self, model_dir: str):
        # Try loading from training_meta.json first
        meta_path = os.path.join(model_dir, "training_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                self._model_name = meta.get("best_model", "catboost")
                self._features = meta.get("features", FEATURE_NAMES)
                logger.info("meta: best_model=%s, %d features", self._model_name, len(self._features))
            except Exception as e:
                logger.error("failed to read training_meta.json: %s", e)

        # Load model based on type
        if self._model_name == "xgboost":
            self._load_xgboost(model_dir)
        elif self._model_name == "lightgbm":
            self._load_lightgbm(model_dir)
        else:
            self._load_catboost(model_dir)

    def _load_catboost(self, model_dir: str):
        path = os.path.join(model_dir, "model.cbm")
        if not os.path.exists(path):
            logger.warning("no catboost model at %s", path)
            return
        try:
            from catboost import CatBoostClassifier
            self._model = CatBoostClassifier()
            self._model.load_model(path)
            self._model_name = "catboost"
            logger.info("loaded catboost model from %s", path)
        except Exception as e:
            logger.error("failed to load catboost model: %s", e)

    def _load_xgboost(self, model_dir: str):
        path = os.path.join(model_dir, "model_xgb.json")
        if not os.path.exists(path):
            logger.warning("no xgboost model at %s", path)
            return
        try:
            from xgboost import XGBClassifier
            self._model = XGBClassifier()
            self._model.load_model(path)
            self._model_name = "xgboost"
            logger.info("loaded xgboost model from %s", path)
        except Exception as e:
            logger.error("failed to load xgboost model: %s", e)

    def _load_lightgbm(self, model_dir: str):
        path = os.path.join(model_dir, "model_lgbm.txt")
        if not os.path.exists(path):
            logger.warning("no lightgbm model at %s", path)
            return
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=path)
            self._model_name = "lightgbm"
            logger.info("loaded lightgbm model from %s", path)
        except Exception as e:
            logger.error("failed to load lightgbm model: %s", e)

    def predict(self, features: dict) -> tuple:
        """Predict direction from feature dict.

        Returns:
            (action, confidence) where action is "buy_yes", "buy_no", or "hold"
            and confidence is 0.0-1.0.
        """
        if self._model is None:
            return "hold", 0.0

        # Build feature vector in correct order from training_meta.json features
        X = [[features.get(name, 0.0) for name in self._features]]

        try:
            if self._model_name == "lightgbm":
                # LightGBM Booster returns raw probabilities
                proba = self._model.predict(X)[0]
                p_up = proba if isinstance(proba, float) else 0.5
            else:
                # CatBoost and XGBoost use predict_proba
                proba = self._model.predict_proba(X)[0]
                p_up = proba[1] if len(proba) > 1 else 0.5

            if p_up > 0.5:
                return "buy_yes", float(p_up)
            elif p_up < 0.5:
                return "buy_no", float(1.0 - p_up)
            else:
                return "hold", 0.5
        except Exception as e:
            logger.error("prediction failed: %s", e)
            return "hold", 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None
