"""Model registry, feature engineering, training and evaluation.

Design notes (kept consistent so Model Comparison and Forecast tabs agree):
  - Every model predicts scaled Close only; the target scaler (`scaler_y`) is
    always a MinMaxScaler fit on Close alone, so inverse-transforming any
    model's prediction is always `scaler_y.inverse_transform(pred)`.
  - "Flat" models (RandomForest, GradientBoosting, LinearRegression, XGBoost,
    LightGBM) see a flattened 60-day Close lookback window concatenated with
    the *previous day's* indicator snapshot (so no same-day leakage).
  - LSTM/GRU see the same lookback window as a multi-channel sequence
    (timesteps x channels) instead of a flat vector.
  - ARIMA is univariate on raw (unscaled) Close and forecasts natively.
  - Heavy optional dependencies (xgboost, lightgbm, tensorflow, statsmodels)
    are imported defensively -- if unavailable, that model is simply left out
    of the registry rather than crashing the app.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from indicators import INDICATOR_COLUMNS

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


LSTM_CHANNEL_COLUMNS = ["Close", "SMA_20", "EMA_20", "RSI_14", "MACD"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nonzero = y_true != 0
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def fit_scalers(train_df: pd.DataFrame):
    """Fit all scalers on TRAINING rows only (no leakage into test/forecast)."""
    scaler_y = MinMaxScaler().fit(train_df[["Close"]])
    scaler_indicators = MinMaxScaler().fit(train_df[INDICATOR_COLUMNS])
    scaler_seq = MinMaxScaler().fit(train_df[LSTM_CHANNEL_COLUMNS])
    return scaler_y, scaler_indicators, scaler_seq


def build_flat_features(feat_df: pd.DataFrame, sequence_length: int, scaler_y, scaler_indicators):
    """Flattened close-window + previous-day indicator snapshot -> for tree/linear models."""
    close_scaled = scaler_y.transform(feat_df[["Close"]]).flatten()
    ind_scaled = scaler_indicators.transform(feat_df[INDICATOR_COLUMNS])

    X, y = [], []
    for i in range(sequence_length, len(feat_df)):
        window = close_scaled[i - sequence_length:i]
        prev_day_indicators = ind_scaled[i - 1]
        X.append(np.concatenate([window, prev_day_indicators]))
        y.append(close_scaled[i])
    return np.array(X), np.array(y)


def build_sequence_features(feat_df: pd.DataFrame, sequence_length: int, scaler_y, scaler_seq):
    """3D (samples, timesteps, channels) window -> for LSTM/GRU."""
    seq_scaled = scaler_seq.transform(feat_df[LSTM_CHANNEL_COLUMNS])
    close_scaled = scaler_y.transform(feat_df[["Close"]]).flatten()

    X, y = [], []
    for i in range(sequence_length, len(feat_df)):
        X.append(seq_scaled[i - sequence_length:i])
        y.append(close_scaled[i])
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------
# Model registry (flat / sklearn-style models only; LSTM & ARIMA handled separately)
# --------------------------------------------------------------------------

def get_flat_model_registry(random_state: int = 42) -> dict:
    registry = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "Linear Regression": LinearRegression(),
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)
    if LIGHTGBM_AVAILABLE:
        registry["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1)
    return registry


def train_flat_models(models: dict, X_train, y_train, X_test, y_test, scaler_y) -> dict:
    """Fit each model, return {name: {'pred': prices, 'metrics': {...}, 'model': fitted_model}}."""
    real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_scaled = model.predict(X_test)
        pred_prices = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        results[name] = {
            "pred": pred_prices,
            "metrics": compute_metrics(real_prices, pred_prices),
            "model": model,
        }
    return results, real_prices


# --------------------------------------------------------------------------
# LSTM / GRU
# --------------------------------------------------------------------------

def build_recurrent_model(cell_type: str, input_shape, units: int = 32):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed; LSTM/GRU unavailable.")
    layer_cls = LSTM if cell_type == "LSTM" else GRU
    model = Sequential([
        Input(shape=input_shape),
        layer_cls(units),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_recurrent_model(cell_type: str, X_train, y_train, X_test, y_test, scaler_y,
                           epochs: int = 8, batch_size: int = 32):
    model = build_recurrent_model(cell_type, input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    pred_scaled = model.predict(X_test, verbose=0).flatten()
    real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_prices = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return {
        "pred": pred_prices,
        "metrics": compute_metrics(real_prices, pred_prices),
        "model": model,
    }, real_prices


# --------------------------------------------------------------------------
# ARIMA
# --------------------------------------------------------------------------

def train_arima(train_close: pd.Series, test_len: int, order=(5, 1, 0)):
    if not ARIMA_AVAILABLE:
        raise RuntimeError("statsmodels is not installed; ARIMA unavailable.")
    fitted = ARIMA(train_close.values, order=order).fit()
    forecast = fitted.forecast(steps=test_len)
    return np.asarray(forecast), fitted


# --------------------------------------------------------------------------
# Walk-forward validation (applied to the fast flat models only, for runtime)
# --------------------------------------------------------------------------

def walk_forward_folds(n_rows: int, sequence_length: int, n_folds: int = 3, min_train_ratio: float = 0.5):
    usable = n_rows - sequence_length
    test_size = max(20, usable // (n_folds + 3))
    folds = []
    for k in range(n_folds):
        test_end = n_rows - k * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end - sequence_length < int(min_train_ratio * usable):
            break
        folds.append((train_end, test_start, test_end))
    return list(reversed(folds))


def run_walk_forward(feat_df: pd.DataFrame, sequence_length: int, n_folds: int = 3,
                      random_state: int = 42):
    """Average RMSE/MAE/MAPE per flat model across expanding-window folds.

    Each fold refits its own scalers on that fold's training slice only, so no
    fold ever leaks information from data after its own test window.
    """
    folds = walk_forward_folds(len(feat_df), sequence_length, n_folds=n_folds)
    fold_metrics = {}  # name -> list of metric dicts

    for train_end, test_start, test_end in folds:
        train_raw = feat_df.iloc[:train_end]
        test_raw = feat_df.iloc[test_start - sequence_length:test_end]

        scaler_y, scaler_indicators, _ = fit_scalers(train_raw)
        X_train, y_train = build_flat_features(train_raw, sequence_length, scaler_y, scaler_indicators)
        X_test, y_test = build_flat_features(test_raw, sequence_length, scaler_y, scaler_indicators)

        models = get_flat_model_registry(random_state=random_state)
        results, _ = train_flat_models(models, X_train, y_train, X_test, y_test, scaler_y)
        for name, res in results.items():
            fold_metrics.setdefault(name, []).append(res["metrics"])

    averaged = {
        name: {metric: float(np.mean([m[metric] for m in metrics_list]))
               for metric in ["RMSE", "MAE", "MAPE"]}
        for name, metrics_list in fold_metrics.items()
    }
    return averaged, len(folds)
