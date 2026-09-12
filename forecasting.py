"""Recursive multi-step forecasting.

Each step predicts the next Close price, appends it to a synthetic history,
then recomputes indicators from that extended history before predicting the
next step. This only works because every indicator in indicators.py is
derived purely from Close (see that module's docstring) -- no Volume/OHLC
dependency to "run out of" future values for.
"""

import numpy as np
import pandas as pd

from indicators import add_indicators, INDICATOR_COLUMNS
from ml_models import LSTM_CHANNEL_COLUMNS


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.BDay(1)


def recursive_forecast_flat(model, feat_df: pd.DataFrame, sequence_length: int,
                             scaler_y, scaler_indicators, horizon: int):
    history_close = feat_df["Close"].copy()
    forecasts = []
    for _ in range(horizon):
        temp_with_ind = add_indicators(pd.DataFrame({"Close": history_close}))
        if len(temp_with_ind) < sequence_length:
            raise RuntimeError("Not enough history to compute indicators for forecasting.")

        close_scaled_full = scaler_y.transform(temp_with_ind[["Close"]]).flatten()
        window = close_scaled_full[-sequence_length:]
        last_indicators = temp_with_ind[INDICATOR_COLUMNS].iloc[[-1]]
        ind_scaled = scaler_indicators.transform(last_indicators).flatten()

        X_next = np.concatenate([window, ind_scaled]).reshape(1, -1)
        pred_scaled = model.predict(X_next)[0]
        pred_price = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])

        forecasts.append(pred_price)
        history_close.loc[_next_business_day(history_close.index[-1])] = pred_price

    future_dates = history_close.index[-horizon:]
    return np.array(forecasts), future_dates


def recursive_forecast_sequence(model, feat_df: pd.DataFrame, sequence_length: int,
                                 scaler_y, scaler_seq, horizon: int):
    history_close = feat_df["Close"].copy()
    forecasts = []
    for _ in range(horizon):
        temp_with_ind = add_indicators(pd.DataFrame({"Close": history_close}))
        if len(temp_with_ind) < sequence_length:
            raise RuntimeError("Not enough history to compute indicators for forecasting.")

        last_window = temp_with_ind[LSTM_CHANNEL_COLUMNS].iloc[-sequence_length:]
        seq_scaled = scaler_seq.transform(last_window)
        X_next = seq_scaled.reshape(1, sequence_length, len(LSTM_CHANNEL_COLUMNS))
        pred_scaled = float(model.predict(X_next, verbose=0).flatten()[0])
        pred_price = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])

        forecasts.append(pred_price)
        history_close.loc[_next_business_day(history_close.index[-1])] = pred_price

    future_dates = history_close.index[-horizon:]
    return np.array(forecasts), future_dates


def heuristic_confidence_band(residual_std: float, horizon: int, z: float = 1.96):
    """Widening band: z * residual_std * sqrt(step). A heuristic, not a fitted
    prediction interval -- uncertainty is assumed to grow with sqrt(time)."""
    steps = np.arange(1, horizon + 1)
    return z * residual_std * np.sqrt(steps)
