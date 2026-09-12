"""Technical indicators derived purely from the Close price.

Kept Close-only (no Volume/OHLC) so that indicator values can be recomputed
from a synthetic close-price series during recursive multi-step forecasting
(see forecasting.py) -- an indicator that depended on Volume could not be
extended into the forecast horizon.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)  # neutral RSI where loss is 0


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


INDICATOR_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "BB_Upper", "BB_Mid", "BB_Lower",
]


def add_indicators(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Return a copy of df with indicator columns added, NaN warm-up rows dropped."""
    out = df.copy()
    close = out[close_col]

    out["SMA_20"] = sma(close, 20)
    out["SMA_50"] = sma(close, 50)
    out["EMA_20"] = ema(close, 20)
    out["RSI_14"] = rsi(close, 14)
    macd_line, signal_line, _ = macd(close)
    out["MACD"] = macd_line
    out["MACD_Signal"] = signal_line
    upper, mid, lower = bollinger_bands(close)
    out["BB_Upper"] = upper
    out["BB_Mid"] = mid
    out["BB_Lower"] = lower

    return out.dropna()
