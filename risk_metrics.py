"""Risk metrics computed from a Close price series."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(close: pd.Series):
    """Return (max_drawdown_pct, drawdown_series)."""
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(drawdown.min()), drawdown


def risk_summary(close: pd.Series, risk_free_rate: float = 0.0) -> dict:
    returns = daily_returns(close)
    dd_pct, dd_series = max_drawdown(close)
    return {
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "max_drawdown": dd_pct,
        "drawdown_series": dd_series,
    }
