# 📈 Stock Price Predictor (ML Based)

A **Streamlit-based web app** that predicts stock prices using Machine Learning.
Two versions live in this repo:

- **`app_v2.py`** — the full-featured version: technical indicators, six models
  (Random Forest, Gradient Boosting, Linear Regression, XGBoost, LightGBM,
  ARIMA, LSTM/GRU), walk-forward validation, recursive multi-step forecasting
  with confidence bands, risk metrics, and Excel/PDF export with a local run
  history.
- **`codex.py`** — the original minimal baseline (ticker + fixed 5-year
  lookback, three models, one chart). Kept as-is for reference.

---

## 🚀 Features (app_v2.py)

- 🔎 Select from **popular US + NSE stocks** (AAPL, GOOGL, AMZN, MSFT, TSLA,
  META, NFLX, NVDA, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS) or enter any
  custom ticker `yfinance` supports
- 📂 Fetch **historical stock data** via Yahoo Finance (`yfinance`), fixed
  5-year lookback
- 📊 **Technical indicators**: SMA(20/50), EMA(20), RSI(14), MACD(12,26,9),
  Bollinger Bands(20, 2σ) — all derived from Close price alone
- 🧠 **Six models** compared on a held-out test set: Random Forest, Gradient
  Boosting, Linear Regression, XGBoost, LightGBM, ARIMA, and a choice of
  LSTM or GRU (heavy optional dependencies degrade gracefully — a model is
  skipped, not a crash, if its library isn't installed)
- 🔁 **Walk-forward validation** (expanding-window folds) for the fast
  tree/linear models, in addition to the single train/test split
- 🔮 **Recursive multi-step forecasting** beyond the last known price, with a
  confidence band (native for ARIMA, a heuristic `±1.96·σ·√step` for the rest)
- ⚠️ **Risk metrics**: annualized volatility, Sharpe ratio, max drawdown
- 📤 **Export**: Excel (predictions + metrics) and PDF report downloads, plus
  a local SQLite run-history log

---

## 🛠️ Tech Stack

- **Python 3.8+**
- [Streamlit](https://streamlit.io/) – Web app framework
- [yFinance](https://pypi.org/project/yfinance/) – Stock market data
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) – Data handling
- [Plotly](https://plotly.com/python/) – Interactive charts (app_v2.py); [Matplotlib](https://matplotlib.org/) for the original codex.py and PDF export images
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [statsmodels](https://www.statsmodels.org/) (ARIMA), [TensorFlow](https://www.tensorflow.org/) (LSTM/GRU) – Models
- [openpyxl](https://openpyxl.readthedocs.io/) / [reportlab](https://www.reportlab.com/) – Excel / PDF export

---

## 📥 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/DS-Hariprakash/Stock-market-future-prediction.git
cd Stock-market-future-prediction
pip install -r requirements.txt
```

`tensorflow-cpu` is the heaviest install here; if you don't need LSTM/GRU you
can remove it from `requirements.txt` — the app detects its absence and
disables just that model.

## ▶️ Usage

```bash
streamlit run app_v2.py     # full-featured version
# or
streamlit run codex.py      # original minimal version
```

Run history is stored locally in `run_history.db` (SQLite, git-ignored) —
nothing is uploaded or shared outside your machine.
