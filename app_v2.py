"""Stock Price Predictor v2 -- full-featured tabbed version.

v1 (codex.py) is kept as the minimal baseline. This version adds: technical
indicators, more models (XGBoost/LightGBM/ARIMA/LSTM-GRU), walk-forward
validation, recursive multi-step forecasting with confidence bands, risk
metrics, and Excel/PDF export with a local run-history log.

Sidebar interaction is kept to what each feature actually requires: ticker
(+ optional custom ticker text), a recurrent-model choice (only shown if
TensorFlow is installed), a forecast horizon, and one "Load Data & Predict"
button that runs the whole pipeline once.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from indicators import add_indicators, INDICATOR_COLUMNS
import ml_models as mlm
import forecasting as fc
import risk_metrics as rm
import export_utils as exp
import db_utils as db

# --- Palette (fixed categorical order; entity -> slot never changes) -----
PALETTE = {
    "Real Price": "#2a78d6",
    "Random Forest": "#eb6834",
    "Gradient Boosting": "#1baf7a",
    "Linear Regression": "#eda100",
    "XGBoost": "#e87ba4",
    "LightGBM": "#008300",
    "ARIMA": "#4a3aa7",
    "LSTM": "#e34948",
    "GRU": "#e34948",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
MUTED = "#898781"

SEQUENCE_LENGTH = 60
LOOKBACK_YEARS = 5
RISK_FREE_RATE = 0.0  # annualized; edit here to change the Sharpe ratio baseline

st.set_page_config(page_title="Stock Price Predictor v2", layout="wide")
st.title("📈 Stock Price Predictor v2 (ML Based)")

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Stock Settings")
popular_stocks = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "NVIDIA (NVDA)": "NVDA",
    "Reliance -- NSE (RELIANCE.NS)": "RELIANCE.NS",
    "TCS -- NSE (TCS.NS)": "TCS.NS",
    "Infosys -- NSE (INFY.NS)": "INFY.NS",
    "HDFC Bank -- NSE (HDFCBANK.NS)": "HDFCBANK.NS",
    "Custom ticker...": None,
}
ticker_label = st.sidebar.selectbox("Select Stock", options=list(popular_stocks.keys()))
if popular_stocks[ticker_label] is None:
    ticker = st.sidebar.text_input(
        "Enter ticker symbol", value="", placeholder="e.g. WIPRO.NS, BTC-USD, GC=F"
    ).strip().upper()
else:
    ticker = popular_stocks[ticker_label]

if mlm.TENSORFLOW_AVAILABLE:
    recurrent_cell = st.sidebar.radio("Recurrent model", ["LSTM", "GRU"], horizontal=True)
else:
    recurrent_cell = None
    st.sidebar.info("TensorFlow not installed -- LSTM/GRU disabled.")

if not mlm.XGBOOST_AVAILABLE:
    st.sidebar.caption("XGBoost not installed -- skipped.")
if not mlm.LIGHTGBM_AVAILABLE:
    st.sidebar.caption("LightGBM not installed -- skipped.")
if not mlm.ARIMA_AVAILABLE:
    st.sidebar.caption("statsmodels not installed -- ARIMA skipped.")

forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 10)

end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.DateOffset(years=LOOKBACK_YEARS)

run_clicked = st.sidebar.button("Load Data & Predict", type="primary")


# --------------------------------------------------------------------------
# Pipeline (runs once per click, cached in session_state)
# --------------------------------------------------------------------------
def line_chart(series_dict: dict, title: str, y_title: str, x=None) -> go.Figure:
    fig = go.Figure()
    for name, y in series_dict.items():
        fig.add_trace(go.Scatter(
            x=x if x is not None else list(range(len(y))),
            y=y, mode="lines", name=name,
            line=dict(color=PALETTE.get(name, MUTED), width=2),
        ))
    fig.update_layout(
        title=title, yaxis_title=y_title, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


if run_clicked:
    if not ticker:
        st.error("Enter a ticker symbol.")
        st.stop()

    with st.spinner(f"Downloading {ticker}..."):
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error(f"No data returned for {ticker}. Check the symbol and try again.")
        st.stop()

    with st.spinner("Computing indicators..."):
        feat_df = add_indicators(data[["Close"]])
        if len(feat_df) < SEQUENCE_LENGTH * 3:
            st.error("Not enough history after indicator warm-up to train reliably. "
                     "Pick a longer-lived ticker.")
            st.stop()

    split_idx = int(0.8 * len(feat_df))
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx - SEQUENCE_LENGTH:]

    scaler_y, scaler_indicators, scaler_seq = mlm.fit_scalers(train_df)

    with st.spinner("Training tree/linear models..."):
        X_train, y_train = mlm.build_flat_features(train_df, SEQUENCE_LENGTH, scaler_y, scaler_indicators)
        X_test, y_test = mlm.build_flat_features(test_df, SEQUENCE_LENGTH, scaler_y, scaler_indicators)
        flat_models = mlm.get_flat_model_registry()
        results, real_prices = mlm.train_flat_models(flat_models, X_train, y_train, X_test, y_test, scaler_y)

    if mlm.ARIMA_AVAILABLE:
        with st.spinner("Training ARIMA..."):
            arima_forecast, arima_fitted = mlm.train_arima(train_df["Close"], len(real_prices))
            results["ARIMA"] = {
                "pred": arima_forecast,
                "metrics": mlm.compute_metrics(real_prices, arima_forecast),
                "model": arima_fitted,
            }

    if mlm.TENSORFLOW_AVAILABLE and recurrent_cell:
        with st.spinner(f"Training {recurrent_cell}..."):
            X_train_seq, y_train_seq = mlm.build_sequence_features(train_df, SEQUENCE_LENGTH, scaler_y, scaler_seq)
            X_test_seq, y_test_seq = mlm.build_sequence_features(test_df, SEQUENCE_LENGTH, scaler_y, scaler_seq)
            rec_result, _ = mlm.train_recurrent_model(
                recurrent_cell, X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_y
            )
            results[recurrent_cell] = rec_result

    with st.spinner("Running walk-forward validation (tree/linear models)..."):
        wf_metrics, wf_folds = mlm.run_walk_forward(feat_df, SEQUENCE_LENGTH, n_folds=3)

    risk = rm.risk_summary(data["Close"], RISK_FREE_RATE)

    test_dates = feat_df.index[split_idx:split_idx + len(real_prices)]

    st.session_state.update({
        "ticker": ticker,
        "data": data,
        "feat_df": feat_df,
        "results": results,
        "real_prices": real_prices,
        "test_dates": test_dates,
        "scaler_y": scaler_y,
        "scaler_indicators": scaler_indicators,
        "scaler_seq": scaler_seq,
        "recurrent_cell": recurrent_cell,
        "wf_metrics": wf_metrics,
        "wf_folds": wf_folds,
        "risk": risk,
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon": forecast_horizon,
    })

if "results" not in st.session_state:
    st.info("Set your options in the sidebar and click **Load Data & Predict** to begin.")
    st.stop()

# --------------------------------------------------------------------------
# Pull cached state for tabs
# --------------------------------------------------------------------------
ticker = st.session_state["ticker"]
data = st.session_state["data"]
feat_df = st.session_state["feat_df"]
results = st.session_state["results"]
real_prices = st.session_state["real_prices"]
test_dates = st.session_state["test_dates"]
scaler_y = st.session_state["scaler_y"]
scaler_indicators = st.session_state["scaler_indicators"]
scaler_seq = st.session_state["scaler_seq"]
recurrent_cell = st.session_state["recurrent_cell"]
wf_metrics = st.session_state["wf_metrics"]
wf_folds = st.session_state["wf_folds"]
risk = st.session_state["risk"]
sequence_length = st.session_state["sequence_length"]

metrics_df = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
best_model_name = metrics_df["RMSE"].idxmin()

predictions_df = pd.DataFrame({"Real Price": real_prices}, index=test_dates)
for name, r in results.items():
    predictions_df[name] = r["pred"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "🧠 Model Comparison", "🔮 Forecast", "⚠️ Risk Metrics", "📤 Export & History"]
)

# --------------------------------------------------------------------------
# Tab 1: Overview
# --------------------------------------------------------------------------
with tab1:
    st.subheader(f"{ticker} -- Close Price & Indicators ({LOOKBACK_YEARS}y)")

    price_fig = line_chart(
        {
            "Real Price": feat_df["Close"].values,
            "SMA 20": feat_df["SMA_20"].values,
            "SMA 50": feat_df["SMA_50"].values,
            "EMA 20": feat_df["EMA_20"].values,
        },
        "Close Price with Moving Averages", "Price", x=feat_df.index,
    )
    price_fig.data[1].line.color = STATUS["warning"]
    price_fig.data[2].line.color = STATUS["serious"]
    price_fig.data[3].line.color = STATUS["good"]
    st.plotly_chart(price_fig, use_container_width=True)

    bb_fig = go.Figure()
    bb_fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["BB_Upper"], name="Upper Band",
                                 line=dict(color=MUTED, width=1, dash="dot")))
    bb_fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["BB_Lower"], name="Lower Band",
                                 line=dict(color=MUTED, width=1, dash="dot"), fill="tonexty",
                                 fillcolor="rgba(137,135,129,0.12)"))
    bb_fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Close"], name="Close",
                                 line=dict(color=PALETTE["Real Price"], width=2)))
    bb_fig.update_layout(title="Bollinger Bands (20, 2σ)", yaxis_title="Price", hovermode="x unified")
    st.plotly_chart(bb_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["RSI_14"], name="RSI 14",
                                      line=dict(color=PALETTE["Real Price"], width=2)))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color=MUTED)
        rsi_fig.add_hline(y=30, line_dash="dash", line_color=MUTED)
        rsi_fig.update_layout(title="RSI (14)", yaxis_title="RSI", yaxis_range=[0, 100],
                               hovermode="x unified")
        st.plotly_chart(rsi_fig, use_container_width=True)
    with col2:
        macd_colors = [PALETTE["Real Price"] if v >= 0 else PALETTE["ARIMA"]
                        for v in (feat_df["MACD"] - feat_df["MACD_Signal"])]
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=feat_df.index, y=feat_df["MACD"] - feat_df["MACD_Signal"],
                                   name="Histogram", marker_color=macd_colors, opacity=0.5))
        macd_fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["MACD"], name="MACD",
                                       line=dict(color=PALETTE["Real Price"], width=2)))
        macd_fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["MACD_Signal"], name="Signal",
                                       line=dict(color=PALETTE["Random Forest"], width=2)))
        macd_fig.update_layout(title="MACD (12, 26, 9)", yaxis_title="MACD", hovermode="x unified")
        st.plotly_chart(macd_fig, use_container_width=True)

    st.markdown(
        "**Logic summary:** SMA/EMA smooth price to show trend direction; Bollinger "
        "Bands (SMA20 ± 2 standard deviations) show volatility range; RSI (14) flags "
        "overbought (>70) / oversold (<30) momentum; MACD (12,26,9) tracks trend "
        "momentum via the gap between two EMAs. All indicators are derived from Close "
        "price only, so they can be recomputed during recursive forecasting (Tab 3)."
    )

# --------------------------------------------------------------------------
# Tab 2: Model Comparison
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Predicted vs Real Price (held-out 20% test set)")
    series = {"Real Price": real_prices}
    series.update({name: r["pred"] for name, r in results.items()})
    st.plotly_chart(line_chart(series, "Model Comparison", "Price", x=test_dates),
                     use_container_width=True)

    st.subheader("Test-set metrics")
    st.dataframe(metrics_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="#cde2fb"))
    st.caption(f"Best model on this test set: **{best_model_name}**")

    st.subheader(f"Walk-forward validation ({wf_folds} expanding-window folds)")
    if wf_metrics:
        st.dataframe(pd.DataFrame(wf_metrics).T)
    else:
        st.info("Not enough history for walk-forward folds on this ticker.")

    st.markdown(
        "**Logic summary:** Tree/linear models see a flattened 60-day Close window "
        "plus yesterday's indicator snapshot; ARIMA fits directly on raw Close and "
        f"forecasts natively; {recurrent_cell or 'LSTM/GRU'} sees the same 60-day "
        "window as a multi-channel sequence. The single test-set table above is one "
        "80/20 split; the walk-forward table re-runs the fast tree/linear models "
        "across multiple expanding-window folds (each with its own scaler fit only "
        "on that fold's training data) for a more robust estimate. ARIMA/LSTM/GRU "
        "are evaluated only on the single split, for runtime."
    )

# --------------------------------------------------------------------------
# Tab 3: Forecast
# --------------------------------------------------------------------------
with tab3:
    horizon = st.session_state["forecast_horizon"]
    model_choice = st.selectbox("Model to forecast with", options=list(results.keys()),
                                 index=list(results.keys()).index(best_model_name))

    with st.spinner(f"Forecasting {horizon} day(s) ahead with {model_choice}..."):
        if model_choice == "ARIMA":
            fitted = results["ARIMA"]["model"]
            fc_result = fitted.get_forecast(steps=horizon)
            forecast_vals = np.asarray(fc_result.predicted_mean)
            ci = np.asarray(fc_result.conf_int(alpha=0.05))
            lower, upper = ci[:, 0], ci[:, 1]
            future_dates = pd.bdate_range(
                start=feat_df.index[-1] + pd.offsets.BDay(1), periods=horizon
            )
        elif model_choice in ("LSTM", "GRU"):
            model = results[model_choice]["model"]
            forecast_vals, future_dates = fc.recursive_forecast_sequence(
                model, feat_df, sequence_length, scaler_y, scaler_seq, horizon
            )
            residual_std = float(np.std(real_prices - results[model_choice]["pred"]))
            band = fc.heuristic_confidence_band(residual_std, horizon)
            lower, upper = forecast_vals - band, forecast_vals + band
        else:
            model = results[model_choice]["model"]
            forecast_vals, future_dates = fc.recursive_forecast_flat(
                model, feat_df, sequence_length, scaler_y, scaler_indicators, horizon
            )
            residual_std = float(np.std(real_prices - results[model_choice]["pred"]))
            band = fc.heuristic_confidence_band(residual_std, horizon)
            lower, upper = forecast_vals - band, forecast_vals + band

    fc_fig = go.Figure()
    recent = feat_df["Close"].iloc[-90:]
    fc_fig.add_trace(go.Scatter(x=recent.index, y=recent.values, name="Recent Close",
                                 line=dict(color=MUTED, width=2)))
    fc_fig.add_trace(go.Scatter(x=future_dates, y=upper, name="Upper band", showlegend=False,
                                 mode="lines", line=dict(width=0)))
    fc_fig.add_trace(go.Scatter(x=future_dates, y=lower, name="Confidence band",
                                 mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(42,120,214,0.15)"))
    fc_fig.add_trace(go.Scatter(x=future_dates, y=forecast_vals, name=f"{model_choice} forecast",
                                 line=dict(color=PALETTE.get(model_choice, PALETTE["Real Price"]), width=2)))
    fc_fig.update_layout(title=f"{ticker} -- {horizon}-day forecast", yaxis_title="Price",
                          hovermode="x unified")
    st.plotly_chart(fc_fig, use_container_width=True)

    st.dataframe(pd.DataFrame({"Forecast": forecast_vals, "Lower": lower, "Upper": upper},
                               index=future_dates))

    st.markdown(
        "**Logic summary:** the forecast is recursive -- each predicted day is fed "
        "back in as history, indicators are recomputed from the growing synthetic "
        "series, and the next day is predicted from that. ARIMA's confidence interval "
        "comes from its own statistical model; every other model's band is a "
        "heuristic (`±1.96 × test-set residual std × √step`) that widens with the "
        "square root of time -- it is not a fitted prediction interval, and forecast "
        "error compounds with each recursive step, so treat longer horizons with "
        "proportionally more skepticism."
    )

# --------------------------------------------------------------------------
# Tab 4: Risk Metrics
# --------------------------------------------------------------------------
with tab4:
    st.subheader(f"{ticker} -- Risk Metrics ({LOOKBACK_YEARS}y)")

    vol = risk["annualized_volatility"]
    sharpe = risk["sharpe_ratio"]
    dd = risk["max_drawdown"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized Volatility", f"{vol * 100:.1f}%")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}",
              help=f"Risk-free rate assumed: {RISK_FREE_RATE * 100:.1f}% annualized")
    c3.metric("Max Drawdown", f"{dd * 100:.1f}%")

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=risk["drawdown_series"].index, y=risk["drawdown_series"].values * 100,
                                 name="Drawdown", line=dict(color=STATUS["critical"], width=2),
                                 fill="tozeroy", fillcolor="rgba(208,59,59,0.15)"))
    dd_fig.update_layout(title="Drawdown from running peak", yaxis_title="Drawdown (%)",
                          hovermode="x unified")
    st.plotly_chart(dd_fig, use_container_width=True)

    st.markdown(
        "**Logic summary:** volatility is the annualized standard deviation of daily "
        "returns (`daily_std × √252`); Sharpe ratio is annualized excess return over "
        "return volatility (risk-free rate is a fixed constant in code, not a live "
        "rate feed); max drawdown is the worst peak-to-trough decline over the "
        f"{LOOKBACK_YEARS}-year window."
    )

# --------------------------------------------------------------------------
# Tab 5: Export & History
# --------------------------------------------------------------------------
with tab5:
    st.subheader("Export current run")
    col1, col2 = st.columns(2)
    with col1:
        excel_bytes = exp.export_to_excel(predictions_df, metrics_df)
        st.download_button("Download Excel (.xlsx)", data=excel_bytes,
                            file_name=f"{ticker}_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2:
        pdf_bytes = exp.export_to_pdf(ticker, predictions_df, metrics_df)
        st.download_button("Download PDF report", data=pdf_bytes,
                            file_name=f"{ticker}_report.pdf", mime="application/pdf")

    st.divider()
    st.subheader("Run history (local SQLite)")
    if st.button("Log this run to history"):
        db.log_run(ticker, best_model_name, metrics_df.loc[best_model_name, "RMSE"],
                   metrics_df.loc[best_model_name, "MAE"], metrics_df.loc[best_model_name, "MAPE"])
        st.success(f"Logged {ticker} run (best model: {best_model_name}).")

    history_df = db.fetch_history()
    st.dataframe(history_df, use_container_width=True)
    if st.button("Clear history", type="secondary") and not history_df.empty:
        db.clear_history()
        st.rerun()

    st.markdown(
        "**Logic summary:** Excel export writes Predictions + Metrics as separate "
        "sheets; the PDF embeds a static chart image plus the metrics table "
        "(reportlab can't render interactive Plotly, so it's re-rendered in "
        "matplotlib for the PDF only). History logging is manual (click the button) "
        "so exploratory clicks don't clutter the log; it's stored in a local "
        "`run_history.db` SQLite file, not shared or uploaded anywhere."
    )
