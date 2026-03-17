from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Task 2 - Stock Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
FEATURES = ["Open", "High", "Low", "Volume"]


@st.cache_data
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("No data found for ticker.")
    df = df.reset_index()
    df["TargetClose"] = df["Close"].shift(-1)
    df = df.dropna().copy()
    df.to_csv(DATA_DIR / f"{ticker.lower()}_historical.csv", index=False)
    return df


def train(df: pd.DataFrame, model_type: str):
    X = df[FEATURES]
    y = df["TargetClose"]
    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates = df["Date"].iloc[split_idx:]

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    result = pd.DataFrame({"Date": dates, "Actual": y_test.values, "Predicted": preds})
    return result, mae, rmse


def main() -> None:
    st.title("Task 2: Predict Future Stock Prices")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    with col2:
        period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2)
    with col3:
        model_type = st.selectbox("Model", ["Linear Regression", "Random Forest"])

    if st.button("Run Prediction"):
        try:
            df = load_data(ticker, period)
            result, mae, rmse = train(df, model_type)

            m1, m2 = st.columns(2)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")

            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result["Date"], result["Actual"], label="Actual", linewidth=2)
            ax.plot(result["Date"], result["Predicted"], label="Predicted", linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.legend()
            plt.xticks(rotation=25)
            st.pyplot(fig)

            st.subheader("Prediction Table")
            st.dataframe(result.tail(20), use_container_width=True)
        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
