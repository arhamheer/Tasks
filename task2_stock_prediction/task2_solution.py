from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


FEATURES = ["Open", "High", "Low", "Volume"]


def load_stock_data(ticker: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No stock data returned for ticker {ticker}.")

    df = df.reset_index()
    df["TargetClose"] = df["Close"].shift(-1)
    df = df.dropna().copy()

    csv_path = DATA_DIR / f"{ticker.lower()}_historical.csv"
    df.to_csv(csv_path, index=False)
    return df


def train_model(df: pd.DataFrame, model_name: str = "linear"):
    X = df[FEATURES]
    y = df["TargetClose"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = df["Date"].iloc[split_idx:]

    if model_name == "forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    results = pd.DataFrame(
        {
            "Date": test_dates,
            "ActualClose": y_test.values,
            "PredictedClose": preds,
        }
    )
    return model, results, mae, rmse


def plot_results(results: pd.DataFrame, ticker: str, model_name: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(results["Date"], results["ActualClose"], label="Actual", linewidth=2)
    plt.plot(results["Date"], results["PredictedClose"], label="Predicted", linewidth=2)
    plt.title(f"{ticker} - Actual vs Predicted Next-Day Close ({model_name})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{ticker.lower()}_{model_name}_actual_vs_pred.png"
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    ticker_symbol = "AAPL"
    selected_model = "linear"  # Change to "forest" for Random Forest

    stock_df = load_stock_data(ticker=ticker_symbol, period="2y")
    _, result_df, mae_score, rmse_score = train_model(stock_df, selected_model)
    plot_results(result_df, ticker_symbol, selected_model)

    print("Model:", selected_model)
    print(f"MAE: {mae_score:.4f}")
    print(f"RMSE: {rmse_score:.4f}")
    print("Saved stock data and prediction plot.")
