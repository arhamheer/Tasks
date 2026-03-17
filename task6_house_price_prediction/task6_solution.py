from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    dataset = fetch_openml(name="house_prices", as_frame=True)
    df = dataset.frame.copy()
    df.to_csv(DATA_DIR / "house_prices.csv", index=False)
    return df


def build_pipeline(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    return Pipeline([("prep", preprocessor), ("model", model)])


def main() -> None:
    df = load_data()

    target_col = "SalePrice"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    out_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
    out_df.to_csv(DATA_DIR / "predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(out_df["Actual"], out_df["Predicted"], alpha=0.6)
    plt.plot(
        [out_df["Actual"].min(), out_df["Actual"].max()],
        [out_df["Actual"].min(), out_df["Actual"].max()],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
