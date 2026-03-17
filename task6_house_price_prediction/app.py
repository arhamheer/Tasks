from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Task 6 - House Price Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@st.cache_data
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

    return Pipeline([("prep", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])


def main() -> None:
    st.title("Task 6: House Price Prediction")

    df = load_data()
    st.dataframe(df.head(), use_container_width=True)

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_pipeline(X)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")

    result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result_df["Actual"], result_df["Predicted"], alpha=0.6)
    minv, maxv = result_df["Actual"].min(), result_df["Actual"].max()
    ax.plot([minv, maxv], [minv, maxv], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    st.subheader("Latest Predictions")
    st.dataframe(result_df.tail(25), use_container_width=True)


if __name__ == "__main__":
    main()
