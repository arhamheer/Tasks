from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Task 3 - Heart Disease", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@st.cache_data
def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(DATA_DIR / "heart_disease.csv", index=False)
    return df


def train_model(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=1200)),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    importance = (
        pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
        .abs()
        .sort_values(ascending=False)
    )

    return model, X_test, y_test, preds, probs, acc, auc, importance


def main() -> None:
    st.title("Task 3: Heart Disease Prediction")

    df = load_df()
    st.subheader("Data Snapshot")
    st.dataframe(df.head(), use_container_width=True)

    model, X_test, y_test, preds, probs, acc, auc, importance = train_model(df)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("ROC-AUC", f"{auc:.4f}")

    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, probs, ax=ax_roc)
    st.pyplot(fig_roc)

    st.subheader("Feature Importance (Logistic Coefficients)")
    st.bar_chart(importance)

    st.subheader("Quick EDA")
    fig_box, ax_box = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="target", y="age", ax=ax_box)
    st.pyplot(fig_box)

    st.subheader("Manual Risk Check")
    with st.form("risk_form"):
        inputs = {}
        for col in X_test.columns:
            default_value = float(df[col].median())
            inputs[col] = st.number_input(col, value=default_value)
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([inputs])
        prob = model.predict_proba(input_df)[0, 1]
        pred = int(prob >= 0.5)
        label = "At Risk" if pred == 1 else "Low Risk"
        st.info(f"Prediction: {label} (probability={prob:.3f})")


if __name__ == "__main__":
    main()
