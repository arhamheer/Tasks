from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

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


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "heart_disease.csv"
    df = pd.read_csv(DATA_URL, names=COLUMNS, na_values="?")
    df.to_csv(csv_path, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["target"] = (cleaned["target"] > 0).astype(int)
    return cleaned


def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="target", y="age")
    plt.title("Age vs Heart Disease")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "age_vs_target.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()


def train_and_evaluate(df: pd.DataFrame):
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

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax1)
    ax1.set_title("Confusion Matrix")
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, probs, ax=ax2)
    ax2.set_title("ROC Curve")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "roc_curve.png", dpi=120)
    plt.close(fig2)

    feature_importance = (
        pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
        .abs()
        .sort_values(ascending=False)
    )

    return model, accuracy, roc_auc, feature_importance


if __name__ == "__main__":
    heart_df = load_data()
    heart_df = clean_data(heart_df)

    print("Shape:", heart_df.shape)
    print("\nHead:")
    print(heart_df.head())
    print("\nInfo:")
    heart_df.info()
    print("\nDescribe:")
    print(heart_df.describe())

    run_eda(heart_df)
    _, acc, auc_score, importance = train_and_evaluate(heart_df)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc_score:.4f}")
    print("\nTop important features:")
    print(importance.head(8))
    print("\nSaved cleaned data and plots.")
