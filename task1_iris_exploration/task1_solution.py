import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def load_iris_dataset() -> pd.DataFrame:
    """Load iris data from seaborn and save a local CSV copy."""
    df = sns.load_dataset("iris")
    csv_path = DATA_DIR / "iris.csv"
    df.to_csv(csv_path, index=False)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe(include="all"))


def create_visualizations(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="petal_length",
        hue="species",
        s=70,
    )
    plt.title("Iris: Sepal Length vs Petal Length")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_plot.png", dpi=120)
    plt.close()

    df.hist(figsize=(10, 7), bins=15, edgecolor="black")
    plt.suptitle("Iris Feature Distributions")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "histograms.png", dpi=120)
    plt.close()

    melted = df.melt(id_vars="species", var_name="feature", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="species")
    plt.title("Iris Box Plots by Feature and Species")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "box_plots.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    iris_df = load_iris_dataset()
    inspect_dataset(iris_df)
    create_visualizations(iris_df)
    print("\nSaved data and visualizations successfully.")
