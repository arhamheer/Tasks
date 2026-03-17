from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Task 1 - Iris Exploration", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = sns.load_dataset("iris")
    df.to_csv(DATA_DIR / "iris.csv", index=False)
    return df


def main() -> None:
    st.title("Task 1: Iris Dataset Exploration")
    st.write("Inspect and visualize the iris dataset with pandas, seaborn, and matplotlib.")

    df = load_data()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Shape")
        st.write(df.shape)
    with col2:
        st.subheader("Columns")
        st.write(df.columns.tolist())

    st.subheader("First Rows")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Info")
    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Non-Null Count": [df[c].notna().sum() for c in df.columns],
            "Dtype": [str(df[c].dtype) for c in df.columns],
        }
    )
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Describe")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    st.subheader("Scatter Plot")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Histograms")
    fig2 = df.hist(figsize=(10, 7), bins=15, edgecolor="black")
    st.pyplot(fig2[0][0].figure)

    st.subheader("Box Plots")
    melted = df.melt(id_vars="species", var_name="feature", value_name="value")
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=melted, x="feature", y="value", hue="species", ax=ax3)
    plt.xticks(rotation=15)
    st.pyplot(fig3)


if __name__ == "__main__":
    main()
