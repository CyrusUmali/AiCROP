import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show_data_analysis(df):
    st.title("Data Analysis")

    st.subheader("Dataset Description")
    st.write(df.describe())

    st.subheader("Correlation Plot")
    plot_correlation(df)

    st.subheader("Distribution Plots")
    plot_distribution(df)

    st.subheader("Outlier Detection")
    plot_outlier_detection(df)

def plot_correlation(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

def plot_distribution(df):
    fig, axs = plt.subplots(1, len(df.columns), figsize=(15, 5))
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axs[i], kde=True)
        axs[i].set_title(f"Distribution of {col}")
    st.pyplot(fig)

def plot_outlier_detection(df):
    fig, axs = plt.subplots(1, len(df.columns), figsize=(15, 5))
    for i, col in enumerate(df.columns):
        sns.boxplot(x=df[col], ax=axs[i])
        axs[i].set_title(f"Outliers in {col}")
    st.pyplot(fig)

def main():
    dataset_path = 'dataset/crop_recommendation.csv'
    df = pd.read_csv(dataset_path)

    show_data_analysis(df)

if __name__ == "__main__":
    main()
