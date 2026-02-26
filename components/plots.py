# components/plots.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column):

    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)


def plot_correlation_heatmap(corr_matrix):

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


def plot_boxplot(df, column):

    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    st.pyplot(fig)


def plot_target_distribution(df, target):

    fig, ax = plt.subplots()
    df[target].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Target Distribution")
    st.pyplot(fig)