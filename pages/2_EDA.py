# pages/2_🔍_EDA.py

import streamlit as st
import pandas as pd
from core.eda_engine import *
from components.plots import *

st.title("🔍 Advanced Exploratory Data Analysis")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("Please upload dataset first.")
    st.stop()

df = st.session_state.dataset
target = st.session_state.target_column
task_type = st.session_state.task_type

tabs = st.tabs([
    "Overview",
    "Target Analysis",
    "Distributions",
    "Relationships",
    "Outliers",
    "Insights"
])

# ------------------ OVERVIEW ------------------
with tabs[0]:

    st.subheader("Statistical Summary (Enhanced)")
    st.dataframe(basic_statistics(df), use_container_width=True)

    st.subheader("Missing Value Summary")
    st.dataframe(missing_summary(df), use_container_width=True)

    st.subheader("Duplicate Rows")
    duplicates = df.duplicated().sum()
    st.write(f"Total Duplicates: {duplicates}")

# ------------------ TARGET ANALYSIS ------------------
with tabs[1]:

    st.subheader("Target Analysis")

    if task_type == "Classification":

        class_dist = detect_class_imbalance(df, target)
        st.write("Class Distribution (%)")
        st.dataframe(class_dist)

        plot_target_distribution(df, target)

        imbalance_ratio = class_dist.min() / class_dist.max()
        if imbalance_ratio < 0.5:
            st.warning("⚠️ Significant class imbalance detected.")

    else:

        st.write("Target Distribution")
        plot_distribution(df, target)

        skewness = df[target].skew()
        st.write(f"Target Skewness: {skewness:.3f}")

        if abs(skewness) > 1:
            st.warning("⚠️ Target is highly skewed.")

# ------------------ DISTRIBUTIONS ------------------
with tabs[2]:

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select Numeric Feature", numeric_cols)
        plot_distribution(df, selected_col)

        st.write("Feature Skewness & Kurtosis")
        st.dataframe(feature_distribution_summary(df))
    else:
        st.info("No numeric columns available.")

# ------------------ RELATIONSHIPS ------------------
with tabs[3]:

    corr = correlation_matrix(df)

    st.subheader("Correlation Heatmap")
    plot_correlation_heatmap(corr)

    st.subheader("Highly Correlated Feature Pairs (> 0.9)")
    high_corr = high_correlation_pairs(corr)
    st.dataframe(high_corr)

    if target in corr.columns:
        st.subheader("Feature Correlation with Target")
        target_corr = feature_target_correlation(corr, target)
        st.dataframe(target_corr)

    st.subheader("VIF (Multicollinearity)")
    st.dataframe(calculate_vif(df))

# ------------------ OUTLIERS ------------------
with tabs[4]:

    outlier_report = detect_outliers_iqr(df)
    st.dataframe(outlier_report)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select Feature for Boxplot", numeric_cols)
        plot_boxplot(df, selected_col)

# ------------------ INSIGHTS ------------------
with tabs[5]:

    st.subheader("Automatic Preprocessing Recommendations")

    recommendations = generate_recommendations(df, target, task_type)

    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("Dataset looks structurally clean.")