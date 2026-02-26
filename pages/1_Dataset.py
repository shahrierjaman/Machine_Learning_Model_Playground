# pages/1_📂_Dataset.py

import streamlit as st
import pandas as pd
import numpy as np
from core.data_validator import validate_dataset
from config import SUPPORTED_TASKS

st.title("📂 Dataset Upload & Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Store original dataset
    st.session_state.original_dataset = df.copy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------------
    # Dataset Validation Report
    # -------------------------
    report = validate_dataset(df)

    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", report["num_rows"])
    col2.metric("Columns", report["num_columns"])
    col3.metric("Missing Values", report["total_missing"])
    col4.metric("Duplicate Rows", report["duplicate_rows"])

    st.markdown("### Data Type Distribution")
    st.write(f"Numeric Columns: {len(report['numeric_columns'])}")
    st.write(f"Categorical Columns: {len(report['categorical_columns'])}")
    st.write(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")

    # Duplicate warning
    if report["duplicate_rows"] > 0:
        st.warning(
            f"⚠️ Dataset contains {report['duplicate_rows']} duplicate rows."
        )

    # Constant columns info
    if report["constant_columns"]:
        st.info(
            f"Constant Columns Detected: {', '.join(report['constant_columns'])}"
        )

    # -------------------------
    # Column Drop Selector
    # -------------------------
    st.subheader("🗑 Column Management")

    columns_to_drop = st.multiselect(
        "Select Columns to Drop",
        df.columns
    )

    df_cleaned = df.drop(columns=columns_to_drop)

    # -------------------------
    # Data Type Conversion Tool
    # -------------------------
    st.subheader("🔄 Data Type Correction")

    selected_column = st.selectbox(
        "Select Column to Convert",
        df_cleaned.columns
    )

    convert_type = st.selectbox(
        "Convert To",
        ["No Change", "Numeric", "Categorical", "Datetime"]
    )

    if convert_type != "No Change":
        try:
            if convert_type == "Numeric":
                df_cleaned[selected_column] = pd.to_numeric(
                    df_cleaned[selected_column], errors="coerce"
                )

            elif convert_type == "Categorical":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype("category")

            elif convert_type == "Datetime":
                df_cleaned[selected_column] = pd.to_datetime(
                    df_cleaned[selected_column], errors="coerce"
                )

            st.success(f"{selected_column} converted to {convert_type}")

        except Exception as e:
            st.error(f"Conversion failed: {e}")

    # -------------------------
    # Target & Task Selection
    # -------------------------
    st.subheader("🎯 Target & Task Configuration")

    target_column = st.selectbox(
        "Select Target Column",
        df_cleaned.columns
    )

    task_type = st.selectbox(
        "Select Task Type",
        SUPPORTED_TASKS
    )

    # -------------------------
    # Confirm Configuration
    # -------------------------
    if st.button("Confirm Configuration"):

        if target_column in columns_to_drop:
            st.error("Target column cannot be dropped.")
        else:
            st.session_state.dataset = df_cleaned
            st.session_state.target_column = target_column
            st.session_state.task_type = task_type
            st.session_state.columns_dropped = columns_to_drop

            st.success("✅ Configuration saved successfully!")

else:
    st.info("Please upload a CSV dataset to begin.")