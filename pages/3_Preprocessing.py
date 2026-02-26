# pages/3_Preprocessing.py

import streamlit as st
from core.preprocessing import build_preprocessing_pipeline
from components.pipeline_visualizer import show_pipeline_config
from config import RANDOM_STATE, DEFAULT_TEST_SIZE

st.title("⚙️ Advanced Preprocessing Pipeline Builder")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("Please upload dataset first.")
    st.stop()

df = st.session_state.dataset
target = st.session_state.target_column

st.sidebar.header("Preprocessing Options")

# -----------------------
# Basic Options
# -----------------------
missing_strategy = st.sidebar.selectbox(
    "Missing Value Strategy",
    ["mean", "median", "most_frequent"]
)

encoding_strategy = st.sidebar.selectbox(
    "Encoding Strategy",
    ["OneHot", "Ordinal", "Frequency", "Target", "None"]
)

scaling_strategy = st.sidebar.selectbox(
    "Scaling Strategy",
    ["StandardScaler", "MinMaxScaler", "RobustScaler", "PowerTransformer", "None"]
)

# -----------------------
# Advanced Options
# -----------------------
st.sidebar.markdown("### Advanced Options")

rare_threshold = st.sidebar.slider(
    "Rare Category Threshold (%)",
    0.0, 10.0, 1.0
) / 100

use_polynomial = st.sidebar.checkbox("Add Polynomial Features (Numeric Only)")

use_variance_threshold = st.sidebar.checkbox("Apply Variance Threshold Feature Selection")

test_size = st.sidebar.slider(
    "Test Size",
    0.1, 0.4,
    DEFAULT_TEST_SIZE
)

# -----------------------
# Build Pipeline
# -----------------------
if st.button("Build Pipeline"):

    pipeline, X_train, X_test, y_train, y_test = build_preprocessing_pipeline(
        df,
        target,
        missing_strategy,
        encoding_strategy,
        scaling_strategy,
        test_size,
        RANDOM_STATE,
        rare_threshold,
        use_polynomial,
        use_variance_threshold
    )

    st.session_state.pipeline = pipeline
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("✅ Advanced pipeline built successfully!")

    show_pipeline_config(
        missing_strategy,
        encoding_strategy,
        scaling_strategy,
        test_size,
        rare_threshold,
        use_polynomial,
        use_variance_threshold
    )