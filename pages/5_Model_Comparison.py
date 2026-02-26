# pages/5_Model_Comparison.py

import streamlit as st
import matplotlib.pyplot as plt
from core.comparator import build_comparison_table

st.title("📊 Model Comparison Dashboard")

if not st.session_state.trained_models:
    st.warning("No trained models found. Please train models first.")
    st.stop()

task = st.session_state.task_type

comparison_df = build_comparison_table(
    st.session_state.trained_models,
    task,
    st.session_state.X_train,
    st.session_state.y_train,
    st.session_state.X_test,
    st.session_state.y_test
)

st.subheader("Model Performance Table")
st.dataframe(comparison_df, use_container_width=True)

# Identify best model (based on sorted table)
best_model_name = comparison_df.iloc[0]["Model"]
st.session_state.best_model = st.session_state.trained_models[best_model_name]

st.success(f"🏆 Best Model: {best_model_name}")

# --------------------------------------------------
# Visual Comparison
# --------------------------------------------------

st.subheader("Performance Comparison Chart")

fig, ax = plt.subplots()

if task == "Classification":
    ax.bar(comparison_df["Model"], comparison_df["Test Accuracy"])
    ax.set_ylabel("Test Accuracy")
else:
    ax.bar(comparison_df["Model"], comparison_df["Test R2"])
    ax.set_ylabel("Test R2")

ax.set_xticks(range(len(comparison_df["Model"])))
ax.set_xticklabels(comparison_df["Model"], rotation=45)

st.pyplot(fig)