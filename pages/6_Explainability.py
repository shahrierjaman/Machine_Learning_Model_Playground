# pages/6_Explainability.py

import streamlit as st
import shap
import matplotlib.pyplot as plt
from core.explainability import get_feature_importance, compute_shap_values

st.title("🧠 Model Explainability Panel")

if "best_model" not in st.session_state or st.session_state.best_model is None:
    st.warning("No best model found. Please compare models first.")
    st.stop()

model = st.session_state.best_model
X_test = st.session_state.X_test

tabs = st.tabs(["Feature Importance", "SHAP Summary", "Individual Prediction"])

# ---------------- FEATURE IMPORTANCE ----------------
with tabs[0]:

    st.subheader("Feature Importance")

    importance_df = get_feature_importance(model)

    if importance_df is not None:
        st.dataframe(importance_df.head(20))

        fig, ax = plt.subplots()
        ax.barh(
            importance_df["Feature"].head(20),
            importance_df["Importance"].head(20)
        )
        ax.invert_yaxis()
        st.pyplot(fig)

    else:
        st.info("Feature importance not available for this model.")


# ---------------- SHAP SUMMARY ----------------
# ---------------- SHAP SUMMARY ----------------
with tabs[1]:

    st.subheader("SHAP Summary Plot")

    sample_size = min(200, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    shap_values = compute_shap_values(model, X_sample)

    if shap_values is not None:
        fig = plt.figure()
        shap.summary_plot(
            shap_values.values,
            shap_values.data,
            feature_names=shap_values.feature_names,
            show=False
        )
        st.pyplot(fig)
    else:
        st.info("SHAP not supported for this model.")


# ---------------- INDIVIDUAL PREDICTION ----------------
with tabs[2]:

    st.subheader("Individual Prediction Explanation")

    index = st.slider("Select Data Point Index", 0, len(X_test)-1, 0)

    single_sample = X_test.iloc[[index]]

    shap_values = compute_shap_values(model, single_sample)

    if shap_values is not None:
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.info("SHAP not supported for this model.")