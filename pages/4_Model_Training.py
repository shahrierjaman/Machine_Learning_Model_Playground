# pages/4_Model_Training.py

import streamlit as st
from core.model_factory import get_model
from core.trainer import train_model
from core.evaluator import *
from config import CLASSIFICATION_MODELS, REGRESSION_MODELS

st.title("🤖 Model Training & Evaluation")

if "pipeline" not in st.session_state:
    st.warning("Please build preprocessing pipeline first.")
    st.stop()

task = st.session_state.task_type

if task == "Classification":
    model_list = CLASSIFICATION_MODELS
else:
    model_list = REGRESSION_MODELS

model_name = st.selectbox("Select Model", model_list)

if st.button("Train Model"):

    model = get_model(model_name, task)

    full_model, training_time = train_model(
        st.session_state.pipeline,
        model,
        st.session_state.X_train,
        st.session_state.y_train
    )

    st.session_state.trained_models[model_name] = full_model

    st.success(f"{model_name} trained successfully!")
    st.write(f"Training Time: {training_time:.2f} seconds")

    # -------------------------------------------------
    # Baseline Comparison
    # -------------------------------------------------
    baseline_score = get_baseline_score(
        task,
        st.session_state.X_train,
        st.session_state.y_train,
        st.session_state.X_test,
        st.session_state.y_test
    )

    st.subheader("Baseline Comparison")
    st.write("Baseline Score:", baseline_score)

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    if task == "Classification":

        metrics, cm, overfit_msg = evaluate_classification(
            full_model,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_test,
            st.session_state.y_test
        )

        st.subheader("Model Performance")
        st.write(metrics)

        st.write("Confusion Matrix")
        st.write(cm)

        st.write(overfit_msg)

    else:

        metrics, overfit_msg = evaluate_regression(
            full_model,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_test,
            st.session_state.y_test
        )

        st.subheader("Model Performance")
        st.write(metrics)
        st.write(overfit_msg)

    # -------------------------------------------------
    # Learning Curve
    # -------------------------------------------------
    st.subheader("Learning Curve")
    fig = plot_learning_curve(
        full_model,
        st.session_state.X_train,
        st.session_state.y_train
    )

    st.pyplot(fig)