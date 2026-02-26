# app.py

import streamlit as st

st.set_page_config(
    page_title="ML Model Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
def initialize_session():
    default_states = {
        "dataset": None,
        "target_column": None,
        "task_type": None,
        "processed_data": None,
        "pipeline": None,
        "trained_models": {},
        "metrics": {},
        "best_model": None
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# Main Page Content
st.title("🤖 Machine Learning Model Playground")
st.markdown("""
Welcome to a professional no-code ML experimentation environment.

Use the sidebar to navigate through:
1. Dataset Upload  
2. Deep EDA  
3. Preprocessing  
4. Model Training  
5. Comparison  
6. Explainability  
7. Prediction  
8. Export  
""")

st.info("Start by uploading a dataset from the Dataset page.")