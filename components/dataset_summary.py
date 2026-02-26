# components/dataset_summary.py

import streamlit as st

def show_dataset_summary(report):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", report["num_rows"])
    col2.metric("Columns", report["num_columns"])
    col3.metric("Total Missing", report["total_missing"])
    col4.metric("Duplicate Rows", report["duplicate_rows"])

    if report["constant_columns"]:
        st.warning(f"Constant Columns Detected: {report['constant_columns']}")