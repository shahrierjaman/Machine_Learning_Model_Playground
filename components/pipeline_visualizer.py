# components/pipeline_visualizer.py

import streamlit as st


def show_pipeline_config(
    missing,
    encoding,
    scaling,
    test_size,
    rare_threshold,
    poly,
    variance
):

    st.subheader("Pipeline Configuration Summary")

    st.markdown(f"""
    **Missing Strategy:** {missing}  
    **Encoding Strategy:** {encoding}  
    **Scaling Strategy:** {scaling}  
    **Test Size:** {test_size}  
    **Rare Category Threshold:** {rare_threshold}  
    **Polynomial Features:** {poly}  
    **Variance Threshold Applied:** {variance}
    """)