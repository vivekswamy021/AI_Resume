# =================================================================
# hiring_dashboard.py
# Contains the UI logic for the Hiring Dashboard
# =================================================================

import streamlit as st
from app_utils import go_to


def hiring_dashboard():
    st.header("🏢 Hiring Company Dashboard")
    st.write("Manage job postings and view candidate applications. (Placeholder for future features)")
    
    nav_col, _ = st.columns([1, 1]) 

    with nav_col:
        if st.button("🚪 Log Out", key="hiring_logout_btn", use_container_width=True):
            go_to("login") 
    
    st.markdown("---")
    st.info("The features for the Hiring Company Dashboard are currently under development. Please check back later!")
