import streamlit as st

# Define the main function for the Hiring Dashboard
# It takes necessary utility functions from app.py as arguments
def hiring_dashboard(go_to):
    st.header("🏢 Hiring Company Dashboard")
    st.write("Manage job postings and view candidate applications. (Placeholder for future features)")
    
    nav_col, _ = st.columns([1, 1]) 

    with nav_col:
        if st.button("🚪 Log Out", key="hiring_logout_btn", use_container_width=True):
            go_to("login")
