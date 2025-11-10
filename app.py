# =================================================================
# main_app.py
# Main Streamlit application file for routing and initialization
# =================================================================

import streamlit as st
# Import core utilities and all dashboard modules
from app_utils import go_to, GROQ_API_KEY
from admin_dashboard_page import admin_dashboard
from candidate_dashboard_page import candidate_dashboard
from hiring_dashboard import hiring_dashboard


# -------------------------
# UI PAGES: Authentication (Login, Signup)
# -------------------------

def login_page():
    st.title("🌐 PragyanAI Job Portal")
    st.header("Login")

    selected_role = st.selectbox(
        "Select Your Role",
        ["Select Role", "Admin Dashboard", "Candidate Dashboard", "Hiring Company Dashboard"],
        key="login_role_select"
    )
    
    st.markdown("---")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if email and password:
            if selected_role == "Select Role":
                st.error("Please select your role before logging in.")
            elif selected_role == "Admin Dashboard":
                st.success("Login successful! Redirecting to Admin Dashboard.")
                go_to("admin_dashboard")
            elif selected_role == "Candidate Dashboard":
                st.success("Login successful! Redirecting to Candidate Dashboard.")
                go_to("candidate_dashboard")
            elif selected_role == "Hiring Company Dashboard":
                st.success("Login successful! Redirecting to Hiring Company Dashboard.")
                go_to("hiring_dashboard")
        else:
            st.error("Please enter both email and password")

    st.markdown("---")
    
    if st.button("Don't have an account? Sign up here"):
        go_to("signup")

def signup_page():
    st.header("Create an Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up", use_container_width=True):
        if password == confirm and email:
            st.success("Signup successful! Please login.")
            go_to("login")
        else:
            st.error("Passwords do not match or email is empty")

    if st.button("Already have an account? Login here"):
        go_to("login")

# -------------------------
# Main App Initialization
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="PragyanAI Job Portal")
    
    # Expose the API Key state needed in other files
    if 'GROQ_API_KEY' not in st.session_state: st.session_state.GROQ_API_KEY = GROQ_API_KEY

    # --- Session State Initialization (Centralized here) ---
    if 'page' not in st.session_state: st.session_state.page = "login"
    if 'parsed' not in st.session_state: st.session_state.parsed = {}
    if 'full_text' not in st.session_state: st.session_state.full_text = ""
    if 'excel_data' not in st.session_state: st.session_state.excel_data = None
    if 'qa_answer' not in st.session_state: st.session_state.qa_answer = ""
    if 'iq_output' not in st.session_state: st.session_state.iq_output = ""
    if 'jd_fit_output' not in st.session_state: st.session_state.jd_fit_output = ""
    if 'admin_jd_list' not in st.session_state: st.session_state.admin_jd_list = [] 
    if 'resumes_to_analyze' not in st.session_state: st.session_state.resumes_to_analyze = [] 
    if 'admin_match_results' not in st.session_state: st.session_state.admin_match_results = [] 
    if 'resume_statuses' not in st.session_state: st.session_state.resume_statuses = {} 
    if 'vendors' not in st.session_state: st.session_state.vendors = []
    if 'vendor_statuses' not in st.session_state: st.session_state.vendor_statuses = {}
    if 'candidate_jd_list' not in st.session_state: st.session_state.candidate_jd_list = []
    if 'candidate_match_results' not in st.session_state: st.session_state.candidate_match_results = []
    if 'candidate_uploaded_resumes' not in st.session_state: st.session_state.candidate_uploaded_resumes = []
    if 'pasted_cv_text' not in st.session_state: st.session_state.pasted_cv_text = "" 
    if 'interview_qa' not in st.session_state: st.session_state.interview_qa = [] 
    if 'evaluation_report' not in st.session_state: st.session_state.evaluation_report = ""
    if 'filtered_jds_display' not in st.session_state: st.session_state.filtered_jds_display = []
    if "last_selected_skills" not in st.session_state: st.session_state.last_selected_skills = []
    if "cv_form_data" not in st.session_state: 
        st.session_state.cv_form_data = {
            "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
            "skills": [], "experience": [], "education": [], "certifications": [], 
            "projects": [], "strength": [], "personal_details": ""
        }


    # --- Page Routing ---
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
    elif st.session_state.page == "admin_dashboard":
        admin_dashboard()
    elif st.session_state.page == "candidate_dashboard":
        candidate_dashboard()
    elif st.session_state.page == "hiring_dashboard":
        hiring_dashboard()

if __name__ == '__main__':
    main()
