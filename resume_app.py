import streamlit as st

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="PragyanAI Project", page_icon="🤖", layout="wide")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🔍 PragyanAI Navigation")
main_choice = st.sidebar.selectbox(
    "Select Dashboard",
    ["Home", "Admin Dashboard", "Candidate Dashboard", "Hiring Company Dashboard"]
)

# -------------------------------
# Home
# -------------------------------
if main_choice == "Home":
    st.title("🤖 PragyanAI Project")
    st.subheader("Welcome to the Unified AI Recruitment Platform")
    st.markdown("""
    ### Features
    - 🔸 Admin, Candidate, and Hiring Company dashboards  
    - 🔸 Resume–JD Matching  
    - 🔸 AI-Assisted SWOT & GAP Analysis  
    - 🔸 Mock Interview Generation  
    - 🔸 Skill Roadmap & Certificate Suggestions  
    """)
    st.info("Select a dashboard from the sidebar to begin.")

# -------------------------------
# Admin Dashboard
# -------------------------------
elif main_choice == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")

    tab = st.tabs(["Login", "Add JD", "Candidate Approvals", "Vendor Approvals", "Statistics"])

    with tab[0]:
        st.subheader("Admin Login")
        admin_user = st.text_input("Username")
        admin_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            st.success(f"Welcome Admin: {admin_user}")

    with tab[1]:
        st.subheader("Add Job Description (JD)")
        jd_source = st.radio("JD Source", ["Upload PDF/DOC", "Paste Text", "LinkedIn URL"])
        if jd_source == "Upload PDF/DOC":
            jd_file = st.file_uploader("Upload JD File", type=["pdf", "docx"])
        elif jd_source == "Paste Text":
            jd_text = st.text_area("Paste JD Content")
        else:
            jd_url = st.text_input("LinkedIn Job URL")
        st.button("Add JD")

    with tab[2]:
        st.subheader("Approve Candidates")
        st.button("Approve Selected Candidates")

    with tab[3]:
        st.subheader("Approve Vendors")
        st.button("Approve Selected Vendors")

    with tab[4]:
        st.subheader("Basic Statistics")
        st.metric("Total JDs", 124)
        st.metric("Candidates Approved", 82)
        st.metric("Vendors Approved", 19)

# -------------------------------
# Candidate Dashboard
# -------------------------------
elif main_choice == "Candidate Dashboard":
    st.title("👤 Candidate Dashboard")

    tab = st.tabs([
        "Login / Signup", "Upload / View CV", "Match CV with JD", 
        "GAP & SWOT Analysis", "Mock Interviews", "Skill Roadmap"
    ])

    with tab[0]:
        st.subheader("Login / Signup")
        st.text_input("Email")
        st.text_input("Password", type="password")
        st.button("Login / Signup")

    with tab[1]:
        st.subheader("Upload or Paste CV")
        cv_method = st.radio("CV Source", ["Upload DOCX/PDF", "Paste Text", "LinkedIn URL"])
        if cv_method == "Upload DOCX/PDF":
            cv_file = st.file_uploader("Upload CV", type=["pdf", "docx"])
        elif cv_method == "Paste Text":
            st.text_area("Paste your CV content here")
        else:
            st.text_input("LinkedIn Profile URL")
        st.button("Save CV")

    with tab[2]:
        st.subheader("Match CV with JD")
        st.selectbox("Select JD", ["JD 1", "JD 2", "JD 3"])
        if st.button("Match Now"):
            st.success("✅ Match Completed! Match Score: 85%")

    with tab[3]:
        st.subheader("GAP Analysis & SWOT")
        st.write("🔹 Strengths, Weaknesses, Opportunities, and Threats shown here after matching.")
        st.button("Run Analysis")

    with tab[4]:
        st.subheader("Mock Interview for JD")
        st.selectbox("Select JD for Interview", ["JD 1", "JD 2", "JD 3"])
        st.button("Start Interview")

    with tab[5]:
        st.subheader("Skill Roadmap & Certifications")
        st.write("💡 Suggested Learning Path based on JD gap analysis.")
        st.button("Generate Skill Plan")

# -------------------------------
# Hiring Company Dashboard
# -------------------------------
elif main_choice == "Hiring Company Dashboard":
    st.title("🏢 Hiring Company Dashboard")

    tab = st.tabs([
        "Login / Signup", "Create JD", "Upload CVs", 
        "Match & Rank Candidates", "Screening & Scheduling", "Tracking"
    ])

    with tab[0]:
        st.subheader("Company Login / Signup")
        st.text_input("Company Email")
        st.text_input("Password", type="password")
        st.button("Login / Signup")

    with tab[1]:
        st.subheader("Create JD")
        jd_method = st.radio("Method", ["Upload Document", "Paste Text", "From LinkedIn"])
        if jd_method == "Upload Document":
            st.file_uploader("Upload JD", type=["pdf", "docx"])
        elif jd_method == "Paste Text":
            st.text_area("Paste JD Content")
        else:
            st.text_input("LinkedIn Job URL")
        st.button("Save JD")

    with tab[2]:
        st.subheader("Upload Candidate CVs")
        st.file_uploader("Upload Individual CV or ZIP", type=["pdf", "docx", "zip"])
        st.button("Upload CVs")

    with tab[3]:
        st.subheader("Match & Rank Candidates")
        st.selectbox("Select JD", ["JD 1", "JD 2", "JD 3"])
        st.button("Run Matching & Ranking")

    with tab[4]:
        st.subheader("Screening & Scheduling")
        st.write("Screen shortlisted candidates and schedule interviews.")
        st.button("Schedule Interview")

    with tab[5]:
        st.subheader("Candidate Tracking")
        st.write("Track progress: Applied → Shortlisted → Selected → Rejected.")
        st.button("Show Tracking Table")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("© 2025 PragyanAI Project | AI Recruitment Platform")
