import streamlit as st
from hashlib import sha256

st.set_page_config(page_title="Role-based Login System", layout="centered")

# -----------------------
# Helper Functions
# -----------------------
def hash_password(password: str) -> str:
    """Return a SHA-256 hashed password."""
    return sha256(password.encode("utf-8")).hexdigest()

def init_session():
    """Initialize session state variables."""
    if "users" not in st.session_state:
        st.session_state.users = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "Admin",
                "fullname": "Default Admin",
                "email": "admin@example.com",
            }
        }
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("current_role", None)
    st.session_state.setdefault("message", "")

def register_user(username, password, role, fullname="", email=""):
    username = username.strip().lower()
    if username == "":
        return False, "Username cannot be empty."
    if username in st.session_state.users:
        return False, "User already exists. Please login or choose another username."
    st.session_state.users[username] = {
        "password": hash_password(password),
        "role": role,
        "fullname": fullname,
        "email": email,
    }
    return True, "Registration successful! You can now login."

def authenticate(username, password, role):
    username = username.strip().lower()
    if username not in st.session_state.users:
        return False, "User does not exist. Please sign up."
    user = st.session_state.users[username]
    if user["role"] != role:
        return False, f"This account is registered as '{user['role']}'. Select the correct role."
    if user["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful."

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.current_role = None
    st.success("Logged out successfully.")
    st.rerun()

# -----------------------
# Login & Signup
# -----------------------
def login_flow(role):
    st.subheader(f"{role} — Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        ok, msg = authenticate(username, password, role)
        if ok:
            st.session_state.logged_in = True
            st.session_state.current_user = username.strip().lower()
            st.session_state.current_role = role
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

def signup_flow(role):
    st.subheader(f"{role} — Sign Up")
    with st.form("signup_form", clear_on_submit=False):
        fullname = st.text_input("Full name")
        email = st.text_input("Email")
        username = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create Account")
    if submitted:
        if password != password2:
            st.error("Passwords do not match.")
            return
        ok, msg = register_user(username, password, role, fullname, email)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

# -----------------------
# Dashboards
# -----------------------
def admin_dashboard():
    st.title("👑 Admin Dashboard")
    st.write(f"Welcome, **{st.session_state.current_user}**!")
    st.divider()

    st.subheader("Registered Users")
    users_data = [
        {"username": u, "role": data["role"], "fullname": data.get("fullname", ""), "email": data.get("email", "")}
        for u, data in st.session_state.users.items()
    ]
    st.table(users_data)

    st.divider()
    st.subheader("Create a New Admin Account")
    new_admin_user = st.text_input("New Admin Username", key="new_admin_user")
    new_admin_pass = st.text_input("New Admin Password", type="password", key="new_admin_pass")
    if st.button("Create Admin"):
        if not new_admin_user or not new_admin_pass:
            st.error("Please fill in both fields.")
        else:
            ok, msg = register_user(new_admin_user, new_admin_pass, "Admin")
            if ok:
                st.success("Admin created successfully!")
            else:
                st.error(msg)

    st.divider()
    if st.button("Logout"):
        logout()

def candidate_dashboard():
    st.title("🎓 Candidate Dashboard")
    st.write(f"Welcome, **{st.session_state.current_user}**!")
    st.markdown("""
    **Features:**
    - View applied jobs  
    - Edit profile & resume  
    - Take mock interviews  
    """)
    if st.button("Logout"):
        logout()

def hiring_dashboard():
    st.title("🏢 Hiring Company Dashboard")
    st.write(f"Welcome, **{st.session_state.current_user}**!")
    st.markdown("""
    **Features:**
    - Post new job openings  
    - View applicants  
    - Schedule interviews  
    """)
    if st.button("Logout"):
        logout()

# -----------------------
# Main App
# -----------------------
def main():
    init_session()
    st.header("🔐 Multi-Role Login System")

    role = st.selectbox("Select your role", ["Admin", "Candidate", "Hiring Company"])

    if st.session_state.logged_in:
        if role != st.session_state.current_role:
            st.warning(f"You are logged in as '{st.session_state.current_role}'. Change dropdown to your role or logout.")
            if st.button("Go to my dashboard"):
                role = st.session_state.current_role
            if st.button("Logout current account"):
                logout()
            return

        # Role-based dashboard
        if role == "Admin":
            admin_dashboard()
        elif role == "Candidate":
            candidate_dashboard()
        elif role == "Hiring Company":
            hiring_dashboard()
        return

    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login_flow(role)
    with tab2:
        signup_flow(role)

    st.markdown("---")
    st.write("💡 Default Admin Login:")
    st.code("Username: admin\nPassword: admin123")

if __name__ == "__main__":
    main()
