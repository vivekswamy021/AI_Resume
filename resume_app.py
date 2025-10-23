# app.py
import streamlit as st
from hashlib import sha256

st.set_page_config(page_title="Role-based Login Demo", layout="centered")

# -----------------------
# Helper functions
# -----------------------
def hash_password(password: str) -> str:
    return sha256(password.encode("utf-8")).hexdigest()

def init_session():
    if "users" not in st.session_state:
        # simple in-memory user store: username -> dict(password_hash, role, fullname, email)
        # Prepopulate a default admin for quick login
        st.session_state.users = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "Admin",
                "fullname": "Default Admin",
                "email": "admin@example.com",
            }
        }
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "current_role" not in st.session_state:
        st.session_state.current_role = None
    if "message" not in st.session_state:
        st.session_state.message = ""

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

def login_flow(role):
    st.subheader(f"{role} — Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        ok, msg = authenticate(username, password, role)
        st.session_state.message = msg
        if ok:
            st.session_state.logged_in = True
            st.session_state.current_user = username.strip().lower()
            st.session_state.current_role = role
            st.success(msg)
            st.experimental_rerun()
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
        submitted = st.form_submit_button("Create account")
    if submitted:
        if password != password2:
            st.error("Passwords do not match.")
            return
        ok, msg = register_user(username, password, role, fullname, email)
        if ok:
            st.success(msg)
            st.info("You can now login using the Login form.")
        else:
            st.error(msg)

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.current_role = None
    st.success("Logged out.")
    st.experimental_rerun()

# -----------------------
# Dashboard pages
# -----------------------
def admin_dashboard():
    st.title("Admin Dashboard")
    st.markdown("Welcome, **Admin**. Use controls below to manage users.")
    st.write("Logged in as:", st.session_state.current_user)
    st.divider()

    st.subheader("Registered users")
    users_list = [
        {"username": u, "role": data["role"], "fullname": data.get("fullname", ""), "email": data.get("email", "")}
        for u, data in st.session_state.users.items()
    ]
    st.table(users_list)

    st.subheader("Create a new admin (quick)")
    c1, c2 = st.columns([2, 1])
    with c1:
        new_admin_user = st.text_input("New admin username", key="new_admin_user")
    with c2:
        new_admin_pass = st.text_input("New admin password", type="password", key="new_admin_pass")
    if st.button("Create admin account"):
        if new_admin_user.strip() == "" or new_admin_pass.strip() == "":
            st.error("Provide both username and password.")
        else:
            ok, msg = register_user(new_admin_user, new_admin_pass, "Admin")
            if ok:
                st.success("Admin account created.")
            else:
                st.error(msg)

    st.divider()
    if st.button("Logout"):
        logout()

def candidate_dashboard():
    st.title("Candidate Dashboard")
    st.write("Welcome,", st.session_state.current_user)
    st.markdown(
        """
        **Candidate features (placeholder)**
        - View applied jobs
        - Edit profile & resume
        - Take assessments
        """
    )
    if st.button("Logout"):
        logout()

def hiring_dashboard():
    st.title("Hiring Company Dashboard")
    st.write("Welcome,", st.session_state.current_user)
    st.markdown(
        """
        **Hiring Company features (placeholder)**
        - Post jobs
        - View applicants
        - Shortlist & schedule interviews
        """
    )
    if st.button("Logout"):
        logout()

# -----------------------
# Main app
# -----------------------
def main():
    init_session()
    st.header("Multi-role Login / Signup Demo")

    st.info("Select a role from the dropdown, then Sign Up or Login for that role.")
    role = st.selectbox("Choose role", ["Admin", "Candidate", "Hiring Company"])

    # If already logged in and role matches, show dashboard
    if st.session_state.logged_in:
        # If logged in but role doesn't match selected role, show info
        if role != st.session_state.current_role:
            st.warning(f"You are logged in as '{st.session_state.current_role}'. Switch the dropdown to that role or logout and login again.")
            if st.button("Go to my dashboard"):
                # show their actual dashboard regardless of dropdown
                if st.session_state.current_role == "Admin":
                    admin_dashboard()
                elif st.session_state.current_role == "Candidate":
                    candidate_dashboard()
                elif st.session_state.current_role == "Hiring Company":
                    hiring_dashboard()
                return
            if st.button("Logout current account"):
                logout()
                return

        # Show dashboard of current role
        if st.session_state.current_role == "Admin":
            admin_dashboard()
        elif st.session_state.current_role == "Candidate":
            candidate_dashboard()
        elif st.session_state.current_role == "Hiring Company":
            hiring_dashboard()
        return

    # Not logged in: show login/signup tabs
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login_flow(role)
    with tab2:
        signup_flow(role)

    st.markdown("---")
    st.write("Tip: Default admin credentials are:")
    st.code("username: admin\npassword: admin123")

    # debug / show current session (optional)
    if st.checkbox("Show session state (debug)"):
        st.write(st.session_state)

if __name__ == "__main__":
    main()
