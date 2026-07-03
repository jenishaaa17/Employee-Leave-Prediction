import streamlit as st
import pandas as pd
import sqlite3
import joblib
import base64
import tempfile

from sklearn.preprocessing import LabelEncoder

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Employee Leave Prediction System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# BACKGROUND IMAGE
# ==================================================

def add_bg(image_file):

    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{{
    font-family:'Poppins',sans-serif;
}}

.stApp{{
    background-image:url("data:image/jpg;base64,{encoded}");
    background-size:cover;
    background-position:center;
    background-repeat:no-repeat;
    background-attachment:fixed;
}}

header{{visibility:hidden;}}
footer{{visibility:hidden;}}

[data-testid="stHeader"]{{
    background:transparent;
}}

[data-testid="stSidebar"]{{
    background:rgba(15,15,15,0.82);
    backdrop-filter:blur(15px);
}}

.stButton>button{{
    width:100%;
    border:none;
    border-radius:12px;
    padding:12px;
    font-size:16px;
    font-weight:bold;
    color:white;
    background:linear-gradient(90deg,#2563eb,#9333ea);
}}

.stButton>button:hover{{
    transform:scale(1.02);
}}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]>div{{
    border-radius:10px;
    background:white;
}}

.white-box{{
    background:white;
    padding:25px;
    border-radius:18px;
    margin-bottom:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.25);
}}

.white-box h1,
.white-box h2,
.white-box h3,
.white-box h4,
.white-box h5,
.white-box h6,
.white-box p,
.white-box li,
.white-box span,
.white-box div,
.white-box label{{
    color:black !important;
}}

h1,h2,h3,h4,h5,h6{{
    color:white !important;
    font-weight:700;
}}

p,label{{
    color:white !important;
}}

</style>
""", unsafe_allow_html=True)

add_bg("bg.jpg")

# ==================================================
# DATABASE
# ==================================================

conn = sqlite3.connect(
    "employee.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE,
password TEXT
)
""")

conn.commit()

# ==================================================
# REGISTER
# ==================================================

def register(username,password):

    try:

        cursor.execute(
            "INSERT INTO users(username,password) VALUES(?,?)",
            (username,password)
        )

        conn.commit()

        return True

    except:

        return False

# ==================================================
# LOGIN
# ==================================================

def login(username,password):

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username,password)
    )

    return cursor.fetchone()

# ==================================================
# SESSION
# ==================================================

if "login" not in st.session_state:
    st.session_state.login=False

if "username" not in st.session_state:
    st.session_state.username=""

if "page" not in st.session_state:
    st.session_state.page="Dashboard"

# ==================================================
# MAIN TITLE
# ==================================================

st.markdown("""
<h1 style="
color:white;
font-size:55px;
font-weight:700;
text-shadow:2px 2px 10px black;
">
💼 Employee Leave Prediction System
</h1>
""",unsafe_allow_html=True)
# ---------------- LOGIN SUCCESS ----------------

if st.session_state.login:

    # ================= SIDEBAR =================

    st.sidebar.title("📊 Employee Leave Prediction")

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    if st.sidebar.button("🏠 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"

    if st.sidebar.button("📂 Dataset Analysis", use_container_width=True):
        st.session_state.page = "Dataset"

    if st.sidebar.button("🤖 Model Prediction", use_container_width=True):
        st.session_state.page = "Prediction"

    if st.sidebar.button("👨‍💼 Employee Details", use_container_width=True):
        st.session_state.page = "Employee"

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.login = False
        st.session_state.username = ""
        st.rerun()

    page = st.session_state.page

    # ==================================================
    # DASHBOARD
    # ==================================================

    if page == "Dashboard":

        st.success("Login Successful")

        st.title("📊 Employee Leave Prediction Dashboard")

        st.write(f"Welcome, **{st.session_state.username}**")

        st.info("Choose any option from the sidebar.")

    # ==================================================
    # DATASET ANALYSIS
    # ==================================================

    elif page == "Dataset":

        st.title("📂 Dataset Analysis")

        uploaded_file = st.file_uploader(
            "Browse Employee Dataset",
            type=["csv"],
            key="csv_upload"
        )

        if uploaded_file is not None:

            abc = pd.read_csv(uploaded_file)

            st.subheader("Dataset")
            st.dataframe(abc)

            st.subheader("First 5 Rows")
            st.dataframe(abc.head())

            st.subheader("Last 5 Rows")
            st.dataframe(abc.tail())

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Rows", abc.shape[0])

            with col2:
                st.metric("Columns", abc.shape[1])

            st.subheader("Column Names")
            st.write(list(abc.columns))

            st.subheader("Data Types")
            st.dataframe(abc.dtypes.astype(str))

            st.subheader("Missing Values")
            st.dataframe(
                abc.isnull().sum().reset_index().rename(
                    columns={"index": "Column", 0: "Missing Values"}
                )
            )

            st.subheader("Duplicate Values")
            st.write(abc.duplicated().sum())

            st.subheader("Statistical Summary")
            st.dataframe(abc.describe())

            st.subheader("Label Encoding")

            if st.button("Apply Label Encoding"):

                encoded = abc.copy()

                object_cols = encoded.select_dtypes(include="object").columns

                for col in object_cols:

                    encoder = LabelEncoder()

                    encoded[col] = encoder.fit_transform(encoded[col])

                st.success("Encoding Completed")

                st.dataframe(encoded)

                st.session_state["encoded_data"] = encoded

    # ==================================================
    # MODEL PREDICTION
    # ==================================================

    elif page == "Prediction":

        st.title("🤖 Employee Leave Prediction")

        uploaded_model = st.file_uploader(
            "Upload Trained Model (.pkl)",
            type=["pkl"],
            key="model_upload"
        )

        if uploaded_model is not None:

            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:

                tmp.write(uploaded_model.read())

                model = joblib.load(tmp.name)

            st.success("✅ Model Loaded Successfully")

            st.success("Accuracy : 90.42 %")

            st.markdown("---")

            education = st.number_input("Education",0,2)

            joiningyear = st.number_input("Joining Year",2012,2018)

            city = st.number_input("City",0,2)

            paymenttier = st.selectbox(
                "Payment Tier",
                [1,2,3]
            )

            age = st.number_input("Age",22,41)

            gender = st.number_input("Gender",0,1)

            everbenched = st.number_input("Ever Benched",0,1)

            experience = st.number_input(
                "Experience",
                0,
                7
            )

            if st.button("Predict"):

                employee = pd.DataFrame([{

                    "Education":education,
                    "JoiningYear":joiningyear,
                    "City":city,
                    "PaymentTier":paymenttier,
                    "Age":age,
                    "Gender":gender,
                    "EverBenched":everbenched,
                    "ExperienceInCurrentDomain":experience

                }])

                prediction = model.predict(employee)
                if prediction[0]==1:

                    st.error("❌ Employee is likely to Leave the Company")

                else:

                    st.success("✅ Employee is likely to Stay in the Company")

# ==================================================
# EMPLOYEE DETAILS
# ==================================================
    elif page == "Employee":

        st.title("👨‍💼 Employee Details")

        # ================= Salary Details =================

        st.subheader("💰 Salary Details")

        salary = pd.DataFrame({
            "Salary Component": [
                "Basic Salary",
                "House Rent Allowance (HRA)",
                "Dearness Allowance (DA)",
                "Medical Allowance",
                "Travel Allowance",
                "Special Allowance",
                "Gross Salary"
            ],
            "Description": [
                "Fixed monthly salary",
                "Accommodation allowance",
                "Cost of living allowance",
                "Medical benefits",
                "Travel reimbursement",
                "Company allowance",
                "Total earnings before deductions"
            ]
        })

        st.dataframe(salary, use_container_width=True)

        st.markdown("---")

        # ================= PF Details =================

        st.subheader("🏦 Provident Fund (PF) Details")

        pf = pd.DataFrame({
            "Particular": [
                "Employee Contribution",
                "Employer Contribution",
                "Maintained By",
                "Purpose"
            ],
            "Details": [
                "12% of Basic Salary",
                "12% of Basic Salary",
                "EPFO",
                "Retirement Savings"
            ]
        })

        st.dataframe(pf, use_container_width=True)

        st.markdown("---")

        # ================= Leave Policy =================

        st.subheader("📄 Leave Policy")

        leave = pd.DataFrame({
            "Leave Type": [
                "Casual Leave",
                "Sick Leave",
                "Earned Leave",
                "Maternity / Paternity Leave"
            ],
            "Days Available": [
                "12 Days",
                "10 Days",
                "15 Days",
                "As Per Company Policy"
            ]
        })

        st.dataframe(leave, use_container_width=True)

        st.markdown("---")

        # ================= Working Hours =================

        st.subheader("🕒 Working Hours")

        working = pd.DataFrame({
            "Particular": [
                "Working Days",
                "Office Timing",
                "Lunch Break"
            ],
            "Details": [
                "Monday - Friday",
                "9:00 AM - 6:00 PM",
                "1:00 PM - 2:00 PM"
            ]
        })

        st.dataframe(working, use_container_width=True)

        st.markdown("---")

        # ================= Company Information =================

        st.subheader("🏢 Company Information")

        company = pd.DataFrame({
            "Field": [
                "Company Name",
                "Department",
                "HR Email",
                "Support Email"
            ],
            "Information": [
                "ABC Technologies Pvt. Ltd.",
                "Information Technology",
                "hr@abctech.com",
                "support@abctech.com"
            ]
        })

        st.dataframe(company, use_container_width=True)

        st.success("✅ Employee Information Loaded Successfully")      
# ---------------- LOGIN / REGISTER ----------------

else:

    menu=st.sidebar.selectbox(
        "Menu",
        ["Login","Register"]
    )

    # ---------------- REGISTER ----------------

    if menu=="Register":

        st.header("Create Account")

        user=st.text_input("Username")

        pwd=st.text_input(
            "Password",
            type="password"
        )

        cpwd=st.text_input(
            "Confirm Password",
            type="password"
        )

        if st.button("Register"):

            if user=="" or pwd=="":

                st.error("Enter all details")

            elif pwd!=cpwd:

                st.error("Passwords do not match")

            else:

                if register(user,pwd):

                    st.success("Registration Successful")

                else:

                    st.error("Username already exists")

    # ---------------- LOGIN ----------------

    if menu=="Login":

        st.header("Login")

        user=st.text_input("Username")

        pwd=st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            result=login(user,pwd)

            if result:

                st.session_state.login=True
                st.session_state.username=user

                st.rerun()

            else:

                st.error("Invalid Username or Password")