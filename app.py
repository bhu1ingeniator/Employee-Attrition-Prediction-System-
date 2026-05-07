# HR Employee Attrition Dashboard
# Built with Streamlit + Machine Learning

# -----------------------------
# Import Libraries
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Main Background */
.stApp {
    background-color: rgb(245, 247, 251);
}

/* Reduce Top Space */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ---------------- Hero Section ---------------- */

.hero-container {

    background: linear-gradient(
        135deg,
        rgb(15, 23, 42),
        rgb(30, 41, 59),
        rgb(37, 99, 235)
    );

    padding: 65px 40px;
    border-radius: 28px;
    color: white;
    text-align: center;
    margin-bottom: 35px;

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.12);

    position: relative;
    overflow: hidden;
}

/* Glass Orb Effect */
.hero-container::before {

    content: "";

    position: absolute;

    width: 320px;
    height: 320px;

    background: rgba(255,255,255,0.06);

    border-radius: 50%;

    top: -120px;
    right: -80px;
}

.hero-title {
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 12px;
    position: relative;
    z-index: 2;
}

.hero-subtitle {

    font-size: 18px;

    color: rgb(219, 234, 254);

    max-width: 780px;

    margin: auto;

    line-height: 1.8;

    position: relative;
    z-index: 2;
}

/* ---------------- Cards ---------------- */

.dashboard-card {

    background-color: white;

    padding: 30px;

    border-radius: 22px;

    box-shadow:
        0px 3px 15px rgba(0,0,0,0.06);
}

.result-card {

    background-color: white;

    padding: 28px;

    border-radius: 22px;

    box-shadow:
        0px 3px 15px rgba(0,0,0,0.06);

    margin-top: 25px;
}

/* ---------------- Button ---------------- */

.stButton > button {

    width: 100%;

    background-color: rgb(59, 130, 246);

    color: white;

    border-radius: 12px;

    border: none;

    padding: 0.9rem;

    font-size: 16px;

    font-weight: 600;

    transition: 0.3s;
}

.stButton > button:hover {

    background-color: rgb(37, 99, 235);

    transform: scale(1.01);
}

/* ---------------- Footer ---------------- */

.footer {

    text-align: center;

    color: gray;

    font-size: 14px;

    margin-top: 60px;

    margin-bottom: 10px;
}

/* ---------------- Responsive ---------------- */

@media (max-width: 768px) {

    .hero-title {
        font-size: 34px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

    .dashboard-card,
    .result-card {
        padding: 20px;
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero-container">

<div class="hero-title">
 HR Attrition Prediction
</div>

<div class="hero-subtitle">

Predict whether an employee is likely to stay in the company or leave based on employee details.

</div>

</div>
""", unsafe_allow_html=True)

# Small Caption
st.caption(
    "AI-powered employee attrition analysis dashboard"
)

# -----------------------------
# Load and Train Model
# -----------------------------
@st.cache_resource
def load_and_train():

    df = pd.read_csv("HR-Employee-Attrition.csv")

    # Drop unnecessary columns
    df.drop(
        columns=['EmployeeCount', 'StandardHours', 'Over18'],
        inplace=True
    )

    # Encode target
    le = LabelEncoder()

    df['Attrition'] = le.fit_transform(
        df['Attrition']
    )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    # Features and Target
    X = df.drop('Attrition', axis=1)

    y = df['Attrition']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # Balance dataset
    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, scaler, X.columns.tolist()


# -----------------------------
# Load Model
# -----------------------------
model, scaler, feature_cols = load_and_train()

# -----------------------------
# Employee Form Card
# -----------------------------
st.markdown(
    "<div class='dashboard-card'>",
    unsafe_allow_html=True
)

st.markdown("## 📝 Employee Information")

c1, c2, c3 = st.columns(3)

# ---------------- Column 1 ----------------
with c1:

    st.markdown("### 👤 Personal Info")

    age = st.slider(
        "Age",
        18, 60, 30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married" ]
    )

    distance = st.slider(
        "Distance From Home",
        1, 30, 5
    )

# ---------------- Column 2 ----------------
with c2:

    st.markdown("### 💼 Job Details")

    job_level = st.slider(
        "Job Level",
        1, 5, 2
    )

    overtime = st.selectbox(
        "Overtime",
        ["Yes", "No"]
    )

    travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        200000,
        5000,
        
    )

# ---------------- Column 3 ----------------
with c3:

    st.markdown("### 📊 Satisfaction")

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1, 4, 3
    )

    env_satisfaction = st.slider(
        "Environment Satisfaction",
        1, 4, 3
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1, 4, 3
    )

    years_at_company = st.slider(
        "Years At Company",
        0, 40, 3
    )

st.write("")

predict_button = st.button(
    "Generate Prediction"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Prepare Input
# -----------------------------
def prepare_input():

    row = {col: 0 for col in feature_cols}

    row['Age'] = age
    row['DistanceFromHome'] = distance
    row['JobLevel'] = job_level
    row['JobSatisfaction'] = job_satisfaction
    row['EnvironmentSatisfaction'] = env_satisfaction
    row['WorkLifeBalance'] = work_life_balance
    row['MonthlyIncome'] = monthly_income
    row['YearsAtCompany'] = years_at_company

    # Encoded Columns
    if f'Gender_{gender}' in row:
        row[f'Gender_{gender}'] = 1

    if f'MaritalStatus_{marital_status}' in row:
        row[f'MaritalStatus_{marital_status}'] = 1

    if overtime == "Yes":
        row['OverTime_Yes'] = 1

    if f'BusinessTravel_{travel}' in row:
        row[f'BusinessTravel_{travel}'] = 1

    input_df = pd.DataFrame([row])

    scaled_input = scaler.transform(input_df)

    return scaled_input

# -----------------------------
# Prediction Section
# -----------------------------
if predict_button:

    input_data = prepare_input()

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    stay_prob = probability[0] * 100

    leave_prob = probability[1] * 100

    st.markdown(
        "<div class='result-card'>",
        unsafe_allow_html=True
    )

    st.markdown("## 📌 Prediction Result")

    # Prediction Result
    if prediction == 1:

        st.error(
            f"⚠️ Employee likely to LEAVE ({leave_prob:.1f}%)"
        )

    else:

        st.success(
            f"✅ Employee likely to STAY ({stay_prob:.1f}%)"
        )

    st.write("")

    # Probability Metrics
    p1, p2 = st.columns(2)

    with p1:
        st.metric(
            "Stay Probability",
            f"{stay_prob:.1f}%"
        )

    with p2:
        st.metric(
            "Leave Probability",
            f"{leave_prob:.1f}%"
        )

    st.write("")

    # ---------------- Charts ----------------
    c1, c2 = st.columns(2)

    # Left Chart
    with c1:

        st.markdown("### 📊 Prediction Analysis")

        chart_data = pd.DataFrame({
            "Status": ["Stay", "Leave"],
            "Probability": [stay_prob, leave_prob]
        })

        st.bar_chart(
            chart_data.set_index("Status")
        )

    # Right Side
    with c2:

        st.markdown("### ⚠️ Risk Factors")

        risks = []

        if overtime == "Yes":
            risks.append("Works overtime frequently")

        if job_satisfaction <= 2:
            risks.append("Low job satisfaction")

        if work_life_balance <= 2:
            risks.append("Poor work-life balance")

        if env_satisfaction <= 2:
            risks.append("Low environment satisfaction")

        if distance > 20:
            risks.append("Long travel distance")

        if monthly_income < 3000:
            risks.append("Low monthly income")

        if risks:

            for risk in risks:
                st.warning(risk)

        else:

            st.success(
                "No major risk factors detected"
            )

    st.markdown("</div>", unsafe_allow_html=True)

else:

    st.info(
        "Enter employee details and click Generate Prediction"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">

Built by Bhuvan • HR Attrition Prediction

</div>
""", unsafe_allow_html=True)
