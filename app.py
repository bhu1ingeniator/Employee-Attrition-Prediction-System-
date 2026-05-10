# =========================================================
# Employee Attrition Prediction Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="👨‍💼",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Main Background */

.stApp {
    background-color: #f4f7fb;
}

/* Remove top space */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* HERO SECTION */

.hero {

    background: linear-gradient(
        135deg,
        rgb(15, 23, 42),
        rgb(30, 41, 59),
        rgb(37, 99, 235)
    );

    padding: 60px;
    border-radius: 28px;
    color: white;
    text-align: center;
    margin-bottom: 30px;

    position: relative;
    overflow: hidden;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.12);
}

.hero::before {

    content: "";

    position: absolute;

    width: 300px;
    height: 300px;

    background: rgba(255,255,255,0.05);

    border-radius: 50%;

    top: -120px;
    right: -80px;
}

.hero-title {

    font-size: 52px;
    font-weight: bold;
    position: relative;
    z-index: 2;
}

.hero-subtitle {

    font-size: 18px;
    margin-top: 15px;
    color: #dbeafe;

    position: relative;
    z-index: 2;
}

/* KPI Cards */

.kpi-card {

    background-color: white;

    padding: 24px;

    border-radius: 20px;

    text-align: center;

    box-shadow: 0px 3px 15px rgba(0,0,0,0.06);
}

.kpi-title {

    font-size: 16px;
    color: gray;
}

.kpi-value {

    font-size: 32px;
    font-weight: bold;
    color: #2563eb;
}

/* Dashboard Cards */

.dashboard-card {

    background-color: white;

    padding: 28px;

    border-radius: 22px;

    box-shadow: 0px 3px 15px rgba(0,0,0,0.06);

    margin-top: 20px;
}

/* Button */

.stButton > button {

    width: 100%;

    background-color: #2563eb;

    color: white;

    border: none;

    border-radius: 12px;

    padding: 0.9rem;

    font-size: 16px;

    font-weight: 600;

    transition: 0.3s;
}

.stButton > button:hover {

    background-color: #1d4ed8;
}

/* Footer */

.footer {

    text-align: center;

    color: gray;

    margin-top: 50px;

    font-size: 14px;
}

/* Responsive */

@media (max-width:768px){

    .hero-title{
        font-size:34px;
    }

    .hero-subtitle{
        font-size:15px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
Employee Attrition Dashboard
</div>

<div class="hero-subtitle">
Predict whether an employee is likely to stay or leave the company
based on employee-related factors.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("HR-Employee-Attrition.csv")

# =========================================================
# KPI SECTION
# =========================================================

total_emp = len(df)

attrition_rate = round(
    (df['Attrition'].value_counts()['Yes'] / total_emp) * 100,
    1
)

avg_income = int(df['MonthlyIncome'].mean())

overtime_rate = round(
    (df['OverTime'].value_counts()['Yes'] / total_emp) * 100,
    1
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Total Employees</div>
    <div class="kpi-value">{total_emp}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Attrition Rate</div>
    <div class="kpi-value">{attrition_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Average Income</div>
    <div class="kpi-value">₹{avg_income}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Overtime Employees</div>
    <div class="kpi-value">{overtime_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREPROCESSING
# =========================================================

data = df.copy()

data.drop(
    columns=['EmployeeCount', 'StandardHours', 'Over18'],
    inplace=True
)

le = LabelEncoder()

data['Attrition'] = le.fit_transform(data['Attrition'])

data = pd.get_dummies(
    data,
    drop_first=True
)

X = data.drop('Attrition', axis=1)
y = data['Attrition']

feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# ANALYTICS SECTION
# =========================================================

st.markdown("""
<div class="dashboard-card">
""", unsafe_allow_html=True)

st.subheader("📊 HR Analytics")

c1, c2 = st.columns(2)

with c1:

    attrition_chart = pd.DataFrame({
        "Status": ["Stay", "Leave"],
        "Count": [
            df['Attrition'].value_counts()['No'],
            df['Attrition'].value_counts()['Yes']
        ]
    })

    st.bar_chart(
        attrition_chart.set_index("Status")
    )

with c2:

    overtime_chart = pd.DataFrame({
        "OverTime": df['OverTime'].value_counts()
    })

    st.bar_chart(overtime_chart)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREDICTION FORM
# =========================================================

st.markdown("""
<div class="dashboard-card">
""", unsafe_allow_html=True)

st.subheader("📝 Employee Prediction Form")

c1, c2, c3 = st.columns(3)

with c1:

    age = st.slider(
        "Age",
        18, 60, 30
    )

    distance = st.slider(
        "Distance From Home",
        1, 30, 5
    )

    income = st.number_input(
        "Monthly Income",
        1000,
        20000,
        5000
    )

with c2:

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1, 4, 3
    )

    work_life = st.slider(
        "Work Life Balance",
        1, 4, 3
    )

with c3:

    years_company = st.slider(
        "Years At Company",
        0, 40, 5
    )

    env_satisfaction = st.slider(
        "Environment Satisfaction",
        1, 4, 3
    )

    travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

predict = st.button("Generate Prediction")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREPARE INPUT
# =========================================================

def prepare_input():

    row = {col: 0 for col in feature_columns}

    row['Age'] = age
    row['DistanceFromHome'] = distance
    row['MonthlyIncome'] = income
    row['JobSatisfaction'] = job_satisfaction
    row['WorkLifeBalance'] = work_life
    row['YearsAtCompany'] = years_company
    row['EnvironmentSatisfaction'] = env_satisfaction

    if overtime == "Yes":
        row['OverTime_Yes'] = 1

    if f'BusinessTravel_{travel}' in row:
        row[f'BusinessTravel_{travel}'] = 1

    input_df = pd.DataFrame([row])

    input_scaled = scaler.transform(input_df)

    return input_scaled

# =========================================================
# PREDICTION
# =========================================================

if predict:

    user_data = prepare_input()

    prediction = model.predict(user_data)[0]

    probability = model.predict_proba(user_data)[0]

    stay_prob = probability[0] * 100
    leave_prob = probability[1] * 100

    st.markdown("""
    <div class="dashboard-card">
    """, unsafe_allow_html=True)

    st.subheader("📌 Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Employee likely to LEAVE ({leave_prob:.1f}%)"
        )

    else:

        st.success(
            f"✅ Employee likely to STAY ({stay_prob:.1f}%)"
        )

    st.write("")

    r1, r2 = st.columns(2)

    with r1:
        st.metric(
            "Stay Probability",
            f"{stay_prob:.1f}%"
        )

    with r2:
        st.metric(
            "Leave Probability",
            f"{leave_prob:.1f}%"
        )

    st.write("")

    chart_data = pd.DataFrame({
        "Status": ["Stay", "Leave"],
        "Probability": [stay_prob, leave_prob]
    })

    st.bar_chart(
        chart_data.set_index("Status")
    )

    st.subheader("⚠️ Risk Analysis")

    risks = []

    if overtime == "Yes":
        risks.append("Employee works overtime frequently")

    if job_satisfaction <= 2:
        risks.append("Low job satisfaction detected")

    if work_life <= 2:
        risks.append("Poor work-life balance")

    if env_satisfaction <= 2:
        risks.append("Low environment satisfaction")

    if distance > 20:
        risks.append("Long distance from office")

    if income < 3000:
        risks.append("Low monthly income")

    if risks:

        for risk in risks:
            st.warning(risk)

    else:

        st.success(
            "No major attrition risk factors detected"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

Built by Bhuvan • Employee Attrition Dashboard

</div>
""", unsafe_allow_html=True)