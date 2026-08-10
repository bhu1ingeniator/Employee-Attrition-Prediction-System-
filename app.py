# Employee Attrition Prediction Dashboard
# Built with Streamlit + Machine Learning

import warnings

import numpy as np
import pandas as pd
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction Dashboard",
    page_icon="👥",
    layout="wide"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: rgb(245, 247, 251);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Hero Section */

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

    box-shadow: 0px 10px 30px rgba(0,0,0,0.12);

    position: relative;
    overflow: hidden;
}

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


/* Cards */

.dashboard-card {
    background-color: white;

    padding: 30px;

    border-radius: 22px;

    box-shadow: 0px 3px 15px rgba(0,0,0,0.06);
}

.result-card {
    background-color: white;

    padding: 28px;

    border-radius: 22px;

    box-shadow: 0px 3px 15px rgba(0,0,0,0.06);

    margin-top: 25px;
}


/* Button */

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


/* Footer */

.footer {
    text-align: center;

    color: gray;

    font-size: 14px;

    margin-top: 60px;

    margin-bottom: 10px;
}


/* Responsive */

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


# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown("""
<div class="hero-container">

<div class="hero-title">
Employee Attrition Prediction
</div>

<div class="hero-subtitle">
Predict whether an employee is likely to stay in the company
or leave based on employee details.
</div>

</div>
""", unsafe_allow_html=True)

st.caption(
    "AI-powered employee attrition analysis dashboard"
)


# --------------------------------------------------
# Dataset and Model Configuration
# --------------------------------------------------

DATA_PATH = "data/HR-Employee-Attrition.csv"

TARGET = "Attrition"

NUMERICAL_FEATURES = [
    "Age",
    "DistanceFromHome",
    "JobLevel",
    "MonthlyIncome",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "MaritalStatus",
    "OverTime",
    "BusinessTravel"
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# Train Model
# --------------------------------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_PATH)

    # Keep only the features used by the dashboard
    df = df[FEATURES + [TARGET]].copy()

    # Convert target into binary values
    df[TARGET] = df[TARGET].map({
        "No": 0,
        "Yes": 1
    })

    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    # Transform training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Balance training data
    smote = SMOTE(random_state=42)

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_processed,
        y_train
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    return model, preprocessor


model, preprocessor = train_model()


# --------------------------------------------------
# Employee Input Form
# --------------------------------------------------

st.markdown(
    "<div class='dashboard-card'>",
    unsafe_allow_html=True
)

st.markdown("## 📝 Employee Information")

c1, c2, c3 = st.columns(3)


# --------------------------------------------------
# Personal Information
# --------------------------------------------------

with c1:

    st.markdown("### 👤 Personal Info")

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    distance = st.slider(
        "Distance From Home",
        1,
        30,
        5
    )


# --------------------------------------------------
# Job Details
# --------------------------------------------------

with c2:

    st.markdown("### 💼 Job Details")

    job_level = st.slider(
        "Job Level",
        1,
        5,
        2
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
        min_value=1000,
        max_value=200000,
        value=5000,
        step=500
    )


# --------------------------------------------------
# Satisfaction
# --------------------------------------------------

with c3:

    st.markdown("### 📊 Satisfaction")

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    env_satisfaction = st.slider(
        "Environment Satisfaction",
        1,
        4,
        3
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1,
        4,
        3
    )

    years_at_company = st.slider(
        "Years At Company",
        0,
        40,
        3
    )


st.write("")

predict_button = st.button(
    "Generate Prediction"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# --------------------------------------------------
# Prepare User Input
# --------------------------------------------------

def prepare_input():

    input_data = pd.DataFrame([{
        "Age": age,
        "DistanceFromHome": distance,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": env_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "Gender": gender,
        "MaritalStatus": marital_status,
        "OverTime": overtime,
        "BusinessTravel": travel
    }])

    return input_data


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    input_data = prepare_input()

    processed_input = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        processed_input
    )[0]

    probability = model.predict_proba(
        processed_input
    )[0]

    stay_prob = probability[0] * 100
    leave_prob = probability[1] * 100


    st.markdown(
        "<div class='result-card'>",
        unsafe_allow_html=True
    )

    st.markdown("## 📌 Prediction Result")


    # Prediction result

    if prediction == 1:

        st.error(
            f"⚠️ Employee likely to LEAVE ({leave_prob:.1f}%)"
        )

    else:

        st.success(
            f"✅ Employee likely to STAY ({stay_prob:.1f}%)"
        )


    st.write("")


    # Probability metrics

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


    # --------------------------------------------------
    # Charts
    # --------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        st.markdown("### 📊 Prediction Analysis")

        chart_data = pd.DataFrame({
            "Status": [
                "Stay",
                "Leave"
            ],
            "Probability": [
                stay_prob,
                leave_prob
            ]
        })

        st.bar_chart(
            chart_data.set_index("Status")
        )


    # --------------------------------------------------
    # Risk Factors
    # --------------------------------------------------

    with c2:

        st.markdown("### ⚠️ Risk Factors")

        risks = []

        if overtime == "Yes":
            risks.append(
                "Works overtime frequently"
            )

        if job_satisfaction <= 2:
            risks.append(
                "Low job satisfaction"
            )

        if work_life_balance <= 2:
            risks.append(
                "Poor work-life balance"
            )

        if env_satisfaction <= 2:
            risks.append(
                "Low environment satisfaction"
            )

        if distance > 20:
            risks.append(
                "Long travel distance"
            )

        if monthly_income < 3000:
            risks.append(
                "Low monthly income"
            )

        if years_at_company < 2:
            risks.append(
                "Short tenure at company"
            )


        if risks:

            for risk in risks:

                st.warning(risk)

        else:

            st.success(
                "No major risk factors detected"
            )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

else:

    st.info(
        "Enter employee details and click Generate Prediction"
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""
<div class="footer">
Built by Bhuvan • HR Attrition Prediction
</div>
""", unsafe_allow_html=True)