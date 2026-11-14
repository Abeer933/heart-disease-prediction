import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("heart_disease_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
st.title("❤️ Heart Disease Prediction System")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Accuracy",
        value="98.54%"
    )

with col2:
    st.metric(
        label="Algorithm",
        value="Random Forest"
    )

with col3:
    st.metric(
        label="Features",
        value="13"
    )

# Sidebar
st.sidebar.title("❤️ Project Information")
st.sidebar.write("Heart Disease Prediction System")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Accuracy: 98.54%")
st.sidebar.write("Developed using Python & Streamlit")

# Main Title
st.markdown(
    """
    <h1 style='text-align:center;color:red;'>
    ❤️ Heart Disease Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Enter patient medical details below to predict heart disease risk.")
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(
        "heart.png",
        width=300
    )
# Two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 50)

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    sex = 1 if gender == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        80,
        250,
        120
    )

    chol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

with col2:

    restecg = st.selectbox(
        "Rest ECG",
        [0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        50,
        250,
        150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3]
    )

# Predict Button
if st.button("🔍 Predict Heart Disease"):

    patient = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(patient)

    try:
        probability = model.predict_proba(patient)
        risk = round(probability[0][1] * 100, 2)
    except:
        risk = "Not Available"

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            f"⚠️ High Risk of Heart Disease\n\nRisk Score: {risk}%"
        )

    else:

        st.success(
            f"✅ Low Risk of Heart Disease\n\nRisk Score: {risk}%"
        )

# About Section
st.markdown("---")

st.header("📖 About Project")

st.write("""
This project predicts the likelihood of heart disease
using Machine Learning.

Algorithm Used:
- Random Forest Classifier

Features Used:
- Age
- Gender
- Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Chest Pain Type
- Exercise Induced Angina
- ECG Results
- Thalassemia

Model Accuracy:
98.54%
""")