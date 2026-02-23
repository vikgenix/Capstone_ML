import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Diabetes Predictor Pro",
    page_icon="🩺",
    layout="wide"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        models = joblib.load("diabetes_prediction_models.pkl")
        lr = models.get('lr')
        dt = models.get('dt')
        sc = models.get('scaler')
        return lr, dt, sc
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

lr, dt, scaler = load_models()

st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction & Risk Analysis</h1>", unsafe_allow_html=True)
st.write("Enter your health metrics below to get a comprehensive risk assessment and diagnosis prediction.")

# Input Layout
tabs = st.tabs(["👤 Demographics", "🏃 Lifestyle", "🩸 Clinical Metrics"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"]) # Female=0, Male=1 usually
    with col2:
        family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("History of Hypertension", ["No", "Yes"])
        cardio_history = st.selectbox("Cardiovascular History", ["No", "Yes"])

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        smoking = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"]) 
        alcohol = st.number_input("Alcohol Consumption (units/week)", min_value=0.0, value=0.0, step=1.0)
        physical_activity = st.number_input("Physical Activity (mins/week)", min_value=0, value=150)
    with col2:
        diet_score = st.slider("Diet Score (0-10)", 0, 10, 5)
        sleep = st.number_input("Sleep Hours/Day", min_value=0.0, max_value=24.0, value=7.0, step=1.0)
        screen_time = st.number_input("Screen Time Hours/Day", min_value=0.0, max_value=24.0, value=4.0, step=1.0)

with tabs[2]:
    col1, col2, col3 = st.columns(3)
    with col1:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        waist_hip = st.number_input("Waist-to-Hip Ratio", min_value=0.5, max_value=1.5, value=0.9)
        sys_bp = st.number_input("Systolic BP", min_value=80, max_value=220, value=120)
        dia_bp = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)
    with col2:
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=72)
        chol_total = st.number_input("Total Cholesterol", min_value=100, value=200)
        hdl = st.number_input("HDL Cholesterol", min_value=20, value=50)
        ldl = st.number_input("LDL Cholesterol", min_value=50, value=130)
    with col3:
        triglycerides = st.number_input("Triglycerides", min_value=50, value=150)
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, value=100)
        post_glucose = st.number_input("Postprandial Glucose (mg/dL)", min_value=50, value=140)
        insulin = st.number_input("Insulin Level", min_value=1, value=15)
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)

# Mapping Categorical Inputs
gender_map = {"Female": 0, "Male": 1}
yes_no_map = {"No": 0, "Yes": 1}
smoking_map = {"Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2}

# Prepare LR inputs (24 features)
lr_inputs = [
    age, gender_map[gender], smoking_map[smoking], alcohol, physical_activity,
    diet_score, sleep, screen_time, yes_no_map[family_history], yes_no_map[hypertension],
    yes_no_map[cardio_history], bmi, waist_hip, sys_bp, dia_bp, heart_rate,
    chol_total, hdl, ldl, triglycerides, fasting_glucose, post_glucose,
    insulin, hba1c
]

if st.button("Analyze Diabetes Risk"):
    # Stage 1: Linear Regression for Risk Score
    input_data_lr = np.array([lr_inputs])
    
    # Handle scaling if scaler is available
    if scaler is not None:
        input_data_lr_prepared = scaler.transform(input_data_lr)
    else:
        input_data_lr_prepared = input_data_lr
        st.warning("⚠️ Logic Error: Scaler not found in model file. Predictions may be inaccurate.")
        
    # Predict and Clip Risk Score to 0-100
    raw_r = lr.predict(input_data_lr_prepared)[0]
    pred_r = np.clip(raw_r, 0, 100)
    
    # Stage 2: Decision Tree for Diagnosis
    # DT was trained on [Scaled Features + pred_r]
    # We must append the pred_r to the already scaled inputs
    input_data_dt = np.append(input_data_lr_prepared, [[pred_r]], axis=1)
    
    prediction_prob = dt.predict_proba(input_data_dt)[0]
    prediction = dt.predict(input_data_dt)[0]
    
    # Results Presentation
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Diabetes Risk Score")
        risk_color = "green" if pred_r < 30 else "orange" if pred_r < 70 else "red"
        st.markdown(f"<h2 style='color: {risk_color};'>{pred_r:.2f} / 100</h2>", unsafe_allow_html=True)
        st.info("The risk score represents a continuous assessment of your diabetic predisposition based on your metabolic and lifestyle factors.")

    with res_col2:
        st.subheader("Diabetes Diagnosis Prediction")
        if prediction == 1:
            st.error("Prediction: DIABETIC DETECTED")
            st.write(f"Confidence: {prediction_prob[1]*100:.1f}%")
        else:
            st.success("Prediction: NO DIABETES DETECTED")
            st.write(f"Confidence: {prediction_prob[0]*100:.1f}%")
        st.warning("Disclaimer: This tool is for educational purposes and should not be used as clinical advice. Always consult a healthcare professional.")

st.markdown("---")
st.caption("Powered by optimized Linear Regression and Decision Tree models.")