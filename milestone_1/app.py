import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Resolve paths relative to this script
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "..", "models")

# Page config
st.set_page_config(
    page_title="Diabetes Predictor Pro",
    page_icon="🩺",
    layout="wide"
)

# Load CSS
_css_path = os.path.join(_BASE_DIR, "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        models = joblib.load(os.path.join(_MODELS_DIR, "diabetes_prediction_models.pkl"))
        linear_regressor = models.get('linear_regressor')
        decision_tree_classifier = models.get('decision_tree_classifier')
        scaler = models.get('scaler')
        return linear_regressor, decision_tree_classifier, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

linear_regressor_model, decision_tree_model, scaler = load_models()

FEATURE_NAMES = [
    'age', 'gender', 'smoking_status', 'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day',
    'screen_time_hours_per_day', 'family_history_diabetes', 'hypertension_history',
    'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 'systolic_bp',
    'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides', 'glucose_fasting', 'glucose_postprandial',
    'insulin_level', 'hba1c'
]

st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction & Risk Analysis</h1>", unsafe_allow_html=True)
st.write("Enter your health metrics below to get a comprehensive risk assessment and diagnosis prediction.")

tabs = st.tabs(["👤 Demographics", "🏃 Lifestyle", "🩸 Clinical Metrics"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"])
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

gender_map  = {"Female": 0, "Male": 1}
yes_no_map  = {"No": 0, "Yes": 1}
smoking_map = {"Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2}

lr_inputs = [
    age, gender_map[gender], smoking_map[smoking], alcohol, physical_activity,
    diet_score, sleep, screen_time, yes_no_map[family_history], yes_no_map[hypertension],
    yes_no_map[cardio_history], bmi, waist_hip, sys_bp, dia_bp, heart_rate,
    chol_total, hdl, ldl, triglycerides, fasting_glucose, post_glucose,
    insulin, hba1c
]

if st.button("🔬 Analyze Diabetes Risk", use_container_width=True):
    input_df_lr = pd.DataFrame([lr_inputs], columns=FEATURE_NAMES)

    if scaler is not None:
        input_data_lr_prepared = scaler.transform(input_df_lr)
    else:
        input_data_lr_prepared = input_df_lr.values

    raw_risk_score = linear_regressor_model.predict(input_data_lr_prepared)[0]
    predicted_risk_score = np.clip(raw_risk_score, 0, 100)

    input_df_dt = pd.DataFrame(input_data_lr_prepared, columns=FEATURE_NAMES)
    input_df_dt['predicted_risk_score'] = predicted_risk_score

    prediction_prob = decision_tree_model.predict_proba(input_df_dt)[0]
    prediction = decision_tree_model.predict(input_df_dt)[0]

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("Diabetes Risk Score")
        risk_color = "green" if predicted_risk_score < 30 else "orange" if predicted_risk_score < 70 else "red"
        st.markdown(f"<h2 style='color: {risk_color};'>{predicted_risk_score:.2f} / 100</h2>", unsafe_allow_html=True)
        st.progress(int(predicted_risk_score))
        st.info("The risk score represents a continuous assessment of your diabetic predisposition.")

    with res_col2:
        st.subheader("Diabetes Diagnosis Prediction")
        if prediction == 1:
            st.error("🔴 Prediction: **DIABETIC DETECTED**")
            st.write(f"Confidence: **{prediction_prob[1]*100:.1f}%**")
        else:
            st.success("🟢 Prediction: **NO DIABETES DETECTED**")
            st.write(f"Confidence: **{prediction_prob[0]*100:.1f}%**")
        st.warning("⚕️ Disclaimer: This tool is for educational purposes only. Always consult a healthcare professional.")

st.markdown("---")
st.caption("Milestone 1 — Powered by Linear Regression + Decision Tree models.")
