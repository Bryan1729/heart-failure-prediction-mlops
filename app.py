import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Heart Failure Predictor", page_icon="🫀", layout="centered")

@st.cache_resource
def get_model():
    return load_model('heart_failure_pipeline')

model = get_model()

st.title("🫀 Heart Failure Prediction App")
st.markdown("Enter the patient's clinical features below to predict the likelihood of a heart disease event.")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Patient Demographics & Vitals")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=['M', 'F'])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=50, max_value=600, value=237)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        
    with col2:
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        chest_pain = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'], help="ASY = Asymptomatic is highest risk based on EDA.")
        resting_ecg = st.selectbox("Resting ECG Results", options=['Normal', 'ST', 'LVH'])
        exercise_angina = st.selectbox("Exercise-Induced Angina", options=['N', 'Y'])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=['Up', 'Flat', 'Down'], help="Flat slope is highly correlated with heart disease.")

    submit_button = st.form_submit_button(label="Predict Heart Disease Risk")


if submit_button:

    input_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }])
    

    prediction_result = predict_model(model, data=input_data)
    

    pred_label = prediction_result['prediction_label'].iloc[0]
    pred_score = prediction_result['prediction_score'].iloc[0] * 100
    

    st.markdown("---")
    st.subheader("Prediction Results")
    
    if pred_label == 1:
        st.error(f"⚠️ **High Risk of Heart Disease Detected.** (Confidence: {pred_score:.1f}%)")
        st.write("Please consult a healthcare professional immediately.")
    else:
        st.success(f"✅ **Low Risk / Normal.** (Confidence: {pred_score:.1f}%)")
        st.write("The model predicts no heart disease event based on these metrics.")