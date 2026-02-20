import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Alzheimer's Diagnostic System", page_icon="🧠", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def get_model():
    return load_model('alzheimers_final_pipeline')

model = get_model()

st.title("🧠 Alzheimer's Disease Real-Time Prediction")
st.markdown("Clinical Decision Support System for early detection and risk assessment.")
st.write("---")

# Helper for Binary Labels
bin_map = {0: "No", 1: "Yes"}

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("Patient Clinical Data Entry")
    
    # We use 4 columns to make the 32 variables look organized and professional
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.caption("📂 Demographics")
        age = st.number_input("Age", 60, 90, 75)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3], format_func=lambda x: {0:"Caucasian", 1:"African American", 2:"Asian", 3:"Other"}[x])
        edu = st.selectbox("Education", [0, 1, 2, 3], format_func=lambda x: {0:"None", 1:"High School", 2:"Bachelor's", 3:"Higher"}[x])
        st.caption("🥗 Lifestyle")
        bmi = st.number_input("BMI", 15.0, 40.0, 25.0)
        smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: bin_map[x])
        alcohol = st.slider("Alcohol Units/Wk", 0.0, 20.0, 5.0)
        physical = st.slider("Activity (Hrs/Wk)", 0.0, 10.0, 3.0)

    with col2:
        st.caption("🏥 Medical History")
        fam_hist = st.selectbox("Family History", [0, 1], format_func=lambda x: bin_map[x])
        cvd = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: bin_map[x])
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: bin_map[x])
        depression = st.selectbox("Depression", [0, 1], format_func=lambda x: bin_map[x])
        head_injury = st.selectbox("Head Injury", [0, 1], format_func=lambda x: bin_map[x])
        hyper = st.selectbox("Hypertension", [0, 1], format_func=lambda x: bin_map[x])
        diet = st.slider("Diet Quality", 0.0, 10.0, 5.0)
        sleep = st.slider("Sleep Quality", 4.0, 10.0, 7.0)

    with col3:
        st.caption("📊 Vital Measurements")
        sbp = st.number_input("Systolic BP", 90, 180, 120)
        dbp = st.number_input("Diastolic BP", 60, 120, 80)
        chol_t = st.number_input("Total Chol.", 150.0, 300.0, 200.0)
        chol_ldl = st.number_input("LDL Chol.", 50.0, 200.0, 100.0)
        chol_hdl = st.number_input("HDL Chol.", 20.0, 100.0, 50.0)
        chol_trig = st.number_input("Triglycerides", 50.0, 400.0, 150.0)
        st.caption("📝 Cognitive Scores")
        mmse = st.slider("MMSE Score", 0.0, 30.0, 24.0)
        func_assess = st.slider("Functional Score", 0.0, 10.0, 5.0)

    with col4:
        st.caption("⚠️ Symptoms")
        mem_comp = st.selectbox("Memory Complaints", [0, 1], format_func=lambda x: bin_map[x])
        behav = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: bin_map[x])
        confusion = st.selectbox("Confusion", [0, 1], format_func=lambda x: bin_map[x])
        disorient = st.selectbox("Disorientation", [0, 1], format_func=lambda x: bin_map[x])
        personality = st.selectbox("Personality Chg", [0, 1], format_func=lambda x: bin_map[x])
        tasks = st.selectbox("Task Difficulty", [0, 1], format_func=lambda x: bin_map[x])
        forget = st.selectbox("Forgetfulness", [0, 1], format_func=lambda x: bin_map[x])
        adl = st.slider("ADL Score", 0.0, 10.0, 5.0)

    st.write("---")
    submit = st.form_submit_button("GENERATE PREDICTION", use_container_width=True)

# --- PREDICTION LOGIC ---
if submit:
    input_dict = {
        'Age': age, 'Gender': gender, 'Ethnicity': ethnicity, 'EducationLevel': edu,
        'BMI': bmi, 'Smoking': smoke, 'AlcoholConsumption': alcohol, 'PhysicalActivity': physical,
        'DietQuality': diet, 'SleepQuality': sleep, 'FamilyHistoryAlzheimers': fam_hist,
        'CardiovascularDisease': cvd, 'Diabetes': diabetes, 'Depression': depression,
        'HeadInjury': head_injury, 'Hypertension': hyper, 'SystolicBP': sbp, 'DiastolicBP': dbp,
        'CholesterolTotal': chol_t, 'CholesterolLDL': chol_ldl, 'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_trig, 'MMSE': mmse, 'FunctionalAssessment': func_assess,
        'MemoryComplaints': mem_comp, 'BehavioralProblems': behav, 'ADL': adl,
        'Confusion': confusion, 'Disorientation': disorient, 'PersonalityChanges': personality,
        'DifficultyCompletingTasks': tasks, 'Forgetfulness': forget
    }
    
    input_df = pd.DataFrame([input_dict])
    prediction = predict_model(model, data=input_df)
    
    label = prediction['prediction_label'].iloc[0]
    score = prediction['prediction_score'].iloc[0]
    
    if label == 1:
        st.error(f"### Result: Positive Risk Detected (Confidence: {score:.2%})")
        st.markdown("Patient exhibits clinical indicators consistent with Alzheimer's Disease.")
    else:
        st.success(f"### Result: Negative / Low Risk (Confidence: {score:.2%})")
        st.markdown("Patient does not currently exhibit significant clinical indicators for Alzheimer's.")