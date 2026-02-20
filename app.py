import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# --- CONFIGURATION ---
st.set_page_config(page_title="Multi-Disease Prediction Hub", page_icon="🏥", layout="wide")

# --- NAVIGATION SIDEBAR ---
st.sidebar.title("🏥 Diagnostic Hub")
st.sidebar.markdown("Choose a diagnostic template based on the clinical data available.")
app_mode = st.sidebar.selectbox("Select Model:", ["Alzheimer's Disease", "Heart Failure"])

# Helper function to load models safely
@st.cache_resource
def get_model(model_name):
    return load_model(model_name)

# Helper for Binary Labels
bin_map = {0: "No", 1: "Yes"}

# ==========================================
# TEMPLATE 1: ALZHEIMER'S DISEASE (COMPREHENSIVE)
# ==========================================
if app_mode == "Alzheimer's Disease":
    st.title("🧠 Alzheimer's Comprehensive Diagnostic Tool")
    st.markdown("This template utilizes 32 clinical parameters for high-precision screening.")

    model = get_model('alzheimers_final_pipeline')

    with st.form("alz_form"):
        tab1, tab2, tab3, tab4 = st.tabs(["Demographics & Lifestyle", "Medical History & Vitals", "Clinical Assessments", "Symptoms"])

        with tab1:
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 60, 90, 75)
            gender = c2.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
            ethnicity = c1.selectbox("Ethnicity", [0, 1, 2, 3], format_func=lambda x: {0:"Caucasian", 1:"African American", 2:"Asian", 3:"Other"}[x])
            edu = c2.selectbox("Education Level", [0, 1, 2, 3], format_func=lambda x: {0:"None", 1:"High School", 2:"Bachelor's", 3:"Higher"}[x])
            bmi = c1.number_input("BMI", 15.0, 40.0, 25.0)
            smoke = c2.selectbox("Smoking Status", [0, 1], format_func=lambda x: bin_map[x])
            alcohol = c1.slider("Weekly Alcohol Units", 0.0, 20.0, 5.0)
            physical = c2.slider("Weekly Physical Activity (Hrs)", 0.0, 10.0, 3.0)
            diet = c1.slider("Diet Quality Score (0-10)", 0.0, 10.0, 5.0)
            sleep = c2.slider("Sleep Quality Score (4-10)", 4.0, 10.0, 7.0)

        with tab2:
            c1, c2 = st.columns(2)
            fam_hist = c1.selectbox("Family History of Alzheimer's", [0, 1], format_func=lambda x: bin_map[x])
            cvd = c2.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: bin_map[x])
            diabetes = c1.selectbox("Diabetes", [0, 1], format_func=lambda x: bin_map[x])
            depression = c2.selectbox("Depression", [0, 1], format_func=lambda x: bin_map[x])
            head_injury = c1.selectbox("History of Head Injury", [0, 1], format_func=lambda x: bin_map[x])
            hyper = c2.selectbox("Hypertension", [0, 1], format_func=lambda x: bin_map[x])
            sbp = c1.number_input("Systolic BP (mmHg)", 90, 180, 120)
            dbp = c2.number_input("Diastolic BP (mmHg)", 60, 120, 80)
            chol_t = c1.number_input("Total Cholesterol (mg/dL)", 150.0, 300.0, 200.0)
            chol_ldl = c2.number_input("LDL Cholesterol (mg/dL)", 50.0, 200.0, 100.0)
            chol_hdl = c1.number_input("HDL Cholesterol (mg/dL)", 20.0, 100.0, 50.0)
            chol_trig = c2.number_input("Triglycerides (mg/dL)", 50.0, 400.0, 150.0)

        with tab3:
            mmse = st.slider("MMSE Score (0-30)", 0.0, 30.0, 24.0)
            func_assess = st.slider("Functional Assessment (0-10)", 0.0, 10.0, 5.0)
            adl = st.slider("ADL Score (0-10)", 0.0, 10.0, 5.0)

        with tab4:
            c1, c2 = st.columns(2)
            mem_comp = c1.selectbox("Memory Complaints", [0, 1], format_func=lambda x: bin_map[x])
            behav = c2.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: bin_map[x])
            confusion = c1.selectbox("Confusion", [0, 1], format_func=lambda x: bin_map[x])
            disorient = c2.selectbox("Disorientation", [0, 1], format_func=lambda x: bin_map[x])
            personality = c1.selectbox("Personality Changes", [0, 1], format_func=lambda x: bin_map[x])
            tasks = c2.selectbox("Difficulty Completing Tasks", [0, 1], format_func=lambda x: bin_map[x])
            forget = st.selectbox("Forgetfulness", [0, 1], format_func=lambda x: bin_map[x])

        submit_alz = st.form_submit_button("Run Alzheimer's Prediction")

    if submit_alz:
        input_data = pd.DataFrame([{
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
        }])
        
        prediction = predict_model(model, data=input_data)
        label = prediction['prediction_label'].iloc[0]
        score = prediction['prediction_score'].iloc[0]
        
        st.markdown("---")
        if label == 1:
            st.error(f"⚠️ **High Risk of Alzheimer's.** (Confidence: {score*100:.2f}%)")
        else:
            st.success(f"✅ **Low Risk / No Alzheimer's.** (Confidence: {score*100:.2f}%)")

# ==========================================
# TEMPLATE 2: HEART FAILURE (Friend's Model)
# ==========================================
elif app_mode == "Heart Failure":
    st.title("🫀 Heart Failure Prediction App")
    st.markdown("Enter the patient's clinical features below to predict heart disease risk.")
    
    model = get_model('heart_failure_pipeline')

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age_h = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=['M', 'F'])
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
            chol = st.number_input("Cholesterol (mm/dl)", 50, 600, 237)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
        with col2:
            max_hr = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
            chest_pain = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
            resting_ecg = st.selectbox("Resting ECG Results", options=['Normal', 'ST', 'LVH'])
            exercise_angina = st.selectbox("Exercise-Induced Angina", options=['N', 'Y'])
            oldpeak = st.number_input("Oldpeak (ST Depression)", -3.0, 10.0, 0.0, 0.1)
            st_slope = st.selectbox("ST Slope", options=['Up', 'Flat', 'Down'])

        submit_heart = st.form_submit_button("Predict Heart Disease Risk")

    if submit_heart:
        input_data_h = pd.DataFrame([{
            'Age': age_h, 'Sex': sex, 'ChestPainType': chest_pain, 'RestingBP': resting_bp,
            'Cholesterol': chol, 'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
            'MaxHR': max_hr, 'ExerciseAngina': exercise_angina, 'Oldpeak': oldpeak, 'ST_Slope': st_slope
        }])
        
        prediction = predict_model(model, data=input_data_h)
        label = prediction['prediction_label'].iloc[0]
        score = prediction['prediction_score'].iloc[0]
        
        st.markdown("---")
        if label == 1:
            st.error(f"⚠️ **High Risk of Heart Disease.** (Confidence: {score*100:.2f}%)")
        else:
            st.success(f"✅ **Low Risk / Normal.** (Confidence: {score*100:.2f}%)")