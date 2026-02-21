import streamlit as st
import numpy as np
import pickle

# --- 1. LOAD THE SYSTEM ---
try:
    with open("mental_health_model.pkl", 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
except FileNotFoundError:
    st.error("⚠️ Error: 'mental_health_model.pkl' not found. Run 'train_model.py' first!")
    st.stop()

# --- 2. APP UI DESIGN ---
st.set_page_config(page_title="Student Mental Health AI", layout="centered")

st.title("🧠 Student Mental Health Predictor")
st.markdown("### AI-Powered Risk Assessment System")
st.info("This tool uses 8 distinct factors to predict the likelihood of student depression.")

# --- 3. INPUT FORM ---
with st.form("risk_form"):
    st.write("#### 📝 Enter Student Details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Academic & Personal Info**")
        age = st.slider("Age", 17, 35, 21)
        year = st.slider("Year of Study", 1, 4, 2)  # Added Year!
        gender = st.selectbox("Gender", ["Female", "Male"])
        cgpa = st.number_input("CGPA (0.00 - 4.00)", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        marital = st.radio("Are you Married?", ["No", "Yes"]) 
        
    with col2:
        st.write("**Mental Health History**")
        anxiety = st.radio("Do you experience Anxiety?", ["No", "Yes"])
        panic = st.radio("Do you have Panic Attacks?", ["No", "Yes"])
        specialist = st.radio("Have you sought specialist treatment before?", ["No", "Yes"]) # Added Specialist!

    submit = st.form_submit_button("🔍 Analyze Risk")

# --- 4. PREDICTION LOGIC ---
if submit:
    # 1. Convert Text Inputs to Numbers
    gen_val = 0 if gender == "Female" else 1
    anx_val = 1 if anxiety == "Yes" else 0
    pan_val = 1 if panic == "Yes" else 0
    mar_val = 1 if marital == "Yes" else 0
    spec_val = 1 if specialist == "Yes" else 0
    
    # 2. Prepare Data Row (MUST MATCH EXACT ORDER IN train_model.py)
    # Order: [Age, Year, CGPA, Gender, Anxiety, Panic, Marital, Specialist]
    raw_data = np.array([[age, year, cgpa, gen_val, anx_val, pan_val, mar_val, spec_val]])
    
    # 3. Scale the Data
    scaled_data = scaler.transform(raw_data)
    
    # 4. Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    # 5. Display Result
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected**")
        st.write(f"Confidence: **{probability*100:.1f}%**")
        st.write("**Suggestion:** Please consider reaching out to university counseling services.")
    else:
        st.success(f"✅ **Low Risk Detected**")
        st.write(f"Confidence: **{probability*100:.1f}%**")
        st.write("**Suggestion:** Keep up the healthy study-life balance!")