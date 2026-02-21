import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained Pipeline
with open('mental_health_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Student Mental Health Prediction System")

# 2. Input Form
with st.form("user_inputs"):
    st.subheader("Personal & Academic Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 17, 40, 20)
        year = st.selectbox("Year of Study", [1, 2, 3, 4])
        cgpa = st.slider("Current CGPA", 0.0, 4.0, 3.2)
        gender = st.selectbox("Gender", ["Female", "Male"])
    
    with col2:
        anxiety = st.radio("Experiencing Anxiety?", ["No", "Yes"])
        panic = st.radio("Experiencing Panic Attacks?", ["No", "Yes"])
        married = st.radio("Marital Status", ["Unmarried", "Married"])
        specialist = st.radio("Previously sought Specialist Treatment?", ["No", "Yes"])
    
    submit = st.form_submit_button("Generate Prediction & Feedback")

# 3. Prediction & Feedback Logic
if submit:
    # Prepare data for model
    input_data = pd.DataFrame([[
        age, year, cgpa,
        0 if gender == "Female" else 1,
        1 if anxiety == "Yes" else 0,
        1 if panic == "Yes" else 0,
        1 if married == "Married" else 0,
        1 if specialist == "Yes" else 0
    ]], columns=['Age', 'Year_Cleaned', 'CGPA_Cleaned', 'Choose your gender', 
                'Do you have Anxiety?', 'Do you have Panic attack?', 
                'Marital status', 'Did you seek any specialist for a treatment?'])

    # Get Prediction and Probability
    prediction = model_pipeline.predict(input_data)[0]
    risk_score = model_pipeline.predict_proba(input_data)[0][1]

    st.divider()

    # --- THE FEEDBACK SYSTEM ---
    if prediction == 1:
        st.error(f"### Result: HIGH RISK DETECTED ({risk_score*100:.1f}%)")
        
        st.subheader("🚨 System Feedback & Actions")
        st.write("Based on the data provided, the system suggests you may be experiencing significant academic or emotional stress.")
        
        st.markdown("""
        **Recommended Steps:**
        * **Seek Professional Support:** Please visit the college counseling department for a confidential chat.
        * **Talk to Someone:** Reach out to a trusted mentor, teacher, or family member.
        * **Prioritize Wellness:** Ensure you are getting at least 7-8 hours of sleep and taking breaks from studies.
        """)
    else:
        st.success(f"### Result: LOW RISK DETECTED ({(1-risk_score)*100:.1f}% Confidence)")
        
        st.subheader("💡 Wellness Maintenance Tips")
        st.write("Your current indicators suggest a stable mental state. Here is how to maintain it:")
        
        st.markdown("""
        **Keep Doing:**
        * **Balanced Schedule:** Continue managing your time between studies and relaxation.
        * **Stay Active:** Engaging in social clubs or sports can help prevent future stress.
        * **Self-Check:** Periodically check in with your feelings, especially during exam weeks.
        """)

st.caption("Disclaimer: This tool provides an AI-based risk screening and is not a clinical diagnosis.")