import numpy as np
import pandas as pd
import pickle
import streamlit as st

rfc = pickle.load(open("Heart disease Prediction.pkl", "rb"))
ss = pickle.load(open("Scaler.pkl", "rb"))



st.title("💓 Heart Disease Risk Prediction")

st.write("Enter the details to check risk of heart disease:")

# Input fields
male = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 18, 100)
currentSmoker = st.selectbox("Currently Smokes?", ["No", "Yes"], help="If you choose 'No', cigarettes per day will be auto set to 0.")

if currentSmoker == "Yes":
    cigsPerDay = st.number_input("Cigarettes per Day", 1, 50)
else:
    cigsPerDay = 0
    st.write("🚭 Since you don't smoke, Cigarettes per Day is set to **0** automatically.")

BPMeds = st.selectbox("On BP Medication?", ["No", "Yes"])
prevalentStroke = st.selectbox("Had Stroke?", ["No", "Yes"])
prevalentHyp = st.selectbox("Hypertension?", ["No", "Yes"])
diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
totChol = st.number_input("Total Cholesterol", 100, 600)
sysBP = st.number_input("Systolic BP", 80, 250)
diaBP = st.number_input("Diastolic BP", 40, 150)
BMI = st.number_input("BMI", 10.0, 60.0)
heartRate = st.number_input("Heart Rate", 40, 180)
glucose = st.number_input("Glucose Level", 50, 300)

if st.button("Predict"):
    # Convert inputs
    input_data = np.array([[
        1 if male == "Male" else 0,
        age,
        1 if currentSmoker == "Yes" else 0,
        cigsPerDay,
        1 if BPMeds == "Yes" else 0,
        1 if prevalentStroke == "Yes" else 0,
        1 if prevalentHyp == "Yes" else 0,
        1 if diabetes == "Yes" else 0,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]])

    input_scaled = ss.transform(input_data)
   
    prediction = rfc.predict(input_scaled)[0]
    proba = rfc.predict_proba(input_scaled)[0][1]

    if prediction == 1:
      st.error(f"⚠️ High Risk of Heart Disease (Confidence: {proba:.2%})")
    else:
      st.success(f"✅ Low Risk of Heart Disease (Confidence: {1 - proba:.2%})")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "💡 <strong>Note:</strong> This is a machine learning-based prediction tool. Please consult a doctor for medical advice.<br>"
    "© 2025 Rajhans Bagri. All Rights Reserved."
    "</div>",
    unsafe_allow_html=True
)

