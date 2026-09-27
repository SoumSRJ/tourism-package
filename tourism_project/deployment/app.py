import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")
model = joblib.load(model_path)

st.title("Tourism Prediction App")
st.write("""
This application predicts  whether a customer will purchase the newly introduced Wellness Tourism Package.
Enter the sensor and configuration data below to get a prediction.
""")

Occupation   = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free lancer"])
Gender     = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100)
MaritalStatus = st.selectbox("MaritalStatus", ["Divorced", "Married","Single"])
TypeofContact    = st.selectbox("TypeofContact", ["Self Enquiry","Company Invited"])
CityTier       = st.selectbox("CityTier", [1, 2, 3])
NumberOfPersonVisiting = st.selectbox("NumberOfPersonVisiting", [1, 2, 3, 4, 5])
NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting", [0, 1, 2, 3])
Designation = st.selectbox("Designation", ["Executive", "Managerial","AVP","Senior Manager","VP"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0)




input_data = pd.DataFrame([{
    "Occupation": Occupation,
    "Gender": Gender,
    "Age": Age,
    "MaritalStatus": MaritalStatus,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome

}])

if st.button("Predict purchase"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
