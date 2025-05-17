import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# Initialize session state variables
if 'clear_triggered' not in st.session_state:
    st.session_state.clear_triggered = False

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the path for the decision tree SMOTE model
decision_tree_smote_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'dtree_sop2.joblib')

# Load the model
decision_tree_smote_model = joblib.load(decision_tree_smote_model_path)

# Function to predict loan status
def predict_loan_status(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    
    # Make prediction using the loaded model
    prediction = decision_tree_smote_model.predict(input_df)[0]
    return prediction

# Center the title using HTML
# Center the title with a border using HTML and CSS
st.markdown("""
    <div style='text-align: center; border: 2px solid white; padding: 10px; border-radius: 15px; background-color: #333;'>
        <h1 style='color: white;'>Loan Approval Prediction</h1>
    </div>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title(" ")
st.title("Loan Application Input Form")

# Create a mapping for user-friendly labels
gender_options = {'Female': 0, 'Male': 1}
marital_status_options = {'Single': 0, 'Married': 1}
dependents_options = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_options = {'High School Graduate': 0, 'College Graduate': 1}
employment_status_options = {'No': 0, 'Yes': 1}
credit_history_options = {'Bad Credit History': 0, 'Good Credit History': 1}
property_area_options = {'Rural': 0, 'Urban': 1}

# Input Fields
gender = st.selectbox("Select Gender:", options=[""] + list(gender_options.keys()), index=0)
married = st.selectbox("Select Marital Status:", options=[""] + list(marital_status_options.keys()), index=0)
dependents = st.selectbox("Select Number of Dependents:", options=[""] + list(dependents_options.keys()), index=0)
education = st.selectbox("Select Education Level:", options=[""] + list(education_options.keys()), index=0)
self_employed = st.selectbox("Are you Self Employed?", options=[""] + list(employment_status_options.keys()), index=0)
applicant_income = st.number_input("Enter Applicant Income (Monthly):", min_value=0.0, value=None)
loan_amount = st.number_input("Enter Loan Amount:", min_value=0.0, value=None)
loan_term = st.number_input("Enter Monthly Loan Term:", min_value=0.0, value=None)
credit_history = st.selectbox("Select Credit History:", options=[""] + list(credit_history_options.keys()), index=0)
property_area = st.selectbox("Select Property Area:", options=[""] + list(property_area_options.keys()), index=0)

# Validation function
def is_valid_input():
    return all([
        gender != "", 
        married != "", 
        dependents != "", 
        education != "", 
        self_employed != "", 
        credit_history != "", 
        property_area != "", 
        applicant_income is not None and applicant_income > 0, 
        loan_amount is not None and loan_amount > 0, 
        loan_term is not None and loan_term > 0
    ])

# Function to reset all fields
def clear_fields():
    # Reset all session state variables to default values
    st.session_state.gender = ""
    st.session_state.married = ""
    st.session_state.dependents = ""
    st.session_state.education = ""
    st.session_state.self_employed = ""
    st.session_state.credit_history = ""
    st.session_state.property_area = ""
    st.session_state.applicant_income = None
    st.session_state.loan_amount = None
    st.session_state.loan_term = None
    
    # Set the flag for triggering a rerun
    st.session_state.clear_triggered = True

    
# Action buttons
col1, col2 = st.columns(2)

with col1:
    # Submit button
    if st.button("Submit", disabled=not is_valid_input()):
        # Calculate derived features
        loan_to_income_ratio = float(loan_amount) / float(applicant_income) if float(applicant_income) > 0 else 0
        
        # Prepare input data for prediction with exactly the columns the model was trained on
        input_data = {
            'Gender': gender_options[gender],
            'Married': marital_status_options[married],
            'Dependents': dependents_options[dependents],
            'Education': education_options[education],
            'Self_Employed': employment_status_options[self_employed],
            'Credit_History': credit_history_options[credit_history],
            'Property_Area': property_area_options[property_area],
            'ApplicantIncomeLog': np.log1p(float(applicant_income)),
            'Loan_to_Income_RatioLog': np.log1p(loan_to_income_ratio),
            'LoanAmountLog': np.log1p(float(loan_amount)),
            'Monthly_Loan_Amount_TermLog': np.log1p(float(loan_term) / 12)
        }

        # Display the input data as text
        st.write("Collected Input Data:")
        st.write(f"Gender: {gender}")
        st.write(f"Marital Status: {married}")
        st.write(f"Number of Dependents: {dependents}")
        st.write(f"Education: {education}")
        st.write(f"Self Employed: {self_employed}")
        st.write(f"Applicant Income: {applicant_income}")
        st.write(f"Loan Amount: {loan_amount}")
        st.write(f"Monthly Loan Term: {loan_term}")
        st.write(f"Credit History: {credit_history}")
        st.write(f"Property Area: {property_area}")
        st.write(f"Loan to Income Ratio: {loan_to_income_ratio:.2f}")

        try:
            # Make predictions
            prediction = predict_loan_status(input_data)
            probability = decision_tree_smote_model.predict_proba(pd.DataFrame([input_data]))[0][1]
            
            st.title("Decision Tree SMOTE Model Prediction")
            if prediction == 1:
                st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
            else:
                st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")

            threshold = 0.7  # Define your threshold
            if probability > threshold:
                st.write(f"The applicant is classified as low risk. (Probability: {probability:.2f})")
            else:
                st.write(f"The applicant is classified as high risk. (Probability: {1 - probability:.2f})")

            # Visualization
            plt.figure(figsize=(8, 5))
            colors = ['#2ecc71', '#e74c3c']  # Green for repayment, Red for default
            plt.bar(['Repayment', 'Default'], [probability, 1 - probability], color=colors)
            plt.title('Loan Repayment Probability')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(plt)
            plt.clf()
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Input data structure:")
            st.write(input_data)

with col2:
    if st.button("Clear"):
        clear_fields()
        