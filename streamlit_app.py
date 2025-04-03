import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the relative paths
decision_tree_model = os.path.join(current_dir, 'SWIFT', 'Models', 'decision_tree_model.pkl')
knn_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'knn_model.pkl')
logistic_regression_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'logistic_regression_model.pkl')
randomforest_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'random_forest_model.pkl')

# Load the models
deicision_tree_model = joblib.load(decision_tree_model)
knn_model = joblib.load(knn_model_path)
logistic_regression_model = joblib.load(logistic_regression_model_path)
randomforest_model = joblib.load(randomforest_model_path)

# Function to predict loan status
def predict_loan_status(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Make prediction using the loaded model
    prediction = deicision_tree_model.predict(input_df)[0]
    return prediction

st.title('Loan Approval Prediction')
st.write('Predict your loan approval chances with machine learning')

# Create a mapping for user-friendly labels
gender_options = {'Female': 0, 'Male': 1}
marital_status_options = {'Single': 0, 'Married': 1}
education_options = {'Graduate': 0, 'Not Graduate': 1}
employment_status_options = {'Unemployed': 0, 'Employed': 1}
credit_history_options = {'No/Bad Credit History': 0, 'Good Credit History': 1}
property_area_options = {'Rural': 0, 'Urban': 1}

# Create two columns for the input form
col1, col2 = st.columns(2)

# Personal information
with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Select Gender:", options=[""] + list(gender_options.keys()), index=0)
    married = st.selectbox("Select Marital Status:", options=[""] + list(marital_status_options.keys()), index=0)
    dependents = st.number_input("Enter Number of Dependents (e.g., 0, 1, 2):", value=None, min_value=0, step=1)
    education = st.selectbox("Select Education Level:", options=[""] + list(education_options.keys()), index=0)
    
    st.subheader("Property Information")
    property_area = st.selectbox("Select Property Area:", options=[""] + list(property_area_options.keys()), index=0)

# Financial information
with col2:
    st.subheader("Financial Information")
    self_employed = st.selectbox("Select Employment Status:", options=[""] + list(employment_status_options.keys()), index=0)
    credit_history = st.selectbox("Select Credit History:", options=[""] + list(credit_history_options.keys()), index=0)
    applicant_income_log = st.number_input("Enter Applicant Income (Monthly in ₱):", min_value=0.0, value=None)
    total_income_log = st.number_input("Enter Total Income (Payroll Amount in ₱):", min_value=0.0, value=None)

# Loan details
st.subheader("Loan Details")
loan_amount_log = st.slider("Enter Loan Amount:", 
                           min_value=1000.0, 
                           max_value=1000000.0, 
                           value=100000.0, 
                           step=1000.0,
                           format="₱ %d")

loan_amount_term_log = st.slider("Enter Loan Amount Term (in Months):", 
                                min_value=1.0, 
                                max_value=360.0, 
                                value=60.0, 
                                step=1.0,
                                format="%d months")

# Validation function
def is_valid_input():
    return all([
        gender != "", 
        married != "", 
        dependents is not None, 
        education != "", 
        self_employed != "", 
        credit_history != "", 
        property_area != "", 
        applicant_income_log is not None and applicant_income_log > 0, 
        loan_amount_log is not None and loan_amount_log > 0, 
        loan_amount_term_log is not None and loan_amount_term_log > 0, 
        total_income_log is not None and total_income_log > 0
    ])

# Function to reset all fields
def clear_fields():
    st.session_state.gender = ""
    st.session_state.married = ""
    st.session_state.dependents = None
    st.session_state.education = ""
    st.session_state.self_employed = ""
    st.session_state.credit_history = ""
    st.session_state.property_area = ""
    st.session_state.applicant_income_log = None
    st.session_state.loan_amount_log = None
    st.session_state.loan_amount_term_log = None
    st.session_state.total_income_log = None
    st.session_state.clear_triggered = True

# Buttons
col1, col2 = st.columns(2)
with col1:
    submit_button = st.button("Submit", disabled=not is_valid_input())
with col2:
    clear_button = st.button("Clear")

# Handle clear button
if clear_button:
    clear_fields()

# Results section    
if submit_button:
    # Prepare input data for prediction
    input_data = {
        'Gender': gender_options[gender],
        'Married': marital_status_options[married],
        'Dependents': dependents,
        'Education': education_options[education],
        'Self_Employed': employment_status_options[self_employed],
        'Credit_History': credit_history_options[credit_history],
        'Property_Area': property_area_options[property_area],
        'ApplicantIncomelog': applicant_income_log,
        'LoanAmountlog': loan_amount_log,
        'Loan_Amount_Term_log': loan_amount_term_log,
        'Total_Income_log': total_income_log
    }

    # Summary section
    st.subheader("Loan Application Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Personal Details:")
        st.write(f"• Gender: {gender}")
        st.write(f"• Marital Status: {married}")
        st.write(f"• Number of Dependents: {dependents}")
        st.write(f"• Education: {education}")
        st.write(f"• Employment Status: {self_employed}")
        
    with col2:
        st.write("Financial Details:")
        st.write(f"• Credit History: {credit_history}")
        st.write(f"• Property Area: {property_area}")
        st.write(f"• Monthly Income: ₱{applicant_income_log:,.2f}")
        st.write(f"• Loan Amount: ₱{loan_amount_log:,.2f}")
        st.write(f"• Loan Term: {loan_amount_term_log} months")
        st.write(f"• Total Income: ₱{total_income_log:,.2f}")

    # Make predictions with each model
    models = {
        "Decision Tree": deicision_tree_model,
        "KNN": knn_model,
        "Logistic Regression": logistic_regression_model,
        "Random Forest": randomforest_model
    }

    for model_name, model in models.items():
        prediction = predict_loan_status(input_data)
        probability = model.predict_proba(pd.DataFrame([input_data]))[0][1]
        
        st.subheader(f"{model_name} Model Prediction")
        col1, col2 = st.columns([2, 1.5])
        
        with col1:
            if prediction == 1:
                st.success("✅ Approval Likely")
                st.write(f"The applicant is likely to pay the loan. (Confidence: {probability:.2f})")
            else:
                st.error("❌ Approval Unlikely")
                st.write(f"The applicant is unlikely to pay the loan. (Confidence: {1 - probability:.2f})")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(['Approval', 'Denial'], [probability, 1 - probability])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title(f'{model_name} Prediction')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                       f'{height:.1%}', ha='center', va='center')
            st.pyplot(fig)
            plt.close(fig)
        
