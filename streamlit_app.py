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
logo_url = "https://raw.githubusercontent.com/DataAnalystJr/Stream-lit/main/SWIFT/swft.png"
st.image(logo_url, width=295)  # Adjust the width as needed
# Title of the app
st.title("Loan Approval Prediction")

# Streamlit app layout
st.title("Loan Application Input Form")

# Create a mapping for user-friendly labels
gender_options = {'Female': 0, 'Male': 1}
marital_status_options = {'Single': 0, 'Married': 1}
education_options = {'Graduate': 0, 'Not Graduate': 1}
employment_status_options = {'Unemployed': 0, 'Employed': 1}
credit_history_options = {'No/Bad Credit History': 0, 'Good Credit History': 1}
property_area_options = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

# Get input data from the user with validation
gender = st.selectbox("Select Gender:", options=[""] + list(gender_options.keys()), index=0)
married = st.selectbox("Select Marital Status:", options=[""] + list(marital_status_options.keys()), index=0)
dependents = st.number_input("Enter Number of Dependents (e.g., 0, 1, 2):", min_value=0, value=0)
education = st.selectbox("Select Education Level:", options=[""] + list(education_options.keys()), index=0)
self_employed = st.selectbox("Select Employment Status:", options=[""] + list(employment_status_options.keys()), index=0)
credit_history = st.selectbox("Select Credit History:", options=[""] + list(credit_history_options.keys()), index=0)
property_area = st.selectbox("Select Property Area:", options=[""] + list(property_area_options.keys()), index=0)
applicant_income_log = st.number_input("Enter Applicant Income (Monthly):", min_value=0.0, value=0.0)
loan_amount_log = st.number_input("Enter Loan Amount:", min_value=0.0, value=0.0)
loan_amount_term_log = st.number_input("Enter Loan Amount Term (in Days):", min_value=0.0, value=0.0)
total_income_log = st.number_input("Enter Total Income (Payroll Amount):", min_value=0.0, value=0.0)
# Output the collected and validated inputs
# Check for empty inputs
# Check for empty inputs and zero values
if (gender == "" or married == "" or education == "" or self_employed == "" or credit_history == "" or property_area == "" or
    applicant_income_log <= 0 or loan_amount_log <= 0 or loan_amount_term_log <= 0 or total_income_log <= 0):
    st.error("Please fill in all the required fields with valid values (greater than 0) before submitting.")
    st.stop()  # Stop further execution if there's an error

if st.button("Submit"):
    # Prepare input data for prediction
    try:
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
    except KeyError as e:
        st.error(f"An error occurred: {str(e)}. Please ensure all fields are filled correctly.")
        st.stop()  # Stop further execution if there's an error
    # Display the input data in a user-friendly format
    st.write("Collected Input Data:")
    for key, value in input_data.items():
        st.write(f"{key}: {value}")

    # Make predictions and display results for each model
    models = {
        "Decision Tree": deicision_tree_model,
        "KNN": knn_model,
        "Logistic Regression": logistic_regression_model,
        "Random Forest": randomforest_model
    }

    for model_name, model in models.items():
        prediction = predict_loan_status(input_data)
        probability = model.predict_proba(pd.DataFrame([input_data]))[0][1]
        st.title(f"{model_name} Model")
        if prediction == 1:
            st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
        else:
            st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")
        
        threshold = 0.7  # Define your threshold
        if probability > threshold:
            st.write(f"The applicant is low risk. (Probability: {probability:.2f})")
        else:
            st.write(f"The applicant is high risk. (Probability: {1 - probability:.2f})")
        
        # Visualization
        plt.figure(figsize=(6, 4))
        plt.bar(['Repayment', 'Default'], [probability, 1 - probability], color=['gray', 'gray'])
        plt.ylabel('Probability')
        st.pyplot(plt)
        plt.clf()


