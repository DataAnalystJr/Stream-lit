import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


import joblib
import os

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the relative paths
knn_model_path = os.path.join(current_dir,'SWIFT', 'Models', 'knn_model.pkl')
logistic_regression_model_path = os.path.join(current_dir,'SWIFT', 'Models', 'logistic_regression_model.pkl')
randomforest_model_path = os.path.join(current_dir,'SWIFT', 'Models', 'random_forest_model.pkl')

# Load the models
knn_model = joblib.load(knn_model_path)
logistic_regression_model = joblib.load(logistic_regression_model_path)
randomforest_model = joblib.load(randomforest_model_path)

# Function to predict loan status
def predict_loan_status(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Make prediction using the loaded model
    prediction = decisiontree_model.predict(input_df)[0]
    prediction = knn_model.predict(input_df)[0]
    prediction = logistic_regression_model.predict(input_df)[0]
    prediction = randomforest_model.predict(input_df)[0]

    # Return the prediction
    return prediction


# Title of the app
st.title("Loan Approval Prediction")


def get_valid_input(prompt, data_type, allowed_values=None):
    # Streamlit does not require a loop for input; we can use the widgets directly
    if allowed_values is not None:
        # Create a selectbox for allowed values
        value = st.selectbox(prompt, allowed_values)
    else:
        # Create a text input for other values
        user_input = st.text_input(prompt)
        if user_input:
            try:
                value = data_type(user_input)
            except ValueError:
                st.error("Invalid input. Please enter a valid value.")
                return None
        else:
            st.error("Input cannot be empty.")
            return None

    return value

# Streamlit app layout
st.title("Loan Application Input Form")
st.title("Loanloan")
# Get input data from the user with validation
gender = get_valid_input("Select Gender:", int, [0, 1])
married = get_valid_input("Select Marital Status:", int, [0, 1])
dependents = get_valid_input("Enter Number of Dependents (e.g., 0, 1, 2):", int)  # Assume no fixed values
education = get_valid_input("Select Education Level:", int, [0, 1])
self_employed = get_valid_input("Select Self Employment Status:", int, [0, 1])
credit_history = get_valid_input("Select Credit History:", float, [0, 1])
property_area = get_valid_input("Select Property Area:", int, [0, 1, 2])
applicant_income_log = get_valid_input("Enter Applicant Income (log-transformed value):", float)
loan_amount_log = get_valid_input("Enter Loan Amount (log-transformed value):", float)
loan_amount_term_log = get_valid_input("Enter Loan Amount Term (log-transformed value):", float)
total_income_log = get_valid_input("Enter Total Income (log-transformed value):", float)

# Output the collected and validated inputs
if st.button("Submit"):
    st.write("\nCollected Input Data:")
    st.write(f"Gender: {gender}")
    st.write(f"Marital Status: {married}")
    st.write(f"Dependents: {dependents}")
    st.write(f"Education: {education}")
    st.write(f"Self-Employed: {self_employed}")
    st.write(f"Credit History: {credit_history}")
    st.write(f"Property Area: {property_area}")
    st.write(f"Applicant Income (log): {applicant_income_log}")
    st.write(f"Loan Amount (log): {loan_amount_log}")
    st.write(f"Loan Amount Term (log): {loan_amount_term_log}")
    st.write(f"Total Income (log): {total_income_log}")
# Prepare input data for prediction
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'ApplicantIncomelog': applicant_income_log,
        'LoanAmountlog': loan_amount_log,
        'Loan_Amount_Term_log': loan_amount_term_log,
        'Total_Income_log': total_income_log
    }
# Create a dictionary for mapping numerical values to textual representations
    text_mapping = {
    'Gender': {0: 'Female', 1: 'Male'},
    'Married': {0: 'No', 1: 'Yes'},
    'Education': {0: 'Graduate', 1: 'Not Graduate'},
    'Self_Employed': {0: 'No', 1: 'Yes'},
    'Property_Area': {0: 'Rural', 1: 'Semiurban', 2: 'Urban'}
    }
        # Create a copy of input_data for display purposes
    display_data = input_data.copy()

    for key, value in display_data.items():
        if key in text_mapping and value in text_mapping[key]:
            display_data[key] = text_mapping[key][value]

    # Transform data into a row-based format for display
    row_data = [{"Feature": key, "Value": value} for key, value in display_data.items()]

    # Display the input data in a row format using pandas DataFrame
    df = pd.DataFrame(row_data)
    st.write(df)

    # Make prediction using KNN MODEL
    prediction = predict_loan_status(input_data)
    probability = knn_model.predict_proba(pd.DataFrame([input_data]))[0][1]

# Print the prediction with probability
    if prediction == 1:
        st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
    else:
        st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")

# Visualization
    plt.figure(figsize=(6, 4))  # Optional: Set figure size
    plt.bar(['Repayment', 'Default'], [probability, 1 - probability])
    plt.title('Loan Repayment Probability')
    plt.ylabel('Probability')
    st.pyplot(plt)
    plt.clf()

    prediction = predict_loan_status(input_data)
    probability = logistic_regression_model.predict_proba(pd.DataFrame([input_data]))[0][1]

# Print the prediction with probability
    if prediction == 1:
        st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
    else:
        st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")

# Visualization
    plt.figure(figsize=(6, 4))  # Optional: Set figure size
    plt.bar(['Repayment', 'Default'], [probability, 1 - probability])
    plt.title('Loan Repayment Probability')
    plt.ylabel('Probability')
    st.pyplot(plt)
    plt.clf()

    prediction = predict_loan_status(input_data)
    probability = randomforest_model.predict_proba(pd.DataFrame([input_data]))[0][1]

# Print the prediction with probability
    if prediction == 1:
        st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
    else:
        st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")

# Visualization
    plt.figure(figsize=(6, 4))  # Optional: Set figure size
    plt.bar(['Repayment', 'Default'], [probability, 1 - probability])
    plt.title('Loan Repayment Probability')
    plt.ylabel('Probability')
    st.pyplot(plt)
    plt.clf()
