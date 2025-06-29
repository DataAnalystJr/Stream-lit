import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the page configuration to wide layout
st.set_page_config(layout="wide")

# Initialize session state variables
if 'clear_triggered' not in st.session_state:
    st.session_state.clear_triggered = False

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))


rf_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'rf_model_with_info.joblib')

# Print the path for debugging
print(f"Looking for model at: {rf_model_path}")
print(f"Current directory: {current_dir}")

# Check if file exists before loading
if not os.path.exists(rf_model_path):
    st.error(f"Model file not found at: {rf_model_path}")
    st.stop()

# Load the model
try:
    model_info = joblib.load(rf_model_path)
    rf_model = model_info['model']  # Extract the model from the dictionary
    print("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Function to predict loan status
def predict_loan_status(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    
    # Make prediction using the loaded model
    prediction = rf_model.predict(input_df)[0]
    return prediction

# Center the title using HTML
# Center the title with a border using HTML and CSS
st.markdown("""
    <div style='text-align: center; border: 2px solid white; padding: 10px; border-radius: 15px; background-color: #333;'>
        <h1 style='color: white;'>Loan Repayment Prediction</h1>
    </div>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title("Loan Application Input Form")

# Create a mapping for user-friendly labels
gender_options = {'Female': 0, 'Male': 1}
marital_status_options = {'Single': 0, 'Married': 1}
dependents_options = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_options = {'High School Graduate': 0, 'College Graduate': 1}
employment_status_options = {'No': 0, 'Yes': 1}
credit_history_options = {'Bad Credit History': 0, 'Good Credit History': 1}
property_area_options = {'Rural': 0, 'Urban': 1}

# Input Fields arranged in columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Select Gender:", options=[""] + list(gender_options.keys()), index=0)
    married = st.selectbox("Select Marital Status:", options=[""] + list(marital_status_options.keys()), index=0)
    dependents = st.selectbox("Select Number of Dependents:", options=[""] + list(dependents_options.keys()), index=0)
    education = st.selectbox("Select Education Level:", options=[""] + list(education_options.keys()), index=0)
    self_employed = st.selectbox("Are you Self Employed?", options=[""] + list(employment_status_options.keys()), index=0)

with col2:
    # Revert to original number_input fields for Applicant Income and Loan Amount
    applicant_income = st.number_input("Enter Applicant Income (Monthly):", 
                                     min_value=0.0, 
                                     value=None,
                                     help="Enter your monthly income before any deductions")
    loan_amount = st.number_input("Enter Loan Amount:", min_value=0.0, value=None)
    loan_term = st.slider("Select Monthly Loan Term (Months):", min_value=1, max_value=100, value=12, help="Select the loan term in months (1-100)")
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
        st.write(f"Applicant Income: {int(applicant_income):,}")
        st.write(f"Loan Amount: {int(loan_amount):,}")
        st.write(f"Monthly Loan Term: {loan_term}")
        st.write(f"Credit History: {credit_history}")
        st.write(f"Property Area: {property_area}")
        st.write(f"Loan to Income Ratio: {loan_to_income_ratio:.2f}")

        try:
            # Make predictions
            prediction = predict_loan_status(input_data)
            probability = rf_model.predict_proba(pd.DataFrame([input_data]))[0][1]
            
            st.title("Random Forest Model Prediction")  # Updated title
            if prediction == 1:
                st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
            else:
                st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")


            # Visualization with minimalist aesthetic
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(14, 7))  # Significantly increased size

            # Use a modern color palette
            colors = ['#4CAF50', '#FF5252']  # Modern green and red

            # Create bars with subtle transparency
            bars = plt.bar(['Repayment', 'Default'],
                         [probability, 1 - probability],
                         color=colors,
                         alpha=0.8,
                         width=0.9)  # Increased bar width

            # Add value labels with clean typography
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom',
                        fontsize=16,  # Increased font size
                        fontweight='medium',
                        color='#2C3E50')

            # Clean up the plot
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)

            # Remove y-axis ticks and labels
            plt.yticks([])
            plt.ylabel('')

            # Style the x-axis
            plt.xticks(fontsize=16, color='#2C3E50')  # Increased font size

            # Add a subtle grid
            plt.grid(axis='y', linestyle='-', alpha=0.1, color='#2C3E50')

            # Set background color to white
            plt.gca().set_facecolor('white')
            plt.gcf().set_facecolor('white')

            # Add a title with modern typography
            plt.title('Loan Repayment Probability',
                     fontsize=18,  # Increased font size
                     fontweight='medium',
                     color='#2C3E50',
                     pad=30)  # Adjusted padding

            # Adjust layout
            plt.tight_layout(pad=2.5) # Adjusted padding

            # Display the plot
            # Removed centering columns as page layout is now wide
            st.pyplot(plt, use_container_width=True)
            plt.clf()
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Input data structure:")
            st.write(input_data)

with col2:
    if st.button("Clear"):
        clear_fields()
        
