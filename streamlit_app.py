import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
import xgboost

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the relative paths
rf_path = os.path.join(current_dir, 'SWIFT', 'Models', 'RF.pkl')
xgb_path = os.path.join(current_dir, 'SWIFT', 'Models', 'XGB.pkl')

# Load the models silently without success messages
rf_model = joblib.load(rf_path)
xgb_model = joblib.load(xgb_path)

# Center the title with a border using HTML and CSS
st.markdown("""
    <style>
    .title-container {
        padding: 2rem 1rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #202020, #202020);
        border-radius: 15px;
        border: 2px solid #585858;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card {
        background-color: transparent;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: transparent;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .loan-slider-container {
        background-color: transparent;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 2rem !important;
        padding: 0;
    }
    div[data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    div[class*="stMarkdown"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 64rem !important;
    }
    section[data-testid="stSidebar"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    div[class*="stVerticalBlock"] {
        gap: 0 !important;
        padding: 0 !important;
    }
    .main > .block-container {
        max-width: 64rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }
    /* New background color styles */
    .stApp {
        background-color: #EBE8DB;
    }
    .stApp > header {
        background-color: #EBE8DB;
    }
    .stApp > div {
        background-color: #EBE8DB;
    }
    .stApp > div > div {
        background-color: #EBE8DB;
    }
    .stApp > div > div > div {
        background-color: #EBE8DB;
    }
    .stApp > div > div > div > div {
        background-color: #EBE8DB;
    }
    </style>
    <div class="title-container">
        <h1 style='color: white;'>Loan Approval Prediction1</h1>
        <p style='color: #aaa; margin-bottom: 0;'>Predict your loan approval chances with machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Create a mapping for user-friendly labels
gender_options = {'M': 1, 'F': 0}
marital_status_options = {'Married': 1, 'Single': 0}
education_options = {'College Graduate': 1, 'High School Graduate': 0}
employment_status_options = {'Yes': 1, 'No': 0}
credit_history_options = {'1': 1, '0': 0}
property_area_options = {'Y': 1, 'N': 0}

# Create two columns for the input form with adjusted ratio
col1, col2 = st.columns([1, 1])

# Personal information
with col1:
    st.subheader("Personal Information")
    Gender = st.selectbox("Customer_Gender:", options=[""] + list(gender_options.keys()), index=0)
    Married = st.selectbox("Married:", options=[""] + list(marital_status_options.keys()), index=0)
    dependents_options = [0, 1, 2, "3+"]
    Dependents = st.selectbox("Dependents:", options=[""] + dependents_options, index=0)
    education = st.selectbox("Education:", options=[""] + list(education_options.keys()), index=0)
    
    st.subheader("Property Information")
    property_area = st.selectbox("Property_Loan_Stat:", options=[""] + list(property_area_options.keys()), index=0)

# Financial information
with col2:
    st.subheader("Financial Information")
    self_employed = st.selectbox("Self_Employed:", options=[""] + list(employment_status_options.keys()), index=0)
    credit_history = st.selectbox("Credit_His:", options=[""] + list(credit_history_options.keys()), index=0)
    applicant_income_log = st.number_input("Monthly_i:", min_value=0.0, value=None, step=1000.0)

# Loan details
st.subheader("Loan Details")
st.write("**ApplicantLoanAmot:**")
loan_amount_log = st.number_input("", 
                           min_value=1000.0, 
                           max_value=1000000.0, 
                           value=100000.0,
                           format="%f")

st.write("**Enter Loan Amount Term (in Months):**")
loan_amount_term_log = st.slider("", 
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
        loan_amount_term_log is not None and loan_amount_term_log > 0
    ])

# Function to reset all fields
def clear_fields():
    # Reset all session state variables to default values
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
    
    # Set the flag for triggering a rerun
    st.session_state.clear_triggered = True

# Action buttons - centered and styled
button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
with button_col2:
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.button("Submit", disabled=not is_valid_input(), use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)

# Handle clear button
if clear_button:
    clear_fields()

# Results section    
if submit_button:
    # Prepare input data for prediction
    input_data = {
        'gender': gender_options[gender],
        'married': marital_status_options[married],
        'dependents': dependents,
        'education': education_options[education],
        'self_employed': employment_status_options[self_employed],
        'credit_history': credit_history_options[credit_history],
        'property_area': property_area_options[property_area],
        'applicant_income_log': applicant_income_log,
        'loan_amount_log': loan_amount_log,
        'loan_amount_term_log': loan_amount_term_log
    }

    # Summary card with collected data
    st.markdown('<div class="card highlight">', unsafe_allow_html=True)
    st.subheader("Loan Application Summary")
    
    # Create two columns for the summary data
    sum_col1, sum_col2 = st.columns(2)
    
    with sum_col1:
        st.write("**Personal Details:**")
        st.write(f"• Gender: {gender}")
        st.write(f"• Marital Status: {married}")
        st.write(f"• Number of Dependents: {dependents}")
        st.write(f"• Education: {education}")
        st.write(f"• Employment Status: {self_employed}")
        
    with sum_col2:
        st.write("**Financial Details:**")
        st.write(f"• Credit History: {credit_history}")
        st.write(f"• Property Area: {property_area}")
        st.write(f"• Monthly Income: ₱{applicant_income_log:,.2f}")
        st.write(f"• Loan Amount: ₱{loan_amount_log:,.2f}")
        st.write(f"• Loan Term: {loan_amount_term_log} months")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Make predictions with each model
    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    for model_name, model in models.items():
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Transform features for both models
        try:
            input_df = transform_features_for_models(input_df)
        except Exception as e:
            st.error(f"Error transforming features for {model_name} model: {str(e)}")
            continue
        
        # Get prediction and probability from the current model
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Error making prediction with {model_name}: {str(e)}")
            continue
        
        # Result card for each model
        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.subheader(f"{model_name} Model Prediction")
        
        # Create columns for text and visualization
        res_col1, res_col2 = st.columns([3, 2])
        
        with res_col1:
            threshold = 0.7  # Define your threshold
            
            if prediction == 1:
                st.markdown(f"<h3 style='color: #3498DB;'>✅ Approval Likely</h3>", unsafe_allow_html=True)
                st.write(f"The applicant is likely to pay the loan. (Confidence: {probability:.2f})")
            else:
                st.markdown(f"<h3 style='color: #F1C40F;'>❌ Approval Unlikely</h3>", unsafe_allow_html=True)
                st.write(f"The applicant is unlikely to pay the loan. (Confidence: {1 - probability:.2f})")

            if probability > threshold:
                st.write(f"**Risk Assessment:** Low risk applicant (Score: {probability:.2f})")
            else:
                st.write(f"**Risk Assessment:** High risk applicant (Score: {1 - probability:.2f})")

        with res_col2:
            # Visualization with better colors and spacing
            fig, ax = plt.subplots(figsize=(4, 4.5))
            
            # Add more space at the top for labels
            plt.subplots_adjust(top=0.8)
            
            # New color theme with blue and yellow
            approval_color = '#3498DB'  # Bright blue
            denial_color = '#F1C40F'    # Bright yellow
            neutral_color = '#95A5A6'   # Modern gray
            
            bars = ax.bar(['Approval', 'Denial'], [probability, 1 - probability], 
                   color=[approval_color if probability > 0.5 else neutral_color, 
                          denial_color if probability <= 0.5 else neutral_color])
            
            # Set background color to light gray
            fig.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            
            # Customize grid and spines
            ax.grid(True, linestyle='--', alpha=0.3, color='#D5D8DC')
            for spine in ax.spines.values():
                spine.set_edgecolor('#ABB2B9')
                spine.set_linewidth(0.5)
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability', color='#2C3E50')
            ax.set_title(f'{model_name} Prediction', pad=20, color='#2C3E50')
            
            # Add percentage labels with more vertical spacing
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2. - 0.05, height + 0.05,
                        f'{height:.1%}', ha='center', va='bottom', color='#2C3E50')
                
            st.pyplot(fig)
            plt.close(fig)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
print("hello")

# Add space at the bottom
st.markdown("<br><br><br>", unsafe_allow_html=True)

# End of file
        
