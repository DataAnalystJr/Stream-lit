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

# Function to transform features for model prediction
def transform_features_for_models(df):
    # Initialize array with 102 features as expected by the models
    transformed_features = np.zeros((1, 102))
    
    # 1. Handle continuous variables first - apply log transformation only once
    transformed_features[0, 0] = df['applicant_income_log'].values[0]  # ApplicantIncomeLog
    transformed_features[0, 1] = df['loan_amount_log'].values[0]  # LoanAmountLog
    transformed_features[0, 2] = df['loan_amount_term_log'].values[0]  # Monthly_Loan_Amount_TermLog
    
    # Calculate and store derived features
    loan_to_income_ratio = df['loan_amount_log'].values[0] / df['applicant_income_log'].values[0]
    transformed_features[0, 3] = loan_to_income_ratio  # Loan_to_Income_RatioLog
    transformed_features[0, 4] = loan_to_income_ratio  # DTI_Log (same as Loan_to_Income_RatioLog)
    
    # 2. Handle categorical variables
    # Gender (2 features)
    gender_val = df['gender'].values[0]
    transformed_features[0, 5] = 1 if gender_val == 1 else 0  # Male
    transformed_features[0, 6] = 1 if gender_val == 0 else 0  # Female
    
    # Married (2 features)
    married_val = df['married'].values[0]
    transformed_features[0, 7] = 1 if married_val == 1 else 0  # Married
    transformed_features[0, 8] = 1 if married_val == 0 else 0  # Single
    
    # Dependents (4 features)
    dependents_val = df['dependents'].values[0]
    transformed_features[0, 9] = 1 if dependents_val == 0 else 0   # 0 dependents
    transformed_features[0, 10] = 1 if dependents_val == 1 else 0  # 1 dependent
    transformed_features[0, 11] = 1 if dependents_val == 2 else 0  # 2 dependents
    transformed_features[0, 12] = 1 if dependents_val == "3+" else 0  # 3+ dependents
    
    # Education (2 features)
    education_val = df['education'].values[0]
    transformed_features[0, 13] = 1 if education_val == 1 else 0  # Graduate
    transformed_features[0, 14] = 1 if education_val == 0 else 0  # Not Graduate
    
    # Self_Employed (2 features)
    self_employed_val = df['self_employed'].values[0]
    transformed_features[0, 15] = 1 if self_employed_val == 1 else 0  # Yes
    transformed_features[0, 16] = 1 if self_employed_val == 0 else 0  # No
    
    # Credit_History (2 features)
    credit_history_val = df['credit_history'].values[0]
    transformed_features[0, 17] = 1 if credit_history_val == 1 else 0  # Good
    transformed_features[0, 18] = 1 if credit_history_val == 0 else 0  # Bad
    
    # Property_Area (2 features)
    property_area_val = df['property_area'].values[0]
    transformed_features[0, 19] = 1 if property_area_val == 1 else 0  # Y
    transformed_features[0, 20] = 1 if property_area_val == 0 else 0  # N
    
    # Add some interaction features (indices 21-101)
    # Income * Loan Amount interaction
    transformed_features[0, 21] = transformed_features[0, 0] * transformed_features[0, 1]
    # Income * Loan Term interaction
    transformed_features[0, 22] = transformed_features[0, 0] * transformed_features[0, 2]
    # Loan Amount * Loan Term interaction
    transformed_features[0, 23] = transformed_features[0, 1] * transformed_features[0, 2]
    
    return transformed_features

# Construct the relative paths1
dt_path = os.path.join(current_dir, 'SWIFT', 'Models', 'decision_tree_smote_model.joblib')
xgb_path = os.path.join(current_dir, 'SWIFT', 'Models', 'XGB.pkl')

# Load the models silently without success messages
dt_model = joblib.load(dt_path)
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
        <h1 style='color: white;'>Loan Approval Prediction </h1>
        <p style='color: #aaa; margin-bottom: 0;'>Predict your loan approval chances with machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Create a mapping for user-friendly labels
gender_options = {'M': 1, 'F': 0}  # Gender
marital_status_options = {'Married': 1, 'Single': 0}  # Married
education_options = {'College Graduate': 1, 'High School Graduate': 0}  # Education
employment_status_options = {'Yes': 1, 'No': 0}  # Self_Employed
credit_history_options = {'Good': 1, 'Bad': 0}  # Credit_History
property_area_options = {'Urban': 1, 'Rural': 0}  # Property_Area

# Create two columns for the input form with adjusted ratio
col1, col2 = st.columns([1, 1])

# Personal information
with col1:
    st.subheader("Personal Information")
    Gender = st.selectbox("Gender:", options=[""] + list(gender_options.keys()), index=0, key="gender")
    Married = st.selectbox("Married:", options=[""] + list(marital_status_options.keys()), index=0, key="married")
    dependents_options = [0, 1, 2, "3+"]
    Dependents = st.selectbox("Dependents:", options=[""] + dependents_options, index=0, key="dependents")
    Education = st.selectbox("Education:", options=[""] + list(education_options.keys()), index=0, key="education")
    
    st.subheader("Property Information")
    property_area = st.selectbox("Property Area:", options=[""] + list(property_area_options.keys()), index=0, key="property_area")

# Financial information
with col2:
    st.subheader("Financial Information")
    self_employed = st.selectbox("Self Employed:", options=[""] + list(employment_status_options.keys()), index=0)
    credit_history = st.selectbox("Credit History:", options=[""] + list(credit_history_options.keys()), index=0)
    applicant_income_log = st.number_input("Monthly Income:", min_value=0.0, value=None, step=1000.0)

# Loan details
st.subheader("Loan Details")
st.write("**Loan Amount:**")
loan_amount_log = st.number_input("", 
                           min_value=1000.0, 
                           max_value=1000000.0, 
                           value=100000.0,
                           format="%f")

st.write("**Loan Term (in Months):**")
loan_amount_term_log = st.slider("", 
                                min_value=1.0, 
                                max_value=160.0, 
                                value=60.0, 
                                step=1.0,
                                format="%d months")

# Validation function
def is_valid_input():
    return all([
        Gender != "", 
        Married != "", 
        Dependents is not None, 
        Education != "", 
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
    for key in ["gender", "married", "dependents", "education", "self_employed", 
                "credit_history", "property_area", "applicant_income_log", 
                "loan_amount_log", "loan_amount_term_log"]:
        if key in st.session_state:
            st.session_state[key] = ""
    
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
        'gender': gender_options[Gender],
        'married': marital_status_options[Married],
        'dependents': Dependents,
        'education': education_options[Education],
        'self_employed': employment_status_options[self_employed],
        'credit_history': credit_history_options[credit_history],
        'property_area': property_area_options[property_area],
        'applicant_income_log': np.log1p(applicant_income_log),
        'loan_amount_log': np.log1p(loan_amount_log),
        'loan_amount_term_log': np.log1p(loan_amount_term_log)
    }

    # Calculate DTI ratio
    monthly_income = applicant_income_log
    monthly_loan_payment = loan_amount_log / loan_amount_term_log
    dti_ratio = (monthly_loan_payment / monthly_income) * 100  # Convert to percentage

    # Summary card with collected data
    st.markdown('<div class="card highlight">', unsafe_allow_html=True)
    st.subheader("Loan Application Summary")
    
    # Create two columns for the summary data
    sum_col1, sum_col2 = st.columns(2)
    
    with sum_col1:
        st.write("**Personal Details:**")
        st.write(f"• Gender: {Gender}")
        st.write(f"• Marital Status: {Married}")
        st.write(f"• Number of Dependents: {Dependents}")
        st.write(f"• Education: {Education}")
        st.write(f"• Employment Status: {self_employed}")
        
    with sum_col2:
        st.write("**Financial Details:**")
        st.write(f"• Credit History: {credit_history}")
        st.write(f"• Property Area: {property_area}")
        st.write(f"• Monthly Income: ₱{monthly_income:,.2f}")
        st.write(f"• Loan Amount: ₱{loan_amount_log:,.2f}")
        st.write(f"• Loan Term: {loan_amount_term_log} months")
        st.write(f"• Monthly Payment: ₱{monthly_loan_payment:,.2f}")
        st.write(f"• Debt-to-Income Ratio: {dti_ratio:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Make predictions with each model
    models = {
        "Decision Tree with SMOTE": dt_model
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