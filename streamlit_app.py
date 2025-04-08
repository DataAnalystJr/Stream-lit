import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# Try to import XGBoost with version check
try:
    import xgboost
    st.write(f"XGBoost version: {xgboost.__version__}")
except ImportError:
    st.error("XGBoost is not installed. Please install it using: pip install xgboost==3.0.0")
    xgboost = None

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the relative paths
XGB_Model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'XGB.pkl')
randomforest_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'RandomForest.pkl')

# Load the models
XGB_Model = None
randomforest_model = None

if xgboost is not None:
    try:
        XGB_Model = joblib.load(XGB_Model_path)
        st.success("XGBoost model loaded successfully")
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        st.error("Please ensure you have the correct version of XGBoost installed (3.0.0 or compatible)")
else:
    st.error("XGBoost is not available. Some features will be disabled.")

try:
    randomforest_model = joblib.load(randomforest_model_path)
    st.success("Random Forest model loaded successfully")
except Exception as e:
    st.error(f"Error loading Random Forest model: {str(e)}")

def transform_features_for_models(input_df, model_name="XGB"):
    """Transform input features to match the expected format for both XGBoost and Random Forest models."""
    # Create dummy variables for categorical columns
    categorical_cols = ['gender', 'married', 'education', 'self_employed', 'credit_history', 'property_area']
    
    if model_name == "XGB":
        # For XGBoost, we need to transform the features to match the expected format
        # First, rename the columns to match the expected format
        input_df = transform_feature_names(input_df)
        
        # Calculate additional features that might be expected by the model
        input_df['Total_Income'] = input_df['ApplicantIncomelog']  # You might want to adjust this
        input_df['EMI'] = input_df['LoanAmountlog'] / input_df['LoanAmountTermlog']  # Monthly EMI
        
        # Ensure columns are in the correct order
        expected_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncomelog', 'LoanAmountlog', 'LoanAmountTermlog',
            'Credit_History', 'Property_Area', 'Total_Income', 'EMI'
        ]
        
        # Reorder columns to match the expected order
        return input_df[expected_columns]
    else:
        # For Random Forest, we need the 102 features transformation
        # Initialize a zero array with 102 features
        transformed = np.zeros(102)
        
        # Map the continuous variables (they will be in the same position)
        transformed[0] = input_df['applicant_income_log'].values[0]
        transformed[1] = input_df['loan_amount_log'].values[0]
        transformed[2] = input_df['loan_amount_term_log'].values[0]
        transformed[3] = input_df['dependents'].values[0]
        
        # Map categorical variables
        # Each categorical variable needs 2 positions (for binary categories)
        start_idx = 4
        for col in categorical_cols:
            val = input_df[col].values[0]
            # Set both positions for each categorical variable
            transformed[start_idx] = 1 if val == 0 else 0  # First category
            transformed[start_idx + 1] = 1 if val == 1 else 0  # Second category
            start_idx += 2  # Move to next pair of positions
        
        return pd.DataFrame([transformed])

def transform_feature_names(input_df):
    """Transform feature names to match the model's expected format."""
    # Create a mapping of current names to expected names
    name_mapping = {
        'applicant_income_log': 'ApplicantIncomelog',
        'credit_history': 'Credit_History',
        'dependents': 'Dependents',
        'education': 'Education',
        'gender': 'Gender',
        'loan_amount_log': 'LoanAmountlog',
        'loan_amount_term_log': 'LoanAmountTermlog',
        'married': 'Married',
        'property_area': 'Property_Area',
        'self_employed': 'Self_Employed'
    }
    
    # Rename columns using the mapping
    return input_df.rename(columns=name_mapping)

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
        <h1 style='color: white;'>Loan Approval Prediction</h1>
        <p style='color: #aaa; margin-bottom: 0;'>Predict your loan approval chances with machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Create a mapping for user-friendly labels
gender_options = {'Female': 0, 'Male': 1}
marital_status_options = {'Single': 0, 'Married': 1}
education_options = {'Graduate': 0, 'Not Graduate': 1}
employment_status_options = {'Unemployed': 0, 'Employed': 1}
credit_history_options = {'No/Bad Credit History': 0, 'Good Credit History': 1}
property_area_options = {'Rural': 0, 'Urban': 1}

# Create two columns for the input form with adjusted ratio
col1, col2 = st.columns([1, 1])

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

# Loan details
st.subheader("Loan Details")
st.write("**Enter Loan Amount:**")
loan_amount_log = st.slider("", 
                           min_value=1000.0, 
                           max_value=500000.0, 
                           value=100000.0, 
                           step=1000.0,
                           format="₱ %d")

st.write("**Enter Loan Amount Term (in Months):**")
loan_amount_term_log = st.slider("", 
                                min_value=1.0, 
                                max_value=160.0, 
                                value=60.0, 
                                step=1.0,
                                format="%d months")

# Calculate monthly payment and debt-to-income ratio
def calculate_monthly_payment(loan_amount, annual_rate, months):
    # Using the loan amortization formula
    monthly_rate = annual_rate / 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    return monthly_payment

# Assuming an annual interest rate of 10% (you can make this configurable if needed)
annual_interest_rate = 0.10

if applicant_income_log and loan_amount_log and loan_amount_term_log:
    monthly_payment = calculate_monthly_payment(loan_amount_log, annual_interest_rate, loan_amount_term_log)
    monthly_income = applicant_income_log
    debt_to_income_ratio = (monthly_payment / monthly_income) * 100
    
    # Display the calculated values
    st.write("**Loan Payment Details:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• Estimated Monthly Payment: ₱{monthly_payment:,.2f}")
    with col2:
        st.write(f"• Debt-to-Income Ratio: {debt_to_income_ratio:.1f}%")
        if debt_to_income_ratio > 43:
            st.warning("⚠️ Debt-to-income ratio is higher than recommended (43%)")
        elif debt_to_income_ratio > 36:
            st.info("ℹ️ Debt-to-income ratio is slightly elevated")
        else:
            st.success("✅ Debt-to-income ratio is within good range")

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
    # Calculate debt-to-income ratio
    monthly_payment = calculate_monthly_payment(loan_amount_log, annual_interest_rate, loan_amount_term_log)
    debt_to_income_ratio = (monthly_payment / applicant_income_log) * 100

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
        'loan_amount_term_log': loan_amount_term_log,
        'debt_to_income_ratio': debt_to_income_ratio
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
    models = {}
    
    if XGB_Model is not None:
        models["XGB"] = XGB_Model
    
    if randomforest_model is not None:
        models["Random Forest"] = randomforest_model

    if not models:
        st.error("No models are available for prediction. Please check the model loading errors above.")
        st.stop()

    for model_name, model in models.items():
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Transform features based on the model type
        try:
            input_df = transform_features_for_models(input_df, model_name)
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
        
