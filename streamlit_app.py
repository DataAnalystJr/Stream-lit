import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load the model
model_path = os.path.join('SWIFT', 'Models', 'decision_tree_smote_model.joblib')
dt_model = load(model_path)

# Define options for categorical variables
gender_options = {"Male": 1, "Female": 0}
marital_status_options = {"Married": 1, "Single": 0}
education_options = {"College Graduate": 1, "High School Graduate": 0}
employment_status_options = {"Yes": 1, "No": 0}
credit_history_options = {"Yes": 1, "No": 0}
property_area_options = {"Urban": 1, "Rural": 0}

# Feature transformation function
def transform_features_for_models(input_data):
    # Initialize array with 20 features
    features = np.zeros(20)
    
    # Continuous variables with log transformation
    features[0] = np.log1p(input_data['ApplicantIncome'])
    features[1] = np.log1p(input_data['LoanAmount'])
    features[2] = np.log1p(input_data['Loan_Amount_Term'])
    
    # Categorical variables
    features[3] = input_data['Gender']
    features[4] = input_data['Married']
    features[5] = input_data['Dependents']
    features[6] = input_data['Education']
    features[7] = input_data['Self_Employed']
    features[8] = input_data['Credit_History']
    features[9] = input_data['Property_Area']
    
    # Derived features
    features[10] = input_data['LoanAmount'] / input_data['ApplicantIncome']  # Loan-to-income ratio
    features[11] = input_data['LoanAmount'] / input_data['Loan_Amount_Term']  # Monthly payment
    features[12] = input_data['ApplicantIncome'] / 12  # Income per month
    
    # Interaction features
    features[13] = input_data['ApplicantIncome'] * input_data['LoanAmount']
    features[14] = input_data['ApplicantIncome'] * input_data['Loan_Amount_Term']
    features[15] = input_data['LoanAmount'] * input_data['Loan_Amount_Term']
    
    # Polynomial features
    features[16] = input_data['ApplicantIncome'] ** 2
    features[17] = input_data['LoanAmount'] ** 2
    features[18] = input_data['Loan_Amount_Term'] ** 2
    
    # DTI ratio
    features[19] = (features[11] / features[12]) * 100  # DTI as percentage
    
    return features.reshape(1, -1)

# Prediction function
def make_prediction(features):
    return dt_model.predict(features)[0], dt_model.predict_proba(features)[0][1]

# Main function
def main():
    st.title("Loan Approval Prediction")
    
    st.subheader("Enter Loan Application Details")
    
    # Create three columns for the input form
    col1, col2, col3 = st.columns(3)

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
        applicant_income = st.number_input("Monthly Income:", min_value=0.0, value=None, step=1000.0)

    # Loan details
    st.subheader("Loan Details")
    st.write("**Loan Amount:**")
    loan_amount = st.number_input("", 
                           min_value=1000.0, 
                           max_value=1000000.0, 
                           value=100000.0,
                           format="%f")

    st.write("**Loan Term (in Months):**")
    loan_amount_term = st.slider("", 
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
            applicant_income is not None and applicant_income > 0, 
            loan_amount is not None and loan_amount > 0, 
            loan_amount_term is not None and loan_amount_term > 0
        ])

    # Function to reset all fields
    def clear_fields():
        # Reset all session state variables to default values
        for key in ["gender", "married", "dependents", "education", "self_employed", 
                    "credit_history", "property_area", "applicant_income", 
                    "loan_amount", "loan_amount_term"]:
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
            'Gender': gender_options[Gender],
            'Married': marital_status_options[Married],
            'Dependents': Dependents,
            'Education': education_options[Education],
            'Self_Employed': employment_status_options[self_employed],
            'Credit_History': credit_history_options[credit_history],
            'Property_Area': property_area_options[property_area],
            'ApplicantIncome': applicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term
        }

        # Calculate DTI ratio
        monthly_income = applicant_income
        monthly_loan_payment = loan_amount / loan_amount_term
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
            st.write(f"• Loan Amount: ₱{loan_amount:,.2f}")
            st.write(f"• Loan Term: {loan_amount_term} months")
            st.write(f"• Monthly Payment: ₱{monthly_loan_payment:,.2f}")
            st.write(f"• Debt-to-Income Ratio: {dti_ratio:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Transform features for prediction
        features = transform_features_for_models(input_data)

        # Make prediction
        prediction, probability = make_prediction(features)

        # Result card for prediction
        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        
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
            ax.set_title('Decision Tree Prediction', pad=20, color='#2C3E50')
            
            # Add percentage labels with more vertical spacing
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2. - 0.05, height + 0.05,
                        f'{height:.1%}', ha='center', va='bottom', color='#2C3E50')
                
            st.pyplot(fig)
            plt.close(fig)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()