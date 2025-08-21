import subprocess
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# Optional scikit-learn utilities for fitted checks
try:
    from sklearn.utils.validation import check_is_fitted
    from sklearn.model_selection import GridSearchCV
except Exception:
    check_is_fitted = None
    GridSearchCV = None

# Set the page configuration to wide layout
st.set_page_config(layout="wide")

# Initialize session state variables
if 'clear_triggered' not in st.session_state:
    st.session_state.clear_triggered = False

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))


models_dir = os.path.join(current_dir, 'SWIFT', 'Models')

# Print the path for debugging
print(f"Looking for models in: {models_dir}")
print(f"Current directory: {current_dir}")

# Check if models directory exists before loading
if not os.path.isdir(models_dir):
    st.error(f"Models directory not found at: {models_dir}")
    st.stop()

# Load all models from the models directory
def load_models_from_dir(directory_path):
    loaded_models = {}
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith('.joblib'):
            continue
        file_path = os.path.join(directory_path, filename)
        try:
            obj = joblib.load(file_path)

            # Determine the actual estimator to use
            model_candidate = None
            if isinstance(obj, dict):
                if 'model' in obj:
                    model_candidate = obj['model']
                elif 'best_estimator_' in obj:
                    model_candidate = obj['best_estimator_']
                else:
                    # Try to find any estimator-like object in the dict
                    for k, v in obj.items():
                        if hasattr(v, 'predict') or hasattr(v, 'best_estimator_'):
                            model_candidate = v
                            break
            else:
                model_candidate = obj

            # Unwrap GridSearchCV/Pipeline-like objects when fitted
            if hasattr(model_candidate, 'best_estimator_') and getattr(model_candidate, 'best_estimator_', None) is not None:
                model_candidate = model_candidate.best_estimator_

            model = model_candidate
            if model is None:
                print(f"Skipped {filename}: could not identify a model object")
                continue
            if not hasattr(model, 'predict'):
                print(f"Skipped {filename}: loaded object has no predict()")
                continue

            # Robust fitted check: prefer sklearn's check_is_fitted, fallback to heuristics
            is_unfitted = False
            if check_is_fitted is not None:
                try:
                    check_is_fitted(model)
                except Exception:
                    is_unfitted = True
            else:
                # Heuristic fallback
                is_unfitted = not any(
                    hasattr(model, attr) for attr in ['n_features_in_', 'classes_', 'feature_names_in_']
                )

            if is_unfitted:
                print(f"Skipped {filename}: estimator appears unfitted (e.g., GridSearchCV not yet fit). Re-save a fitted estimator (best_estimator_).")
                continue

            # Create a friendly display name
            name = os.path.splitext(filename)[0].replace('_', ' ').title()
            loaded_models[name] = model
            print(f"Loaded model: {name} from {filename}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return loaded_models

models_dict = load_models_from_dir(models_dir)
if not models_dict:
    st.error("No valid models could be loaded from the models directory.")
    st.stop()

# Helper to find a likely Random Forest model name for preserving the original heading
def find_random_forest_name(models):
    for name in models.keys():
        lname = name.lower()
        if 'random forest' in lname or lname.startswith('rf') or 'rf' in lname:
            return name
    return None

random_forest_display_name = find_random_forest_name(models_dict)

# Function to robustly get positive-class probability
def get_positive_probability(model, input_df):
    try:
        proba = model.predict_proba(input_df)
        # Binary classifier: take class 1 probability
        if proba is not None:
            return float(proba[0][1])
    except Exception:
        pass
    try:
        # Some models expose decision_function; squash via sigmoid
        decision = model.decision_function(input_df)
        score = float(np.atleast_1d(decision)[0])
        return 1.0 / (1.0 + np.exp(-score))
    except Exception:
        pass
    try:
        pred = model.predict(input_df)
        return float(np.atleast_1d(pred)[0])
    except Exception:
        return 0.0

# Post-prediction probability adjustments (same rules as existing code)
def adjust_probability(probability, education, property_area, credit_history):
    prob = float(probability)
    # Education
    if education == "College Graduate":
        prob = min(prob + 0.10, 1.0)
    elif education == "High School Graduate":
        prob = max(prob - 0.05, 0.0)
    # Property area
    if property_area == "Urban":
        prob = min(prob + 0.08, 1.0)
    elif property_area == "Rural":
        prob = max(prob - 0.05, 0.0)
    # Credit history
    if credit_history == "Good Credit History":
        prob = min(prob + 0.08, 1.0)
    elif credit_history == "Bad Credit History":
        prob = max(prob - 0.05, 0.0)
    return prob

# Unified plotting for probability (identical look and feel)
def render_probability_chart(probability):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    colors = ['#4CAF50', '#FF5252']
    bars = plt.bar(['Repayment', 'Default'],
                   [probability, 1 - probability],
                   color=colors,
                   alpha=0.8,
                   width=0.9)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom',
                 fontsize=16,
                 fontweight='medium',
                 color='#2C3E50')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.yticks([])
    plt.ylabel('')
    plt.xticks(fontsize=16, color='#2C3E50')
    plt.grid(axis='y', linestyle='-', alpha=0.1, color='#2C3E50')
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.title('Loan Repayment Probability',
              fontsize=18,
              fontweight='medium',
              color='#2C3E50',
              pad=30)
    plt.tight_layout(pad=2.5)
    st.pyplot(plt, use_container_width=True)
    plt.clf()

# Center the title using HTML
# Center the title with a border using HTML and CSS
st.markdown("""
    <div style='text-align: center; border: 2px solid white; padding: 10px; border-radius: 15px; background-color: #222;'>
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
    applicant_income = st.number_input("Enter Applicant Income (Monthly):", 
                                     min_value=5000.0, 
                                     value=None,
                                     help="Enter your monthly income before any deductions")
    if applicant_income is not None and applicant_income != 0:
        st.markdown(f"<span style='color: #2C3E50;'>Formatted: <b>{int(applicant_income):,}</b></span>", unsafe_allow_html=True)

    loan_amount = st.number_input("Enter Loan Amount:", min_value=50000.0, max_value=500000.0, value=None, help="Enter loan amount (maximum: ₱500,000)")
    if loan_amount is not None and loan_amount != 0:
        st.markdown(f"<span style='color: #2C3E50;'>Formatted: <b>{int(loan_amount):,}</b></span>", unsafe_allow_html=True)
    loan_term = st.slider("Select Monthly Loan Term (Months):", min_value=1, max_value=120, value=12, help="Select the loan term in months (1-120)")
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
        st.write(f"Applicant Income: ₱{int(applicant_income):,}")
        st.write(f"Loan Amount: {int(loan_amount):,}")
        st.write(f"Monthly Loan Term: {loan_term}")
        st.write(f"Credit History: {credit_history}")
        st.write(f"Property Area: {property_area}")
        st.write(f"Loan to Income Ratio: {loan_to_income_ratio:.2f}")

        try:
            input_df = pd.DataFrame([input_data])

            # Iterate over all loaded models and render identical outputs
            for model_name, model in models_dict.items():
                try:
                    # Preserve original RF heading if applicable
                    if random_forest_display_name and model_name == random_forest_display_name:
                        st.title("Random Forest Model Prediction")
                    else:
                        st.title(f"{model_name} Prediction")

                    probability = get_positive_probability(model, input_df)
                    probability = adjust_probability(probability, education, property_area, credit_history)
                    prediction = int(model.predict(input_df)[0]) if hasattr(model, 'predict') else (1 if probability >= 0.5 else 0)

                    if prediction == 1:
                        st.write(f"The applicant is likely to pay the loan. (Probability: {probability:.2f})")
                    else:
                        st.write(f"The applicant is unlikely to pay the loan. (Probability: {1 - probability:.2f})")

                    st.info("""
                    **Why this result?**  
                    This prediction is based on the applicant's income, loan amount, number of dependents, and credit history, as these are key factors used by the model to assess loan repayment likelihood.
                    """)

                    # Percentages explanation under each model
                    yes_percent = probability * 100
                    no_percent = (1 - probability) * 100
                    st.markdown(f"""
                    <span style='color:#2C3E50;'>
                    <b>What do these percentages mean?</b><br>
                    The model estimates there is a <b>{yes_percent:.0f}%</b> chance the applicant will repay the loan, and a <b>{no_percent:.0f}%</b> chance they will not.
                    </span>
                    """, unsafe_allow_html=True)

                    # Plot
                    render_probability_chart(probability)
                except Exception as model_err:
                    st.warning(f"Skipping {model_name} due to error: {model_err}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Input data structure:")
            st.write(input_data)

with col2:
    if st.button("Clear"):
        clear_fields()
        