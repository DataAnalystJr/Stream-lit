import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Set the page configuration to wide layout
st.set_page_config(layout="wide")

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'SWIFT', 'Models')

# Check if models directory exists before loading
if not os.path.isdir(models_dir):
    st.error(f"Models directory not found at: {models_dir}")
    st.stop()

# Function to format model names with proper capitalization of abbreviations
def format_model_name(filename):
    """Format model names with proper capitalization of abbreviations like SMOTE, VAES, SOP2, GANS"""
    # Remove .joblib extension and replace underscores with spaces
    name = os.path.splitext(filename)[0].replace('_', ' ')
    
    # Define abbreviations that should be capitalized
    abbreviations = ['SMOTE', 'VAES', 'SOP2', 'GANS', 'RF', 'KNN', 'XGBoost']
    
    # Split the name into words
    words = name.split()
    
    # Process each word
    formatted_words = []
    for word in words:
        # Check if the word (case-insensitive) matches any abbreviation
        word_lower = word.lower()
        matched_abbrev = None
        for abbrev in abbreviations:
            if word_lower == abbrev.lower():
                matched_abbrev = abbrev
                break
        
        if matched_abbrev:
            # Use the exact abbreviation
            formatted_words.append(matched_abbrev)
        else:
            # Capitalize first letter, lowercase the rest
            formatted_words.append(word.capitalize())
    
    return ' '.join(formatted_words)

# Load all models from the models directory
def load_models_from_dir(directory_path):
    loaded_models = {}
    st.info(f"Looking for models in: {directory_path}")
    
    # List all files in directory
    try:
        all_files = os.listdir(directory_path)
        st.info(f"Files found in directory: {all_files}")
    except Exception as e:
        st.error(f"Error listing directory: {str(e)}")
        return loaded_models
    
    for filename in all_files:
        if not filename.lower().endswith('.joblib'):
            continue
        file_path = os.path.join(directory_path, filename)
        st.info(f"Trying to load: {filename}")
        
        try:
            obj = joblib.load(file_path)
            st.success(f"Successfully loaded: {filename}")
            
            # Extract model from object
            model = obj
            if isinstance(obj, dict):
                if 'model' in obj:
                    model = obj['model']
                elif 'best_estimator_' in obj:
                    model = obj['best_estimator_']
            
            # Check if model is valid
            if not hasattr(model, 'predict'):
                st.warning(f"Model {filename} has no predict method")
                continue
                
            # Check if model is fitted
            try:
                check_is_fitted(model)
                st.success(f"Model {filename} is fitted and ready")
            except Exception as e:
                st.warning(f"Model {filename} is not fitted: {str(e)}")
                continue

            # Create display name and store model
            name = format_model_name(filename)
            loaded_models[name] = model
            st.success(f"Added model: {name}")
            
        except Exception as e:
            st.error(f"Failed to load {filename}: {str(e)}")
            # Log the specific error for debugging
            st.error(f"Error details: {type(e).__name__}: {str(e)}")
            continue
    
    st.info(f"Total models loaded: {len(loaded_models)}")
    return loaded_models

models_dict = load_models_from_dir(models_dir)
if not models_dict:
    st.error("No valid models could be loaded from the models directory.")
    st.stop()

# Helper to find Random Forest models and create proper display names
def get_model_display_name(model_name):
    """Get the proper display name for a model, handling Random Forest variants"""
    lname = model_name.lower()
    
    # Check if it's a Random Forest variant
    if 'random forest' in lname:
        # Handle specific Random Forest variants
        if 'smote' in lname:
            return "Random Forest SMOTE"
        elif 'vaes' in lname:
            return "Random Forest VAES"
        elif 'sop2' in lname:
            return "Random Forest (Recommended)"
        elif 'gans' in lname:
            return "Random Forest GANS"
        else:
            return "Random Forest"
    elif lname.startswith('rf') or 'rf' in lname:
        return "Random Forest"
    else:
        return model_name

# No need for random_forest_display_name variable anymore

# Function to get positive-class probability
def get_positive_probability(model, input_df):
    try:
        proba = model.predict_proba(input_df)
        return float(proba[0][1])
    except Exception:
        try:
            decision = model.decision_function(input_df)
            score = float(np.atleast_1d(decision)[0])
            return 1.0 / (1.0 + np.exp(-score))
        except Exception:
            return 0.5

# Post-prediction probability adjustments
def adjust_probability(probability, education, property_area, credit_history):
    prob = float(probability)
    
    # Education adjustments
    if education == "College Graduate":
        prob = min(prob + 0.10, 1.0)
    elif education == "Elementary Graduate":
        prob = max(prob - 0.15, 0.0)
    
    # Property area adjustments
    if property_area == "Urban":
        prob = min(prob + 0.08, 1.0)
    
    # Credit history adjustments
    if credit_history == "Good Credit History":
        prob = min(prob + 0.08, 1.0)
    
    return prob

# Plot probability chart
def render_probability_chart(probability):
    plt.figure(figsize=(14, 7))
    colors = ['#4CAF50', '#FF5252']
    bars = plt.bar(['Yes', 'No'], [probability, 1 - probability], color=colors, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1%}',
                 ha='center', va='bottom', fontsize=16, fontweight='medium')
    
    plt.gca().spines[:].set_visible(False)
    plt.yticks([])
    plt.xticks(fontsize=16)
    plt.title('Loan Repayment Probability', fontsize=18, fontweight='medium', pad=30)
    plt.tight_layout()
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
education_options = {'Elementary Graduate': 0, 'High School Graduate': 1, 'College Graduate': 2}
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
    st.rerun()

    
# Action buttons
col1, col2 = st.columns(2)

with col1:
    # Submit button
    if st.button("Submit", disabled=not is_valid_input()):
        # Calculate derived features
        loan_to_income_ratio = float(loan_amount) / float(applicant_income) if float(applicant_income) > 0 else 0
        
        # Prepare input data for prediction with exactly the columns the model was trained on
        # Map education to original binary values: Elementary=0, High School=0, College=1
        education_mapping = {'Elementary Graduate': 0, 'High School Graduate': 0, 'College Graduate': 1}
        
        input_data = {
            'Gender': gender_options[gender],
            'Married': marital_status_options[married],
            'Dependents': dependents_options[dependents],
            'Education': education_mapping[education],  # Use the mapping for model compatibility
            'Self_Employed': employment_status_options[self_employed],
            'Credit_History': credit_history_options[credit_history],
            'Property_Area': property_area_options[property_area],
            'ApplicantIncomeLog': np.log1p(float(applicant_income)),
            'Loan_to_Income_RatioLog': np.log1p(loan_to_income_ratio),
            'LoanAmountLog': np.log1p(float(loan_amount)),
            'Monthly_Loan_Amount_TermLog': np.log1p(float(loan_term) / 12)
        }

        try:
            input_df = pd.DataFrame([input_data])

            # Display input summary
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Income:** ₱{int(applicant_income):,}")
                st.write(f"**Loan:** {int(loan_amount):,}")
                st.write(f"**Term:** {loan_term} months")
            with col2:
                st.write(f"**Education:** {education}")
                st.write(f"**Credit:** {credit_history}")
                st.write(f"**Area:** {property_area}")

            # Add explanation for the results (only once)
            st.info("""
            **Why these results?**  
            These predictions are based on the applicant's income, loan amount, number of dependents, and credit history, as these are key factors used by the models to assess loan repayment likelihood.
            """)

            # Model predictions
            st.info("**Model Predictions**")
            
            def display_model_prediction(model_name, model, column):
                with column:
                    try:
                        display_name = get_model_display_name(model_name)
                        st.subheader(f"{display_name} Prediction")

                        probability = get_positive_probability(model, input_df)
                        probability = adjust_probability(probability, education, property_area, credit_history)
                        prediction = 1 if probability >= 0.5 else 0

                        if prediction == 1:
                            st.success(f"✅ **APPROVED** - {probability:.1%}")
                        else:
                            st.error(f"❌ **REJECTED** - {1 - probability:.1%}")
                        
                        st.markdown(f"**Repayment:** {probability:.0%} | **Default:** {1 - probability:.0%}")
                        render_probability_chart(probability)
                        
                    except Exception:
                        st.warning(f"Error with {model_name}")
            
            # Custom sorting for specific model order
            def custom_model_sort_key(model_name):
                lname = model_name.lower()
                
                # Row 1: Decision Tree and Gradient Boosting
                if 'decision tree' in lname:
                    return 1
                elif 'gradient boosting' in lname:
                    return 2
                
                # Row 2: KNN and Logistic Regression
                elif 'knn' in lname:
                    return 3
                elif 'logistic regression' in lname:
                    return 4
                
                # Row 3: XGBoost and Random Forest (Recommended)
                elif 'xgboost' in lname:
                    return 5
                elif 'sop2' in lname:
                    return 6
                
                # Row 4: VAES, SMOTE, and GANS
                elif 'vaes' in lname:
                    return 7
                elif 'smote' in lname:
                    return 8
                elif 'gans' in lname:
                    return 9
                
                # Any other models
                else:
                    return 0
            
            # Display models in custom order
            model_names = sorted(list(models_dict.keys()), key=custom_model_sort_key)
            for i in range(0, len(model_names), 2):
                col1, col2 = st.columns(2)
                display_model_prediction(model_names[i], models_dict[model_names[i]], col1)
                if i + 1 < len(model_names):
                    display_model_prediction(model_names[i + 1], models_dict[model_names[i + 1]], col2)
                if i + 2 < len(model_names):
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Input data structure:")
            st.write(input_data)

with col2:
    if st.button("Clear"):
        clear_fields()
        