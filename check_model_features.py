import joblib
import os
import pandas as pd
import numpy as np

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the path for the decision tree SMOTE model
model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'decision_tree_smote_model.joblib')

# Load the model
model = joblib.load(model_path)

# Create a sample input with all possible features
sample_input = {
    'Gender': 1,
    'Married': 1,
    'Dependents': 2,
    'Education': 1,
    'Self_Employed': 0,
    'ApplicantIncome': 5000,
    'LoanAmount': 100000,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 1,
    'ApplicantIncome_log': np.log1p(5000),
    'LoanAmount_log': np.log1p(100000),
    'Loan_Amount_Term_log': np.log1p(360),
    'ApplicantIncome_log_2': np.log1p(5000) ** 2,
    'LoanAmount_log_2': np.log1p(100000) ** 2,
    'Loan_Amount_Term_log_2': np.log1p(360) ** 2,
    'EMI': 100000/360,
    'EMI_log': np.log1p(100000/360),
    'Balance_Income': 5000 - (100000/360),
    'Balance_Income_log': np.log1p(5000 - (100000/360))
}

# Try different feature combinations
feature_sets = [
    # Basic features only
    ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
     'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
     'Property_Area'],
    
    # Basic + log features
    ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
     'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
     'Property_Area', 'ApplicantIncome_log', 'LoanAmount_log', 'Loan_Amount_Term_log'],
    
    # Basic + log + squared log features
    ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
     'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
     'Property_Area', 'ApplicantIncome_log', 'LoanAmount_log', 'Loan_Amount_Term_log',
     'ApplicantIncome_log_2', 'LoanAmount_log_2', 'Loan_Amount_Term_log_2'],
    
    # All features
    list(sample_input.keys())
]

print("Testing different feature combinations...")
for i, features in enumerate(feature_sets):
    try:
        # Create DataFrame with selected features
        df = pd.DataFrame({f: [sample_input[f]] for f in features})
        # Try to make prediction
        model.predict(df)
        print(f"\nSuccess with {len(features)} features:")
        for f in features:
            print(f"- {f}")
    except Exception as e:
        print(f"\nFailed with {len(features)} features: {str(e)}") 