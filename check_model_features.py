import joblib
import pandas as pd

# Load the model
model = joblib.load('SWIFT/Models/decision_tree_smote_model.joblib')

# Print feature names
print("Model expects these features in order:")
print(model.feature_names_in_)

# Create a sample input with all zeros to test
sample_input = pd.DataFrame({
    'Gender': [0],
    'Married': [0],
    'Dependents': [0],
    'Education': [0],
    'Self_Employed': [0],
    'ApplicantIncome': [0],
    'LoanAmount': [0],
    'Loan_Amount_Term': [0],
    'Credit_History': [0],
    'Property_Area': [0]
})

# Try to predict with sample input
try:
    prediction = model.predict(sample_input)
    print("\nPrediction successful with sample input")
except Exception as e:
    print("\nError with sample input:")
    print(e) 