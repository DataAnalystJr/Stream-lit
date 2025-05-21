import joblib
import os
import numpy as np
import pandas as pd

def test_random_forest_features():
    # Get the current directory
    current_dir = os.path.dirname(__file__)
    
    # Load the model
    model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'random_forest_noise.joblib')
    model = joblib.load(model_path)
    
    # Print model type
    print("\nModel Type:", type(model).__name__)
    
    # Print number of trees
    print("\nNumber of trees:", model.n_estimators)
    
    # Print feature importances
    print("\nFeature Importances:")
    feature_names = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'Credit_History', 'Property_Area', 'ApplicantIncomeLog',
        'Loan_to_Income_RatioLog', 'LoanAmountLog', 'Monthly_Loan_Amount_TermLog'
    ]
    
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")
    
    # Test model with sample data
    print("\nTesting model with sample data:")
    sample_data = {
        'Gender': 1,
        'Married': 1,
        'Dependents': 2,
        'Education': 1,
        'Self_Employed': 0,
        'Credit_History': 1,
        'Property_Area': 1,
        'ApplicantIncomeLog': np.log1p(5000),
        'Loan_to_Income_RatioLog': np.log1p(0.5),
        'LoanAmountLog': np.log1p(100000),
        'Monthly_Loan_Amount_TermLog': np.log1p(360/12)
    }
    
    # Convert to DataFrame with only the required features in the correct order
    sample_df = pd.DataFrame([sample_data])[feature_names]
    
    # Make prediction
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0]
    
    print(f"\nPrediction (0=Default, 1=Repay): {prediction}")
    print(f"Probability of repayment: {probability[1]:.4f}")
    
    # Print model parameters
    print("\nModel Parameters:")
    for param, value in model.get_params().items():
        print(f"{param}: {value}")

if __name__ == "__main__":
    test_random_forest_features() 