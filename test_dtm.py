import joblib
import os
import pandas as pd
import numpy as np

# Get the current directory
current_dir = os.path.dirname(__file__)

# Path to the Decision Tree model
dtm_path = os.path.join(current_dir, 'SWIFT', 'Models', 'DTM.pkl')

try:
    # Try to load the model
    print("Attempting to load DTM model...")
    dtm_model = joblib.load(dtm_path)
    print("✅ DTM model loaded successfully!")
    
    # Create sample input with our current feature names
    sample_input = {
        'gender': 1,  # lowercase
        'married': 1,  # lowercase
        'dependents': 2,  # lowercase
        'education': 0,  # lowercase
        'self_employed': 0,  # lowercase
        'credit_history': 1,  # lowercase
        'property_area': 1,  # lowercase
        'applicant_income_log': 50000,  # lowercase
        'loan_amount_log': 100000,  # lowercase
        'loan_amount_term_log': 360  # lowercase
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([sample_input])
    
    # Print the columns we're trying to use
    print("\nOur input features:")
    print(input_df.columns.tolist())
    
    # Try to make a prediction
    print("\nAttempting to make a prediction...")
    try:
        prediction = dtm_model.predict(input_df)
        print("✅ Prediction successful!")
        print("Prediction value:", prediction[0])
        
        probability = dtm_model.predict_proba(input_df)
        print("Probability:", probability[0])
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
    
    # Check model features
    print("\nModel features:")
    if hasattr(dtm_model, 'feature_names_in_'):
        print("Feature names:", dtm_model.feature_names_in_)
    else:
        print("Model doesn't have feature names stored")
    
    # Check number of features expected
    print("\nNumber of features expected:", dtm_model.n_features_in_)
    
except Exception as e:
    print(f"❌ Error: {str(e)}") 