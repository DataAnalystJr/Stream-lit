import joblib
import os
import numpy as np
import pandas as pd

# Get the current directory
current_dir = os.path.dirname(__file__)

# Construct the path to XGB model
xgb_path = os.path.join(current_dir, 'SWIFT', 'Models', 'XGB.pkl')

try:
    # Load the model
    print("Loading XGBoost model...")
    xgb_model = joblib.load(xgb_path)
    
    # Print model information
    print("\nXGBoost Model Information:")
    print("-" * 50)
    print(f"Number of features expected: {xgb_model.n_features_in_}")
    
    # Get feature names and importances
    feature_names = xgb_model.feature_names_in_
    importances = xgb_model.feature_importances_
    
    # Create DataFrame and sort by importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Print top features
    print("\nTop 102 features by importance:")
    print("-" * 50)
    pd.set_option('display.max_rows', None)
    print(feature_importance_df.to_string(index=False))
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 