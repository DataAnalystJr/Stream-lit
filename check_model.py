import joblib
import pandas as pd
import os

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the path for the decision tree SMOTE model
model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'dtree_sop2.joblib')

try:
    # Load the model
    model = joblib.load(model_path)
    
    print("Model loaded successfully!")
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    
    # Try to access feature names if available
    if hasattr(model, 'feature_names_in_'):
        print("\nFeature names:")
        print(model.feature_names_in_)
    
    # Try to access feature importances if available
    if hasattr(model, 'feature_importances_'):
        print("\nFeature importances:")
        for feature, importance in zip(model.feature_names_in_, model.feature_importances_):
            print(f"{feature}: {importance:.4f}")
    
    # Try to access the tree structure if it's a decision tree
    if hasattr(model, 'tree_'):
        print("\nTree structure:")
        print(f"Number of nodes: {model.tree_.node_count}")
        print(f"Number of leaves: {model.tree_.n_leaves}")
    
    # If the model was saved with training data, try to access it
    if hasattr(model, 'X_train_'):
        print("\nTraining data shape:", model.X_train_.shape)
        print("\nFirst few rows of training data:")
        print(pd.DataFrame(model.X_train_, columns=model.feature_names_in_).head())
    
except Exception as e:
    print(f"Error loading or inspecting model: {str(e)}")
