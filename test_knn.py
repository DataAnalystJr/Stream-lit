import joblib
import os

# Get the current directory
current_dir = os.path.dirname(__file__)

# Path to the KNN model
knn_model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'KNN.pkl')

try:
    # Try to load the model
    print("Attempting to load KNN model...")
    knn_model = joblib.load(knn_model_path)
    print("✅ KNN model loaded successfully!")
    
    # Check if the model has the required methods
    print("\nChecking model attributes:")
    print(f"Has predict method: {'predict' in dir(knn_model)}")
    print(f"Has predict_proba method: {'predict_proba' in dir(knn_model)}")
    
    # Try to get model parameters
    print("\nModel parameters:")
    print(knn_model.get_params())
    
except Exception as e:
    print(f"❌ Error loading KNN model: {str(e)}") 