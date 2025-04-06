import joblib
import os
import sys

# Get the current directory
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'SWIFT', 'Models', 'KNN.pkl')

print(f"Current directory: {current_dir}")
print(f"Model path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")

try:
    # Load the model
    print("\nAttempting to load model...")
    knn_model = joblib.load(model_path)
    
    # Print model information
    print("\nKNN Model Information:")
    print("-" * 50)
    print(f"Model type: {type(knn_model)}")
    print(f"Number of neighbors: {getattr(knn_model, 'n_neighbors', 'N/A')}")
    print(f"Weights: {getattr(knn_model, 'weights', 'N/A')}")
    print(f"Algorithm: {getattr(knn_model, 'algorithm', 'N/A')}")
    print(f"Leaf size: {getattr(knn_model, 'leaf_size', 'N/A')}")
    print(f"Metric: {getattr(knn_model, 'metric', 'N/A')}")
    print(f"Metric params: {getattr(knn_model, 'metric_params', 'N/A')}")
    print(f"Number of features seen during fit: {getattr(knn_model, 'n_features_in_', 'N/A')}")
    print(f"Number of samples seen during fit: {getattr(knn_model, 'n_samples_fit_', 'N/A')}")
    
except Exception as e:
    print(f"\nError loading model: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc() 