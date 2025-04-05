import os
import joblib

# Get the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(current_dir, 'SWIFT', 'Models')
xgboost_path = os.path.join(model_dir, 'KGB.pkl')

print(f"Current directory: {current_dir}")
print(f"Model directory: {model_dir}")
print(f"XGBoost model path: {xgboost_path}")
print(f"File exists: {os.path.exists(xgboost_path)}")

try:
    model = joblib.load(xgboost_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}") 