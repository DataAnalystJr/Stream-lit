import joblib
import pandas as pd

# Load the model
model = joblib.load('SWIFT/Models/decision_tree_smote_model.joblib')

# Print feature names and count
print("Number of features expected:", len(model.feature_names_in_))
print("\nModel expects these features in order:")
for i, feature in enumerate(model.feature_names_in_):
    print(f"{i+1}. {feature}")

# Create a sample input with all zeros to test
sample_input = pd.DataFrame({feature: [0] for feature in model.feature_names_in_})

# Try to predict with sample input
try:
    prediction = model.predict(sample_input)
    print("\nPrediction successful with sample input")
except Exception as e:
    print("\nError with sample input:")
    print(e) 