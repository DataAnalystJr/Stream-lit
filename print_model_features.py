import joblib

model = joblib.load('SWIFT/Models/decision_tree_smote_model.joblib')
print("Model expects these features (in order):")
print(getattr(model, 'feature_names_in_', 'No feature_names_in_ attribute found')) 