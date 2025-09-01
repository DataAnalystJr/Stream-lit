import os
import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Function to format model names with proper capitalization of abbreviations
def format_model_name(filename):
    """Format model names with proper capitalization of abbreviations like SMOTE, VAES, SOP2, GANS"""
    # Remove .joblib extension and replace underscores with spaces
    name = os.path.splitext(filename)[0].replace('_', ' ')
    
    # Define abbreviations that should be capitalized
    abbreviations = ['SMOTE', 'VAES', 'SOP2', 'GANS', 'RF', 'KNN', 'XGBoost']
    
    # Split the name into words
    words = name.split()
    
    # Process each word
    formatted_words = []
    for word in words:
        # Check if the word (case-insensitive) matches any abbreviation
        word_lower = word.lower()
        matched_abbrev = None
        for abbrev in abbreviations:
            if word_lower == abbrev.lower():
                matched_abbrev = abbrev
                break
        
        if matched_abbrev:
            # Use the exact abbreviation
            formatted_words.append(matched_abbrev)
        else:
            # Capitalize first letter, lowercase the rest
            formatted_words.append(word.capitalize())
    
    return ' '.join(formatted_words)

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'SWIFT', 'Models')

print(f"Testing models from directory: {models_dir}")
print("=" * 60)

# Check if models directory exists
if not os.path.isdir(models_dir):
    print(f"❌ ERROR: Models directory not found at: {models_dir}")
    exit(1)

# List all .joblib files
joblib_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.joblib')]
print(f"Found {len(joblib_files)} .joblib files:")
for f in joblib_files:
    print(f"  - {f}")

print("\n" + "=" * 60)
print("TESTING MODEL LOADING AND FUNCTIONALITY")
print("=" * 60)

# Test data (same structure as the app)
test_data = {
    'Gender': 1,
    'Married': 1,
    'Dependents': 0,
    'Education': 1,
    'Self_Employed': 0,
    'Credit_History': 1,
    'Property_Area': 1,
    'ApplicantIncomeLog': np.log1p(50000),
    'Loan_to_Income_RatioLog': np.log1p(2.0),
    'LoanAmountLog': np.log1p(100000),
    'Monthly_Loan_Amount_TermLog': np.log1p(12)
}

test_df = pd.DataFrame([test_data])

# Test each model
working_models = []
failed_models = []

for filename in joblib_files:
    print(f"\n🔍 Testing: {filename}")
    print("-" * 40)
    
    try:
        # Load the model
        file_path = os.path.join(models_dir, filename)
        obj = joblib.load(file_path)
        print(f"✅ Successfully loaded {filename}")
        
        # Determine the actual estimator to use
        model_candidate = None
        if isinstance(obj, dict):
            if 'model' in obj:
                model_candidate = obj['model']
                print(f"  📦 Found 'model' key in dictionary")
            elif 'best_estimator_' in obj:
                model_candidate = obj['best_estimator_']
                print(f"  📦 Found 'best_estimator_' key in dictionary")
            else:
                # Try to find any estimator-like object in the dict
                for k, v in obj.items():
                    if hasattr(v, 'predict') or hasattr(v, 'best_estimator_'):
                        model_candidate = v
                        print(f"  📦 Found estimator in key '{k}'")
                        break
        else:
            model_candidate = obj
            print(f"  📦 Loaded object directly (not a dictionary)")
        
        # Unwrap GridSearchCV/Pipeline-like objects when fitted
        if hasattr(model_candidate, 'best_estimator_') and getattr(model_candidate, 'best_estimator_', None) is not None:
            model_candidate = model_candidate.best_estimator_
            print(f"  🔄 Unwrapped best_estimator_ from GridSearchCV/Pipeline")
        
        model = model_candidate
        if model is None:
            print(f"❌ Could not identify a model object in {filename}")
            failed_models.append((filename, "Could not identify model object"))
            continue
            
        if not hasattr(model, 'predict'):
            print(f"❌ Loaded object has no predict() method")
            failed_models.append((filename, "No predict() method"))
            continue
        
        # Check if model is fitted
        try:
            check_is_fitted(model)
            print(f"✅ Model is properly fitted")
        except Exception as e:
            print(f"⚠️  Model appears unfitted: {e}")
            # Try heuristic check
            is_unfitted = not any(
                hasattr(model, attr) for attr in ['n_features_in_', 'classes_', 'feature_names_in_']
            )
            if is_unfitted:
                print(f"❌ Model is definitely unfitted - skipping")
                failed_models.append((filename, "Model is unfitted"))
                continue
            else:
                print(f"✅ Model appears to be fitted (heuristic check passed)")
        
        # Test prediction
        try:
            prediction = model.predict(test_df)
            print(f"✅ Predict method works: {prediction}")
        except Exception as e:
            print(f"❌ Predict method failed: {e}")
            failed_models.append((filename, f"Predict failed: {e}"))
            continue
        
        # Test probability prediction if available
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(test_df)
                print(f"✅ Predict_proba works: {proba}")
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(test_df)
                print(f"✅ Decision_function works: {decision}")
            else:
                print(f"ℹ️  No probability method available")
        except Exception as e:
            print(f"⚠️  Probability method failed: {e}")
        
        # Test with different input shapes
        try:
            # Test with single row
            single_row = test_df.iloc[0:1]
            pred_single = model.predict(single_row)
            print(f"✅ Single row prediction works: {pred_single}")
        except Exception as e:
            print(f"⚠️  Single row prediction failed: {e}")
        
        # Check model attributes
        print(f"  📊 Model type: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            print(f"  📊 Features expected: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            print(f"  📊 Classes: {model.classes_}")
        
        working_models.append(filename)
        print(f"✅ {filename} is working correctly!")
        
    except Exception as e:
        print(f"❌ Failed to load/test {filename}: {e}")
        failed_models.append((filename, f"Load/Test failed: {e}"))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"✅ Working models: {len(working_models)}")
for model in working_models:
    formatted_name = format_model_name(model)
    print(f"  - {formatted_name} (from {model})")

print(f"\n❌ Failed models: {len(failed_models)}")
for model, error in failed_models:
    formatted_name = format_model_name(model)
    print(f"  - {formatted_name} (from {model}): {error}")

print(f"\n📊 Success rate: {len(working_models)}/{len(joblib_files)} ({len(working_models)/len(joblib_files)*100:.1f}%)")

if failed_models:
    print(f"\n🔧 RECOMMENDATIONS:")
    print(f"1. Check if failed models are properly trained and saved")
    print(f"2. Ensure models are saved with joblib.dump() after training")
    print(f"3. For GridSearchCV models, save the best_estimator_ attribute")
    print(f"4. Verify the input data structure matches what the models expect")

print("\n" + "=" * 60) 