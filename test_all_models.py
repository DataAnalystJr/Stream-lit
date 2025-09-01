
import os
import joblib
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

def test_model_loading():
    """Test loading all models to ensure they work with current numpy version"""
    print("🧪 Testing Model Loading...")
    print(f"Current numpy version: {np.__version__}")
    print(f"Current joblib version: {joblib.__version__}")
    print("-" * 50)
    
    # Get models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'SWIFT', 'Models')
    
    if not os.path.isdir(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    print(f"📁 Models directory: {models_dir}")
    print("-" * 50)
    
    # Test each model
    success_count = 0
    total_count = 0
    
    for filename in os.listdir(models_dir):
        if not filename.lower().endswith('.joblib'):
            continue
            
        total_count += 1
        file_path = os.path.join(models_dir, filename)
        print(f"🔍 Testing: {filename}")
        
        try:
            # Load the model
            obj = joblib.load(file_path)
            print(f"  ✅ Loaded successfully")
            
            # Extract model from object
            model = obj
            if isinstance(obj, dict):
                if 'model' in obj:
                    model = obj['model']
                elif 'best_estimator_' in obj:
                    model = obj['best_estimator_']
            
            # Check if model is valid
            if not hasattr(model, 'predict'):
                print(f"  ⚠️  No predict method")
                continue
                
            # Check if model is fitted
            try:
                check_is_fitted(model)
                print(f"  ✅ Model is fitted and ready")
                success_count += 1
            except Exception as e:
                print(f"  ⚠️  Model not fitted: {str(e)}")
                
        except Exception as e:
            print(f"  ❌ Failed to load: {str(e)}")
    
    print("-" * 50)
    print(f"📊 Results: {success_count}/{total_count} models loaded successfully")
    
    if success_count == total_count:
        print("🎉 All models working perfectly!")
    else:
        print("⚠️  Some models have issues")
    
    return success_count == total_count

if __name__ == "__main__":
    test_model_loading() 