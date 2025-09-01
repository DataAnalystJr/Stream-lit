import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def create_test_model():
    """Create a simple test model to verify the setup works"""
    print("🔧 Creating test model...")
    
    # Create synthetic data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save the model
    test_model_path = "SWIFT/Models/test_model.joblib"
    joblib.dump(model, test_model_path)
    
    print(f"✅ Test model created and saved to: {test_model_path}")
    
    # Test loading it back
    try:
        loaded_model = joblib.load(test_model_path)
        prediction = loaded_model.predict(X[:1])
        print(f"✅ Test model loads and predicts successfully: {prediction}")
        return True
    except Exception as e:
        print(f"❌ Failed to load test model: {e}")
        return False

if __name__ == "__main__":
    create_test_model()
