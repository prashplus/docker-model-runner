"""
Example: Creating and Adding Custom Models
This script demonstrates how to train a custom model and prepare it for the Docker Model Runner
"""

import pickle
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os

def create_classification_model():
    """Create a custom classification model"""
    print("Creating classification model...")
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy: {accuracy:.3f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/custom_classifier_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    return model, X_test[:5]  # Return model and sample data for testing

def create_regression_model():
    """Create a custom regression model"""
    print("\nCreating regression model...")
    
    # Generate synthetic dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Regression MSE: {mse:.3f}")
    
    # Save model
    model_path = "models/custom_regressor_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    return model, X_test[:5]  # Return model and sample data for testing

def create_simple_logistic_model():
    """Create a simple logistic regression model"""
    print("\nCreating logistic regression model...")
    
    # Generate binary classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Evaluate
    accuracy = model.score(X, y)
    print(f"Logistic regression accuracy: {accuracy:.3f}")
    
    # Save model
    model_path = "models/logistic_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    return model, X[:5]  # Return model and sample data for testing

def test_model_loading():
    """Test loading the saved models"""
    print("\n" + "="*50)
    print("Testing Model Loading")
    print("="*50)
    
    model_files = [
        "models/custom_classifier_model.pkl",
        "models/custom_regressor_model.pkl", 
        "models/logistic_model.pkl"
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f"✓ Successfully loaded: {model_path}")
                print(f"  Model type: {type(model).__name__}")
                
                # Test with dummy data
                if "regressor" in model_path:
                    dummy_data = np.random.randn(1, 5)
                else:
                    dummy_data = np.random.randn(1, 4 if "logistic" in model_path else 10)
                
                prediction = model.predict(dummy_data)
                print(f"  Test prediction: {prediction[0]}")
                
            except Exception as e:
                print(f"✗ Failed to load {model_path}: {e}")
        else:
            print(f"✗ File not found: {model_path}")

def generate_usage_examples():
    """Generate usage examples for the README"""
    print("\n" + "="*50)
    print("Usage Examples")
    print("="*50)
    
    examples = {
        "custom_classifier": {
            "data": [[1.2, -0.5, 0.8, 2.1, -1.0, 0.3, 1.5, -0.8, 0.9, -0.2]],
            "description": "Multi-class classification with 10 features"
        },
        "custom_regressor": {
            "data": [[1.0, 2.0, -1.5, 0.8, 2.3]],
            "description": "Regression with 5 features"
        },
        "logistic": {
            "data": [[1.0, 2.0, 3.0, 4.0]],
            "description": "Binary classification with 4 features"
        }
    }
    
    for model_name, info in examples.items():
        print(f"\n# {info['description']}")
        print("curl -X POST \"http://localhost:8000/predict\" \\")
        print("     -H \"Content-Type: application/json\" \\")
        print(f"     -d '{{")
        print(f"       \"data\": {info['data']},")
        print(f"       \"model_name\": \"{model_name}\"")
        print(f"     }}\'")

def main():
    """Main function"""
    print("Custom Model Creation Example")
    print("="*50)
    
    # Create different types of models
    clf_model, clf_sample = create_classification_model()
    reg_model, reg_sample = create_regression_model()
    log_model, log_sample = create_simple_logistic_model()
    
    # Test loading
    test_model_loading()
    
    # Generate usage examples
    generate_usage_examples()
    
    print("\n" + "="*50)
    print("Next Steps:")
    print("1. Build and run the Docker container:")
    print("   docker-compose up --build")
    print("\n2. Load your custom models:")
    print("   curl -X POST \"http://localhost:8000/models/custom_classifier/load\"")
    print("   curl -X POST \"http://localhost:8000/models/custom_regressor/load\"")
    print("   curl -X POST \"http://localhost:8000/models/logistic/load\"")
    print("\n3. Test predictions with the examples above")
    print("="*50)

if __name__ == "__main__":
    main()
