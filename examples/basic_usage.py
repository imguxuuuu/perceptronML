"""
Basic usage example for the Perceptron implementation.

This script demonstrates how to:
1. Load data
2. Initialize and train a perceptron
3. Make predictions
4. Evaluate performance
"""

import numpy as np
import pandas as pd
from src.perceptron import Perceptron
from src.utils import load_data, evaluate_model


def main():
    print("=" * 60)
    print("Perceptron Learning Algorithm - Basic Usage Example")
    print("=" * 60)
    
    # Load the synthetic dataset
    print("\n1. Loading synthetic dataset...")
    X_train, X_test, y_train, y_test = load_data('data/iris_binary.csv')
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Initialize the perceptron
    print("\n2. Initializing perceptron...")
    perceptron = Perceptron(learning_rate=0.1, n_epochs=30)
    print(f"   Learning rate: {perceptron.learning_rate}")
    print(f"   Epochs: {perceptron.n_epochs}")
    
    # Train the model
    print("\n3. Training the perceptron...")
    perceptron.fit(X_train, y_train)
    print(f"   Training completed!")
    print(f"   Final weights: {perceptron.weights[:5]}...")  # Show first 5 weights
    print(f"   Bias: {perceptron.bias}")
    
    # Make predictions
    print("\n4. Making predictions on test set...")
    y_pred = perceptron.predict(X_test)
    
    # Evaluate the model
    print("\n5. Evaluating model performance...")
    accuracy = evaluate_model(y_test, y_pred)
    print(f"   Test Accuracy: {accuracy:.2%}")
    
    # Show some example predictions
    print("\n6. Sample predictions:")
    print("   " + "-" * 50)
    print(f"   {'True Label':<15} {'Predicted':<15} {'Correct?'}")
    print("   " + "-" * 50)
    for i in range(min(10, len(y_test))):
        correct = "✓" if y_test[i] == y_pred[i] else "✗"
        print(f"   {y_test[i]:<15} {y_pred[i]:<15} {correct}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()