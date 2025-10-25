import argparse
import numpy as np
from src.perceptron import Perceptron
from src.utils import load_breast_cancer, standardize


def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save-model', type=str, default='model.npz')
    args = parser.parse_args()

    # Load and standardize dataset
    X, y = load_breast_cancer()
    X = standardize(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize and train model
    model = Perceptron(n_features=X.shape[1], learning_rate=args.learning_rate)
    history = model.fit(X_train, y_train, epochs=args.epochs, verbose=True)

    # Evaluate
    acc = np.mean(model.predict(X_test) == y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Save trained model
    model.save(args.save_model)
    print(f"Saved model to {args.save_model}")


if __name__ == '__main__':
    main()
