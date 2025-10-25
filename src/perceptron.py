import numpy as np


class Perceptron:
    """
    Simple Perceptron classifier.

    Binary classification with labels {0,1}.
    Uses learning rule:
        w += lr * (y_true - y_pred) * x
        b += lr * (y_true - y_pred)
    """

    def __init__(self, n_features, learning_rate=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate

    def predict_raw(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        raw = self.predict_raw(X)
        return (raw > 0).astype(int)

    def fit(self, X, y, epochs=50, shuffle=True, verbose=False):
        n_samples = X.shape[0]
        history = {"accuracy": []}

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
                X, y = X[indices], y[indices]

            for xi, yi in zip(X, y):
                pred = int((np.dot(xi, self.w) + self.b) > 0)
                error = yi - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error

            y_pred = self.predict(X)
            acc = np.mean(y_pred == y)
            history["accuracy"].append(acc)
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch+1}/{epochs} — acc: {acc:.4f}")

        return history

    def save(self, path):
        np.savez(path, w=self.w, b=self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(n_features=data['w'].shape[0])
        model.w = data['w']
        model.b = float(data['b'])
        return model
