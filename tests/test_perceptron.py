import numpy as np
from src.perceptron import Perceptron


def test_predict_and_update():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([1, 1, 1, 0])
    model = Perceptron(n_features=2, learning_rate=0.1)
    history = model.fit(X, y, epochs=10, shuffle=False)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.mean(preds == y) >= 0.75
