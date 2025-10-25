import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.perceptron import Perceptron
from src.utils import load_iris_binary, standardize


def plot_decision_boundary(model, X, y, out=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')

    # Draw decision line
    if abs(model.w[1]) > 1e-6:
        x_vals = np.array([x_min, x_max])
        y_vals = -(model.w[0] * x_vals + model.b) / model.w[1]
        plt.plot(x_vals, y_vals, '--k')

    plt.xlabel('sepal_length (standardized)')
    plt.ylabel('sepal_width (standardized)')
    plt.title('Perceptron Decision Boundary')
    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved figure to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.npz')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    X, y = load_iris_binary()
    X = standardize(X)
    model = Perceptron.load(args.model)

    plot_decision_boundary(model, X, y, out=args.out)


if __name__ == '__main__':
    main()
