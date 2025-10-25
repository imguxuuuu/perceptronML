import pandas as pd
import numpy as np


def load_iris_binary(path='data/iris_binary.csv'):
    df = pd.read_csv(path)
    X = df[["sepal_length", "sepal_width"]].values
    y = df["label"].values
    return X, y


def load_breast_cancer(path='data/breast_cancer_clean.csv'):
    df = pd.read_csv(path)
    # Keep two features for visualization simplicity
    X = df[["radius_mean", "texture_mean"]].values
    y = df["label"].values
    return X, y


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / sigma