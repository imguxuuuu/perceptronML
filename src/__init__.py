# src/__init__.py
"""
Perceptron Learning Algorithm - From Scratch

A complete implementation of the perceptron algorithm without
using high-level machine learning frameworks.
"""

__version__ = "1.0.0"
__author__ = "Guru R"

from .perceptron import Perceptron
from .utils import load_data, evaluate_model

__all__ = ['Perceptron', 'load_data', 'evaluate_model']