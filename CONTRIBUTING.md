# Contributing to Perceptron From Scratch

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](https://github.com/<your-username>/perceptron-from-scratch/issues) section
2. If not, create a new issue with:
   - A clear, descriptive title
   - Detailed description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/<your-username>/perceptron-from-scratch.git
   cd perceptron-from-scratch
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Provide a clear description of your changes

## Code Style Guidelines

- Follow [PEP 8](https://pep8.org/) style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Comment complex logic

### Example Function Style

```python
def train_perceptron(X, y, learning_rate=0.01, epochs=100):
    """
    Train a perceptron on the given data.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : numpy.ndarray
        Target vector of shape (n_samples,)
    learning_rate : float, optional
        Learning rate for weight updates (default: 0.01)
    epochs : int, optional
        Number of training iterations (default: 100)
    
    Returns
    -------
    weights : numpy.ndarray
        Trained weight vector
    """
    # Implementation here
    pass
```

## Testing Guidelines

- Write tests for all new functions
- Ensure all tests pass before submitting
- Aim for meaningful test coverage
- Use descriptive test names

```python
def test_perceptron_learns_linear_separation():
    """Test that perceptron converges on linearly separable data."""
    # Test implementation
    pass
```

## Documentation

- Update README.md if adding new features
- Add inline comments for complex logic
- Include docstrings for all public functions
- Update examples if behavior changes

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Reach out via LinkedIn (see README for contact info)

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to learn and improve together.

Thank you for contributing! 🎉