# Perceptron Learning Algorithm — From Scratch

A complete implementation of the Perceptron Learning Algorithm built from first principles using Python, NumPy, and Matplotlib. This project demonstrates binary classification on both synthetic linearly separable data and real-world medical diagnostics data.

## Overview

This project implements the perceptron—the foundational building block of modern neural networks—without relying on high-level machine learning frameworks. Through hands-on implementation, it explores:

- Manual weight updates using the perceptron learning rule
- Binary classification with activation functions
- Decision boundary visualization
- Performance comparison between ideal and real-world datasets

## Datasets

### 1. Synthetic Dataset (Iris-inspired)
- **File:** `data/iris_binary.csv`
- **Purpose:** Validation and visualization of algorithm convergence
- **Characteristics:** Linearly separable binary classification problem

### 2. Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Raw file:** `data/wdbc.data`
- **Processed file:** `data/breast_cancer_clean.csv`
- **Features:** 30 continuous attributes (mean radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Binary classification
  - `M` → 1 (Malignant)
  - `B` → 0 (Benign)

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/perceptron-from-scratch.git
   cd perceptron-from-scratch
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train on the synthetic dataset:
```bash
python -m src.train --learning-rate 0.1 --epochs 30 --save-model model_synthetic.npz
```

Convert and train on the breast cancer dataset:
```bash
python data/convert_breast_cancer.py
python -m src.train --learning-rate 0.01 --epochs 50 --save-model model_breast.npz
```

### Visualization

Visualize decision boundaries:
```bash
# Synthetic dataset
python -m src.visualize --model model_synthetic.npz

# Breast cancer dataset
python -m src.visualize --model model_breast.npz --out breast_boundary.png
```

### Testing

Run the test suite:
```bash
pytest -q
```

## Project Structure

```
perceptron-from-scratch/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   ├── iris_binary.csv              # Synthetic training data
│   ├── wdbc.data                    # Raw breast cancer data
│   ├── breast_cancer_clean.csv      # Processed breast cancer data
│   └── convert_breast_cancer.py     # Data preprocessing script
├── examples/
│   └── basic_usage.py
├── src/
│   ├── __init__.py
│   ├── perceptron.py                # Core perceptron implementation
│   ├── utils.py                     # Utility functions
│   ├── train.py                     # Training script
│   └── visualize.py                 # Visualization script
├── tests/
│   ├── __init__.py
│   └── test_perceptron.py           # Unit tests
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Technical Details

### Dependencies

| Purpose | Library |
|---------|---------|
| Numerical Computation | NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |
| Testing | Pytest |

### Algorithm Implementation

The perceptron uses the following learning rule:

```
w = w + α(y - ŷ)x
```

where:
- `w` = weight vector
- `α` = learning rate
- `y` = true label
- `ŷ` = predicted label
- `x` = input features

The activation function is a simple step function that outputs 1 if the weighted sum is positive, otherwise 0.

## Key Findings

- **Synthetic Data:** Demonstrates perfect convergence on linearly separable data
- **Real Data:** Shows limitations of linear classifiers on complex, real-world medical data
- **Interpretability:** Visualizations reveal how decision boundaries evolve during training
- **Educational Value:** Provides deep insight into the mechanics of gradient-free learning

## References

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6), 386-408.
- Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.
- Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995). "Breast Cancer Wisconsin (Diagnostic) Data Set."

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Gururaghuraman Sethuraman**  
Python Developer | Erasmus Scholar | Nanotechnology & Electronics  
[LinkedIn](https://www.linkedin.com/in/gururaghuraman/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- UCI Machine Learning Repository for providing the breast cancer dataset
- The scientific community for foundational research in neural networks and machine learning