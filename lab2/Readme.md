# Deep Learning Lab 2: Handwritten Digit Recognition with CNN and MLP

## Project Overview
This project implements and compares different Convolutional Neural Network (CNN) and Multi-Layer Perceptron (MLP) architectures for handwritten digit classification using the MNIST dataset. The main focus is on understanding the impact of different activation functions, optimizers, and regularization techniques.

## Objective
- Build and refine CNN/MLP models for digit classification
- Compare activation functions (Sigmoid, Tanh, ReLU)
- Evaluate different optimizers (SGD, SGD with Momentum, Adam)
- Analyze the effects of Batch Normalization and Dropout

## Dataset
- **MNIST**: 28x28 grayscale images of handwritten digits (0-9)
- **Training samples used**: 10,000 (reduced for faster execution)
- **Test samples used**: 2,000

## Project Structure
```
├── lab2.ipynb   # Main notebook
├── lab_summary.txt                   # Complete findings summary
└── README.md                         # This file
```

## Model Architectures

### CNN Base Architecture
```
Input (28x28x1)
  ↓
Conv2D (32 filters, 3x3, activation)
  ↓
Conv2D (64 filters, 3x3, activation)
  ↓
MaxPooling2D (2x2)
  ↓
Dropout (0.25)
  ↓
Flatten
  ↓
Dense (128 units, activation)
  ↓
Dense (10 units, Softmax)
```

### MLP Base Architecture
```
Input (28x28x1)
  ↓
Flatten (784)
  ↓
Dense (256)
BatchNormalization (optional)
Activation (ReLU)
  ↓
Dense (128)
BatchNormalization (optional)
Activation (ReLU)
  ↓
Dense (10, Softmax)
```

## Experiments Conducted

### Task 1: Activation Function Challenge
Compared three activation functions with Adam optimizer:
- **Sigmoid**: 11.70% accuracy (vanishing gradient problem)
- **Tanh**: 96.50% accuracy (better than sigmoid)
- **ReLU**: 96.95% accuracy (best performance)

**Key Finding**: ReLU avoids vanishing gradients and converges fastest.

### Task 2: Optimizer Showdown
Compared optimizers using ReLU activation:
- **SGD**: 90.45% accuracy (slowest, unstable)
- **SGD + Momentum**: 96.45% accuracy (smoother convergence)
- **Adam**: 97.15% accuracy (fastest, best performance)

**Key Finding**: Adam's adaptive learning rates provide superior convergence speed.

### Task 3: Regularization Effects
Tested Batch Normalization and Dropout combinations:
- **No BN, No Dropout**: 94.30% accuracy
- **No BN, Dropout=0.1**: 94.45% accuracy
- **With BN, Dropout=0.25**: 94.35% accuracy

**Key Finding**: BN stabilizes training; Dropout prevents overfitting.

## Required Model Configurations

| Model | Architecture | Optimizer | Epochs | Accuracy |
|-------|-------------|-----------|--------|----------|
| CNN-1 | 128 FC units | Adam | 10 | 97.00% |
| MLP-1 | 512-256-128 | SGD | 20 | 88.90% |
| MLP-2 | 256 units | Adam | 15 | 94.45% |

## Installation & Requirements

### Prerequisites
```bash
pip install tensorflow numpy matplotlib pandas
```

### Required Libraries
- TensorFlow/Keras 2.x
- NumPy
- Matplotlib
- Pandas (optional, for data display)

## How to Run

1. **Clone or download the project**

2. **Open Jupyter Notebook**
```bash
jupyter notebook lab2_mnist_classification.ipynb
```

3. **Run all cells**
   - The notebook will automatically download MNIST dataset
   - Training takes approximately 15-20 minutes
   - Plots and summary will be generated automatically

4. **View Results**
   - Check the generated PNG files for visualizations
   - Open `lab_summary.txt` for complete findings

## Results Summary

### Best Configurations
- **Best Activation**: ReLU (96.95%)
- **Best Optimizer**: Adam (97.15%)
- **Best Model**: CNN-1 with Adam (97.00%)

### Key Insights
1. **ReLU** is superior for deep networks - no vanishing gradient
2. **Adam** converges faster than SGD variants
3. **Batch Normalization** improves training stability
4. **Dropout** helps prevent overfitting
5. **CNNs** outperform MLPs for image classification

## Visualizations

### Generated Plots
1. **task1_activation_comparison.png**
   - Training/Validation Loss curves
   - Training/Validation Accuracy curves
   - Comparison of Sigmoid, Tanh, and ReLU

2. **task2_optimizer_comparison.png**
   - Training/Validation Loss curves
   - Training/Validation Accuracy curves
   - Comparison of SGD, SGD+Momentum, and Adam

3. **task3_bn_dropout_comparison.png**
   - Training/Validation Loss curves
   - Training/Validation Accuracy curves
   - Comparison of different BN/Dropout configurations

## Observations & Conclusions

### Activation Functions
- **Sigmoid**: Severe vanishing gradient (10% accuracy ~random guessing)
- **Tanh**: Zero-centered outputs, better than sigmoid
- **ReLU**: f(x) = max(0,x), eliminates negative gradients, fastest convergence

### Optimizers
- **SGD**: Fixed learning rate, slow and unstable
- **SGD + Momentum**: Dampens oscillations, smoother descent
- **Adam**: Adaptive learning rates per parameter, optimal performance

### Regularization
- **Batch Normalization**: Normalizes layer inputs, stabilizes training
- **Dropout**: Randomly deactivates neurons, prevents co-adaptation

## Author
**Student Name**: Sadhana Devarajan
**Institution**: Sardar Vallabhbhai National Institute of Technology, Surat  
**Course**: AI302 - Deep Learning  
**Lab**: Practical 2

## References
1. MNIST Database: http://yann.lecun.com/exdb/mnist/
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Deep Learning Book by Ian Goodfellow

## License
This project is submitted as part of academic coursework at SVNIT Surat.

---
**Note**: This implementation uses reduced dataset size (10k training samples) for faster execution during lab sessions. For production use, train on the full 60k training samples.