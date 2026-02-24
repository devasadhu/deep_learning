# Deep Learning Lab - Practical 1

**Sardar Vallabhbhai National Institute of Technology**  
Surat-395007

**Department of Artificial Intelligence**

---

## Lab Information

- **Course**: Deep Learning (AI302)
- **Lab**: Practical 1 - Introduction to PyTorch and Neural Networks
- **Submitted By**: Sadhana Devarajan
- **Roll No**: U23AI003
- **Branch**: Artificial Intelligence
- **Date**: January 16, 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Task 1: PyTorch Tensors and Basic Operations](#task-1-pytorch-tensors-and-basic-operations)
3. [Task 2: Linear Algebra Operations with TensorFlow](#task-2-linear-algebra-operations-with-tensorflow)
4. [Task 3: AND/OR Gates using Perceptron](#task-3-andor-gates-using-perceptron)
5. [Task 4: XOR Problem using Neural Network](#task-4-xor-problem-using-neural-network)
6. [Task 5: Neural Network for Regression](#task-5-neural-network-for-regression)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Introduction

This report presents the implementation and results of Lab Practical 1 for the Deep Learning course (AI302). The lab focuses on fundamental concepts in deep learning including PyTorch tensors, linear algebra operations, perceptron implementation, and neural network architectures for solving logical and regression problems.

### Objectives

- Understand PyTorch tensors, initialization methods, and data types
- Perform tensor operations and explore automatic differentiation
- Implement linear algebra operations using TensorFlow
- Design and train perceptrons for AND/OR gate logic
- Solve the XOR problem using multi-layer neural networks
- Implement a neural network for regression tasks

---

## Task 1: PyTorch Tensors and Basic Operations

### Theory

PyTorch tensors are multi-dimensional arrays similar to NumPy arrays but with GPU acceleration capabilities. The Autograd system in PyTorch enables automatic differentiation, which is fundamental for training neural networks through backpropagation.

### Implementation

The implementation covers three main aspects:

#### Tensor Initialization

Multiple initialization methods were explored:
- Creating tensors from Python lists
- Zero and one tensors using `torch.zeros()` and `torch.ones()`
- Random tensors using `torch.rand()`
- Range tensors using `torch.arange()`
- Tensors with specific data types (float32, int64, bool)

#### Tensor Operations

Various operations were performed:
- Arithmetic operations: addition, subtraction, multiplication
- Matrix multiplication using `@` operator
- Broadcasting for operations on tensors of different shapes
- Indexing and slicing
- Reshaping using `reshape()` and `view()`

#### Automatic Differentiation

The Autograd system was demonstrated by computing gradients:

**Function**: `z = x² + 2y + 3`

**Gradients computed**:
- ∂z/∂x = 2x = 4 (at x=2)
- ∂z/∂y = 2

### Results

- Successfully created tensors of various types and shapes
- All arithmetic and matrix operations executed correctly
- Autograd computed gradients accurately with values matching theoretical expectations
- Mean gradient for vector operations: 0.6667 (verified as 2/3)

---

## Task 2: Linear Algebra Operations with TensorFlow

### Theory

Linear algebra forms the mathematical foundation of neural networks. TensorFlow provides efficient implementations of matrix operations essential for deep learning computations.

### Implementation

The following operations were implemented using TensorFlow:

**Given matrices**:
```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]
```

### Results

#### Matrix Multiplication
```
A × B = [[19, 22],
         [43, 50]]
```

#### Element-wise Operations

**Addition**:
```
A + B = [[6,  8],
         [10, 12]]
```

**Multiplication**:
```
A ⊙ B = [[5,  12],
         [21, 32]]
```

#### Matrix Properties

- **Transpose**: A^T = [[1, 3], [2, 4]]
- **Determinant**: det(A) = -2.0
- **Inverse**: A^(-1) = [[-2, 1], [1.5, -0.5]]
- **Eigenvalues and eigenvectors**: Successfully computed using `tf.linalg.eigh()`

---

## Task 3: AND/OR Gates using Perceptron

### Theory

A perceptron is a single-layer neural network capable of learning linearly separable functions. The AND and OR gates are linearly separable, making them suitable for perceptron implementation.

**Perceptron model**: `y = σ(w^T x + b)`

**Sigmoid activation function**: `σ(z) = 1 / (1 + e^(-z))`

### Implementation

- **Architecture**: 2 inputs → 1 output with sigmoid activation
- **Loss function**: Binary Cross-Entropy Loss
- **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate 0.1
- **Training epochs**: 1000

### Results

#### AND Gate

| Input   | Target | Prediction |
|---------|--------|------------|
| [0, 0]  | 0      | 0.0082     |
| [0, 1]  | 0      | 0.1493     |
| [1, 0]  | 0      | 0.1492     |
| [1, 1]  | 1      | 0.7869     |

**Final Loss**: 0.1429

#### OR Gate

| Input   | Target | Prediction |
|---------|--------|------------|
| [0, 0]  | 0      | 0.1925     |
| [0, 1]  | 1      | 0.9258     |
| [1, 0]  | 1      | 0.9250     |
| [1, 1]  | 1      | 0.9985     |

**Final Loss**: 0.0927

### Analysis

Both gates were learned successfully. The perceptron achieved near-perfect classification for linearly separable logic functions, demonstrating its capability for simple pattern recognition tasks.

---

## Task 4: XOR Problem using Neural Network

### Theory

The XOR problem is not linearly separable and cannot be solved by a single-layer perceptron. This limitation led to the development of multi-layer neural networks. A neural network with at least one hidden layer can learn the XOR function.

**XOR truth table**:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

### Network Architecture

- **Input layer**: 2 neurons
- **Hidden layer**: 4 neurons with sigmoid activation
- **Output layer**: 1 neuron with sigmoid activation
- **Loss function**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate 0.1
- **Training epochs**: 5000

### Results

| Input   | Target | Prediction | Result  |
|---------|--------|------------|---------|
| [0, 0]  | 0      | 0.0000     | ✓       |
| [0, 1]  | 1      | 1.0000     | ✓       |
| [1, 0]  | 1      | 1.0000     | ✓       |
| [1, 1]  | 0      | 0.0000     | ✓       |

**Final Loss**: 0.0000 (perfect convergence)

### Training Progress

| Epoch | Loss   |
|-------|--------|
| 1000  | 0.0005 |
| 2000  | 0.0002 |
| 3000  | 0.0001 |
| 4000  | 0.0000 |
| 5000  | 0.0000 |

### Analysis

The multi-layer neural network successfully learned the XOR function with 100% accuracy. The loss decreased steadily from initial values to virtually zero, and all predictions matched the expected outputs perfectly. This demonstrates the power of hidden layers in learning non-linear decision boundaries.

---

## Task 5: Neural Network for Regression

### Theory

Regression problems involve predicting continuous values rather than discrete classes. Neural networks can approximate any continuous function given sufficient capacity (Universal Approximation Theorem).

### Problem Statement

Generate synthetic data following the linear relationship:

**y = 2x + 3 + ε**

where ε ~ N(0, 4) represents Gaussian noise.

### Network Architecture

- **Input layer**: 1 neuron
- **Hidden layer 1**: 10 neurons with ReLU activation
- **Hidden layer 2**: 10 neurons with ReLU activation
- **Output layer**: 1 neuron (linear activation)
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.01
- **Training data**: 100 samples from range [-10, 10]

### Results

#### Training Progress

| Epoch | Loss (MSE) |
|-------|------------|
| 200   | 3.2029     |
| 400   | 3.1017     |
| 600   | 3.0740     |
| 800   | 3.0511     |
| 1000  | 3.0349     |

#### Sample Predictions

| Input (x) | Prediction (y) |
|-----------|----------------|
| -10.00    | -16.69         |
| -5.79     | -9.04          |
| -1.58     | -1.10          |
| 2.63      | 8.53           |
| 6.84      | 16.61          |

### Analysis

The neural network successfully learned the underlying linear relationship despite the added noise. The predictions closely follow the expected pattern y ≈ 2x + 3. The final MSE of approximately 3.03 is close to the variance of the noise (σ² = 4), indicating good model fit without overfitting.

---

## Conclusion

This lab practical successfully demonstrated fundamental concepts in deep learning:

1. **PyTorch Fundamentals**: Mastered tensor operations, broadcasting, and automatic differentiation using Autograd

2. **Linear Algebra**: Implemented essential matrix operations using TensorFlow, including multiplication, transpose, inverse, and eigendecomposition

3. **Perceptron Learning**: Successfully trained single-layer networks for linearly separable problems (AND/OR gates)

4. **Multi-layer Networks**: Solved the non-linearly separable XOR problem with 100% accuracy using a hidden layer

5. **Regression Modeling**: Built a neural network for continuous value prediction with good generalization

### Key Learnings

- Single-layer perceptrons are limited to linearly separable problems
- Hidden layers enable learning of complex, non-linear patterns
- Proper architecture choice (number of neurons, activation functions) is crucial
- Optimizer selection (SGD vs Adam) significantly impacts convergence
- Neural networks can effectively handle both classification and regression tasks

### Future Work

- Experiment with different activation functions (ReLU, tanh, LeakyReLU)
- Implement regularization techniques to prevent overfitting
- Explore convolutional and recurrent architectures
- Apply learned concepts to real-world datasets

---

## References

1. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
2. TensorFlow API Documentation: https://www.tensorflow.org/api_docs/python/tf
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press
4. Nielsen, M. (2015). *Neural Networks and Deep Learning*

---

**End of Report**