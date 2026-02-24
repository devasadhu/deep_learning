# Deep Learning Lab 3: Comparative Analysis of CNN Architectures

**Course:** AI302 - Deep Learning  
**Institution:** Sardar Vallabhbhai National Institute of Technology, Surat  
**Department:** Artificial Intelligence

## 📋 Overview

This project implements and compares multiple state-of-the-art CNN architectures, loss functions, and optimization strategies on the CIFAR-10 dataset. The lab is divided into three parts:

1. **Architecture Comparison** - Evaluate 8 CNN models
2. **Loss Function Analysis** - Compare BCE, Focal Loss, and ArcFace
3. **Feature Visualization** - t-SNE analysis of learned features

## 🏗️ Architecture Implementations

The following CNN architectures are implemented from scratch:

- **LeNet-5** (1998) - 83,126 parameters
- **AlexNet** (2012) - 21,289,546 parameters  
- **VGGNet** (2014) - 9,762,890 parameters
- **ResNet-50** (2015) - 23,520,842 parameters
- **ResNet-101** (2015) - 42,512,970 parameters
- **MobileNet** (2017) - 3,217,226 parameters
- **EfficientNet** (2019) - 711,626 parameters
- **InceptionV3** (2015) - 912,298 parameters

## 📊 Dataset

**CIFAR-10:**
- 50,000 training images
- 10,000 test images
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Image size: 32×32×3

## 🎯 Key Results

### Part 1: Best Architecture
- **Winner:** InceptionV3
- **Test Accuracy:** 85.67%
- **Parameters:** 912,298 (excellent efficiency)

### Part 2: Best Loss Function Configuration
- **Winner:** AlexNet + SGD + Focal Loss
- **Test Accuracy:** 82.63%
- **Training Epochs:** 20

### Part 3: Feature Clustering
- t-SNE visualizations show distinct class clustering
- ArcFace produces more separated feature embeddings
- Better separation correlates with improved classification

## 📁 Project Structure
```
.
├── CNN_Comparative_Analysis.ipynb          # Main implementation notebook
├── README.md                    # This file
├── REPORT.md                    # Detailed analysis report
```

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn tqdm
```

### Execution
1. Open `CNN_Comparative_Analysis.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Modify the **Configuration** section to customize experiments:
   - Change dataset (MNIST/FashionMNIST/CIFAR10)
   - Select models to train
   - Adjust epochs and batch size
   - Enable/disable specific parts

### Configuration Example
```python
DATASET = 'CIFAR10'
BATCH_SIZE = 128
PART1_EPOCHS = 10
PART1_MODELS = ['LeNet-5', 'AlexNet', 'VGGNet', 'ResNet-50', 
                'ResNet-101', 'MobileNet', 'EfficientNet', 'InceptionV3']
```

## 📈 Experiments Conducted

### Part 1: Architecture Comparison (10 epochs each)
All 8 models trained on CIFAR-10 with:
- Optimizer: Adam (lr=0.001)
- Loss: Cross-Entropy
- Batch Size: 128

### Part 2: Loss Function Comparison
| Model | Optimizer | Epochs | Loss Function | Test Accuracy |
|-------|-----------|--------|---------------|---------------|
| VGGNet | Adam | 10 | BCE | ~79% |
| AlexNet | SGD | 20 | Focal Loss | 82.63% |
| ResNet-50 | Adam | 15 | ArcFace | ~81% |

### Part 3: t-SNE Visualization
- Extracted 2000 test samples
- Visualized feature embeddings for each loss function
- Compared cluster separation quality

## 🔍 Key Findings

1. **InceptionV3** achieves best accuracy with moderate parameters due to multi-scale feature extraction
2. **Focal Loss** effectively handles hard examples, improving AlexNet performance
3. **ArcFace** creates more discriminative features with better inter-class separation
4. **Deeper ≠ Better**: ResNet-101 doesn't significantly outperform ResNet-50 on CIFAR-10
5. **Efficiency Champion**: EfficientNet provides good accuracy (80%+) with only 711K parameters

## 🛠️ Technologies Used

- **Framework:** PyTorch 2.0+
- **Data:** torchvision datasets
- **Visualization:** matplotlib, seaborn
- **Dimensionality Reduction:** scikit-learn (t-SNE)
- **Hardware:** NVIDIA Tesla T4 GPU (Google Colab)

## 📝 Loss Functions Implemented

### 1. Cross-Entropy (BCE)
Standard classification loss function

### 2. Focal Loss
```python
FL = -α(1-pt)^γ * log(pt)
```
Focuses on hard-to-classify examples (α=1, γ=2)

### 3. ArcFace Loss
```python
L = -log(e^(s*cos(θ+m)) / (e^(s*cos(θ+m)) + Σe^(s*cos(θ))))
```
Additive angular margin loss for enhanced feature discrimination (s=30, m=0.5)

## 👥 Author

**Student Name:** Sadhana
**Roll Number:** U23AI003 
**Course:** AI302 - Deep Learning  

## 📅 Submission Details

**Lab Number:** 3  
**Date:** January 2026  
**Institution:** SVNIT Surat

## 📄 License

This project is submitted as part of academic coursework at SVNIT Surat.
