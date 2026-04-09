# Hybrid Quantum + PyTorch Model for Breast Cancer Prediction

## Overview

This project implements a **hybrid quantum-classical machine learning model** to classify tumors as **malignant or benign**. It combines a **parameterized quantum circuit (PennyLane)** with **PyTorch** to explore how quantum layers can be integrated into modern ML workflows.

## Objective

To build and evaluate a binary classification model using a hybrid approach that leverages both quantum computing concepts and classical deep learning techniques.

##Dataset

* Source: `sklearn.datasets.load_breast_cancer`
* Total samples: 569
* Features: 30
* Classes:

  * 0 → Malignant
  * 1 → Benign

## Approach

### 1. Data Preprocessing

* Train-test split
* Feature scaling using `StandardScaler`
* Dimensionality reduction using **PCA** (reduced to 4 features for quantum circuit)

### 2. Quantum Layer

* Implemented using **PennyLane**
* Used:

  * Angle Embedding for feature encoding
  * Strongly Entangling Layers
* Outputs expectation values from qubits

### 3. Hybrid Model (PyTorch)

* Classical layer → Quantum layer → Classical output layer
* Activation: ReLU + Sigmoid
* Loss: Binary Cross Entropy
* Optimizer: Adam

## Tech Stack

* Python
* PyTorch
* PennyLane
* Scikit-learn
* NumPy
* Matplotlib

## Results

* Model trained successfully using hybrid architecture
* Achieved solid classification performance on test data
* Training loss decreased consistently over epochs

## Outputs

All outputs are saved in the `results/` folder:

* 📉 `training_loss.png` → Loss vs epochs
* 📊 `dataset_pca_plot.png` → 2D visualization of dataset
* 📝 `model_results.txt` → Accuracy, classification report, confusion matrix


## Project Structure

```
Quantum-Tumor-Prediction/
│
├── results/
│   ├── training_loss.png
│   ├── dataset_pca_plot.png
│   └── model_results.txt
│
├── Tumor_Prediction.py
├── README.md
├── requirements.txt
```


## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the project

```
python Tumor_Prediction.py
```



## Why this Project Matters

This project demonstrates:

* Integration of **quantum computing with machine learning**
* Practical use of **hybrid models**
* Understanding of **feature reduction for quantum systems**
* End-to-end ML pipeline from preprocessing to evaluation

## Key Takeaways

* Quantum circuits can be used as trainable components
* Hybrid models combine strengths of quantum and classical systems
* Real-world datasets can be adapted for quantum ML

## Future Improvements

* Compare performance with classical ML models
* Tune hyperparameters for better accuracy
* Deploy using Streamlit for interactive use
* Experiment with more qubits and deeper circuits

## Author

Devarshi Trivedi
MS Business Analytics and AI | UT DALLAS
