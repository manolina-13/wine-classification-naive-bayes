# Wine Classification using Naive Bayes

This repository contains a practical machine learning notebook that performs multiclass classification on the **Wine dataset** using two Naive Bayes variants:

- **Gaussian Naive Bayes (GNB)**
- **Multinomial Naive Bayes (MNB)**

The notebook demonstrates a complete mini-workflow from dataset exploration to model evaluation and conclusion.

## Repository File

- `wine-classification-naive-bayes.ipynb` — main notebook containing data exploration, model training, and comparison.

## Project Objective

The goal is to classify wine samples into their correct classes and compare how different Naive Bayes assumptions affect model performance on this dataset.

Specifically, the notebook aims to:

1. Load and inspect the Wine dataset.
2. Visualize and understand feature distributions.
3. Split data into train/test sets.
4. Train both GaussianNB and MultinomialNB models.
5. Compare their accuracies.
6. Explain why one model is more suitable for this type of data.

## Dataset Information

The dataset is loaded from **scikit-learn** (`sklearn.datasets.load_wine`).

It contains:

- **178 samples**
- **13 numerical chemical features** (continuous values)
- **3 target classes** (wine categories)

Because the features are continuous, the dataset is naturally more aligned with Gaussian-style assumptions.

## Workflow Used in the Notebook

### 1) Data Loading

The notebook imports the wine dataset and required libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`).

### 2) Data Exploration

- Checks dataset structure (feature names, targets, target names).
- Creates a DataFrame for easier inspection.
- Plots histograms of features to observe distributions.

### 3) Train-Test Split

Uses `train_test_split` with a 70:30 split (`test_size=0.3`) and fixed `random_state=42` for reproducibility.

### 4) Model Training

- Trains **GaussianNB** on training data.
- Trains **MultinomialNB** on the same training data.

### 5) Evaluation

Computes test accuracy for both models and compares results.

## Brief Theory

## Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem:

$$P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}$$

where:

- $C$ is a class label,
- $X$ is the feature vector.

The "naive" assumption is that features are conditionally independent given the class.

### Gaussian Naive Bayes (GNB)

- Assumes each feature follows a **normal (Gaussian) distribution** within each class.
- Works well for continuous-valued features.

### Multinomial Naive Bayes (MNB)

- Designed for **count/discrete data** (e.g., text word counts).
- Usually more suitable for NLP and frequency-based features.

## Why GaussianNB performs better here

The Wine dataset features are real-valued continuous measurements, so GaussianNB's assumptions fit better than MultinomialNB's discrete count assumption. This is why GaussianNB generally achieves higher accuracy in this notebook.

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## How to Run

1. Open `wine-classification-naive-bayes.ipynb` in VS Code or Jupyter Notebook.
2. Run all cells from top to bottom.
3. Review printed accuracy values and final conclusion.

## Output Summary

The notebook reports model accuracy and provides a final interpretation comparing GaussianNB and MultinomialNB for this dataset.
