# Breast Cancer Classification — Logistic Regression

**Author:** Germán Fragola Ramos

A binary classification project to predict whether a breast tumor is malignant or benign, using the Wisconsin Breast Cancer dataset.

---

## Overview

The goal of this project is to build a machine learning model capable of classifying breast tumors based on 30 numerical features derived from cell nucleus measurements. The target variable (`diagnosis`) has two possible values: **M** (malignant) and **B** (benign).

---

## Dataset

- **Source:** Wisconsin Breast Cancer Dataset
- **Samples:** 569
- **Features:** 30 numerical variables (radius, texture, perimeter, area, smoothness, etc.)
- **Target:** `diagnosis` — binary (M / B)

---

## Methodology

1. **Exploratory Data Analysis** — correlation heatmap to identify the most relevant features
2. **Data cleaning** — removed irrelevant columns and handled missing values
3. **Preprocessing** — applied `RobustScaler` to normalize features (chosen over StandardScaler to handle outliers, which are common in biological data)
4. **Modeling** — Logistic Regression with `max_iter=1000`
5. **Evaluation** — classification report, confusion matrix, cross-validation, and ROC curve
6. **Synthetic data validation** — generated synthetic samples using multivariate Gaussian distribution to test model generalization

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 98% |
| Recall — Malignant | 98% |
| Cross-validation accuracy | 97.5% ± 1.6% |

The model correctly identified 98% of malignant cases. Cross-validation confirms the result is consistent and not a product of overfitting.

> In a medical diagnosis context, recall on the malignant class is the most critical metric — missing a cancer case is a more serious error than a false positive.

---

## Libraries

```
numpy pandas matplotlib seaborn scikit-learn
```

---

## How to run

1. Clone the repository
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```
3. Open `breast_cancer_classification.ipynb` and run all cells
