# 🌸 Data Classification Using AI — KNN on Iris Dataset

> **DecodeLabs Industrial Training Kit | Batch 2026 | Project 2**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project demonstrates the complete **supervised learning pipeline** using the **K-Nearest Neighbors (KNN)** algorithm to classify Iris flower species. It covers every step from raw data to evaluation — following the IPO (Input → Process → Output) framework taught in the DecodeLabs AI training.

### Pipeline

```
Raw Data → EDA → Feature Scaling → Train-Test Split → KNN Model → Confusion Matrix + F1 Score
```

---

## 🗂️ Project Structure

```
iris-knn-classifier/
│
├── iris_knn_classifier.ipynb   # Main Jupyter Notebook (complete pipeline)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation


---

## 📊 Dataset: The Iris Benchmark

| Property | Value |
|---|---|
| Samples | 150 (Balanced — 50 per class) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Source | `sklearn.datasets.load_iris` |

---

## ⚙️ Pipeline Steps

1. **Load & Explore** — Load Iris dataset, check shape, class distribution, missing values
2. **EDA** — Pairplot, correlation heatmap, class distribution chart
3. **Feature Scaling** — `StandardScaler` (mean=0, variance=1) — essential for KNN
4. **Train-Test Split** — 80/20 split, stratified + shuffled to remove order bias
5. **Optimal K** — Elbow method across K=1 to K=20 (error rate & F1 score curves)
6. **Train KNN** — `KNeighborsClassifier(n_neighbors=5)` via scikit-learn
7. **Evaluate** — Accuracy, F1 Score (weighted & macro), full classification report
8. **Confusion Matrix** — Raw counts + normalized (per-class percentage)
9. **Decision Boundary** — 2D visualization using petal features
10. **Custom Predictions** — Predict species + confidence for new flower measurements

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | ~96–100% |
| F1 Score (weighted) | ~0.97+ |
| F1 Score (macro) | ~0.97+ |

---

## 🔑 Key Concepts Covered

| Concept | Explanation |
|---|---|
| **Supervised Learning** | Machine learns decision logic from labeled historical data |
| **Feature Scaling** | Mandatory for distance-based algorithms — prevents feature dominance |
| **Stratified Split** | Maintains class balance in both train and test sets |
| **K Selection (Elbow)** | K=1 → overfitting, K=100 → underfitting; optimal K at the elbow |
| **Accuracy Mirage** | On imbalanced data, accuracy misleads — F1 Score is more reliable |
| **Confusion Matrix** | Exposes TP, FP, FN, TN — honest per-class performance breakdown |
| **Precision vs Recall** | Trade-off: Precision = spam filters, Recall = medical diagnosis |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/iris-knn-classifier.git
cd iris-knn-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook iris_knn_classifier.ipynb
```

Or open directly in **VS Code** with the Jupyter extension.

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 💡 Extend This Project

- [ ] Compare KNN with Decision Tree, SVM, Logistic Regression
- [ ] Implement cross-validation (k-fold) instead of single train-test split
- [ ] Add hyperparameter tuning with `GridSearchCV`
- [ ] Try on a different dataset (Wine, Breast Cancer, Titanic)
- [ ] Deploy as a simple web app using Streamlit

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Iris Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)

