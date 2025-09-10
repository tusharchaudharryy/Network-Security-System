# Credit Card Fraud Detection 🚨

A machine learning project to detect fraudulent credit card transactions using multiple models, trained and evaluated on real-world data.

---

## 📋 Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Project Structure](#project-structure)  
- [Models & Techniques](#models--techniques)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Results](#results)  
- [Future Work](#future-work)  
- [Contributing](#contributing)  
- [License](#license)

---

## Overview

As digital payments surge, so does the risk of credit card fraud. This project leverages ML to detect fraudulent transactions by analyzing patterns within a labeled dataset—focusing on accuracy, reliability, and interpretability.

---

## Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Details**:
  - ~ 284,807 transactions over 2 days in 2013 (European users)
  - 31 features: `V1–V28` (PCA components), `Time`, `Amount`
  - Target column `Class`: 1 = fraud, 0 = legitimate  
- **Imbalance**: Only 492 fraud cases (~0.17%)

---

## Project Structure

```text
.
├── data/                 # Raw and processed datasets
├── notebooks/            # Exploratory data analysis & model prototyping
├── src/                  # Training and evaluation scripts
├── models/               # Saved trained models
├── reports/              # Plots and performance figures
└── README.md             # Project overview
```

---

## Models & Techniques

The project implements and compares several machine learning algorithms:

| Model                 | Description |
|----------------------|-------------|
| Logistic Regression  | Baseline linear classifier |
| Support Vector Machine (SVM) | Robust decision boundary classifier |
| K-Nearest Neighbors (KNN) | Instance-based classifier |
| Decision Tree        | Simple interpretable model |
| Random Forest        | Ensemble of decision trees |
| AdaBoost & XGBoost   | Boosting-based ensemble methods |

Preprocessing & optimization strategies include:
- **Feature scaling** (e.g., RobustScaler)
- **Handling imbalance** with SMOTE, SMOTEENN, and downsampling
- **Feature selection** using Recursive Feature Elimination (RFE)
- **Hyperparameter tuning** via GridSearchCV

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tusharchaudharryy/ML_Project_Credit_Card_Fraud_Detection.git
   cd ML_Project_Credit_Card_Fraud_Detection
   ```

2. (Optional) Create a virtual env:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Place `creditcard.csv` in `data/`.
2. Run notebooks for exploration and feature engineering.
3. Train models:
   ```bash
   python src/train_models.py
   ```
4. Evaluate and visualize reports in `reports/`.

---

## Evaluation Metrics

Given the data imbalance, focus is on:

- **Accuracy**  
- **Precision, Recall, F1‑Score**  
- **ROC-AUC** & **Precision‑Recall AUC**

---

## Results

Summary of model performance (example):

| Model              | Accuracy | Precision | Recall | F1‑Score | AUC‑ROC | AUC‑PR |
|-------------------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression | 99.9% | 0.85 | 0.76 | 0.80 | 0.98 | 0.60 |
| Decision Tree     | …        | …         | …      | …        | …       | …      |
| Random Forest     | …        | …         | …      | …        | …       | …      |
| XGBoost           | …        | …         | …      | …        | …       | …      |

---

## Future Work

Potential improvements:

- Use **time‑series / sequential modeling** (e.g., LSTM, HMM) to track transaction patterns  
- Deploy in real‑time systems with streaming data  
- Add features: merchant info, geolocation, device metadata  
- Experiment with **federated learning** or advanced imbalanced algorithms

---

## Contributing

1. Fork the repo  
2. Create a feature branch `git checkout -b feature/YourFeature`  
3. Commit changes and push  
4. Open a pull request—discuss, iterate, and merge!

---

## License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 📬 Contact

Created by **Tushar Chaudharryy**.  
For questions or feedback, feel free to open an issue or reach out via GitHub.

---
