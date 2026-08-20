# Customer Churn Prediction Using Machine Learning

Machine learning project to predict customer churn using Logistic Regression and Random Forest, with SMOTE for class imbalance handling. Built on the Telco Customer Churn dataset.

## 1. Problem Statement

Customer churn refers to customers who stop using a company's service. For subscription-based businesses such as telecom providers, retaining existing customers is significantly cheaper than acquiring new ones. This project builds a machine learning model to predict whether a customer is likely to churn, based on their account information, service usage, and billing details.

## 2. Dataset Description

| Attribute | Details |
|---|---|
| Dataset name | Telco Customer Churn |
| Source | Kaggle (IBM Sample Dataset) |
| Number of records | 7,043 customers |
| Number of features | 20 input features + 1 target (Churn) |
| Target variable | Churn (Yes / No) |
| Class balance | ~73% No Churn, ~27% Churn (imbalanced) |

## 3. Methodology

**Data Cleaning** — Converted `TotalCharges` to numeric (blank entries became NaN, filled with median). Dropped `customerID`.

**EDA** — Visualized churn distribution, churn by contract type, and churn by tenure.

**Feature Encoding** — Mapped target to 0/1, one-hot encoded categorical features.

**Train-Test Split** — 80/20 split with `stratify=y` to preserve class ratio.

**SMOTE** — Applied to training set only, balancing 4,139 vs 4,139 samples per class.

**Feature Scaling** — StandardScaler applied before Logistic Regression; Random Forest trained on unscaled data.

**Models** — Logistic Regression and Random Forest.

## 4. Results

| Metric | Logistic Regression | Random Forest | Better Model |
|---|---|---|---|
| Accuracy | 76.08% | 77.50% | Random Forest |
| Precision (Churn) | 0.54 | 0.57 | Random Forest |
| Recall (Churn) | 0.63 | 0.60 | Logistic Regression |
| F1-score (Churn) | 0.58 | 0.58 | Tie |

Random Forest achieved higher overall accuracy, but Logistic Regression caught more actual churners (higher recall) — an important trade-off, since missing a churner is costlier than a false alarm.

## 5. Key Insights

- Contract type is one of the strongest predictors — month-to-month customers churn far more than those on longer contracts
- Newer customers (low tenure) churn more frequently
- Class imbalance means accuracy alone isn't enough — precision/recall/F1 give a fuller picture

## 6. Conclusion

Both models performed reasonably well, with Random Forest ahead on accuracy and Logistic Regression ahead on churn recall. Future work could include hyperparameter tuning and trying additional algorithms like XGBoost.

## 7. Tools Used

Python (Google Colab) · pandas · numpy · matplotlib · seaborn · scikit-learn · imbalanced-learn (SMOTE)
