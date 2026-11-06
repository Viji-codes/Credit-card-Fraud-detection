
# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression and Random Forest.

## About the Dataset
- Source: Kaggle, "Credit Card Fraud Detection" (mlg-ulb)
- 284,807 transactions made by European cardholders over two days
- Only 492 are fraud (about 0.17%), so the data is highly imbalanced
- Features V1 to V28 are anonymised (PCA); Time, Amount and Class are the other columns
- The CSV is not included here because of its size. Download it from Kaggle and place it next to the notebook.

## What This Project Does
- Explores the data (fraud vs normal, amount distribution, patterns over time)
- Scales Time and Amount and splits the data 80/20 with stratification
- Trains Logistic Regression and Random Forest
- Evaluates with a confusion matrix, ROC curve, precision, recall and AUC

## Results (Random Forest)
| Metric | Score |
|--------|-------|
| Fraud Precision | __% |
| Fraud Recall | __% |
| ROC AUC | __ |

Accuracy alone is misleading on imbalanced data. A model that always predicts "Normal" would still score about 99.8%, so precision and recall on the fraud class matter more.

## Charts


![Fraud Analysis](fraud_analysis.png)




![Model Results](model_results.png)



## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab

## How to Run
1. Open the notebook in Google Colab
2. Upload creditcard.csv (or creditcard.csv.zip) to the Colab files panel
3. Click Runtime, then Run all

## Limitations and Future Work
- Try SMOTE or class weights to handle the imbalance
- Test XGBoost and tune hyperparameters
- Use precision-recall curves for a fairer evaluation

## Author
Vijaya Lakshmi, B.E CSE
GitHub: github.com/Viji-codes
