# Santander-Customer-Transaction-Prediction
Work in  Progress

# Santander Customer Transaction Prediction

Binary classification project based on the Santander Kaggle competition dataset, focused on predicting whether a customer will make a future transaction using anonymized numerical features.

This project was developed for learning and experimentation purposes rather than leaderboard optimization. The main objective was to practice real-world machine learning workflows, model evaluation under class imbalance, hyperparameter optimization, and analytical interpretation of predictive models.

The project includes:
- Exploratory Data Analysis (EDA)
- Baseline and ensemble models
- Hyperparameter optimization with RandomizedSearchCV
- ROC-AUC and PR-AUC evaluation
- Threshold analysis and classification tradeoffs
- Model comparison between Logistic Regression, Random Forest and XGBoost

---

# Project Objectives

- Explore and understand the structure of an anonymized tabular dataset
- Compare baseline and ensemble classification models
- Evaluate model performance under class imbalance conditions
- Analyze ROC-AUC, PR-AUC, precision and recall tradeoffs
- Study the impact of classification thresholds on predictive behavior
- Practice end-to-end machine learning workflows and model evaluation

---

# Dataset

The dataset comes from the Santander Customer Transaction Prediction Kaggle competition and contains anonymized numerical features.

## Dataset Characteristics

- 200 anonymized numerical variables
- Binary target variable
- Imbalanced dataset
- No missing values detected
- Large tabular dataset focused on binary classification

The official competition metric was ROC-AUC.

---

# Model Performance

Three classification models were evaluated and compared:

- Logistic Regression (baseline model)
- Random Forest
- XGBoost

Performance was evaluated mainly using:
- ROC-AUC (official competition metric)
- PR-AUC (important under class imbalance)
- Recall and Precision
- Confusion Matrix analysis

## Model Comparison

| Model | Accuracy | ROC-AUC | PR-AUC | Recall (Class 1) |
|---|---|---|---|---|
| Logistic Regression | 0.783 | 0.860 | 0.500 | 0.78 |
| Random Forest | 0.903 | 0.832 | 0.424 | 0.05 |
| XGBoost | 0.904 | 0.876 | 0.541 | 0.51 |

## Key Observations

- Logistic Regression achieved strong recall despite being the simplest model.
- Random Forest produced high overall accuracy but performed poorly on the minority class due to severe class imbalance.
- XGBoost achieved the best overall ranking performance with the highest ROC-AUC and PR-AUC scores.
- PR-AUC proved especially useful for evaluating performance under imbalanced classification conditions.

---

# Hyperparameter Optimization

RandomizedSearchCV was used for Random Forest and XGBoost hyperparameter optimization using a stratified 10% sample of the dataset in order to reduce computational cost.

The best XGBoost configuration included:
- Regularization parameters (`gamma`, `reg_lambda`)
- Controlled tree depth
- Subsampling strategies
- Balanced class weighting

Future improvements may include:
- Bayesian optimization
- Additional feature engineering


