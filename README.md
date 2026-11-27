# Telco Customer Churn Prediction — ML Classification Study

## a. Problem Statement
Customer attrition (churn) directly impacts recurring revenue in subscription-based telecommunications businesses. The objective of this project is to build, evaluate, and compare five supervised machine learning classification models to predict whether an existing telecom customer will discontinue their service (`Churn = Yes`) or remain retained (`Churn = No`) based on demographic, account tenure, billing, and subscribed service attributes. Early identification of high-risk customers enables targeted retention interventions.

## b. Dataset Description
* **Dataset Name & Source:** Telco Customer Churn Dataset ([IBM Sample Data / Kaggle Repository](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
* **Problem Type:** Binary Classification
* **Dataset Dimensions:** 7,043 instances and 19 predictive features (after removing the non-predictive `customerID` identifier column), satisfying the $\ge 1,000$ instances and $\ge 12$ features criteria.
* **Target Variable:** `Churn` (Encoded as `0` for `No` and `1` for `Yes`)
* **Feature Types:**
  * **Numerical Features (4):** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
  * **Categorical Features (15):** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
* **Missing Values & Duplicates:** Detected 11 missing values in `TotalCharges` (corresponding to brand-new customers with `tenure == 0`) and checked for duplicate records (0 exact duplicates found after dropping `customerID`).
* **Class Distribution & Imbalance:** The target variable exhibits moderate class imbalance: **73.46% `No` (5,174 instances)** vs. **26.54% `Yes` (1,869 instances)**. Because of this ~73:27 skew, Accuracy alone can be misleading, making **ROC-AUC** and **Matthews Correlation Coefficient (MCC)** critical evaluation metrics.
* **Data Preprocessing Pipeline:**
  * Applied a stratified 80:20 train-test split (`5,634` training samples, `1,409` testing samples) with a fixed random seed (`random_state=42`) to preserve exact class proportions.
  * Fitted all transformations strictly on the training set (`X_train`) via `ColumnTransformer` to prevent data leakage: median imputation + `StandardScaler` for numerical features, and most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical features.

## c. GitHub Repository Link
* **Repository URL:** [https://github.com/TapaswiniHaldar/tabular-churn-classifier](https://github.com/TapaswiniHaldar/tabular-churn-classifier)

### Execution Instructions
```bash
git clone [https://github.com/TapaswiniHaldar/tabular-churn-classifier.git](https://github.com/TapaswiniHaldar/tabular-churn-classifier.git)
cd tabular-churn-classifier
pip install -r requirements.txt
jupyter notebook telco_churn_classification.ipynb
