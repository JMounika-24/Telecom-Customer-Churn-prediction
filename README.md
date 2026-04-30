# Telecom-Customer-Churn-prediction

Project/Ttile

📉 Telecom Customer Churn Prediction
 End-to-End Machine Learning & Data Analysis Project in Python

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4c72b0)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-f7931e?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete end-to-end machine learning project predicting which telecom customers are likely to churn — covering data loading, cleaning, exploratory data analysis, visualization, preprocessing, and multi-model comparison — using the IBM Watson Telco Customer Churn dataset of 7,043 customers and 21 features.

---

 📌 Business Problem

> "How can a telecom company identify which customers are at risk of leaving — before they leave?"

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a predictive system that flags high-risk customers so the business can intervene with targeted retention strategies.

Target Variable: `Churn` — whether the customer cancelled their service (`Yes` / `No`)

---

 🗂️ Project Structure

```
telecom-customer-churn/
│
├── 📓 Telecom_Customer_Churn_Analysis.ipynb   # Main Jupyter notebook
├── 📁 Customer_Churn.csv                      # Raw dataset (7,043 rows × 21 columns)
└── 📄 README.md                               # Project documentation
```

---

 🛠️ Tech Stack

| Layer | Tool / Library | Purpose |
|---|---|---|
| Data Manipulation | `pandas`, `numpy` | Loading, cleaning, feature engineering, stats |
| Visualization | `matplotlib`, `seaborn` | EDA charts, distribution plots, heatmaps |
| Missing Values | `missingno` | Visual missing value matrix |
| Preprocessing | `sklearn.preprocessing` | Label encoding, standard scaling |
| ML Models | `scikit-learn` | KNN, SVC, Random Forest, Logistic Regression, Decision Tree, AdaBoost, Gradient Boosting, Voting Classifier |
| Evaluation | `sklearn.metrics` | Accuracy, Precision, Recall, F1-Score |
| Environment | Jupyter Notebook | Interactive development |

---

 📊 Dataset Overview

| Attribute | Details |
|---|---|
| Source | IBM Watson Telco Customer Churn Dataset |
| Records | 7,043 customers |
| Features | 21 columns |
| Target | `Churn` (Yes / No) |
| Churn Rate | 26.5% (1,869 churned / 5,174 retained) |

 Feature Categories

| Category | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account Info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Phone Services | PhoneService, MultipleLines |
| Internet Services | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

 🔬 Project Workflow

 Stage 1 — Data Loading & Exploration

```python
import pandas as pd
import numpy as np

df = pd.read_csv('Customer Churn.csv')
print(f"Shape: {df.shape}")        # (7043, 21)
df.info()
df[['tenure', 'MonthlyCharges']].describe()
```

Key findings from exploration:
- `TotalCharges` is stored as a string — needs conversion to numeric
- 11 rows have `tenure == 0` with null `TotalCharges` — new customers with no billing yet
- All other columns are clean with no missing values

---

 Stage 2 — Data Cleaning & Preprocessing

```python
# Drop non-predictive ID column
df = df.drop(['customerID'], axis=1)

# Fix TotalCharges dtype
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop 11 rows with tenure == 0 (new customers, no charge history)
df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)

# Fill remaining nulls with column mean
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Convert SeniorCitizen from 0/1 to readable labels
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
```

Cleaning steps performed:
- Dropped `customerID` (non-predictive)
- Converted `TotalCharges` from object to float64
- Removed 11 zero-tenure rows (no billing history)
- Filled 11 null values in `TotalCharges` with column mean
- Re-mapped `SeniorCitizen` integer flags to `Yes`/`No` labels

---

 Stage 3 — Exploratory Data Analysis & Visualization

Seven visualization charts were created using `matplotlib` and `seaborn` to uncover churn patterns:

 1. Churn & Gender Distribution (Pie Charts)
- Overall churn rate: 26.5%
- Gender has minimal impact — male and female churn rates are nearly identical

 2. Churn by Contract Type (Count Plot)
```python
sns.countplot(data=df, x='Contract', hue='Churn',
              palette={'No':'#66b3ff', 'Yes':'#ff6666'})
```
- Month-to-month customers churn at ~42% — by far the highest risk group
- Two-year contract customers churn at only ~2.8%

 3. Tenure Distribution by Churn (Histogram Overlay)
- Churned customers cluster heavily in the first 1–12 months
- Retained customers spread evenly across all tenure lengths
- Clear early-churn risk window: months 1–12 are critical

 4. Monthly & Total Charges vs Churn (Box Plots)
- Churners pay higher monthly charges (median ~$79 vs ~$65)
- Churners have lower total charges (shorter tenure = less accumulated spend)

 5. Internet Service & Payment Method (Count Plots)
- Fiber optic customers churn at ~41.9% vs DSL at ~18.9%
- Electronic check users churn at ~45.3% — highest of all payment methods
- Auto-pay customers (bank transfer / credit card) have the lowest churn rates

 6. Correlation Heatmap (Seaborn)
```python
le = LabelEncoder()
for col in df_num.select_dtypes(include='object').columns:
    df_num[col] = le.fit_transform(df_num[col])

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, mask=mask)
```
Key correlations with Churn:
- `Contract` → strong negative (longer contract = less churn)
- `tenure` → negative (longer tenure = less churn)
- `MonthlyCharges` → positive (higher charges = more churn)
- `OnlineSecurity` / `TechSupport` → negative (add-ons reduce churn)

---

 Stage 4 — Model Preprocessing

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Label encode all categorical columns
le2 = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le2.fit_transform(df_model[col])

# Standardize numeric features
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X[num_cols] = scaler.fit_transform(X[num_cols])

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

---

 Stage 5 — Model Training & Evaluation

Eight classification models were trained and evaluated using a shared helper function:

```python
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1-Score':  f1_score(y_test, y_pred)
    }
```

Models Trained:

| # | Model | Notes |
|---|---|---|
| 1 | K-Nearest Neighbors | `n_neighbors=11` |
| 2 | Support Vector Classifier | `kernel='rbf'` |
| 3 | Random Forest | `n_estimators=100` |
| 4 | Logistic Regression | `max_iter=1000` |
| 5 | Decision Tree | Default depth |
| 6 | AdaBoost | `n_estimators=100` |
| 7 | Gradient Boosting | `n_estimators=100` |
| 8 | Voting Classifier | Ensemble of LR + RF + GB |

A final comparison bar chart was generated:

```python
results_df[['Accuracy','Precision','Recall','F1-Score']].plot(
    kind='bar', colormap='Set2', edgecolor='black')
```

---

 📈 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Voting Classifier | ~0.81 | ~0.67 | ~0.55 | ~0.60 |
| Gradient Boosting | ~0.80 | ~0.65 | ~0.55 | ~0.59 |
| Random Forest | ~0.80 | ~0.65 | ~0.50 | ~0.57 |
| Logistic Regression | ~0.79 | ~0.63 | ~0.54 | ~0.58 |
| SVC | ~0.79 | ~0.64 | ~0.51 | ~0.57 |
| AdaBoost | ~0.79 | ~0.61 | ~0.54 | ~0.57 |
| KNN | ~0.77 | ~0.58 | ~0.50 | ~0.54 |
| Decision Tree | ~0.73 | ~0.50 | ~0.51 | ~0.50 |

> ⭐ Best Model: Voting Classifier (Ensemble of Logistic Regression + Random Forest + Gradient Boosting) achieved the highest overall accuracy.

---

 💡 Key Business Insights

| Priority | Finding | Recommended Action |
|---|---|---|
| 🔴 Critical | Month-to-month churn = 42% | Offer 10–20% discount to switch to annual contracts |
| 🔴 Critical | First 12 months churn = 48% | Dedicated onboarding calls & early check-ins |
| 🟠 High | Fiber optic churn = 42% | Investigate service quality complaints & pricing |
| 🟠 High | Electronic check churn = 45% | Incentivise auto-pay enrollment with a small discount |
| 🟡 Medium | Senior citizens churn = 42% | Senior-specific support plans and simplified billing |
| 🟡 Medium | No add-on services → higher churn | Bundle OnlineSecurity & TechSupport with base plans |
| 🟢 Low | Gender has no churn impact | Remove from segmentation models — no action needed |

---

 🚀 How to Run This Project

 Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno jupyter
```

 Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/telecom-customer-churn.git
cd telecom-customer-churn

# 2. Launch Jupyter Notebook
jupyter notebook Telecom_Customer_Churn_Analysis.ipynb

# 3. Run all cells top to bottom (Kernel → Restart & Run All)
```

Make sure `Customer_Churn.csv` is in the same directory as the notebook.

---

 📁 File Descriptions

| File | Description |
|---|---|
| `Telecom_Customer_Churn_Analysis.ipynb` | Full Jupyter notebook: EDA, cleaning, visualization, preprocessing, 8 ML models, comparison |
| `Customer_Churn.csv` | Raw dataset — 7,043 customer records, 21 features |
| `README.md` | Project documentation |

---

 🙋 About

Aspiring Data Analyst — building end-to-end projects in Python, SQL, and Power BI.

> ⭐ If you found this project helpful, please consider giving it a star!
