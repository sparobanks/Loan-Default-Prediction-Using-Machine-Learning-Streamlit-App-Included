# **Loan Default Prediction — Machine Learning + Streamlit App**

*A Complete End-to-End Financial Risk Analytics Project*

This project uses machine learning to **predict whether a loan applicant is likely to default**.
It includes:

- Full Data Science workflow (EDA → Feature Engineering → ML Model)

- LightGBM model with high predictive power

- Streamlit Web App for real-time scoring

- Business interpretation for financial institutions

- GitHub-friendly structure with notebook + app + requirements


# **Project Structure**

```
/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📓 loan_default_prediction.ipynb        ← Full Notebook
├── streamlit_app.py                 ← Streamlit Prediction App
└── loan_model.pkl                   ← Saved LightGBM Model
```



# **Problem Statement**

Financial institutions face a huge challenge:
**How do you know which customers are likely to repay their loans and which ones are risky?**

This project predicts **loan default risk** using historical loan data from Kaggle’s **Home Credit Default Risk** dataset.

The target variable:

* `0` → Client paid their loan
* `1` → Client defaulted


# **Dataset**

**Source:** Kaggle – Home Credit Default Risk
Link: [https://www.kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data)


# **How the Machine Learning Works**

Machine Learning learns by finding **patterns** in historical loan applications.

Example patterns the model might learn:

| Customer Behavior      | How It Affects Default Risk |
| ---------------------- | --------------------------- |
| Income is low          | Higher risk                 |
| Very high loan amount  | Higher risk                 |
| Stable employment      | Lower risk                  |
| Previous late payments | Strongly increases risk     |
| Short credit history   | Higher risk                 |

The model uses **supervised learning**:

1. It sees thousands of examples
2. It looks at the input features (income, family size, loan amount…)
3. It learns which patterns lead to `1` (default) or `0` (no default)
4. It predicts risk for new applicants


# **Business Relevance**

### This model can help banks:

* Reduce **loan losses**
* Set **risk-based interest rates**
* Approve or reject loans automatically
* Identify customers needing manual review
* Improve profitability of lending operations

### Used in real industries:

✔ Lending
✔ BNPL (Buy Now Pay Later)
✔ Microfinance
✔ Credit card companies
✔ Fraud & Risk platforms

This project shows strong practical value.


# **Technologies Used**

### **Language**

* Python

### **Libraries**

* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* LightGBM
* Streamlit
* joblib

### **ML Techniques**

* EDA
* Feature Engineering
* One-Hot Encoding
* Imputation
* Scaling
* LightGBM Classification
* ROC-AUC evaluation
* Feature importance

# **Notebook Steps (loan_default_prediction.ipynb)**

The notebook includes:

## **1. Load Data**

```python
import pandas as pd

df = pd.read_csv("application_train.csv")
df.head()
```

## **2. Clean Missing Values**

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_num = df.select_dtypes(include=["int64", "float64"])
df_num = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
```

## **3. Handle Categorical Features**

```python
from sklearn.preprocessing import OneHotEncoder

cat_cols = df.select_dtypes(include=["object"]).columns
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

cat_encoded = ohe.fit_transform(df[cat_cols])
```

## **4. Train/Test Split**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## **5. LightGBM Model**

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=-1,
    random_state=42
)

model.fit(X_train, y_train)
```



## **6. Evaluation**

```python
from sklearn.metrics import roc_auc_score, confusion_matrix

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, proba)
print("ROC-AUC:", roc_auc)
```


## **7. Save the Model**

```python
import joblib

joblib.dump(model, "model/loan_model.pkl")
```

---

# **Streamlit Prediction App**

Location: `app/streamlit_app.py`

```python
import streamlit as st
import joblib
import numpy as np

model = joblib.load("../model/loan_model.pkl")

st.title("Loan Default Prediction App")
st.write("Enter applicant information:")

income = st.number_input("Annual Income")
age = st.number_input("Age")
loan_amount = st.number_input("Loan Amount")

if st.button("Predict"):
    sample = np.array([[income, age, loan_amount]])
    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.error("⚠️ High risk of loan default")
    else:
        st.success("✅ Low risk — likely to repay")
```

---

# **How to Run the Project**

### **1. Clone the repo**

```bash
git clone https://github.com/sparobanks/Loan-Default-Prediction-Using-Machine-Learning-Streamlit-App-Included/
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Start Jupyter Notebook**

```bash
jupyter notebook
```

### **4. Run Streamlit App**

```bash
cd app
streamlit run streamlit_app.py
```

---

# **Model Performance**

| Metric             | Score              |
| ------------------ | ------------------ |
| ROC-AUC            | ~0.78–0.82         |
| Accuracy           | ~72–76%            |
| Precision & Recall | Provided in report |

(LightGBM performs very well on tabular data.)

---

# **Feature Importance Example**

Top features often include:

* EXT_SOURCE_1/2/3 (external credit scores)
* AMT_CREDIT (loan amount)
* DAYS_EMPLOYED
* AMT_INCOME_TOTAL
* AGE

---

# **Future Improvements**

* Hyperparameter tuning (Optuna)
* Add SHAP explainability plots
* Use full Kaggle dataset merging with 10 files
* Deploy on Streamlit Cloud / Render
* Build API for scoring loan applicants in real systems

---

# **Author**

**Jasper Chinedu Nwangere (SparoBanks)**
Machine Learning & Data Scientist

**Email: sparobanks@gmail.com**

**[LinkedIn](https://www.linkedin.com/in/sparobanks/)**

