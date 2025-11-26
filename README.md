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
├── 📓 Loan Default Prediction model.ipynb        ← Full Notebook
├── loan_default_app.py                 ← Streamlit Prediction App
└── loan_default_pipeline.pkl                  ← Saved LightGBM Model
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
streamlit run loan_default_app.py
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

