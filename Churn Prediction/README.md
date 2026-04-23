# 🔮 Telco Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**An end-to-end Machine Learning project to predict customer churn for a telecom company — from raw data to a live Streamlit web app.**

[Live Demo](#-live-demo) • [Project Structure](#-project-structure) • [Results](#-model-results) • [Run Locally](#-how-to-run)

</div>

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Losing a customer is far more expensive than retaining one. This project builds a machine learning system that:

- **Predicts** whether a customer is likely to churn
- **Explains** the key risk factors driving that prediction
- **Recommends** business actions to retain at-risk customers

> **Business Impact:** By identifying high-risk customers early, a telecom company can proactively offer retention deals — reducing revenue loss significantly.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,032 customers |
| Features | 21 original → 29 after engineering |
| Target | `Churn` (Yes / No) |
| Class Balance | 73.4% No Churn / 26.6% Churn (imbalanced) |

---

## 🔍 Key EDA Findings

| Finding | Insight |
|---|---|
| **Contract Type** | Month-to-month customers churn at ~43% vs only ~3% for 2-year contracts |
| **Tenure** | New customers (≤12 months) churn at 47.7% vs 17.1% for older customers |
| **Monthly Charges** | Churned customers pay ~$74/month avg vs ~$61 for loyal customers |
| **Services** | Customers with 0 services churn at ~52% vs ~5% with 6 services |
| **Payment Method** | Electronic check users have the highest churn rate |

---

## ⚙️ Feature Engineering

5 new features created from domain knowledge:

| Feature | Logic | Correlation with Churn |
|---|---|---|
| `charge_per_tenure` | MonthlyCharges / (tenure + 1) — cost efficiency | **+0.424** (strongest!) |
| `is_new_customer` | tenure ≤ 12 months → flag as 1 | **+0.320** |
| `has_no_protection` | No OnlineSecurity AND no TechSupport | **+0.183** |
| `total_services` | Count of add-on services subscribed | -0.088 |
| `payment_consistency` | TotalCharges / (tenure × MonthlyCharges) | -0.038 |

---

## 🤖 Model Results

4 models trained and compared on the same train-test split (80-20, stratified):

| Model | Accuracy | F1-Score | AUC-ROC | CV F1 (5-fold) |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | **80.03%** | **58.74%** | **83.81%** | ~57% |
| XGBoost | 76.90% | 54.42% | 81.13% | ~55% |
| Random Forest | 77.54% | 52.98% | 81.96% | ~53% |
| Decision Tree | 70.50% | 46.17% | 63.20% | ~46% |

> **Why Logistic Regression won?**
> The churn patterns in this dataset are largely linearly separable — contract type, tenure, and charges create clean decision boundaries. Complex models (XGBoost, RF) don't gain much with only ~7K samples. LR also benefits from being less prone to overfitting on this scale.

> **Why F1-Score over Accuracy?**
> With 73-27 class imbalance, a model that always predicts "No Churn" achieves 73% accuracy — yet catches zero churners. F1-Score and AUC-ROC are the real indicators of performance here.

---

## 🗂️ Project Structure

```
telco-churn-prediction/
│
├── app/
│   └── app.py                         ← Streamlit web application
│
├── data/
│   ├── Telco-Customer-Churn.csv       ← Raw dataset (original)
│   ├── telco_fully_cleaned.csv        ← After cleaning + feature engineering
│   ├── X_train.csv / X_test.csv       ← Train-test split (features)
│   └── y_train.csv / y_test.csv       ← Train-test split (target)
│
├── Interview/
│   └── Interview_Prep_Churn_Acknobit.docx  ← Interview Q&A document
│
├── models/
│   ├── best_model.pkl                 ← Logistic Regression (best model)
│   └── scaler.pkl                     ← StandardScaler (fitted)
│
├── notebook/
│   ├── 01_EDA.ipynb                   ← Exploratory Data Analysis
│   ├── 02_Data_Cleaning.ipynb         ← Cleaning + Preprocessing
│   ├── 03_Feature_Engineering.ipynb   ← 5 new features + scaling
│   ├── 04_Model_Building.ipynb        ← 4 models + evaluation
│   └── 05_Streamlit_App.ipynb         ← App deployment on Colab
│
├── plots/
│   ├── 01_churn_distribution.png
│   ├── 02_contract_vs_churn.png
│   ├── 03_tenure_vs_churn.png
│   ├── 04_charges_vs_churn.png
│   ├── 05_engineered_features.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   ├── 08_model_comparison.png
│   └── 09_feature_importance.png
│
├── src/
│   └── (helper scripts / utility functions)
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Option 1: Run on Google Colab (Recommended)

```
1. Clone or download this repository
2. Upload to Google Drive at: My Drive/Capstone Project/Churn Prediction/
3. Open notebooks from the notebook/ folder in order: 01 → 02 → 03 → 04 → 05
4. For the Streamlit app: get a free ngrok token from ngrok.com
5. Run 05_Streamlit_App.ipynb → public URL will be generated
```

### Option 2: Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order (notebook/01 → notebook/04)

# 4. Launch Streamlit app
streamlit run app/app.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
streamlit>=1.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
pyngrok>=5.1.0
```

---

## 🖥️ Live Demo

The Streamlit app takes customer information as input and returns:

- ✅ **Churn probability** (0–100%)
- ✅ **Risk classification** (High / Low)
- ✅ **Key risk factors** (what's driving the prediction)
- ✅ **Business recommendations** (what action to take)

**App Screenshots:**

> *(Add screenshots of your running app here)*
> Tip: Run the app → take screenshot → drag into this README on GitHub

---

## 📈 ML Pipeline Summary

```
Raw Data (7043 rows)
      ↓
Data Cleaning
  • TotalCharges: string → float
  • 11 blank rows dropped
  • customerID removed
  • Binary + OHE encoding
      ↓
Feature Engineering
  • 5 new features created
  • Train-Test Split (80-20, stratified)
  • StandardScaler applied
      ↓
Model Training
  • Logistic Regression ← Best
  • Decision Tree
  • Random Forest
  • XGBoost
      ↓
Evaluation
  • F1-Score: 58.74%
  • AUC-ROC: 83.81%
      ↓
Deployment
  • Streamlit Web App
  • Real-time predictions
```

---

## 💡 Key Learnings

- **Class imbalance** must be handled — accuracy is a misleading metric here
- **Feature engineering** matters more than model complexity — `charge_per_tenure` was the strongest predictor (correlation +0.424)
- **Data quality issues** hide in plain sight — `TotalCharges` stored as string with blank spaces caused multiple downstream errors
- **Simple models** can outperform complex ones on small, structured tabular datasets
- **Business context** drives metric choice — Recall matters more than Precision for churn (missing a churner costs more than a false alarm)

---

## 👨‍💻 Author

**Shaurab**
Acknobit — AI/ML Training & Coaching

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
<b>Built with ❤️ as part of Acknobit's Data Science Capstone Program</b>
</div>
