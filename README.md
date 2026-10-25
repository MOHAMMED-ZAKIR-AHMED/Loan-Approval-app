# 🏦 LoanLens: Loan Approval Predictor

An interactive **Streamlit** web app that predicts whether a loan application will be approved. Users pick one of five trained machine learning classifiers, enter applicant details, and instantly get an approval probability, personalised tips and model-by-model comparison.

🔗 **Live app:** `<https://loan-approval-classification-app.streamlit.app/>`
💻 **Code:** `<https://github.com/MOHAMMED-ZAKIR-AHMED/loan-approval-app>`

> ⚠️ This is a learning based project. Predictions are not real lending decisions.

---

## 📌 Table of Contents
1. [Features](#-features)
2. [Screenshots](#-screenshots)
3. [Dataset](#-dataset)
4. [Data Preprocessing](#-data-preprocessing)
5. [Models and Results](#-models-and-results)
6. [How the App Works](#-how-the-app-works)
7. [Project Structure](#-project-structure)
8. [Installation and Usage](#-installation-and-usage)
9. [Limitations](#-limitations)
10. [Tech Stack](#-tech-stack)

---

## ✨ Features

- **Model selector:** choose Logistic Regression, Decision Tree, Random Forest, XGBoost or KNN from the sidebar, with each model's metrics shown.
- **Approval probability gauge:** a clear Approved or Rejected verdict with a percentage, not just a yes/no.
- **All-model vote:** see what all five models say about the same applicant side by side.
- **Personalised tips:** flags risk factors such as a past default, a high loan-to-income ratio, a low credit score or a high interest rate.
- **What-if analysis:** see how the approval chance changes as the loan amount changes.
- **Feature importance:** shows which columns drive the selected model's decisions.
- **Model comparison dashboard:** metrics table, grouped bar chart and radar chart.
- **How It Works page:** a plain-English explanation of the app, the models and the metrics.
- **Prediction history:** every prediction in a session is logged and can be downloaded as CSV.
- **Input safety check:** warns when a request falls outside the range the models were trained on.

---

## 🖼️ Screenshots

 (<img width="1902" height="865" alt="Screenshot 2026-10-25 231125" src="https://github.com/user-attachments/assets/db64bc91-412c-4f46-a016-47c88fdba27b" />
 <img width="1897" height="837" alt="image" src="https://github.com/user-attachments/assets/dba9b983-d413-43ca-9aab-8cf7622bfc9e" />

---

## 📊 Dataset

- **Task:** binary classification
- **Rows:** about 45,000 loan applications
- **Target column:** `loan_status` (`1` = Approved, `0` = Rejected)
- **Class balance:** 35,000 rejected vs 10,000 approved (about 22% approved), so the data is imbalanced and F1 and recall matter alongside accuracy

### Input columns used for training (19 features)

| Column | Meaning |
|---|---|
| `person_age` | Applicant age |
| `person_education` | Education level (ordinal, 1 to 5) |
| `person_income` | Annual income |
| `person_emp_exp` | Years of work experience |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate (%) |
| `loan_percent_income` | Loan amount divided by income |
| `cb_person_cred_hist_length` | Length of credit history (years) |
| `credit_score` | Credit score |
| `previous_loan_defaults_on_file` | Past default on file (1 = yes, 0 = no) |
| `person_gender_male` | 1 = male, 0 = female |
| `person_home_ownership_OTHER` / `_OWN` / `_RENT` | One-hot home ownership (all 0 = Mortgage) |
| `loan_intent_EDUCATION` / `_HOMEIMPROVEMENT` / `_MEDICAL` / `_PERSONAL` / `_VENTURE` | One-hot loan purpose (all 0 = Debt Consolidation) |

---

## 🧹 Data Preprocessing

| Column type | Method |
|---|---|
| Education | Ordinal encoding: Associate = 1, High School = 2, Bachelor = 3, Master = 4, Doctorate = 5 |
| Previous default | Binary encoding: Yes = 1, No = 0 |
| Gender, home ownership, loan purpose | One-hot encoding with the first category dropped |
| Numeric columns | Used as they are (no scaling applied) |

The app rebuilds these exact transformations from the form inputs, so the models receive features in the same order and format they were trained on.

---

## 🤖 Models and Results

Five classifiers were trained on the same train/test split and evaluated on the test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.82 | 0.67 | 0.39 | 0.50 |
| Decision Tree | 0.90 | 0.77 | 0.78 | 0.78 |
| Random Forest | 0.92 | 0.89 | 0.77 | 0.83 |
| **XGBoost** | **0.93** | **0.89** | **0.80** | **0.844** |
| KNN (k = 3) | 0.82 | 0.62 | 0.53 | 0.57 |

**Takeaways**
- **XGBoost** is the best model overall, followed closely by Random Forest.
- **Logistic Regression** has the lowest recall (0.39): it misses many applicants who should be approved.
- **KNN** performs below the tree models, partly because it is distance-based and the features were not scaled.

### What the metrics mean
- **Accuracy:** share of all predictions that were correct.
- **Precision:** when the model says "Approved", how often it is right.
- **Recall:** of all truly approved applicants, how many the model found.
- **F1 Score:** a balance between precision and recall, useful for imbalanced data.

---

## 🧠 How the App Works

1. **User input:** the user fills in personal, loan and credit details in the form. Loan-as-%-of-income is calculated automatically.
2. **Feature building:** the inputs are converted into the 19 numeric columns the models expect (encoding, one-hot columns, correct column order).
3. **Prediction:** the chosen model returns a probability of approval using `predict_proba`. A probability of 50% or more is shown as **Approved**.
4. **Explanation:** the app shows a gauge, personalised tips, a vote from all five models, a what-if curve and feature importance.

---

## 📁 Project Structure

```
loan-approval-app/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── logistic_model.pkl      # Trained Logistic Regression
├── Decision_model.pkl      # Trained Decision Tree
├── Random_model.pkl        # Trained Random Forest
├── xgboost_model.pkl       # Trained XGBoost
├── knn_model.pkl           # Trained KNN
└── README.md
```

---

## 🚀 Installation and Usage

**1. Clone the repository**
```bash
git clone [https://github.com/MOHAMMED-ZAKIR-AHMED/loan-approval-app.git]
cd loan-approval-app
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Using your own trained models

Save your models from the notebook with the same file names:

```python
import joblib

joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(Decision_model, "Decision_model.pkl")
joblib.dump(Random_model,   "Random_model.pkl")
joblib.dump(xgboost_model,  "xgboost_model.pkl")
joblib.dump(Knn_model,      "Knn_model.pkl")
```

Use the same `scikit-learn` and `xgboost` versions as in `requirements.txt`, because models saved with one version may fail to load or behave differently in another.

---

## ⚠️ Limitations

- Trained on one dataset, so results may not reflect real lending decisions.
- Predictions are unreliable for values far outside the training range (for example a loan of 10 times the yearly income).
- Logistic Regression and KNN were trained without feature scaling, which limits their performance.
- Demographic columns such as gender are present in the data. A real system would need fairness checks before any use.
- The 50% decision threshold is fixed. With imbalanced data, a tuned threshold could improve recall.

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Plotly · Streamlit · Joblib · GitHub

---

## 👤 Author

**Mohammed Zakir Ahmed**
GitHub: [Zakir Ahmed](https://github.com/MOHAMMED-ZAKIR-AHMED/) · LinkedIn: [Zakir Ahmed](https://www.linkedin.com/in/zakir00/)

If you found this project useful, please ⭐ the repository!
