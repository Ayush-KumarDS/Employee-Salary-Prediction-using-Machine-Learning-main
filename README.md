# 💼 Employee Salary Prediction ML App

![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## 📌 Project Overview

This project predicts whether an employee earns **≤50K or >50K** using demographic and work-related features from the **UCI Adult dataset**, deployed as an interactive web app using **Streamlit**.

---

## 🚀 Features

- Data cleaning and preprocessing pipeline
- Trained multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (**Best: ~86% accuracy**)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Pipeline integration with scikit-learn
- Streamlit app with:
  - Sidebar input for employee features
  - Batch prediction via CSV upload
  - Professional background theme and clean UI

---

## 🛠️ Technologies Used

- **Languages & Libraries:** Python, pandas, numpy, seaborn, matplotlib
- **Machine Learning:** scikit-learn (Pipeline, ColumnTransformer, models)
- **Deployment:** Streamlit
- **Serialization:** joblib

---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **File:** `adult.csv`
- **Description:** Contains demographic and employment data for salary classification tasks.

---

## 🧠 Key Learnings

✔️ End-to-end ML project workflow  
✔️ Data preprocessing and handling missing values  
✔️ Feature engineering and encoding  
✔️ Model training, evaluation, and selection  
✔️ Deploying ML models as web apps using Streamlit

---

## 📝 Algorithm

1. Import libraries and load dataset
2. Replace missing values and drop duplicates
3. Remove outliers for age and educational-num
4. Encode categorical columns with LabelEncoder
5. Define feature set `x` and target `y`
6. Split data into training and test sets
7. Build a pipeline:
   - Numeric: StandardScaler
   - Categorical: OneHotEncoder
8. Train multiple models and evaluate their performance
9. Select and save the best model using joblib
10. Build and deploy Streamlit app with interactive UI for predictions

---

## 💻 How to Run Locally

1. **Clone this repo**

```bash
git clone https://github.com/yourusername/employee-salary-prediction.git
cd employee-salary-prediction
