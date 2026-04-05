# Customer-Churn-Analysis-With-Maven-Dataset
# 📊 Customer Churn Analysis using Machine Learning

## 🚀 Project Overview
This project focuses on analyzing customer churn using Machine Learning techniques. The goal is to identify key factors that influence customer churn and build predictive models to classify whether a customer is likely to churn.

A Streamlit dashboard is developed to visualize insights and model performance interactively.

---

## 🎯 Objectives
- Understand customer behavior through data analysis
- Identify key factors affecting churn (cost, services, tenure, etc.)
- Build predictive ML models
- Compare model performance
- Visualize results using an interactive UI

---

## 📁 Dataset
Dataset used:
- Telecom Customer Churn Dataset (Maven Analytics)

Features include:
- Customer demographics (Age, Gender, Location)
- Services (Internet, Tech Support, Streaming)
- Billing (Monthly Charges, Total Charges)
- Customer status (Churned / Stayed)

---

## 🧠 Machine Learning Models Used

### 📈 Logistic Regression
- Used as a baseline model
- Helps interpret feature influence using weights

### 🌳 Decision Tree
- Provides rule-based understanding of churn
- Easy to interpret but prone to overfitting

### 🌲 Random Forest (Main Model)
- Combines multiple decision trees
- Provides better accuracy and generalization
- Used for feature importance analysis

---

## ⚙️ Data Preprocessing
- Handled missing values (categorical → "Unknown", numerical → median)
- Dropped irrelevant columns (Customer ID, Churn Reason, etc.)
- Converted categorical data using One-Hot Encoding
- Scaled data for better model performance

---

## 📊 Model Performance

| Model | Accuracy |
|------|---------|
| Logistic Regression | ~78% |
| Decision Tree | ~81% |
| Random Forest | ~83% |

---

## 🔥 Key Insights

- Customers with **higher monthly charges** are more likely to churn
- **Short tenure customers** have higher churn rates
- Lack of **technical support and services** increases churn probability
- Contract type and engagement play a major role in retention

---

## 📊 Streamlit Dashboard Features

- Dataset overview
- Churn distribution visualization
- Model performance display
- Feature importance analysis

---

## ▶️ How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
