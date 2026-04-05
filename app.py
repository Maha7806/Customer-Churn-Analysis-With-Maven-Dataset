import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Customer Churn Analysis Dashboard")
st.write("Analyze and predict customer churn using ML 🚀")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("shilongzhuang/telecom-customer-churn-by-maven-analytics")
    file = os.path.join(path, "telecom_customer_churn.csv")
    df = pd.read_csv(file)
    return df

df = load_data()

# -----------------------------
# PREPROCESSING
# -----------------------------
df['Churn'] = df['Customer Status'].apply(lambda x: 1 if x == 'Churned' else 0)

df.drop(['Customer ID', 'Churn Category', 'Churn Reason', 'Customer Status'], axis=1, inplace=True)

# Handle missing values
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64','float64']).columns

df[cat_cols] = df[cat_cols].fillna("Unknown")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Encode
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# SPLIT DATA
# -----------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (for safety)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose Section", [
    "Dataset Overview",
    "EDA",
    "Model Performance",
    "Feature Importance"
])

# -----------------------------
# 1. DATASET OVERVIEW
# -----------------------------
if option == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Shape")
    st.write(df.shape)

# -----------------------------
# 2. EDA
# -----------------------------
elif option == "EDA":
    st.subheader("Churn Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)

    st.subheader("Monthly Charge vs Churn")

    fig, ax = plt.subplots()
    sns.boxplot(x=y, y=df['Monthly Charge'], ax=ax)
    st.pyplot(fig)

# -----------------------------
# 3. MODEL PERFORMANCE
# -----------------------------
elif option == "Model Performance":
    st.subheader("Model Accuracy")
    st.success(f"Random Forest Accuracy: {accuracy:.2f}")

# -----------------------------
# 4. FEATURE IMPORTANCE
# -----------------------------
elif option == "Feature Importance":
    st.subheader("Top 10 Important Features")

    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()
    top_features.plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)

    st.write(top_features)