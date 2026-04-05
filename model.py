import kagglehub
path = kagglehub.dataset_download("shilongzhuang/telecom-customer-churn-by-maven-analytics")
print(path)
import pandas as pd
import os
files = os.listdir(path)
print(files)
df = pd.read_csv(os.path.join(path,files[0]))
print(df.info())
print(df.head())
print(df.columns)
print(df.shape[1])
print(df.isnull().sum())
print(df.nunique(dropna=False))
df["churn"] = df["Customer Status"].apply(lambda x: 1 if x == "Churned" else 0)
df.drop(["Customer ID" , "Churn Category" , "Customer Status", "Churn Reason"] , axis = 1 , inplace = True)
# Categorical columns
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# Numerical columns
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = pd.get_dummies(df, drop_first = True)
from sklearn.model_selection import train_test_split
x = df.drop("churn" , axis = 1)
y = df["churn"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 1000)
lr.fit(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train ,y_train)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train , y_train)
from sklearn.metrics import accuracy_score
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(x_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt.predict(x_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(x_test)))    
lr_weights = pd.Series(lr.coef_[0], index=x.columns)
print(lr_weights.sort_values(ascending=False).head(10))
importance = pd.Series(rf.feature_importances_, index=x.columns)
print(importance.sort_values(ascending=False).head(10))