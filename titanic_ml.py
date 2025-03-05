# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
train = pd.read_csv("train.csv")

# Drop unnecessary columns
train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing values
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train["Fare"] = train["Fare"].fillna(train["Fare"].median())

# Convert categorical variables
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)

# Bin Age into categories
bins = [0, 12, 18, 35, 50, 80]
labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
train["AgeGroup"] = pd.cut(train["Age"], bins=bins, labels=labels)
train = pd.get_dummies(train, columns=["AgeGroup"], drop_first=True)

# Bin Fare into categories
train["FareGroup"] = pd.qcut(train["Fare"], 4, labels=["Low", "Medium", "High", "Very High"])
train = pd.get_dummies(train, columns=["FareGroup"], drop_first=True)

# Drop original Age and Fare columns
train.drop(["Age", "Fare"], axis=1, inplace=True)

# Ensure all columns are numeric
train = train.astype(int)

# Split dataset into features (X) and target variable (y)
X = train.drop("Survived", axis=1)
y = train["Survived"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")

# Feature Importance Plot
importances = model.feature_importances_
features = np.array(X.columns)

# Sort features by importance
sorted_indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.bar(features[sorted_indices], importances[sorted_indices])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance in Decision Tree")
plt.xticks(rotation=45)
plt.show()
