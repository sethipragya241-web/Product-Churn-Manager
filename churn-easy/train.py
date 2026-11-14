import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv("data/customers.csv")

# 2. Fill missing values
df = df.fillna(df.mean())

# 3. Split data: X = inputs, y = answer (churn)
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a normal Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("NORMAL MODEL")
print(classification_report(y_test, model.predict(X_test)))

# 5. Improve it with GridSearchCV (tries different settings)
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    {"max_depth": [3, 5, 7], "min_samples_leaf": [1, 5, 10]}, cv=5)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("TUNED MODEL", grid.best_params_)
print(classification_report(y_test, best.predict(X_test)))

# 6. Save the model
joblib.dump(best, "model.pkl")
