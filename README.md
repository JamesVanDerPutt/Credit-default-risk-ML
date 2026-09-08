# Credit-default-risk-ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import shap
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Load user dataset
# ---------------------------------------------------------

print("Enter the path to your dataset file (CSV or Excel):")
file_path = input().strip()

try:
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("File must be .csv, .xls, or .xlsx")

except Exception as e:
    print(f"Error loading file: {e}")
    exit()

print("\nFile loaded successfully!")
print("Columns detected:")
print(df.columns.tolist())
print()

# ---------------------------------------------------------
# 2. User selects feature columns + target column
# ---------------------------------------------------------

print("Enter feature column names (comma-separated):")
feature_cols = [c.strip() for c in input().split(",")]

print("Enter the target column name:")
target_col = input().strip()

# Validate
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"Error: Missing feature columns: {missing}")
    exit()

if target_col not in df.columns:
    print(f"Error: Target column '{target_col}' not found.")
    exit()

# ---------------------------------------------------------
# 3. Prepare data
# ---------------------------------------------------------

X = df[feature_cols]
y = df[target_col]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Define models + hyperparameters
# ---------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10]
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    },
    "Gradient Boosting": {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200]
    }
}

# ---------------------------------------------------------
# 5. Train + evaluate models
# ---------------------------------------------------------

results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")

    grid = GridSearchCV(model, param_grids[name], cv=3, scoring="roc_auc")
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"Best params: {grid.best_params_}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    results[name] = {
        "model": best_model,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    }

# ---------------------------------------------------------
# 6. Select best model
# ---------------------------------------------------------

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = results[best_name]["model"]

print(f"\n=== BEST MODEL: {best_name} ===")
print(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}")

# ---------------------------------------------------------
# 7. Plot ROC curves
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. Feature importance (Random Forest or Gradient Boosting)
# ---------------------------------------------------------

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), X.columns[idx], rotation=90)
    plt.title(f"Feature Importance ({best_name})")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 9. SHAP explainability
# ---------------------------------------------------------

print("\nGenerating SHAP values...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Predict new customer profile
# ---------------------------------------------------------

print("\nWould you like to enter a new customer profile? (yes/no)")
if input().strip().lower() == "yes":

    new_data = {}
    print("\nEnter values for each feature column:")

    for col in feature_cols:
        val = input(f"{col}: ").strip()
        new_data[col] = val

    new_df = pd.DataFrame([new_data])
    new_df = pd.get_dummies(new_df)
    new_df = new_df.reindex(columns=X.columns, fill_value=0)

    new_scaled = scaler.transform(new_df)
    prob = best_model.predict_proba(new_scaled)[0][1]

    print(f"\nPredicted probability of default: {prob:.4f}")

