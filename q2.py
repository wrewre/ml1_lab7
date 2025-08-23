import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ----------------------
# Load dataset
# ----------------------
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv")
X = df.drop(columns=["LABEL"])
y = df["LABEL"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# Helper function to evaluate models
# ----------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute metrics
    results = {
        "Train Accuracy": accuracy_score(y_train, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Train Precision": precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "Test Precision": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "Train Recall": recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "Test Recall": recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "Train F1": f1_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "Test F1": f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    }
    return results

# ----------------------
# Classifier functions
# ----------------------
def run_svm():
    return evaluate_model(SVC(kernel='rbf', random_state=42), X_train, y_train, X_test, y_test)

def run_decision_tree():
    return evaluate_model(DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test)

def run_random_forest():
    return evaluate_model(RandomForestClassifier(random_state=42), X_train, y_train, X_test, y_test)

def run_adaboost():
    return evaluate_model(AdaBoostClassifier(random_state=42), X_train, y_train, X_test, y_test)

def run_naive_bayes():
    return evaluate_model(GaussianNB(), X_train, y_train, X_test, y_test)

def run_mlp():
    return evaluate_model(MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42), X_train, y_train, X_test, y_test)

def run_xgboost():
    return evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), X_train, y_train, X_test, y_test)

def run_catboost():
    return evaluate_model(CatBoostClassifier(verbose=0, random_state=42), X_train, y_train, X_test, y_test)

# ----------------------
# Run all models and collect results
# ----------------------
results = {}
results["SVM"] = run_svm()
results["Decision Tree"] = run_decision_tree()
results["Random Forest"] = run_random_forest()
results["AdaBoost"] = run_adaboost()
results["Naive Bayes"] = run_naive_bayes()
results["MLP"] = run_mlp()
results["XGBoost"] = run_xgboost()
results["CatBoost"] = run_catboost()

# Convert results into DataFrame for easy comparison
results_df = pd.DataFrame(results).T

# Print table
print("\nModel Performance Comparison:\n")
print(results_df)
