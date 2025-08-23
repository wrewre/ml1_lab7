import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Try importing XGBoost and CatBoost safely
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

try:
    from catboost import CatBoostClassifier
    cat_available = True
except ImportError:
    cat_available = False


data = pd.read_csv("DCT_mal.csv").head(150)

# Drop rows with NaN values
data = data.dropna()

# Separate features and labels
X = data.drop("LABEL", axis=1)
y = data["LABEL"]

# Encode labels into consecutive integers
y = LabelEncoder().fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Helper function to calculate metrics
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "Test Precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "Train Recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "Test Recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "Train F1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "Test F1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    }

# Individual functions for each classifier
def run_svm():
    # Support Vector Machine classifier
    return evaluate_model(SVC(kernel="linear", random_state=42))

def run_decision_tree():
    # Decision Tree classifier
    return evaluate_model(DecisionTreeClassifier(random_state=42))

def run_random_forest():
    # Random Forest classifier
    return evaluate_model(RandomForestClassifier(random_state=42))

def run_adaboost():
    # AdaBoost classifier
    return evaluate_model(AdaBoostClassifier(random_state=42))

def run_naive_bayes():
    # Naive Bayes classifier
    return evaluate_model(GaussianNB())

def run_mlp():
    # Multi-layer Perceptron classifier
    return evaluate_model(MLPClassifier(max_iter=500, random_state=42))

def run_xgboost():
    # XGBoost classifier (only if available)
    if xgb_available:
        return evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    else:
        return {m: None for m in ["Train Accuracy","Test Accuracy","Train Precision","Test Precision","Train Recall","Test Recall","Train F1","Test F1"]}

def run_catboost():
    # CatBoost classifier (only if available)
    if cat_available:
        return evaluate_model(CatBoostClassifier(verbose=0, random_state=42))
    else:
        return {m: None for m in ["Train Accuracy","Test Accuracy","Train Precision","Test Precision","Train Recall","Test Recall","Train F1","Test F1"]}

# Run all classifiers and collect results
results = {}
results["SVM"] = run_svm()
results["Decision Tree"] = run_decision_tree()
results["Random Forest"] = run_random_forest()
results["AdaBoost"] = run_adaboost()
results["Naive Bayes"] = run_naive_bayes()
results["MLP"] = run_mlp()
results["XGBoost"] = run_xgboost()
results["CatBoost"] = run_catboost()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).T
print("\nClassification Results:\n")
print(results_df.round(4))
