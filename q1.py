import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings('ignore')

def hyperparameter_tuning_analysis(file_path):
    df = pd.read_csv(file_path).head(150)
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    class_counts = y.value_counts()
    print("Class distribution:", class_counts.to_dict())
    min_class_size = 2
    valid_classes = class_counts[class_counts >= min_class_size].index
    if len(valid_classes) < len(class_counts):
        print("Removing classes with <", min_class_size, "samples")
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]
        print("Dataset reduced to", len(y), "samples with classes:", y.value_counts().to_dict())
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        stratify_used = True
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        stratify_used = False
        print("Warning: Using regular split instead of stratified split due to class imbalance")
    dataset_info = {
        'total_samples': len(y),
        'total_features': X.shape[1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'class_distribution': y.value_counts().to_dict(),
        'stratify_used': stratify_used
    }
    perceptron_params = {
        'penalty': [None, 'l2', 'l1'],
        'alpha': uniform(0.001, 0.01),
        'max_iter': [500, 1000],
        'eta0': [0.1, 1.0],
        'early_stopping': [True, False]
    }
    perceptron = Perceptron(random_state=42)
    start_time = time.time()
    perceptron_search = RandomizedSearchCV(
        estimator=perceptron,
        param_distributions=perceptron_params,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    perceptron_search.fit(X_train, y_train)
    perceptron_time = time.time() - start_time
    perceptron_pred = perceptron_search.predict(X_test)
    perceptron_accuracy = accuracy_score(y_test, perceptron_pred)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    start_time = time.time()
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=8,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rf_search.fit(X_train, y_train)
    rf_time = time.time() - start_time
    rf_pred = rf_search.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(random_state=42)
    start_time = time.time()
    svm_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=svm_params,
        n_iter=6,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    svm_search.fit(X_train, y_train)
    svm_time = time.time() - start_time
    svm_pred = svm_search.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    models_to_analyze = {
        'Tuned_Perceptron': perceptron_search.best_estimator_,
        'Tuned_Random_Forest': rf_search.best_estimator_,
        'Tuned_SVM': svm_search.best_estimator_
    }
    cv_results = {}
    for name, model in models_to_analyze.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        cv_results[name] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
    feature_importance = None
    if hasattr(rf_search.best_estimator_, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
    results_comparison = {
        'Perceptron': {'cv_score': perceptron_search.best_score_, 'test_accuracy': perceptron_accuracy, 'time': perceptron_time},
        'Random_Forest': {'cv_score': rf_search.best_score_, 'test_accuracy': rf_accuracy, 'time': rf_time},
        'SVM': {'cv_score': svm_search.best_score_, 'test_accuracy': svm_accuracy, 'time': svm_time}
    }
    best_model_name = max(results_comparison.keys(), key=lambda x: results_comparison[x]['test_accuracy'])
    if best_model_name == 'Perceptron':
        best_model = perceptron_search.best_estimator_
        best_params = perceptron_search.best_params_
        best_predictions = perceptron_pred
    elif best_model_name == 'Random_Forest':
        best_model = rf_search.best_estimator_
        best_params = rf_search.best_params_
        best_predictions = rf_pred
    else:
        best_model = svm_search.best_estimator_
        best_params = svm_search.best_params_
        best_predictions = svm_pred
    confusion_mat = confusion_matrix(y_test, best_predictions)
    class_report = classification_report(y_test, best_predictions)
    return {
        'dataset_info': dataset_info,
        'model_results': {
            'perceptron': {
                'best_params': perceptron_search.best_params_,
                'cv_score': perceptron_search.best_score_,
                'test_accuracy': perceptron_accuracy,
                'training_time': perceptron_time
            },
            'random_forest': {
                'best_params': rf_search.best_params_,
                'cv_score': rf_search.best_score_,
                'test_accuracy': rf_accuracy,
                'training_time': rf_time
            },
            'svm': {
                'best_params': svm_search.best_params_,
                'cv_score': svm_search.best_score_,
                'test_accuracy': svm_accuracy,
                'training_time': svm_time
            }
        },
        'cv_analysis': cv_results,
        'feature_importance': feature_importance,
        'best_model': {
            'name': best_model_name,
            'model': best_model,
            'parameters': best_params,
            'confusion_matrix': confusion_mat,
            'classification_report': class_report
        },
        'comparison_summary': results_comparison
    }

file_path = r"DCT_mal.csv"
results = hyperparameter_tuning_analysis(file_path)
print("=" * 60)
print("A2: HYPERPARAMETER TUNING USING RANDOMIZEDSEARCHCV")
print("=" * 60)
print("\nDataset Information:")
print("Total Samples:", results['dataset_info']['total_samples'])
print("Total Features:", results['dataset_info']['total_features'])
print("Training Size:", results['dataset_info']['train_size'])
print("Test Size:", results['dataset_info']['test_size'])
print("Class Distribution:", results['dataset_info']['class_distribution'])
print("Stratified Split Used:", results['dataset_info']['stratify_used'])
print("\nModel Performance Summary:")
print("=" * 60)
for model_name, metrics in results['model_results'].items():
    print(model_name.upper() + ":")
    print("  Best CV Score:", round(metrics['cv_score'], 4))
    print("  Test Accuracy:", round(metrics['test_accuracy'], 4))
    print("  Training Time:", round(metrics['training_time'], 2), "s")
    print("  Best Parameters:", metrics['best_params'])
    print()
print("Cross-Validation Analysis (3-fold):")
print("=" * 60)
for model_name, cv_data in results['cv_analysis'].items():
    print(model_name + ":")
    print("  Mean CV Score:", round(cv_data['mean'], 4), "(+/-", round(cv_data['std'] * 2, 4), ")")
    scores_str = [str(round(score, 4)) for score in cv_data['scores']]
    print("  Individual Scores:", scores_str)
    print()
if results['feature_importance'] is not None:
    print("Top 10 Most Important Features (Random Forest):")
    print("=" * 60)
    for idx, row in results['feature_importance'].head(10).iterrows():
        print(row['feature'] + ":", round(row['importance'], 4))
    print()
best = results['best_model']
print("Best Performing Model:", best['name'])
print("=" * 60)
print("Best Parameters:", best['parameters'])
print()
print("Confusion Matrix:")
print(best['confusion_matrix'])
print()
print("Classification Report:")
print(best['classification_report'])
print("Model Ranking by Test Accuracy:")
print("=" * 60)
sorted_models = sorted(results['comparison_summary'].items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
for i, (model, metrics) in enumerate(sorted_models, 1):
    print(str(i) + ".", model, "| Test Accuracy:", round(metrics['test_accuracy'], 4), "| CV Score:", round(metrics['cv_score'], 4), "| Time:", round(metrics['time'], 2), "s")

