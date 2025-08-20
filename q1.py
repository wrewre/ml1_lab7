import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# Load your dataset
df = pd.read_excel("/mnt/data/DCT_MAL(shortened_version).xlsx")

# Separate features (X) and target (y)
X = df.drop("target", axis=1)   # replace "target" with your actual target column name
y = df["target"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Use RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf, 
    param_distributions=param_dist,
    n_iter=30,  # number of parameter settings to try
    cv=5,       # 5-fold cross-validation
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)

# Get best parameters
print("Best Parameters:", random_search.best_params_)

# Evaluate on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
