
import csv
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. Load the dataset
# --------------------------------

columns = [
    'age', 'workclass', 'fnlwgt', 'education',
    'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

X_data = []
y_data = []

with open(r'C:\Users\cusua\Downloads\Data\Data\ISE464\Projects\adult.data', 'r') as file:
    reader = csv.reader(file, skipinitialspace=True)

    for row in reader:

        if len(row) != 15:
            continue

        row = [value.strip() for value in row]

        # Separate target from features
        features = row[:14]
        income = row[14]

        # Convert numerical features
        numerical_indices = [0, 2, 4, 10, 11, 12]

        for index in numerical_indices:
            features[index] = float(features[index])

        X_data.append(features)

        if income == '<=50K':
            y_data.append(0)
        elif income == '>50K':
            y_data.append(1)

# Convert to NumPy arrays
X = np.array(X_data, dtype=object)
y = np.array(y_data)

print("Dataset loaded successfully")
print("Dataset shape:", X.shape)

# --------------------------------
# 2. Identify feature types
# --------------------------------

numerical_features = [0, 2, 4, 10, 11, 12]

categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]

# --------------------------------
# 3. Preprocessing
# --------------------------------

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(
        strategy='constant',
        fill_value='Missing'
    )),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# --------------------------------
# 4. Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------
# 5. Random Forest pipeline
# --------------------------------

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])

# --------------------------------
# 6. Train model
# --------------------------------

rf_model.fit(X_train, y_train)

# --------------------------------
# 7. Make predictions
# --------------------------------

y_pred = rf_model.predict(X_test)

# --------------------------------
# 8. Evaluate model
# --------------------------------

print("\nRandom Forest Results")

print("Training Accuracy:",
      rf_model.score(X_train, y_train))

print("Testing Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['<=50K', '>50K']
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------
# 9. Hyperparameter tuning
# --------------------------------

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# --------------------------------
# 10. Evaluate optimized model
# --------------------------------

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print("\nOptimized Random Forest Results")

print("Best Parameters:")
print(grid_search.best_params_)

print("Best Cross-Validation Accuracy:",
      grid_search.best_score_)

print("Optimized Testing Accuracy:",
      accuracy_score(y_test, y_pred_best))

print("\nOptimized Classification Report:")
print(classification_report(
    y_test,
    y_pred_best,
    target_names=['<=50K', '>50K']
))

print("\nOptimized Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))