
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. Load the dataset
# --------------------------------

train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

columns = [
    'age', 'workclass', 'fnlwgt', 'education',
    'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

train = pd.read_csv(
    train_url,
    names=columns,
    skipinitialspace=True
)
test = pd.read_csv(
    test_url,
    names=columns,
    skiprows=1,
    skipinitialspace=True
)

# Treat the dataset's '?' missing-value markers as actual NaNs.
train = train.replace('?', pd.NA)
test = test.replace('?', pd.NA)

# Convert numeric columns in each split, keeping their original feature order.
numerical_indices = [0, 2, 4, 10, 11, 12]
for index in numerical_indices:
    column = columns[index]
    train[column] = pd.to_numeric(train[column], errors='coerce')
    test[column] = pd.to_numeric(test[column], errors='coerce')

# Use the official UCI train and test files as separate datasets.
X_train = train.iloc[:, :14].to_numpy(dtype=object)
X_test = test.iloc[:, :14].to_numpy(dtype=object)
y_train = train['income'].str.strip().map({'<=50K': 0, '>50K': 1}).to_numpy()
y_test = (
    test['income'].str.strip().str.rstrip('.').map({'<=50K': 0, '>50K': 1}).to_numpy()
)

print("Dataset loaded successfully")
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

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
# Additional preprocessing for Logistic Regression and SVM
# --------------------------------

numerical_transformer_scaled = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer_scaled = Pipeline(steps=[
    ('imputer', SimpleImputer(
        strategy='constant',
        fill_value='Missing'
    )),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

scaled_preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer_scaled, numerical_features),
    ('cat', categorical_transformer_scaled, categorical_features)
])

# --------------------------------
# 4. Use the supplied train-test split
# --------------------------------

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

# --------------------------------
# 11. Logistic Regression
# --------------------------------

lr_model = Pipeline(steps=[
    ('preprocessor', scaled_preprocessor),
    ('classifier', LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# Train model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate model
print("\nLogistic Regression Results")

print("Training Accuracy:",
      lr_model.score(X_train, y_train))

print("Testing Accuracy:",
      accuracy_score(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_lr,
    target_names=['<=50K', '>50K']
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# --------------------------------
# 12. Support Vector Machine
# --------------------------------

from sklearn.svm import LinearSVC

svm_model = Pipeline(steps=[
    ('preprocessor', scaled_preprocessor),
    ('classifier', LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    ))
])

# Train model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate model
print("\nSupport Vector Machine Results")

print("Training Accuracy:",
      svm_model.score(X_train, y_train))

print("Testing Accuracy:",
      accuracy_score(y_test, y_pred_svm))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_svm,
    target_names=['<=50K', '>50K']
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# --------------------------------
# 13. Model Comparison
# --------------------------------

print("\nMODEL COMPARISON")

models = ['Random Forest', 'Logistic Regression', 'SVM']

accuracies = [
    accuracy_score(y_test, y_pred_best),
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_svm)
]

for i in range(len(models)):
    print(models[i], ":", round(accuracies[i], 4))
