
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
    skipinitialspace=True,
    na_values='?'
)
test = pd.read_csv(
    test_url,
    names=columns,
    skiprows=1,
    skipinitialspace=True,
    na_values='?'
)

# GDP per capita values supplied for each native-country category.
gdp_per_capita = {
    'United-States': 27694.85,
    'Cambodia': 302.13,
    'England': 21443,
    'Puerto-Rico': 10876.42,
    'Canada': 19997.62,
    'Germany': 27201.96,
    'Outlying-US(Guam-USVI-etc)': 7675,
    'India': 347.73,
    'Japan': 39933.52,
    'Greece': 10937.02,
    'South': 10385.39,
    'China': 475.68,
    'Cuba': 2613.29,
    'Iran': 1167.76,
    'Honduras': 832.06,
    'Philippines': 1059.98,
    'Italy': 19382.20,
    'Poland': 2874.83,
    'Jamaica': 2202.86,
    'Vietnam': 229.86,
    'Mexico': 6202.37,
    'Portugal': 9977.32,
    'Ireland': 15902.93,
    'France': 23359.90,
    'Dominican-Republic': 1891.67,
    'Laos': 320.97,
    'Ecuador': 1996.86,
    'Taiwan': 12009.75,
    'Haiti': 1806.82,
    'Columbia': 2325.98,
    'Hungary': 4187.03,
    'Guatemala': 1292.67,
    'Nicaragua': 849.18,
    'Scotland': 19709,
    'Thailand': 2517.71,
    'Yugoslavia': 2113,
    'El-Salvador': 1343.93,
    'Trinadad&Tobago': 3839.96,
    'Peru': 1879.37,
    'Hong': 22502.53,
    'Holand-Netherlands': 24682.59,
}

for dataset in (train, test):
    country = dataset['native-country'].str.strip().str.rstrip('.')
    dataset['GDP-per-capita'] = country.map(gdp_per_capita)
# Unknown countries use the numeric pipeline's training median for missing GDP.

# Remove unused features from both splits.
train = train.drop(columns=['education', 'fnlwgt'])
test = test.drop(columns=['education', 'fnlwgt'])

# Select features by name so dropping columns cannot shift their meaning.
numerical_features = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'GDP-per-capita'
]
categorical_features = [
    'workclass', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]
#continent, native country GDP, occupation scaling by average salary

for column in numerical_features:
    train[column] = pd.to_numeric(train[column], errors='coerce')
    test[column] = pd.to_numeric(test[column], errors='coerce')

# Keep the target out of the model inputs.
X_train = train.drop(columns=['income'])
X_test = test.drop(columns=['income'])

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

# Feature names are defined above and shared by both preprocessors.

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

# Tune the complete pipeline using only training data.
lr_param_grid = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__class_weight': [None, 'balanced']
}
lr_grid_search = GridSearchCV(
    lr_model,
    lr_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
lr_grid_search.fit(X_train, y_train)
lr_model = lr_grid_search.best_estimator_

print("\nLogistic Regression Best Parameters:")
print(lr_grid_search.best_params_)
print("Best Cross-Validation Accuracy:", lr_grid_search.best_score_)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate model
print("\nOptimized Logistic Regression Results")

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


svm_model = Pipeline(steps=[
    ('preprocessor', scaled_preprocessor),
    ('classifier', SVC(
        kernel='rbf',
        C=1.0,
        max_iter=-1,
        random_state=42
    ))
])

# Tune the complete pipeline using only training data.
svm_param_grid = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__class_weight': [None, 'balanced']
}
svm_grid_search = GridSearchCV(
    svm_model,
    svm_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
svm_grid_search.fit(X_train, y_train)
svm_model = svm_grid_search.best_estimator_

print("\nSupport Vector Machine Best Parameters:")
print(svm_grid_search.best_params_)
print("Best Cross-Validation Accuracy:", svm_grid_search.best_score_)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate model
print("\nOptimized Support Vector Machine Results")

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

models = ['Optimized Random Forest', 'Optimized Logistic Regression', 'Optimized SVM']

accuracies = [
    accuracy_score(y_test, y_pred_best),
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_svm)
]

for i in range(len(models)):
    print(models[i], ":", round(accuracies[i], 4))

# --------------------------------
# 14. Training vs Testing Accuracy
# --------------------------------

print("\nTRAINING VS TESTING ACCURACY")

model_names = [
    'Random Forest',
    'Optimized Random Forest',
    'Optimized Logistic Regression',
    'Optimized SVM'
]

trained_models = [
    rf_model,
    best_model,
    lr_model,
    svm_model
]

for i in range(len(trained_models)):

    train_accuracy = trained_models[i].score(
        X_train, y_train
    )

    test_accuracy = trained_models[i].score(
        X_test, y_test
    )

    print("\n", model_names[i])
    print("Training Accuracy:", round(train_accuracy, 4))
    print("Testing Accuracy:", round(test_accuracy, 4))
    print("Difference:", round(
        test_accuracy - train_accuracy, 4
    ))
