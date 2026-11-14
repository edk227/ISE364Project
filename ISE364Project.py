import pandas as pd
#IMPORT DATA
train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

#Add column names
columns = [
    "age", "workclass", "pop_total", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

train = pd.read_csv(train_url, names=columns, skipinitialspace=True)
test = pd.read_csv(test_url, names=columns, skiprows=1, skipinitialspace=True)

#PRE-PROCESSING DATA
#Remove repeat information
train = train.drop(columns=["education"])
test = test.drop(columns=["education"])

#Find missing values
train = train.replace("?", pd.NA)
test = test.replace("?", pd.NA)
print("Train missing values:")
print(train.isna().sum().loc[lambda counts: counts > 0])
print("Test missing values:")
print(test.isna().sum().loc[lambda counts: counts > 0])

# One-hot Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
#remove target variable
X_train = train.drop(columns=["income"])
X_test = test.drop(columns=["income"])
y_train = train["income"]
y_test = test["income"].str.rstrip(".")  # test labels have a period

categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns # get categorical variables

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_encoded = encoder.fit_transform(X_train[categorical_cols]) #encode training data
test_encoded = encoder.transform(X_test[categorical_cols]) #encode test data
encoded_cols = encoder.get_feature_names_out(categorical_cols) # preserve column names and row indices
train_encoded = pd.DataFrame(
    train_encoded, columns=encoded_cols, index=X_train.index
)
test_encoded = pd.DataFrame(
    test_encoded, columns=encoded_cols, index=X_test.index
)

# combine rest of features with encoded features
X_train = pd.concat([X_train.drop(columns=categorical_cols), train_encoded], axis=1)
X_test = pd.concat([X_test.drop(columns=categorical_cols), test_encoded], axis=1)

print(X_train.head())

# Imputation

# Scaling Data

# MACHINE LEARNING MODELS
# Logistic Regression

# Random Forest

# SVM