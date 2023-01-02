import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Read the data
X_full = pd.read_csv("train.csv", index_col="Id")
X_test_full = pd.read_csv("test.csv", index_col="Id")

# Remove missing target
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = X_full.SalePrice
X_full.drop(["SalePrice"], axis=1, inplace=True)

# Break off vdalidation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y, train_size=0.8, test_size=0.2, random_state=0
)

# find cardinality
categorical_cols = [
    col
    for col in X_train_full.columns
    if X_train_full[col].nunique() < 10 and X_train_full[col].dtype == "Object"
]
numerical_cols = [
    col
    for col in X_train_full.columns
    if X_train_full[col].dtype in ["int64", "float64"]
]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()


X_train.head()


# Preprocessing for numerical data
# SimpleImputer() -> Approach 2 (Imputation) 用其他数值代替
numerical_transformer = SimpleImputer(strategy="constant")

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print("MAE:", mean_absolute_error(y_valid, preds))
# MAE: 17861.780102739725


# ===================== Step 1: Improve the performance
# Part A
# define the own preprocessing steps and random forest model.

# Preprocessing for numerical data
# 数值型数据的预处理
numerical_transformer = SimpleImputer(
    # strategy='constant') # MAE too high
    strategy="median"
)

# Preprocessing for categorical data
# 分类型数据的预处理
# 分类数据处理有两部分：填补和编码，可以用管道捆绑
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

# Bundle preprocessing for numerical and categorical data
# 用ColumnTransformer捆绑数值型和分类型数据的预处理
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)


# Part B
# have defined a pipeline in Part A that achieves lower MAE than the code above.
# You're encouraged to take the time here and try out many different approaches, to see how low you can get the MAE!

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print("MAE:", score)


# Step 2: Generate test predictions
# use the trained model to generate predictions with the test data.

# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print("MAE:", score)
