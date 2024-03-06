# # Set up code checking
# import os
# if not os.path.exists("../input/train.csv"):
#     os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
#     os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
# from learntools.core import binder
# binder.bind(globals())
# from learntools.ml_intermediate.ex5 import *
# print("Setup Complete")


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# read the data
train_data = pd.read_csv("train.csv", index_col="Id")
test_data = pd.read_csv("test.csv", index_col="Id")

# remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = train_data.SalePrice
train_data.drop(["SalePrice"], axis=1, inplace=True)

numeric_cols = [
    col for col in train_data.columns if train_data[col].dtype in ["int64", "float64"]
]

X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()


X.head()
# 	MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageArea	WoodDeckSF	OpenPorchSF	EnclosedPorch	3SsnPorch	ScreenPorch	PoolArea	MiscVal	MoSold	YrSold
# Id
# 1	60	65.0	8450	7	5	2003	2003	196.0	706	0	...	548	0	61	0	0	0	0	0	2	2008
# 2	20	80.0	9600	6	8	1976	1976	0.0	978	0	...	460	298	0	0	0	0	0	0	5	2007
# 3	60	68.0	11250	7	5	2001	2002	162.0	486	0	...	608	0	42	0	0	0	0	0	9	2008
# 4	70	60.0	9550	7	5	1915	1970	0.0	216	0	...	642	0	35	272	0	0	0	0	2	2006
# 5	60	84.0	14260	8	5	2000	2000	350.0	655	0	...	836	192	84	0	0	0	0	0	12	2008
# 5 rows × 36 columns


my_pipeline = Pipeline(
    steps=[
        ("processor", SimpleImputer()),
        ("model", RandomForestRegressor(n_estimators=50, random_state=0)),
    ]
)


# ============ submit

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print("MAE:", score)

preds_test = my_pipeline.predict(X_test)
output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
output.to_csv("submission.csv", index=False)


# Multiple by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(
    my_pipeline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
print("Average MAE score:", scores.mean())
# Average MAE score: 18276.410356164386


# Step 1: Write a useful function
# use cross-validation to select parameters for a machine learning model.

# writing a function get_score() that reports the average (over three cross-validation folds) MAE of a machine learning pipeline that uses:

# the data in X and y to create folds,
# SimpleImputer() (with all parameters left as default) to replace missing values, and
# RandomForestRegressor() (with random_state=0) to fit a random forest model.
# The n_estimators parameter supplied to get_score() is used when setting the number of trees in the random forest model.


def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline = Pipeline(
        steps=[
            ("processor", SimpleImputer()),
            ("model", RandomForestRegressor(n_estimators, random_state=0)),
        ]
    )

    scores = -1 * cross_val_score(
        my_pipeline, X, y, cv=3, scoring="neg_mean_absolute_error"
    )
    return scores.mean()


# Step 2: Test different parameter values
# use the function in Step 1 to evaluate the model performance corresponding to eight different values for the number of trees in the random forest: 50, 100, 150, ..., 300, 350, 400.
# Store your results in a Python dictionary results, where results[i] is the average MAE returned by get_score(i).
results = {}
for i in range(1, 9):
    results[50 * i] = get_score(50 * i)


# visualize your results from Step 2. Run the code without changes.
# %matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()


# Step 3: Find the best parameter value
# Given the results, which value for n_estimators seems best for the random forest model? Use your answer to set the value of n_estimators_best.

n_estimators_best = 200
