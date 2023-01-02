# =============== Set up code checking
# import os
# if not os.path.exists("../input/train.csv"):
#     os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
#     os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
# from learntools.core import binder
# binder.bind(globals())
# from learntools.ml_intermediate.ex3 import *
# print("Setup Complete")


# =============== function reports the mean absolute error (MAE) from a random forest model.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# =============== load the training and validation sets in X_train, X_valid, y_train, and y_valid.
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv("train.csv", index_col="Id")
X_test = pd.read_csv("test.csv", index_col="Id")  # print(X.head())

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = X.SalePrice
X.drop(["SalePrice"], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)


# =============== Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

X_train.head()
# 	MSSubClass	MSZoning	LotArea	Street	LotShape	LandContour	Utilities	LotConfig	LandSlope	Neighborhood	...	OpenPorchSF	EnclosedPorch	3SsnPorch	ScreenPorch	PoolArea	MiscVal	MoSold	YrSold	SaleType	SaleCondition
# Id
# 619	20	RL	11694	Pave	Reg	Lvl	AllPub	Inside	Gtl	NridgHt	...	108	0	0	260	0	0	7	2007	New	Partial
# 871	20	RL	6600	Pave	Reg	Lvl	AllPub	Inside	Gtl	NAmes	...	0	0	0	0	0	0	8	2009	WD	Normal
# 93	30	RL	13360	Pave	IR1	HLS	AllPub	Inside	Gtl	Crawfor	...	0	44	0	0	0	0	8	2009	WD	Normal
# 818	20	RL	13265	Pave	IR1	Lvl	AllPub	CulDSac	Gtl	Mitchel	...	59	0	0	0	0	0	7	2008	WD	Normal
# 303	20	RL	13704	Pave	IR1	Lvl	AllPub	Corner	Gtl	CollgCr	...	81	0	0	0	0	0	1	2006	WD	Normal


# =============== the dataset contains both numerical and categorical variables.
# need to encode the categorical data before training a model.
num_cols = [
    col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]
]

s = X_train.dtypes == "object"
obj_cols = list(s[s].index)


# =============== Step 1: Solution: Drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

print("--- MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
# MAE from Approach 1 (Drop categorical variables):
# 17837.82570776256
print(
    "Unique values in 'Condition2' column in training data:",
    X_train["Condition2"].unique(),
)
print(
    "Unique values in 'Condition2' column in validation data:",
    X_valid["Condition2"].unique(),
)
print("\n==================== ")
# Unique values in 'Condition2' column in training data: ['Norm' 'PosA' 'Feedr' 'PosN' 'Artery' 'RRAe']
# Unique values in 'Condition2' column in validation data: ['Norm' 'Feedr''PosN' 'Artery' 'RRAn' 'RRNn' ]


# =============== Step 2: Ordinal encoding
# Part A
# fit an ordinal encoder to the training data, and then use it to transform both the training and validation data,
from sklearn.preprocessing import OrdinalEncoder

label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
# print(label_X_train.head())
# print(label_X_valid.head())

# Fitting an ordinal encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data.
# ordinal_encoder = OrdinalEncoder()
# label_X_train[obj_cols] = ordinal_encoder.fit_transform(X_train[obj_cols])
# label_X_valid[obj_cols] = ordinal_encoder.transform(X_valid[obj_cols])
# In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them.
# Notice that the 'Condition2' column in the validation data contains the values 'RRAn' and 'RRNn', but these don't appear in the training data -- thus, if we try to use an ordinal encoder with scikit-learn, the code will throw an error.


# This is a common problem for real-world data, and there are many approaches to fixing this issue.
# For instance, you can write a custom ordinal encoder to deal with new categories.
# The simplest approach, however, is to drop the problematic categorical columns.
# Run the code cell below to save the problematic columns to a Python list bad_label_cols. Likewise, columns that can be safely ordinal encoded are stored in good_label_cols.

# Categorical columns in the training data
obj_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [
    col for col in obj_cols if set(X_valid[col]).issubset(set(X_train[col]))
]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(obj_cols) - set(good_label_cols))

print("Categorical columns that will be ordinal encoded:", good_label_cols)
print("Categorical columns that will be dropped from the dataset:", bad_label_cols)
# Categorical columns that will be ordinal encoded: ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition']
# Categorical columns that will be dropped from the dataset: ['Condition2', 'RoofMatl', 'Functional']


# Part B
# ordinal encode the data in X_train and X_valid.
# Set the preprocessed DataFrames to label_X_train and label_X_valid, respectively.
# drop the categorical columns in bad_label_cols from the dataset.
# You should ordinal encode the categorical columns in good_label_cols.
from sklearn.preprocessing import OrdinalEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])
print("--- MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# 17098.01649543379
print("\n==================== ")


# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), obj_cols))
d = dict(zip(obj_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
# [('Street', 2),
#  ('Utilities', 2),
#  ('CentralAir', 2),
#  ('LandSlope', 3),
#  ('PavedDrive', 3),
#  ('LotShape', 4),
#  ('LandContour', 4),
#  ('ExterQual', 4),
#  ('KitchenQual', 4),
#  ('MSZoning', 5),
#  ('LotConfig', 5),
#  ('BldgType', 5),
#  ('ExterCond', 5),
#  ('HeatingQC', 5),
#  ('Condition2', 6),
#  ('RoofStyle', 6),
#  ('Foundation', 6),
#  ('Heating', 6),
#  ('Functional', 6),
#  ('SaleCondition', 6),
#  ('RoofMatl', 7),
#  ('HouseStyle', 8),
#  ('Condition1', 9),
#  ('SaleType', 9),
#  ('Exterior1st', 15),
#  ('Exterior2nd', 16),
#  ('Neighborhood', 25)]


# # =============== Step 3: Investigating cardinality
# Part A
# The output above shows, for each column with categorical data, the number of unique values in the column. For instance, the 'Street' column in the training data has two unique values: 'Grvl' and 'Pave', corresponding to a gravel road and a paved road, respectively.
# We refer to the number of unique entries of a categorical variable as the cardinality of that categorical variable.
# For instance, the 'Street' variable has cardinality 2.

# Fill in the line below:
# How many categorical variables in the training data have cardinality greater than 10?
cardinality = [
    col
    for col in X_train.columns
    if X_train[col].nunique() > 10 and X_train[col].dtype == "object"
]
# print(len(cardinality))
high_cardinality_numcols = 3
# How many columns are needed to one-hot encode the 'Neighborhood' variable in the training data?
#  ('Neighborhood', 25)]
num_cols_neighborhood = 25


# Part B
# For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset.
# For this reason, we typically will only one-hot encode columns with relatively low cardinality.
# Then, high cardinality columns can either be dropped from the dataset, or we can use ordinal encoding.

# As an example, consider a dataset with 10,000 rows, and containing one categorical column with 100 unique entries.
# If this column is replaced with the corresponding one-hot encoding, how many entries are added to the dataset?
# If we instead replace the column with the ordinal encoding, how many entries are added?
# Use your answers to fill in the lines below.
# Fill in the line below:
# How many entries are added to the dataset by replacing the column with a one-hot encoding?
OH_entries_added = 100 * 10000 - 10000
# print(OH_entries_added)
# How many entries are added to the dataset by replacing the column with an ordinal encoding?
label_entries_added = 0


# experiment with one-hot encoding. But, instead of encoding all of the categorical variables in the dataset, you'll only create a one-hot encoding for columns with cardinality less than 10.

# Run the code cell below without changes to set low_cardinality_cols to a Python list containing the columns that will be one-hot encoded.
# Likewise, high_cardinality_cols contains a list of categorical columns that will be dropped from the dataset.

low_cardinality_cols = [col for col in obj_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(obj_cols) - set(low_cardinality_cols))

print("Categorical columns that will be one-hot encoded:", low_cardinality_cols)
print(
    "Categorical columns that will be dropped from the dataset:", high_cardinality_cols
)
# Categorical columns that will be one-hot encoded: ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
# Categorical columns that will be dropped from the dataset: ['Exterior1st', 'Exterior2nd', 'Neighborhood']


# =============== Step 4: One-hot encoding
# Use the next code cell to one-hot encode the data in X_train and X_valid.
# Set the preprocessed DataFrames to OH_X_train and OH_X_valid, respectively.
# The full list of categorical columns in the dataset can be found in the Python list obj_cols.
# You should only one-hot encode the categorical columns in low_cardinality_cols.
# All other categorical columns should be dropped from the dataset.

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
obj_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
obj_X_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

print("obj_X_train " + str(obj_X_train.shape))
print("obj_X_valid " + str(obj_X_valid.shape))

# One-hot encoding removed index; put it back
obj_X_train.index = X_train.index
obj_X_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(obj_cols, axis=1)
num_X_valid = X_valid.drop(obj_cols, axis=1)


# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, obj_X_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, obj_X_valid], axis=1)

print("OH_X_train " + str(OH_X_train.shape))
print("OH_X_valid " + str(OH_X_valid.shape))

print("--- MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# 17525.345719178084
print("\n==================== ")


# Fills NA/NaN values using the forward fill method (fill)
X_test = X_test.fillna(method="ffill")
print("X_test " + str(X_test.shape))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))
OH_cols_test.index = X_test.index
num_X_test = X_test.drop(obj_cols, axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)


# Fill in the line below: get test predictions
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train, y_train)
preds_test = model.predict(OH_X_test)

# Save test predictions to file
output = pd.DataFrame({"Id": OH_X_test.index, "SalePrice": preds_test})
output.to_csv("submission.csv", index=False)
