---
title: AIML - 2nd - Intermediate Machine Learning
date: 2021-08-11 11:11:11 -0400
categories: [51AIML, MLNote]
tags: [AIML]
toc: true
---

- [ML - Intermediate Machine Learning](#ml---intermediate-machine-learning)
  - [Missing Values](#missing-values)
    - [Three Approaches](#three-approaches)
    - [Approach 1 (Drop Columns with Missing Values) 删除无数值列](#approach-1-drop-columns-with-missing-values-删除无数值列)
    - [Approach 2 (Imputation) 用其他数值代替](#approach-2-imputation-用其他数值代替)
    - [Approach 3 (An Extension to Imputation)](#approach-3-an-extension-to-imputation)
    - [summary](#summary)
  - [categorical variable](#categorical-variable)
    - [Three Approaches](#three-approaches-1)
    - [get list of categorical variables](#get-list-of-categorical-variables)
    - [Drop Categorical Variables 删除空值](#drop-categorical-variables-删除空值)
    - [Ordinal Encoding 转换成数值](#ordinal-encoding-转换成数值)
    - [One-Hot Encoding](#one-hot-encoding)
    - [conclusion](#conclusion)
  - [Pipeline](#pipeline)
    - [Step 1: Define Preprocessing Steps](#step-1-define-preprocessing-steps)
    - [Step 2: Define the Model](#step-2-define-the-model)
    - [Step 3: Create and Evaluate the Pipeline](#step-3-create-and-evaluate-the-pipeline)
    - [summary](#summary-1)
  - [Cross-Validation](#cross-validation)
    - [Example](#example)
    - [Conclusion](#conclusion-1)
  - [XGBoost](#xgboost)
    - [Gradient Boosting 坡度提升](#gradient-boosting-坡度提升)
      - [loading the training and validation data](#loading-the-training-and-validation-data)
      - [work with XGBoost library.](#work-with-xgboost-library)
      - [make predictions and evaluate the model.](#make-predictions-and-evaluate-the-model)
    - [Parameter Tuning](#parameter-tuning)
      - [`n_estimators` times of modeling cycle](#n_estimators-times-of-modeling-cycle)
      - [`early_stopping_rounds` Early stopping causes the model to stop iterating](#early_stopping_rounds-early-stopping-causes-the-model-to-stop-iterating)
      - [`earning_rate` multiply the predictions](#earning_rate-multiply-the-predictions)
      - [`n_jobs` build the models faster](#n_jobs-build-the-models-faster)
      - [Conclusion](#conclusion-2)
  - [data leakagae](#data-leakagae)
    - [Target leakage](#target-leakage)
    - [Train-Test Contamination](#train-test-contamination)
    - [To detect and remove target leakage.](#to-detect-and-remove-target-leakage)
    - [Conclusion](#conclusion-3)

---

# ML - Intermediate Machine Learning

> Welcome to Kaggle Learn's Intermediate Machine Learning micro-course!

> If you have some background in machine learning and you'd like to learn how to quickly improve the quality of the models, you're in the right place! In this micro-course, you will accelerate the machine learning expertise by learning how to:

- tackle data types often found in real-world datasets (**missing values, categorical variables**),
- design **pipelines** to improve the quality of the machine learning code,
- use advanced techniques for model validation (**cross-validation**),
- build state-of-the-art models that are widely used to win Kaggle competitions (**XGBoost**), and
- avoid common and important data science mistakes (**leakage**).

---

## Missing Values

Most machine learning libraries (including scikit-learn) give an error if you try to build a model using data with missing values. So you'll need to choose one of the strategies below.



example:

```py
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```


### Three Approaches

### Approach 1 (Drop Columns with Missing Values) 删除无数值列
- simplest option is to drop columns with missing values.
- Unless most values in the dropped columns are missing, the model loses access to a lot of (potentially useful!) information with this approach.

> extreme example:
> consider a dataset with 10,000 rows, where one important column is missing a single entry.
> This approach would drop the column entirely!

![Sax80za-1](https://i.imgur.com/EjJsTGh.png)

- Since are working with both training and validation sets, are careful to drop the same columns in both DataFrames.

```py
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# MAE from Approach 1 (Drop columns with missing values):
# 183550.22137772635
```



### Approach 2 (Imputation) 用其他数值代替
- Imputation **fills in the missing values with some number**
  - For instance, can fill in the `mean value` along each column.
- The imputed value won't be exactly right in most cases, but it usually leads to more accurate models than you would get from dropping the column entirely.


![4BpnlPA](https://i.imgur.com/Ukc1i7n.png)

- use `SimpleImputer` to replace missing values with the `mean value` along each column.
- filling in the `mean value` generally performs quite well
  - but this varies by dataset
- While statisticians have experimented with more complex ways to determine imputed values (such as regression imputation, for instance), the complex strategies typically give no additional benefit once you plug the results into sophisticated machine learning models.

```py
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# MAE from Approach 2 (Imputation):
# 178166.46269899711

```


### Approach 3 (An Extension to Imputation)

- Imputation is the standard approach, and it usually works well.
  - However, imputed values may be systematically above or below their actual values
  - (which weren't collected in the dataset).
  - Or rows with missing values may be unique in some other way.
- In that case, make better predictions by **considering which values were originally missing**.
  - impute the missing values, as before.
  - And for each column with missing entries in the original dataset, add a new column that shows the location of the imputed entries.
- In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.

![UWOyg4a](https://i.imgur.com/Os0pAeH.png)

```py
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
# MAE from Approach 3 (An Extension to Imputation):
# 178927.503183954
```

---


### summary

```py
# Shape of training data (num_rows, num_columns)
print(X_train.shape)
# (10864, 12)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Car               49
# BuildingArea    5156
# YearBuilt       4307
# dtype: int64

```


---

## categorical variable

A **categorical variable** takes only a limited number of values.

> Consider a survey that asks how often you eat breakfast and provides four options: "Never", "Rarely", "Most days", or "Every day".
> In this case, the data is categorical
> because responses fall into a fixed set of categories.

> If people responded to a survey about which what brand of car they owned, the responses would fall into categories like "Honda", "Toyota", and "Ford".
> In this case, the data is also categorical.

- You will get an error if you try to plug these variables into most machine learning models in Python without preprocessing them first.

There are three approaches to prepare the categorical data.



---

### Three Approaches


```py
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 10
                        and X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns
                  if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train.head()
# Type	Method	Regionname	Rooms	Distance	Postcode	Bedroom2	Bathroom	Landsize	Latitude	Longtitude	Propertycount
# 12167	u	S	Southern Metropolitan	1	5.0	3182.0	1.0	1.0	0.0	-37.85984	144.9867	13240.0
# 6524	h	SA	Western Metropolitan	2	8.0	3016.0	2.0	2.0	193.0	-37.85800	144.9005	6380.0
# 8413	h	S	Western Metropolitan	3	12.6	3020.0	3.0	1.0	555.0	-37.79880	144.8220	3755.0
# 2919	u	SP	Northern Metropolitan	3	13.0	3046.0	3.0	1.0	265.0	-37.70830	144.9158	8870.0
# 6043	h	S	Western Metropolitan	3	13.3	3020.0	3.0	1.0	673.0	-37.76230	144.8272	4217.0


# get list of all categorical variables in the training data.
# checking the data type (or dtype) of each column.
# The object dtype indicates a column has text (there are other things it could theoretically be, but that's unimportant for our purposes). For this dataset, the columns with text -> indicate categorical variables.
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
# ['Type', 'Method', 'Regionname']



# Define Function function score_dataset()
# to Measure Quality of Each Approach
# to compare the three different approaches to dealing with categorical variables.
# This function reports the mean absolute error (MAE) from a random forest model.
# In general, want the MAE to be as low as possible!
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```


---

### get list of categorical variables

```py
# checking the data type (or dtype) of each column.
# The object dtype indicates a column has text (there are other things it could theoretically be, but that's unimportant for our purposes). For this dataset, the columns with text -> indicate categorical variables.
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
# ['Type', 'Method', 'Regionname']
```

---

### Drop Categorical Variables 删除空值

- The easiest approach to dealing with categorical variables is to simply remove them from the dataset.
- This approach will only work well if the columns did not contain useful information.

drop the object columns with the `select_dtypes()` method.

```py
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
# MAE from Approach 1 (Drop categorical variables):
# 175703.48185157913
```

---

### Ordinal Encoding 转换成数值

- assigns each unique value to a different integer.

![tEogUAr](https://i.imgur.com/ILMPr32.png)

> assumes an ordering of the categories:
> "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

- This assumption makes sense in this example, because there is an indisputable ranking to the categories.
- Not all categorical variables have a clear ordering in the values, but refer to those that do as ordinal variables.
- For tree-based models (like decision trees and random forests), you can expect **ordinal encoding** to work well with **ordinal variables**.

- Scikit-learn has a `OrdinalEncoder` class that can be used to get ordinal encodings.
- loop over the **categorical variables** and apply the `ordinal encoder` separately to each column.

- for each column, randomly assign each unique value to a different integer.
- This is a common approach that is simpler than providing custom labels;
- however, can expect an additional boost in performance if provide better-informed labels for all ordinal variables.


```py
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# MAE from Approach 2 (Ordinal Encoding):
# 165936.40548390493
```

---


### One-Hot Encoding

- One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data.

![TW5m0aJ](https://i.imgur.com/z88ShxY.png)

> In the original dataset, "Color" is a categorical variable with three categories: "Red", "Yellow", and "Green".
> The corresponding one-hot encoding contains one column for each possible value, and one row for each row in the original dataset.
> Wherever the original value was "Red", put a 1 in the "Red" column;
> if the original value was "Yellow", put a 1 in the "Yellow" column, and so on.

- In contrast to ordinal encoding, one-hot encoding **does not assume an ordering of the categories**.

- can expect this approach to work particularly well
  - if there is no clear ordering in the categorical data
  - (e.g., "Red" is neither more nor less than "Yellow").
  - refer to **categorical variables** without an `intrinsic ranking` as **nominal variables**.

- it does not perform well if the categorical variable takes on a large number of values
- (i.e., you generally won't use it for variables taking more than 15 different values).


- use the `OneHotEncoder` class from scikit-learn to get one-hot encodings.
- There are a number of parameters that can be used to customize its behavior.
  - `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data
  - `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
- To use the encoder, supply only the **categorical columns** that want to be one-hot encoded.
- For instance, to encode the training data, supply `X_train[object_cols]`.
- (object_cols in the code cell below is a list of the column names with **categorical data**, and so `X_train[object_cols]` contains all of the categorical data in the training set.)

```py
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_obj_cols_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_obj_cols_X_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_obj_cols_X_train.index = X_train.index
OH_obj_cols_X_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
OH_num_cols_X_train = X_train.drop(object_cols, axis=1)
OH_num_cols_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([OH_num_cols_X_train, OH_obj_cols_X_train], axis=1)
OH_X_valid = pd.concat([OH_num_cols_X_valid, OH_obj_cols_X_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# MAE from Approach 3 (One-Hot Encoding):
# 166089.4893009678
```


---


### conclusion

- In this case, dropping the categorical columns (Approach 1) performed worst, since it had the highest MAE score.
- As for the other two approaches, since the returned MAE scores are so close in value, there doesn't appear to be any meaningful benefit to one over the other.
- In general, one-hot encoding (Approach 3) will typically perform best,
- and dropping the categorical columns (Approach 1) typically performs worst,
- but it varies on a case-by-case basis.

```py

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
X = pd.read_csv('train.csv', index_col='Id')
X_test = pd.read_csv('test.csv', index_col='Id')\
# print(X.head())

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)


# =============== Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

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
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

s = (X_train.dtypes == 'object')
obj_cols = list(s[s].index)




# =============== Step 1: Solution: Drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("--- MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
# MAE from Approach 1 (Drop categorical variables):
# 17837.82570776256
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("Unique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
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
# Notice that the 'Condition2' column in the validation data contains the values 'RRAn' and 'RRNn', but these don't appear in the training data -- thus, if try to use an ordinal encoder with scikit-learn, the code will throw an error.


# This is a common problem for real-world data, and there are many approaches to fixing this issue.
# For instance, you can write a custom ordinal encoder to deal with new categories.
# The simplest approach, however, is to drop the problematic categorical columns.
# Run the code cell below to save the problematic columns to a Python list bad_label_cols. Likewise, columns that can be safely ordinal encoded are stored in good_label_cols.

# Categorical columns in the training data
obj_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in obj_cols
                    if set(X_valid[col]).issubset(set(X_train[col]))]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(obj_cols)-set(good_label_cols))

print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('Categorical columns that will be dropped from the dataset:', bad_label_cols)
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
# refer to the number of unique entries of a categorical variable as the cardinality of that categorical variable.
# For instance, the 'Street' variable has cardinality 2.

# Fill in the line below:
# How many categorical variables in the training data have cardinality greater than 10?
cardinality = [col for col in X_train.columns
                if X_train[col].nunique() > 10
                and X_train[col].dtype == "object"]
# print(len(cardinality))
high_cardinality_numcols = 3
# How many columns are needed to one-hot encode the 'Neighborhood' variable in the training data?
#  ('Neighborhood', 25)]
num_cols_neighborhood = 25



# Part B
# For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset.
# For this reason, typically will only one-hot encode columns with relatively low cardinality.
# Then, high cardinality columns can either be dropped from the dataset, or can use ordinal encoding.

# As an example, consider a dataset with 10,000 rows, and containing one categorical column with 100 unique entries.
# If this column is replaced with the corresponding one-hot encoding, how many entries are added to the dataset?
# If instead replace the column with the ordinal encoding, how many entries are added?
# Use the answers to fill in the lines below.
# Fill in the line below:
# How many entries are added to the dataset by replacing the column with a one-hot encoding?
OH_entries_added = 100*10000-10000
# print(OH_entries_added)
# How many entries are added to the dataset by replacing the column with an ordinal encoding?
label_entries_added = 0



# experiment with one-hot encoding. But, instead of encoding all of the categorical variables in the dataset, you'll only create a one-hot encoding for columns with cardinality less than 10.

# Run the code cell below without changes to set low_cardinality_cols to a Python list containing the columns that will be one-hot encoded.
# Likewise, high_cardinality_cols contains a list of categorical columns that will be dropped from the dataset.

low_cardinality_cols = [col for col in obj_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(obj_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('Categorical columns that will be dropped from the dataset:', high_cardinality_cols)
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
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
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

# print("OH_X_train " + str(OH_X_train.shape))
# print("OH_X_valid " + str(OH_X_valid.shape))
print("--- MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# 17525.345719178084
print("\n==================== ")
```


---



## Pipeline


pipelines have some important benefits. Those include:
- **Cleaner Code**: Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of the training and validation data at each step.
- **Fewer Bugs**: There are fewer opportunities to misapply a step or forget a preprocessing step.
- **Easier to Productioniz**e: It can be surprisingly hard to transition a model from a prototype to something deployable at scale. won't go into the many related concerns here, but pipelines can help.
- **More Options for Model Validation**: You will see an example in the next tutorial, which covers cross-validation.


```py
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].nunique() < 10
                    and X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns
                  if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train.head()
# Type	Method	Regionname	Rooms	Distance	Postcode	Bedroom2	Bathroom	Car	Landsize	BuildingArea	YearBuilt	Latitude	Longtitude	Propertycount
# 12167	u	S	Southern Metropolitan	1	5.0	3182.0	1.0	1.0	1.0	0.0	NaN	1940.0	-37.85984	144.9867	13240.0
# 6524	h	SA	Western Metropolitan	2	8.0	3016.0	2.0	2.0	1.0	193.0	NaN	NaN	-37.85800	144.9005	6380.0
# 8413	h	S	Western Metropolitan	3	12.6	3020.0	3.0	1.0	1.0	555.0	NaN	NaN	-37.79880	144.8220	3755.0
# 2919	u	SP	Northern Metropolitan	3	13.0	3046.0	3.0	1.0	1.0	265.0	NaN	1995.0	-37.70830	144.9158	8870.0
# 6043	h	S	Western Metropolitan	3	13.3	3020.0	3.0	1.0	2.0	673.0	673.0	1970.0	-37.76230	144.8272	4217.0
```

construct the full pipeline in three steps.


### Step 1: Define Preprocessing Steps

- Similar to how a pipeline bundles together preprocessing and modeling steps
- use the `ColumnTransformer` class to bundle together different preprocessing steps.
  - imputes missing values in numerical data
  - imputes missing values and applies a one-hot encoding to categorical data.

```py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(
  steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```



### Step 2: Define the Model
- define a random forest model with the familiar `RandomForestRegressor` class.

```py
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
```


### Step 3: Create and Evaluate the Pipeline
- use the `Pipeline` class to define a pipeline that bundles the preprocessing and modeling steps.
- With the pipeline,
  - preprocess the `training data` and fit the model in a single line of code.
    - (without a pipeline, have to do imputation, one-hot encoding, and model training in separate steps.
    - This becomes especially messy if have to deal with both numerical and categorical variables!)
  - supply the unprocessed features in `X_valid` to the `predict(`) command, and the pipeline automatically preprocesses the features before generating predictions.
    - (without a pipeline, have to remember to preprocess the validation data before making predictions.)

```py
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(
  steps=[
    ('preprocessor', preprocessor),
    ('model', model)
  ]
)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
# MAE: 160679.18917034855
```


### summary

```py
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove missing target
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off vdalidation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# find cardinality
categorical_cols = [col for col in X_train_full.columns
            if X_train_full[col].nunique() < 10
            and X_train_full[col].dtype == "Object"]
numerical_cols = [col for col in X_train_full.columns
            if X_train_full[col].dtype in ['int64', 'float64']]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()


X_train.head()



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
# SimpleImputer() -> Approach 2 (Imputation) 用其他数值代替
numerical_transformer = SimpleImputer(
    strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
# MAE: 17861.780102739725





# ===================== Step 1: Improve the performance
# Part A
# define the own preprocessing steps and random forest model.

# Preprocessing for numerical data
# 数值型数据的预处理
numerical_transformer = SimpleImputer(
    # strategy='constant') # MAE too high
    strategy='median')

# Preprocessing for categorical data
# 分类型数据的预处理
# 分类数据处理有两部分：填补和编码，可以用管道捆绑
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))
    ]
)

# Bundle preprocessing for numerical and categorical data
# 用ColumnTransformer捆绑数值型和分类型数据的预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)


# Part B
# have defined a pipeline in Part A that achieves lower MAE than the code above.
# You're encouraged to take the time here and try out many different approaches, to see how low you can get the MAE!

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)




# Step 2: Generate test predictions
# use the trained model to generate predictions with the test data.

# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```



---


## Cross-Validation

> use cross-validation for better measures of model performance.

Machine learning is an iterative process.
- choices about
  - what predictive variables to use,
  - what types of models to use,
  - what arguments to supply to those models, etc.
- made these choices in a `data-driven way` by measuring model quality with a `validation (or holdout) set`.

Drawbacks of this approach.
- dataset with 5000 rows.
- keep about 20% of the data as a validation dataset (1000 rows)
- this leaves some random chance in determining model scores.
  - a model might do well on one set of 1000 rows,
  - even if it would be inaccurate on a different 1000 rows.

- At an extreme, you could imagine having only 1 row of data in the validation set.
- If you compare alternative models, which one makes the best predictions on a single data point will be mostly a matter of luck!


In general,
- the larger the validation set,
- the less randomness (aka "noise") there is in our measure of model quality,
- and the more reliable it will be.


Unfortunately, can only get a large validation set by removing rows from our training data,
- and smaller training datasets mean worse models!

![9k60cVA](https://i.imgur.com/NhQDSfR.png)

**cross-validation**
- run our modeling process on different subsets of the data to get multiple measures of model quality.
- For example, could begin by dividing the data into 5 pieces, each 20% of the full dataset.
- In this case, say that have broken the data into 5 **"folds"**.
- Then, run one experiment for each fold:
  - In Experiment 1, use the `first fold as a validation (or holdout) set` and everything else as training data. This gives us a measure of model quality based on a **20% holdout set**.
  - In Experiment 2, hold out data from the `second fold` (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
  - repeat this process, using every fold once as the holdout set.
- Putting this together,
  - 100% of the data is used as holdout at some point,
  - end up with a **measure of model quality** that is `based on all of the rows in the dataset` (even if don't use all rows simultaneously).


> Cross-validation gives a more accurate measure of model quality, which is especially important if you are making a lot of modeling decisions.
> **it take longer to run**, because it estimates multiple models (one for each fold).

When to use cross-validation:
- For small datasets
  - where extra computational burden isn't a big deal
  - run cross-validation.
- For larger datasets,
  - a single validation set is sufficient.
  -  the code will run faster,
  -  and you may have enough data that there's little need to re-use some of it for holdout.

There's no simple threshold for what constitutes a large vs. small dataset.
- But if the model takes a couple minutes or less to run, it's probably worth switching to cross-validation.

Alternatively, you can run cross-validation and see if the scores for each experiment seem close.
- If each experiment yields the same results, a single validation set is probably sufficient.



### Example

1. load the input data in X and the output data in y.

```py
import pandas as pd

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price
```


2. define a pipeline that uses an `imputer` to fill in missing values and a random forest model to make predictions.

   - While it's possible to do cross-validation without pipelines, it is quite difficult!
   - Using a pipeline will make the code remarkably straightforward.

```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(
  steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))])


# obtain the cross-validation scores with the cross_val_score() function from scikit-learn.
# set the number of folds with the cv parameter.
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
# MAE scores:
# [301628.7893587  303164.4782723  287298.331666   236061.84754543   260383.45111427]
```


The scoring parameter chooses a measure of model quality to report:
- in this case, chose `negative mean absolute error (MAE)`.
- The docs for scikit-learn show a list of options.
- It is a little surprising that specify negative MAE.
- Scikit-learn has a convention where all metrics are defined so a high number is better.
- Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere.
- typically want a single measure of model quality to compare alternative models. So take the average across experiments.

```py
print("Average MAE score (across experiments):")
print(scores.mean())
# Average MAE score (across experiments):
# 277707.3795913405
```


### Conclusion
- Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code: note that no longer need to keep track of separate training and validation sets. So, especially for small datasets, it's a good improvement!



---


## XGBoost

build and optimize models with gradient boosting.

- **ensemble method**
  - ensemble methods `combine the predictions of several models`
    - (e.g., several trees, in the case of random forests)

- refer to the `random forest method` as an "**ensemble method**".
  - predictions with the `random forest method`
  - achieves better performance than a single decision tree
  - by averaging the predictions of many decision trees.

- another ensemble method called `gradient boosting`

---

### Gradient Boosting 坡度提升

![MvCGENh](https://i.imgur.com/lIhYFt9.png)

- a method that goes through cycles to `iteratively add models into an ensemble`.
- It begins by initializing the ensemble with a **single model**,
  - whose predictions can be pretty naive.
  - (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

- Then, start the cycle:
  - First, use the current ensemble to generate predictions for each observation in the dataset.
    - To make a prediction, add the predictions from all models in the ensemble.
    - These predictions are used to calculate a **loss function** (like mean squared error, for instance).
  - Then, use the **loss function** to fit a new model that will be added to the ensemble.
    - Specifically, determine model parameters so that adding this new model to the ensemble will reduce the loss.
    - "gradient": use gradient descent on the loss function to determine the parameters in this new model
  - Finally, add the new model to ensemble, and repeat!


#### loading the training and validation data

```py
# begin by loading the training and validation data in X_train, X_valid, y_train, and y_valid.
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
```


#### work with XGBoost library.

- XGBoost stands for **extreme gradient boosting**
- an implementation of gradient boosting with several additional features focused on performance and speed.
- (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.)

- import the scikit-learn API for XGBoost (`xgboost.XGBRegressor`).
- to build and fit a model just as would in scikit-learn.

```py
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012,
#              max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
```

#### make predictions and evaluate the model.

```py
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
# Mean Absolute Error: 238794.73582819404
```

---

### Parameter Tuning
- XGBoost has a few parameters that can dramatically affect accuracy and training speed.

---

#### `n_estimators` times of modeling cycle

- how many times to go through the modeling cycle described above.
- It is equal to the number of models that include in the ensemble.
- Too low a value causes `underfitting`
  - which leads to inaccurate predictions on both training data and test data.
- Too high a value causes `overfitting`
  - which causes accurate predictions on training data, but inaccurate predictions on test data (which is what care about).
- Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

set the number of models in the ensemble:

```py
my_model = XGBRegressor(n_estimators=500)

my_model.fit(X_train, y_train)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=500, n_jobs=4, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
```

---


#### `early_stopping_rounds` Early stopping causes the model to stop iterating

- a way to automatically find the ideal value for `n_estimators`.
- Early stopping causes the model to stop iterating
  - when the validation score stops improving,
  - even if aren't at the hard stop for n_estimators.
- It's smart to set a `high value` for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating.

Since random chance sometimes causes a single round where validation scores don't improve
- need to specify a number for how many rounds of straight deterioration 直接恶化 to allow before stopping.
- Setting `early_stopping_rounds=5` is a reasonable choice.
- In this case, stop after 5 straight rounds of deteriorating 恶化 validation scores.

When using early_stopping_rounds, you also need to set aside some data for **calculating the validation scores**
- this is done by setting the `eval_set` parameter.
- If you later want to fit a model with all of the data, set n_estimators to whatever value you found to be optimal when run with early stopping.


```py
my_model = XGBRegressor(n_estimators=500)

my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=500, n_jobs=4, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
```

---

#### `earning_rate` multiply the predictions

- Instead of getting predictions by simply `adding up the predictions from each component model`,
- can `multiply the predictions from each model by a small number` (known as the learning rate) before adding them in.

This means each tree add to the ensemble helps us less.
- So, can set a higher value for `n_estimators` without overfitting.
- If use `early stopping`, the appropriate number of trees will be determined automatically.

In general, a `small learning rate` and `large number of estimators` will yield more accurate **XGBoost models**,
- though it will also take the model longer to train
- since it does more iterations through the cycle.
- As default, XGBoost sets `learning_rate=0.1`

Modifying the example above to change the learning rate yields the following code:

```py
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.05, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=1000, n_jobs=4, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
```



---

#### `n_jobs` build the models faster

- On larger datasets, runtime is a consideration
- use `parallelism` to build the models faster.
- It's common to set the parameter `n_jobs` equal to `the number of cores on the machine`.
- On smaller datasets, this won't help.

The resulting model won't be any better, so `micro-optimizing` for fitting time is typically nothing but a distraction.
- But, it's useful in large datasets where you would otherwise spend a long time waiting during the `fit` command.


Here's the modified example:

```py
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.05, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=1000, n_jobs=4, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
```


#### Conclusion
XGBoost is a the leading software library for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). With careful parameter tuning, you can train highly accurate models.


---


## data leakagae

- If you don't know how to prevent data leakagae, leakage will come up frequently, and it will ruin your models in subtle and dangerous ways.
- this is one of the most important concepts for practicing data scientists.


Data leakage
- happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction.
- This leads to high performance on the training set (and possibly even the validation data),
- but the model will perform poorly in production.

**leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.**

2 main types of leakage:
- target leakage: **predictors** include `data that will not be available at the time you make predictions`.
- train-test contamination.

---

### Target leakage

![y7hfTYe-1](https://i.imgur.com/Hg0OZDn.png)

- when your **predictors** include `data that will not be available at the time you make predictions`.
- think about target leakage in terms of the `timing or chronological order` that data becomes available, not merely whether a feature helps make good predictions.

example
- Imagine you want to predict who will get sick with pneumonia.
- The top few rows of your raw data look like this:

```
got_pneumonia	| age	| weight	| male	| took_antibiotic_medicine	...
False	          65	   100     	False    	False	...
False	          72	   130     	True	    False	...
True	          58	   100     	False    	True	...
```

People take antibiotic medicines after getting pneumonia in order to recover.
- The raw data shows a strong relationship between those columns,
- but `took_antibiotic_medicine` is frequently changed after the value for `got_pneumonia` is determined.
- This is **target leakage**.

> The model would see that anyone who has a value of False for `took_antibiotic_medicine` didn't have pneumonia.

Since `validation data` comes from the `same source` as `training data`,
- the pattern will repeat itself in validation,
- and the model will have great **validation (or cross-validation) scores**.

> But the model will be very inaccurate when subsequently deployed in the real world, because even patients who will get pneumonia won't have received antibiotics yet when we need to make predictions about their future health.

To prevent this type of data leakage
- any `variable` **updated (or created) after the target value** is realized should be **excluded**.

---

### Train-Test Contamination
- aren't careful to `distinguish training data from validation data`
- Recall that **validation** is meant to be a measure of `how the model does on data that it hasn't considered before`
- You can corrupt this process in subtle 微妙的 ways if the validation data affects the preprocessing behavior
- This is sometimes called `train-test contamination`

example
- imagine you run preprocessing (like fitting an imputer for missing values) before calling `train_test_split()`.
- The end result?
- Your model may get good validation scores, giving you great confidence in it,
- but perform poorly when you deploy it to make decisions.

After all, you incorporated data from the validation or test data into how you make predictions, so they may do well on that particular data even if it can't generalize to new data. This problem becomes even more subtle/dangerous when you do more complex feature engineering.

If your validation is based on a simple `train-test split`, exclude the validation data from any type of fitting, including the fitting of preprocessing steps.
- This is easier if you use scikit-learn pipelines.
- When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline!

---

### To detect and remove target leakage.

- use a dataset about credit card applications and skip the basic data set-up code.
- The end result is that information about each credit card application is stored in a DataFrame `X`.
- We'll use it to predict which applications were accepted in a Series `y`.

```py
import pandas as pd

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv',
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()
# Number of rows in the dataset: 1319
# reports	age	income	share	expenditure	owner	selfemp	dependents	months	majorcards	active
# 0	0	37.66667	4.5200	0.033270	124.983300	True	False	3	54	1	12
# 1	0	33.25000	2.4200	0.005217	9.854167	False	False	3	34	1	13
# 2	0	33.66667	4.5000	0.004156	15.000000	True	False	4	58	1	5
# 3	0	30.50000	2.5400	0.065214	137.869200	False	False	0	25	1	7
# 4	0	32.16667	9.7867	0.067051	546.503300	True	False	2	64	1	5
```

Since this is a small dataset
- use `cross-validation` to ensure accurate measures of model quality.

```py
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())
# Cross-validation accuracy: 0.980294
```

it's very rare to find models that are accurate 98% of the time.
- It happens, but it's uncommon
- we should inspect the data more closely for target leakage.

Here is a summary of the data, which you can also find under the data tab:

- `card`: 1 if credit card application accepted, 0 if not
- `reports`: Number of major derogatory reports
- `age`: Age n years plus twelfths of a year
- `income`: Yearly income (divided by 10,000)
- `share`: Ratio of monthly credit card expenditure to yearly income
- `expenditure`: Average monthly credit card expenditure
- `owner`: 1 if owns home, 0 if rents
- `selfempl`: 1 if self-employed, 0 if not
- `dependents`: 1 + number of dependents
- `months`: Months living at current address
- `majorcards`: Number of major credit cards held
- `active`: Number of active credit accounts

A few variables look suspicious.
- For example, does `expenditure` mean expenditure on this card or on cards used before applying?

At this point, **basic data comparisons** can be very helpful:

```py
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who received a card and had no expenditures: %.2f' %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who did not receive a card and had no expenditures: %.2f' %((expenditures_noncardholders == 0).mean()))
# Fraction of those who         receive a card and had no expenditures: 0.02
# Fraction of those who did not receive a card and had no expenditures: 1.00
```

As shown above:
- everyone who did not receive a card had no expenditures,
- while only 2% of those who received a card had no expenditures.
- It's not surprising that our model appeared to have a high accuracy.
- But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

Since share is `partially determined by expenditure`, it should be excluded too.

- The variables `active` and `majorcards` are a little less clear, but from the description, they sound concerning.
- In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.


run a model without target leakage as follows:

```py
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, cv=5, scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
# Cross-val accuracy: 0.836989
```


This accuracy is quite a bit lower, which might be disappointing.
- However, we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model would likely do much worse than that (in spite of its higher apparent score in cross-validation).

---


### Conclusion
- Data leakage can be multi-million dollar mistake in many data science applications.
- **Careful separation of training and validation data** can prevent `train-test contamination`,
- and **pipelines** can help `implement this separation`.
- Likewise, a combination of caution, common sense, and data exploration can help identify target leakage.


















.
