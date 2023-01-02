---
title: ML lab 02 - Predict home prices in Iowa
date: 2021-08-11 11:11:11 -0400
categories: [00CodeNote, MLNote]
tags: [ML]
toc: true
---

- [ML lab - Home data](#ml-lab---home-data)
  - [Basic](#basic)
    - [Step 1: Evaluate several models](#step-1-evaluate-several-models)
    - [Step 2: Generate test predictions](#step-2-generate-test-predictions)
  - [Missing value](#missing-value)
    - [Step 1: Preliminary investigation](#step-1-preliminary-investigation)
    - [Step 2: Drop columns with missing values](#step-2-drop-columns-with-missing-values)
    - [Step 3: Imputation](#step-3-imputation)
    - [Step 4: Generate test predictions](#step-4-generate-test-predictions)
      - [Part A](#part-a)
      - [Part B](#part-b)


- ref
  - https://www.kaggle.com/learn/intermediate-machine-learning


---


# ML lab - Home data

work with data from the Housing Prices Competition for Kaggle Learn Users to predict home prices in Iowa using 79 explanatory variables describing (almost) every aspect of the homes.

In this exercise, you will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course).


## Basic

basic approach for the ML

1. Set up code checking
2. Read the data `X_data/test_full = pd.read_csv()`
3. Obtain target `y_data = X_data_full.SalePrice`
4. Obtain redictors `X_data/test = X_data_full[A,B,C,D].copy()`
5. Break off validation set from training data `X_train, X_valid, y_train, y_valid = train_test_split()`
6. test head `X_train.head()`
7. defines different random forest models `model = RandomForestRegressor(n_estimators=50, random_state=0)`
8. define a function returns the mean absolute error (MAE) from the validation set.

  ```py
  def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
      model.fit(X_t, y_t)
      preds = model.predict(X_v)
      return mean_absolute_error(y_v, preds)
  ​
  for i in range(0, len(models)):
      mae = score_model(models[i])
      print("Model %d MAE: %d" % (i+1, mae))
  ```

9. best_model `my_model = RandomForestRegressor(n_estimators=100, random_state=0]=56)`
10. Generate test predictions

    ```py
    my_model.fit(X, y)
    preds_test = my_model.predict(X_test)
    output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
    output.to_csv('submission.csv', index=False)
    ```


```py
# =============================== Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")

from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")



# =============================== setup ML module
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_data_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y_data = X_data_full.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X_data = X_data_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X_data, y_data, train_size=0.8, test_size=0.2, random_state=0)

X_train.head()
#      LotArea	YearBuilt	1stFlrSF	2ndFlrSF	FullBath	BedroomAbvGr	TotRmsAbvGrd
# Id
# 619	11694	2007	1828	0	2	3	9
# 871	6600	1962	894	0	1	2	5
# 93	13360	1921	964	0	1	2	5
# 818	13265	2002	1689	0	2	3	7
# 303	13704	2001	1541	0	2	3	6


# =============================== defines five different random forest models
from sklearn.ensemble import RandomForestRegressor
# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, random_state=0, criterion='mae')
model_4 = RandomForestRegressor(n_estimators=200, random_state=0, min_samples_split=20)
model_5 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=7)
models = [model_1, model_2, model_3, model_4, model_5]


# =============================== select the best model out of the five
# define a function score_model(), returns the mean absolute error (MAE) from the validation set.
# the best model will obtain the lowest MAE.
from sklearn.metrics import mean_absolute_error
​
# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)
​
for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
# Model 1 MAE: 24015
# Model 2 MAE: 23740
# Model 3 MAE: 23528
# Model 4 MAE: 23996
# Model 5 MAE: 23706
```



### Step 1: Evaluate several models

```py
# Fill in the best model
best_model = model_3
```


### Step 2: Generate test predictions


```py
# Define a model
my_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
```





## Missing value

1. Set up code checking
2. Read the data `X_data/test_full = pd.read_csv()`
3. Remove rows with missing target, separate target from predictors

  ```py
  X_data_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
  y = X_data_full.SalePrice

  X_data_full.drop(['SalePrice'], axis=1, inplace=True)
  X = X_data_full.select_dtypes(exclude=['object'])
  X_test = X_test_full.select_dtypes(exclude=['object'])
  ```

4. Break off validation set from training data `X_train, X_valid, y_train, y_valid = train_test_split()`
5. test head `X_train.head()`

6. **Preliminary investigation**
   1. Shape of training data (num_rows, num_columns) `print(X_train.shape)`
   2. check Number of missing values in each column of training data

      ```py
      missing_val_count_by_column = (X_train.isnull().sum())
      print(missing_val_count_by_column[missing_val_count_by_column > 0])
      ```
   3. Function for comparing different approachesop054

      ```py
      def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
          model.fit(X_t, y_t)
          preds = model.predict(X_v)
          return mean_absolute_error(y_v, preds)
      ```

7. **Drop columns with missing values**

  ```py
  cols_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
  reduced_X_train = X_train.drop(cols_missing, axis=1)
  reduced_X_valid = X_valid.drop(cols_missing, axis=1)
  print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
  # MAE (Drop columns with missing values): 17837.82570776256
  ```

8. **Imputation** impute missing values with the mean value along each column.

  ```py
  from sklearn.impute import SimpleImputer
  my_imputer = SimpleImputer()
  imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
  imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

  imputed_X_train.columns = X_train.columns
  imputed_X_valid.columns = X_valid.columns

  print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
  # MAE (Imputation): 18062.894611872147
  ```

9. **Generate test predictions**
   1. Part A: preprocess the training and validation data.
   2. Part B: preprocess test data

      ```py
      final_imputer = SimpleImputer(strategy='median')
      final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
      final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

      final_X_train.columns = X_train.columns
      final_X_valid.columns = X_valid.columns

      final_X_test = pd.DataFrame(final_imputer.transform(X_test))
      preds_test = model.predict(final_X_test)

      output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
      output.to_csv('submission.csv', index=False)
      ```



```py

# =============================== delete the columns
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_data_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_data_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_data_full.SalePrice
X_data_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_data_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
  X, y, train_size=0.8, test_size=0.2, random_state=0)

# print the first five rows of the data.
X_train.head()
#   	MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageArea	WoodDeckSF	OpenPorchSF	EnclosedPorch	3SsnPorch	ScreenPorch	PoolArea	MiscVal	MoSold	YrSold
# Id
# 619	20	90.0	11694	9	5	2007	2007	452.0	48	0	...	774	0	108	0	0	260	0	0	7	2007
# 871	20	60.0	6600	5	5	1962	1962	0.0	0	0	...	308	0	0	0	0	0	0	0	8	2009
# 93	30	80.0	13360	5	7	1921	2006	0.0	713	0	...	432	0	0	44	0	0	0	0	8	2009
# 818	20	NaN	13265	8	5	2002	2002	148.0	1218	0	...	857	150	59	0	0	0	0	0	7	2008
# 303	20	118.0	13704	7	5	2001	2002	150.0	0	0	...	843	468	81	0	0	0	0	0	1	2006
# 5 rows × 36 columns
```


### Step 1: Preliminary investigation

```py
# Shape of training data (num_rows, num_columns)
print(X_train.shape)
# (1168, 36)
​
# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
# LotFrontage    212
# MasVnrArea       6
# GarageYrBlt     58
# dtype: int64

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)\
```

Since there are relatively few missing entries in the data
- (the column with the greatest percentage of missing values is missing less than 20% of its entries)
- we can expect that dropping columns is unlikely to yield good results.
- This is because we'd be throwing away a lot of valuable data, and so imputation will likely perform better.


### Step 2: Drop columns with missing values

> preprocess the data in `X_train` and `X_valid` to remove columns with missing values.
> Set the preprocessed DataFrames to `reduced_X_train` and reduced_X_valid, respectively.


```py
# get names of columns with missing values
cols_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_missing, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# MAE (Drop columns with missing values):
# 17837.82570776256
```



### Step 3: Imputation

> impute missing values with the mean value along each column.
> Set the preprocessed DataFrames to imputed_X_train and imputed_X_valid.
> Make sure that the column names match those in `X_train` and X_valid.


```py
from sklearn.impute import SimpleImputer

# imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# MAE (Imputation):
# 18062.894611872147
```


there are so few missing values in the dataset, we'd expect imputation to perform better than dropping columns entirely.
- However, we see that dropping columns performs slightly better!
- While this can probably partially be attributed to noise in the dataset, another potential explanation is that the imputation method is not a great match to this dataset.
- That is, maybe instead of filling in the mean value, it makes more sense to set every missing value to a value of 0, to fill in the most frequently encountered value, or to use some other method.
- For instance, consider the GarageYrBlt column (which indicates the year that the garage was built). It's likely that in some cases, a missing value could indicate a house that does not have a garage.
- Does it make more sense to fill in the median value along each column in this case? Or could we get better results by filling in the minimum value along each column?
- It's not quite clear what's best in this case, but perhaps we can rule out some options immediately - for instance, setting missing values in this column to 0 is likely to yield horrible results!


### Step 4: Generate test predictions

#### Part A

- Use the next code cell to preprocess the training and validation data.
- Set the preprocessed DataFrames to final_X_train and final_X_valid.

You can use any approach of your choosing here! in order for this step to be marked as correct, you need only ensure:
- the preprocessed DataFrames have the same number of columns,
- the preprocessed DataFrames have no missing values,
- `final_X_train` and `y_train` have the same number of rows,
- `final_X_valid` and `y_valid` have the same number of rows.


```py
# Preprocessed training and validation features
final_X_train = reduced_X_train
final_X_valid = reduced_X_valid

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))
# MAE (Your approach):
# 17837.82570776256
```


#### Part B

- Use the next code cell to preprocess your test data.
- Make sure that you use a method that agrees with how you preprocessed the training and validation data, and set the preprocessed test features to final_X_test.

Then, use the preprocessed test features and the trained model to generate test predictions in preds_test.

In order for this step to be marked correct, you need only ensure:
- the preprocessed test DataFrame has no missing values, and
- final_X_test has the same number of rows as X_test.

.
