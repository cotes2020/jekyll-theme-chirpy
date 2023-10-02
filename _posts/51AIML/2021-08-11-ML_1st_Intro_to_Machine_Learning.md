---
title: AIML - 1st - Intro to Machine Learning
date: 2021-08-11 11:11:11 -0400
categories: [51AIML, MLNote]
tags: [AIML]
toc: true
---

- [Intro to Machine Learning](#intro-to-machine-learning)
  - [How Models Work](#how-models-work)
  - [over all](#over-all)
  - [Your First Machine Learning Model - `DecisionTreeRegressor`](#your-first-machine-learning-model---decisiontreeregressor)
  - [Model Validation](#model-validation)
    - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Underfitting and Overfitting](#underfitting-and-overfitting)
  - [Random Forests -`RandomForestRegressor`](#random-forests--randomforestregressor)
  - [example](#example)


---

# Intro to Machine Learning

---

## How Models Work

First Decision Trees
- fitting or training the model.
  - capturing patterns from data is called
  - We use data to decide how to break the houses into two groups, and then again to determine the predicted price in each group.
- The data used to fit the model is called the training data.
- After the model has been fit, apply it to new data to predict
- The point at the bottom where we make a prediction is called a leaf.


The steps to building and using a model are:
- Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- Fit: Capture patterns from provided data. This is the heart of modeling.
- Predict: Just what it sounds like
- Evaluate: Determine how accurate the model's predictions are.


---

## over all


0. notebook


```py
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

# Set up filepaths
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
```


1. get the select data

```py
import pandas as pd

# save filepath to variable
test_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

# pd.read_csv(path)
# read the data and store data in DataFrame titled test_data
test_data = pd.read_csv(test_file_path)

# .describe()
# print a summary of the data in Melbourne data
test_data.describe()

# Selecting Data for Modeling
test_data.columns

# Selecting The Prediction Target
y = test_data.Price

# Choosing "Features"
test_features = ['Rooms', 'Bathroom', 'Landsize', 'Latitude', 'Longtitude']
X = test_data[test_features]
X.describe()

# shows the top few rows.
X.head()
```

2. setup ML


```py
# defining a decision tree model with scikit-learn
# fitting it with the features and target variable.
from sklearn.tree import DecisionTreeRegressor

# ====================== Define model
# Specify a number for random_state to ensure same results each run
test_model = DecisionTreeRegressor(random_state=1)

# Fit model
test_model.fit(X, y)
# Predict
predicted_prices = test_model.predict(X)


print("Making predictions for the following 5 houses:")
print(X.head())
#    Rooms  Bathroom  Landsize  Latitude  Longtitude
# 1      2       1.0     156.0   -37.8079    144.9934
# 2      3       2.0     134.0   -37.8093    144.9944
# 4      4       1.0     120.0   -37.8072    144.9941
# 6      3       2.0     245.0   -37.8024    144.9993
# 7      2       1.0     256.0   -37.8060    144.9954


# .predict()
print("The predictions are")
print(test_model.predict(X.head()))
# The predictions are
# [1035000. 1465000. 1600000. 1876000. 1636000.]
```


3. calculate MAE
   1. use train_data to get the model
   2. use val_X to predict preds_val_y
   3. mae = mean_absolute_error(val_y, preds_val_y)


```py
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# ====================== calculate the mean absolute error 1
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y, predicted_prices)


# ====================== calculate the mean absolute error 2
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
test_model = DecisionTreeRegressor()

# Fit model
test_model.fit(train_X, train_y)

# Predict
val_predictions = test_model.predict(val_X)

# get predicted prices on validation data
print(mean_absolute_error(val_y, val_predictions))

```

4. setup the leaf

```py
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

# Check your answer
step_2.check()
```


5. Random Forests

```py
# build a random forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```







---

## Your First Machine Learning Model - `DecisionTreeRegressor`


Building Model
- Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.


```py
import pandas as pd

test_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
test_data = pd.read_csv(test_file_path)

# Selecting Data for Modeling
test_data.columns

# dropna drops missing values (think of na as "not available")
test_data = test_data.dropna(axis=0)

# Selecting The Prediction Target
y = test_data.Price

# Choosing "Features"
test_features = ['Rooms', 'Bathroom', 'Landsize', 'Latitude', 'Longtitude']
X = test_data[test_features]
X.describe()

# shows the top few rows.
X.head()

# defining a decision tree model with scikit-learn
# fitting it with the features and target variable.
from sklearn.tree import DecisionTreeRegressor

# Define model.
# Specify a number for random_state to ensure same results each run
test_model = DecisionTreeRegressor(random_state=1)

# Fit model
test_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
#    Rooms  Bathroom  Landsize  Latitude  Longtitude
# 1      2       1.0     156.0   -37.8079    144.9934
# 2      3       2.0     134.0   -37.8093    144.9944
# 4      4       1.0     120.0   -37.8072    144.9941
# 6      3       2.0     245.0   -37.8024    144.9993
# 7      2       1.0     256.0   -37.8060    144.9954
print("The predictions are")
print(test_model.predict(X.head()))
# The predictions are
# [1035000. 1465000. 1600000. 1876000. 1636000.]

```

---


## Model Validation

> The prediction error is:
>
> error=actual−predicted


### Mean Absolute Error (MAE)

summarizing model quality

To calculate MAE:

```py

import pandas as pd

# ====================== Load data
test_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
test_data = pd.read_csv(test_file_path)

# Filter rows with missing price values
filtered_test_data = test_data.dropna(axis=0)

# Choose target and features
y = filtered_test_data.Price
test_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',  'YearBuilt', 'Latitude', 'Longtitude']
X = filtered_test_data[test_features]


# ====================== Define model
from sklearn.tree import DecisionTreeRegressor
test_model = DecisionTreeRegressor()

# Fit model
test_model.fit(X, y)

# Predict
predicted_prices = test_model.predict(X)


# ====================== calculate the mean absolute error:
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y, predicted_prices)
```


The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.


The scikit-learn library has a function `train_test_split` to break up the data into two pieces.
- use some data as training data to fit the model
- and use the other data as validation data to calculate `mean_absolute_error`.


```py
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
test_model = DecisionTreeRegressor()

# Fit model
test_model.fit(train_X, train_y)

# Predict
val_predictions = test_model.predict(val_X)

# get predicted prices on validation data
print(mean_absolute_error(val_y, val_predictions))
```


---


## Underfitting and Overfitting

**Experimenting With Different Models**
Now that you have a reliable way to measure model accuracy, you can experiment with alternative models and see which gives the best predictions. But what alternatives do you have for models?

In scikit-learn's documentation that the **decision tree model** has many options. The most important options determine the tree's **depth**, a measure of how many splits it makes before coming to a prediction. This is a relatively shallow tree


**overfitting**
- When we divide the houses amongst many leaves, we also have fewer houses in each leaf.
- Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
- This is a phenomenon called overfitting, where a model matches the training data almost perfectly, but does poorly in validation and other new data.


**underfitting**
- if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.
- if a tree divides houses into only 2 or 4, each group still has a wide variety of houses.
- Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason).
- When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.


Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting. Visually, we want the low point of the (red) validation curve in the figure below.


![2q85n9s](https://i.imgur.com/WUSECIc.png)

There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.

We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:


```py

# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Data Loading Code Runs At This Point
import pandas as pd
# Load data
test_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
test_data = pd.read_csv(test_file_path)
# Filter rows with missing values
filtered_test_data = test_data.dropna(axis=0)
# Choose target and features
y = filtered_test_data.Price
test_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Latitude', 'Longtitude']
X = filtered_test_data[test_features]



# split data into training and validation data, for both features and target
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)



# The data is loaded into train_X, val_X, train_y and val_y using the code you've already seen (and which you've already written).
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# Max leaf nodes: 5  		 Mean Absolute Error:  347380
# Max leaf nodes: 50  		 Mean Absolute Error:  258171
# Max leaf nodes: 500  		 Mean Absolute Error:  243495
# Max leaf nodes: 5000  		 Mean Absolute Error:  254983
# Of the options listed, 500 is the optimal number of leaves.
```

Step 2: Fit Model Using All Data

```py
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

# Check your answer
step_2.check()
```


---


## Random Forests -`RandomForestRegressor`


- Decision trees leave you with a difficult decision.
- A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few houses at its leaf.
- But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.


- many models have clever ideas that can lead to better performance. We'll look at the **random forest** as an example.
- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree.
- It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.


```py
import pandas as pd
# Load data
test_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
test_data = pd.read_csv(test_file_path)
# Filter rows with missing values
test_data = test_data.dropna(axis=0)
# Choose target and features
y = test_data.Price
test_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Latitude', 'Longtitude']
X = test_data[test_features]


from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# build a random forest model similarly to how we built a decision tree in scikit-learn - this time using the RandomForestRegressor class instead of DecisionTreeRegressor.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```




---

## example


```py
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

# Set up filepaths
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")


# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select columns corresponding to features, and preview the data
y = home_data.SalePrice
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
```

The code cell above trains a Random Forest model on train_X and train_y.

Use the code cell below to build a Random Forest model and train it on all of X and y.

```py


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)

# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```






.
