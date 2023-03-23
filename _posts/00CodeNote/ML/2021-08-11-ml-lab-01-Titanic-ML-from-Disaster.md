---
title: ML lab 01 - Titanic-ML-from-Disaster
date: 2021-08-11 11:11:11 -0400
categories: [00CodeNote, MLNote]
tags: [ML]
toc: true
---

- [ML Lab - Titanic - Machine Learning from Disaster](#ml-lab---titanic---machine-learning-from-disaster)

---

# ML Lab - Titanic - Machine Learning from Disaster


1. the file

```py
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# /kaggle/input/titanic/train.csv
# /kaggle/input/titanic/test.csv
# /kaggle/input/titanic/gender_submission.csv

```



2. list the head info

```py
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
# PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
# 0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
# 1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
# 2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
# 3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
# 4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

```



3. calculate the rate

```py

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women*100)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men*100)

```



4. display the data

```py

train_data.describe()
test_data.describe()


c=train_data.columns.to_list()
c
# ['PassengerId',
#  'Survived',
#  'Pclass',
#  'Name',
#  'Sex',
#  'Age',
#  'SibSp',
#  'Parch',
#  'Ticket',
#  'Fare',
#  'Cabin',
#  'Embarked']


r=['PassengerId','Survived','Name','Ticket','Sex','Embarked','Cabin']
for i in r:
    c.remove(i)
c
# ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']


for col in c:
    train_data[col] = train_data[col].fillna(train_data[col].median())
    test_data[col] = test_data[col].fillna(test_data[col].median())


import matplotlib.pyplot as plt
import seaborn as sns

figure = plt.figure(figsize=(12, 6))
sns.heatmap(train_data.corr(), annot=True, cmap=plt.cm.cool)
plt.tight_layout()
plt.xlabel('Corr')
plt.show()
```

![__results___10_0](https://i.imgur.com/BTydiRL.png)

5. submit the notebook

```py
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch","Fare","Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
```





.
