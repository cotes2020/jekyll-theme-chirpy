---
title: Cross Validation (교차검증)
author: nanac0516
date: 2020-11-26 14:00:00 +0800
categories: [Book, 파이썬 머신러닝 완벽 가이드]
tags: [Cross validation]
---
##  교차검증
고정된 학습 데이터와 테스트 데이터로 분할하여 평가를 하다보면 테스트 데이터에만 최적으로 적합하게끔 편향되게 모델링을 하는 위험이 있다.
이런 문제를 개선하기 위해 등장한 것이 교차검증이다.
대부분의 ML 모델의 성능 평가는 교차 검증을 기반으로 1차 평가를 한 뒤에 최종적으로 테스트 데이터로 평가를 하는 방식으로 이루어진다.

### K 폴드 교차 검증
K개의 데이터 폴드 세트를 만들어서 각 폴드 세트에서 학습데이터와 검증데이터로 나누어 K번만큼 반복적으로 학습을 시키는 방법이다.
K=5 라고 한다면 이 5개의 평가를 평균한 결과를 가지고 예측 성과를 평가한다.
![gh-pages-sources](/assets/img/sample/k_fold_CV.png)
#### 코드
``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])

n_iter = 0

# KFold객체의 split( ) 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환  
for train_index, test_index  in kfold.split(features):
    # kfold.split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    #학습 및 예측
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n_iter += 1
    # 반복 시 마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))
```

### Stratified K 폴드
  특정 label값이 너무 많거나 혹은 너무 적은 경우 값의 분포가 한 쪽으로 치우지게 되는 불균형한 형태를 띄게 된다. 이러한 불균형한 (imbalanced) 분포도를 가진 데이터 집합을 위한 K폴드 방식이 stratified K 폴드이다. 대출 사기 데이터를 예로 볼 수 있다.\
  K 폴드로 분할된 label 데이터 세트가 전체 label 값의 분포도를 잘 반영하게끔 해주는 방법이다. 따라서 K폴드와 달리 splilt() 함수를 적용할 때 label 데이터 세트도 넣어주어야 한다.
  일반적으로 분류에서는 K-fold 대신 Stratified K-fold를 사용하여야 한다. 회귀에서는 이산형의 레이블 값이 아니라 연속형이기 때문에 Stratified K-fold가 지원되지 않는다.

#### 코드
```python
dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy=[]

# StratifiedKFold의 split( ) 호출시 반드시 레이블 데이터 셋도 추가 입력 필요  
for train_index, test_index  in skfold.split(features, label):
    # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    #학습 및 예측
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)

    # 반복 시 마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 계산
print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy))

```
### K-fold vs Stratified K-fold
##### K fold로 데이터 분할
  첫번째 폴드의 경우 학습 데이터에 label이 1 또는 2인 데이터만이 들어가 있으니 0의 경우는 전혀 학습하지 못하는 문제 발생.
``` python
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()

kfold = KFold(n_splits=3)
# kfold.split(X)는 폴드 세트를 5번 반복할 때마다 달라지는 학습/테스트 용 데이터 로우 인덱스 번호 반환.
n_iter =0
for train_index, test_index  in kfold.split(iris_df):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
```
```
## 교차 검증: 1
학습 레이블 데이터 분포:
 2    50
1    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    50
Name: label, dtype: int64
## 교차 검증: 2
학습 레이블 데이터 분포:
 2    50
0    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 1    50
Name: label, dtype: int64
## 교차 검증: 3
학습 레이블 데이터 분포:
 1    50
0    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    50
Name: label, dtype: int64
```
#### stratified K fold로 데이터 분할
  label 0,1,2인 데이터들이 각 폴드에 고르게 데이터가 분포됨.

``` python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
```
```
## 교차 검증: 1
학습 레이블 데이터 분포:
 2    33
1    33
0    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    17
1    17
0    17
Name: label, dtype: int64
## 교차 검증: 2
학습 레이블 데이터 분포:
 2    33
1    33
0    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    17
1    17
0    17
Name: label, dtype: int64
## 교차 검증: 3
학습 레이블 데이터 분포:
 2    34
1    34
0    34
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    16
1    16
0    16
Name: label, dtype: int64
```
### cross_val_score()
사이킷런에서 제공하는 API\
Classifier가 입력되면 Stratified K-fold방식으로, Regressor가 입력되면 K-fold방식으로 데이터 분할을 진행
``` python
scores=cross_val_score(dt_clf,data,label,scoring='accuracy',cv=3)
# 성능 지표는 정확도(accuracy) , 교차 검증 세트는 3개
```
