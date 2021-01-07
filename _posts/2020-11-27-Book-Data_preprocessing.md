---
layout: post
title:  "Data preprocessing with scikit-learn (데이터전처리)"
date:   2020-11-27T21:40:52-05:00
author: nanac0516
categories: 파이썬 머신러닝 완벽 가이드
tags: data_prepocessing
---
## 데이터 전처리

문자열 값은 인코딩돼서 숫자형으로 반환하여야 한다. 일반적으로 문자열은 카테고리형 피처와 텍스트형 피처로 나뉜다.

## 데이터 인코딩

### 레이블 인코딩

카테고리 피처를 코드형 숫자 값으로 변환하는 방법이다. 예를 들어 TV : 1, 냉장고 : 2, 전자레인지 : 3 이런식으로 변환. → 숫자 값은 크고 작음에 대한 특성이 작용하기 때문에 예측 성능이 떨어지는 경우도 있다. 따라서 선형회귀같은 알고리즘에는 적용해서는 안되며, 트리 계열의 알고리즘에서는 별문제가 없다.

```python
from sklearn.preprocessing import LabelEncoder
items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
# LabelEncoder를 객체로 생성한 후, fit()과 transform()으로 레이블 인코딩 수행.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.tranform(items)
print('인코딩 변환값:',labels)
print('인코딩 클래스:',encoder.classes_)
print('디코딩 원본값:',encoder.inverse_transform(labels)
```

### 원 핫 인코딩

피처 값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 칼럼에만 1을 표시하고 나머지 칼럼에는 0을 표시하는 방법이다.

사이킷런에서 제공하는 OneHotEncoder 클래스로 쉽게 변환이 가능하나, 먼저 LabelEncoder를 이용해 숫자형으로 변환해주고 2차원의 데이터 형태로 reshape을 해줘야하는 특징이 있다.

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np
items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
# 먼저 숫자 값으로 변환을 위해 LabelEncoder로 변환합니다.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
# 2차원 데이터로 변환합니다.
labels = labels.reshape(-1,1)

#원-핫 인코딩을 적용합니다.
oh_encoder = OneHotEncode()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)
```

판다스에서는  get_dummies() 라는 API를 제공하여 숫자형으로 변환할 필요 없이 바로 변환이 가능하다.

```python
import pandas as pd
df = pd.DataFrame({'item':['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
pd.get_dummies(df)
```

## 피처 스케일링과 정규화

서로 다른 변수의 값 범위를 일정 수준으로 맞추는 작업으로 대표적으로는 표준화와 정규화가 있다.

표준화 : 데이터의 피처가 각각 평균이 0, 분산이 1인 가우시안 정규분로를 가지게끔 변환하는 것

$$x_i~new=\frac{x_i-mean(x)}{stdev(x)}$$

정규화 : 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해주는 것

$$x_i~new=\frac{x_i-min(x)}{max(x)-min(x)}$$

사이킷런의 Normalizer : 선형대수의 정규화 개념, 개별 벡터의 크기를 맞추기 위해 변환하는 것

$$x_i~new = \frac{x_i}{\sqrt{x_i^2+y_i^2+z_i^2}}$$

### StandardScaler

몇몇 알고리즘들이 데이터가 가우시안 분포를 가지고 있다고 가정하고 구현하기 때문에 표준화 작업이 매우 중요하다. - RBF 커널을 이용한 SVM, Linear Regression, Logistic Regression

```python
from sklearn.datasets import load_iris
import pandas as pd
# 붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환합니다.
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)

print('feature 들의 평균 값')
print(iris_df.mean())
print('\nfeature 들의 분산 값')
print(iris_df.var())

==============================================================================
==============================================================================

from sklearn.preprocessing import StandardScaler

# StandardScaler객체 생성
scaler = StandardScaler()
# StandardScaler로 데이터 세트 변환. fit()과 transform() 호출.
scaler.fit(iris_df)
iris_scaled = scaler.tranform(iris_df)

# transform()시 스케일 변환된 데이터 세트가 NumPy ndarray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print('feature들의 평균 값')
print(iris_df_scaled.mean())
print('\nfeature 들의 분산 값')
print(iris_df_scaled.var())
```

### MinMaxScaler

데이터값을 0과 1사이의 범위 값으로 변환시켜준다.(음수값이 있다면 -1과 1사이의 값)

```python
from sklearn.prepocessing import MinMaxScaler

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_df_scaled,columns=iris.feature_names)
print('feature들의 최솟값')
print(iris_df_scaled.min())
print('\nfeature들의 최댓값')
print(iris_df_scaled.max())
```

### + RobustScaler

 StandardScaler와 비슷하지만 더불어 이상치에 영향을 받지 않게끔 변환해주는 방법이다.

평균과 분산대신 중앙값과 사분위수를 사용한다.

$$x_i~new = \frac{x_i-median(x)}{ Q_3-Q_1(\text{Interquartile Range})}$$

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_df_scaled,columns=iris.feature_names)
print('feature들의 최솟값')
print(iris_df_scaled.min())
print('\nfeature들의 최댓값')
print(iris_df_scaled.max())
```

### 학습데이터와ㅏ 테스트 데이터의 스케일링 변환 시 유의점

1. 가능하다면 전체 데이터의 스케일링 변환을 적용한 뒤 학습과 테스트 데이터로 분리

    → 학습데이터 따로 테스트 데이터 따로 스케일링을 하면 원본과 스케일링이 다르게 되는 문제 발생

2. 1이 여의치 않다면 테스트 데이터 변환 시에는 fit()이나 fit_transform()을 적용하지 않고 학습 데이터로 이미 fit()된 Scaler 객체를 이용해 transform()으로 변환
